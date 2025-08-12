import os
import yaml
import torch
import numpy as np
import ray
from typing import Dict, List
import time
import logging
from tqdm import tqdm
import wandb
import argparse
import torch.nn.functional as F
import pickle
import os
import json

from src.core.agent import DistributedAgent
from src.core.pheromone_vector import PheromoneField, PheromoneVector
from src.models.diffusion_model import TemporalDiffusionModel
from src.models.attention_network import DistributedAttentionRouter
from src.utils.normalization import PheromoneNormalizer
from src.utils.metrics import MetricsTracker
from src.utils.visualization import ExperimentVisualizer
from src.utils.memory_manager import MemoryManager, BatchMemoryOptimizer

# CUDA 로깅 간소화
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # CUDA 비동기 실행 허용
os.environ['NCCL_DEBUG'] = 'WARN'  # NCCL 로깅 레벨 감소
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'  # C++ 스택 트레이스 비활성화

# 로깅 레벨을 WARNING으로 설정하여 CUDA 로그 감소
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 메인 로거만 INFO 유지

# PyTorch 로깅 비활성화
torch.backends.cudnn.benchmark = False  # 벤치마크 비활성화
if hasattr(torch._C, '_set_print_stacktraces_on_fatal_signal'):
    torch._C._set_print_stacktraces_on_fatal_signal(False)


"""
설정 파일을 기반으로 전체 시뮬레이션을 설정하고 실행하는 메인 스크립트입니다.
"""


class ExperimentRunner:
    """Main experiment runner"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        # GPU 사용 가능성 확인 및 디바이스 설정
        requested_device = self.config['experiment']['device']
        if 'cuda' in requested_device and not torch.cuda.is_available():
            logger.warning(f"CUDA가 사용 불가능합니다. CPU로 전환합니다.")
            self.device = torch.device('cpu')
            self.config['experiment']['device'] = 'cpu'
        else:
            self.device = torch.device(requested_device)
            if 'cuda' in requested_device:
                # GPU 메모리 최적화 설정
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.7)  # GPU 메모리 70%로 제한
                # 로깅 비활성화된 상태에서만 GPU 정보 출력
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f"GPU 디바이스 확인됨: {torch.cuda.get_device_name(0)}")
        
        self.setup_experiment()
        
    def setup_experiment(self):
        """Initialize experiment components"""
        # Initialize Ray with reduced resource allocation
        if ray.is_initialized():
            ray.shutdown()
        
        # 메모리 사용량 최적화를 위한 Ray 설정
        ray_config = {
            "num_cpus": min(4, os.cpu_count()),  # CPU 사용량 제한
            "object_store_memory": 2 * 1024**3,  # 2GB object store
            "_system_config": {
                "object_spilling_config": json.dumps({
                    "type": "filesystem",
                    "params": {"directory_path": "/tmp/spill"}
                })
            }
        }
        
        if "cuda" in self.config['experiment']['device']:
            ray_config["num_gpus"] = 1  # GPU 리소스는 정수 단위로 할당
        else:
            ray_config["num_gpus"] = 0
            
        ray.init(**ray_config)
        
        # Initialize models and ensure CUDA usage
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
            
        self.diffusion_model = TemporalDiffusionModel(
            decay_factor=self.config['pheromone']['decay_rate']
        ).to(self.device)
        self.attention_router = DistributedAttentionRouter(
            embed_dim=self.config['attention']['embed_dim'],
            num_heads=self.config['attention']['num_heads']
        ).to(self.device)
        
        # Ensure models are in eval mode for inference and use GPU
        self.diffusion_model.eval()
        self.attention_router.eval()
        
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'모델이 {self.device} 장치로 이동되었습니다.')
        
        # Initialize utilities
        self.normalizer = PheromoneNormalizer()
        self.metrics_tracker = MetricsTracker()
        log_dir = self.config['experiment']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        self.visualizer = ExperimentVisualizer(log_dir)
        
        # Initialize memory management
        memory_config = self.config.get('memory_management', {})
        self.memory_manager = MemoryManager(
            max_memory_percent=memory_config.get('max_memory_percent', 85.0),
            warning_memory_percent=memory_config.get('warning_memory_percent', 75.0),
            cleanup_threshold_percent=memory_config.get('cleanup_threshold_percent', 80.0)
        )
        self.batch_optimizer = BatchMemoryOptimizer(self.memory_manager)
        self.memory_monitor_interval = memory_config.get('memory_monitor_interval', 50)
        
        # Initialize pheromone field
        map_size = tuple(self.config['environment']['map_size'])
        self.pheromone_field = PheromoneField(map_size, self.config['pheromone']['decay_rate'])
        
        # Initialize agents with reduced count for memory efficiency
        self.agents = self.create_agents()
        
        # 메모리 최적화를 위한 배치 크기 동적 조정
        self.batch_size = min(len(self.agents), 4)  # 최대 4개 에이전트씩 배치 처리
        
        # Setup logging
        if self.config['monitoring'].get('use_wandb', False):
            wandb.init(project="digital_pheromone_mas", config=self.config)
            
    def create_agents(self) -> List:
        """Create distributed agents"""
        num_agents = self.config['environment']['num_agents']
        p_dims = self.config['pheromone']['dimensions']
        pheromone_dim = p_dims['behavior'] + p_dims['emotion'] + p_dims['social'] + p_dims['context']
        
        use_gpu = "cuda" in self.config['experiment']['device']
        
        # GPU 및 CPU 리소스 할당 최적화
        agent_options = {"num_cpus": 0.25}  # CPU 요구량을 낮춰 더 많은 액터 생성 허용
        if use_gpu and num_agents > 0:
            # Ensure total GPU allocation does not exceed 1.0
            gpu_per_agent = 1.0 / num_agents
            agent_options["num_gpus"] = gpu_per_agent
            
        AgentActor = DistributedAgent.options(**agent_options)

        agents = []
        for i in range(num_agents):
            agent_config = {
                'map_size': self.config['environment']['map_size'],
                'pheromone_dim': pheromone_dim,
                'num_agents': num_agents,
                'device': self.config['experiment']['device']
            }
            agents.append(AgentActor.remote(i, agent_config))
            
        return agents
        
    def run_timestep(self, t: int) -> Dict:
        """Run single timestep of simulation"""
        timestep_metrics = {}
        comp_load = {} # 연산 부하 측정용

        # Phase 1: Agents perceive pheromones and decide actions
        start_time = time.perf_counter()
        field_dict = {pos: pheromones for pos, pheromones in self.pheromone_field.field.items()}
        
        encoded_pheromones_list = []
        actions = []
        
        for i in range(0, len(self.agents), self.batch_size):
            batch_agents = self.agents[i:i+self.batch_size]
            perception_futures = [agent.perceive_pheromones.remote(field_dict) for agent in batch_agents]
            batch_encoded = ray.get(perception_futures)
            action_futures = [agent.decide_action.remote(encoded) for agent, encoded in zip(batch_agents, batch_encoded)]
            batch_actions = ray.get(action_futures)
            encoded_pheromones_list.extend(batch_encoded)
            actions.extend(batch_actions)
            if i % (self.batch_size * 2) == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        comp_load['action_decision'] = time.perf_counter() - start_time

        # Phase 1.5: Execute actions and update agent states
        start_time = time.perf_counter()
        environment_state = {'field_density': len(self.pheromone_field.field), 'timestep': t}
        action_results = []
        for i in range(0, len(self.agents), self.batch_size):
            batch_agents = self.agents[i:i+self.batch_size]
            batch_actions = actions[i:i+self.batch_size]
            execution_futures = [agent.execute_action.remote(action, environment_state) for agent, action in zip(batch_agents, batch_actions)]
            batch_results = ray.get(execution_futures)
            action_results.extend(batch_results)
        comp_load['action_execution'] = time.perf_counter() - start_time
        
        successful_actions = sum(1 for result in action_results if result['success'])
        total_reward = sum(result['reward'] for result in action_results)
        timestep_metrics['success_rate'] = successful_actions / len(action_results) if action_results else 0
        timestep_metrics['average_reward'] = total_reward / len(action_results) if action_results else 0
        
        # Phase 2: Agents emit pheromones
        start_time = time.perf_counter()
        new_pheromones = []
        for i in range(0, len(self.agents), self.batch_size):
            batch_agents = self.agents[i:i+self.batch_size]
            agent_futures = [agent.emit_pheromone.remote() for agent in batch_agents]
            batch_pheromones = ray.get(agent_futures)
            new_pheromones.extend(batch_pheromones)
        comp_load['pheromone_emission'] = time.perf_counter() - start_time

        # Phase 3: Update pheromone field
        start_time = time.perf_counter()
        for i, pheromone in enumerate(new_pheromones):
            agent_base_x = (i * 20) % self.config['environment']['map_size'][0]
            agent_base_y = (i * 15) % self.config['environment']['map_size'][1]
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    pos_x = max(0, min(self.config['environment']['map_size'][0]-1, agent_base_x + dx))
                    pos_y = max(0, min(self.config['environment']['map_size'][1]-1, agent_base_y + dy))
                    position = (pos_x, pos_y)
                    manhattan_distance = abs(dx) + abs(dy)
                    distance_weight = max(0.4, 1.0 - manhattan_distance * 0.12)
                    enhanced_pheromone = PheromoneVector(
                        behavior=np.clip(pheromone.behavior * distance_weight * 4.0, 0, 10.0),
                        emotion=np.clip(pheromone.emotion * distance_weight * 4.0, 0, 10.0),
                        social=np.clip(pheromone.social * distance_weight * 4.0, 0, 10.0),
                        context=np.clip(pheromone.context * distance_weight * 4.0, 0, 10.0),
                        timestamp=pheromone.timestamp,
                        agent_id=pheromone.agent_id
                    )
                    self.pheromone_field.deposit(position, enhanced_pheromone)
        comp_load['pheromone_deposit'] = time.perf_counter() - start_time

        # Phase 4: Diffusion and Decay
        start_time = time.perf_counter()
        import psutil
        memory_usage = psutil.virtual_memory().percent
        if memory_usage < 80.0 and t % 2 == 0:
            self.pheromone_field.diffuse(radius=4)
        
        field_tensor = self.create_field_tensor()
        decay_interval = 12 if memory_usage < 70.0 else 20
        if field_tensor.nelement() > 0 and t % decay_interval == 0:
            try:
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                    field_tensor_reshaped = field_tensor.permute(0, 2, 3, 1).reshape(1, -1, field_tensor.shape[1])
                    diffused_field_reshaped = self.diffusion_model(field_tensor_reshaped, t)
                    diffused_field = diffused_field_reshaped.reshape(1, *self.config['environment']['map_size'], -1).permute(0, 3, 1, 2)
                    if diffused_field.dtype == torch.float16:
                        diffused_field = diffused_field.float()
                    self.update_field_from_tensor(diffused_field)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"GPU 메모리 부족으로 확산 모델 건너뜀: {e}")
                    torch.cuda.empty_cache()
                else:
                    raise e
        
        if t % 5 == 0:
            lifecycle_config = self.config['pheromone'].get('lifecycle', {})
            self.pheromone_field.decay_all(
                min_magnitude_threshold=lifecycle_config.get('min_magnitude_threshold', 0.01),
                max_lifetime_seconds=lifecycle_config.get('max_lifetime_seconds', 30.0)
            )
        comp_load['diffusion_decay'] = time.perf_counter() - start_time
        timestep_metrics['computation_overhead'] = comp_load

        # Phase 5: Communication and Social Interactions
        if t > 0 and t % self.config['hyperparameters']['communication_period'][0] == 0:
            start_time = time.perf_counter()
            comm_metrics = self.execute_communication_round()
            self.metrics_tracker.update(communication_overhead=comm_metrics)
            comp_load['communication'] = time.perf_counter() - start_time

            start_time = time.perf_counter()
            self.execute_social_interactions(actions, action_results)
            comp_load['social_interaction'] = time.perf_counter() - start_time
            
        # Collect metrics
        if field_tensor.nelement() > 0:
            field_numpy = field_tensor.float().cpu().numpy() if field_tensor.dtype == torch.float16 else field_tensor.cpu().numpy()
            timestep_metrics['shannon_entropy'] = self.metrics_tracker.compute_shannon_entropy(field_numpy)
            del field_numpy
        
        return timestep_metrics
        
    def create_field_tensor(self) -> torch.Tensor:
        """Convert pheromone field to tensor with memory optimization"""
        H, W = self.config['environment']['map_size']
        p_dims = self.config['pheromone']['dimensions']
        dim_count = sum(p_dims.values())
        
        dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        field_tensor = torch.zeros(1, dim_count, H, W, device=self.device, dtype=dtype)
        
        positions = list(self.pheromone_field.field.keys())
        batch_size = 50
        
        for i in range(0, len(positions), batch_size):
            batch_positions = positions[i:i+batch_size]
            for pos in batch_positions:
                pheromones = self.pheromone_field.field[pos]
                if pheromones:
                    x, y = pos
                    aggregated = pheromones[0]
                    for p in pheromones[1:]:
                        aggregated = aggregated + p
                    tensor_vector = aggregated.to_tensor(device=self.device).to(dtype)
                    field_tensor[0, :, x, y] = tensor_vector
            if i % (batch_size * 4) == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        return field_tensor
        
    def update_field_from_tensor(self, tensor: torch.Tensor):
        """Update pheromone field from tensor"""
        tensor = tensor.cpu().numpy()[0]
        H, W = tensor.shape[1:]
        p_dims_config = self.config['pheromone']['dimensions']
        
        new_field = {}
        for x in range(H):
            for y in range(W):
                if np.any(tensor[:, x, y] > 1e-4):
                    vector = tensor[:, x, y]
                    p_dims = list(p_dims_config.values())
                    sections = np.cumsum(p_dims)
                    pheromone = PheromoneVector(
                        behavior=vector[0:sections[0]],
                        emotion=vector[sections[0]:sections[1]],
                        social=vector[sections[1]:sections[2]],
                        context=vector[sections[2]:sections[3]],
                        timestamp=time.time(),
                        agent_id=-1
                    )
                    new_field[(x, y)] = [pheromone]
                    
        self.pheromone_field.field = new_field
        
    def execute_communication_round(self) -> Dict:
        """Execute communication and return overhead metrics."""
        num_agents = len(self.agents)
        embed_dim = self.config['attention']['embed_dim']
        
        agent_embeddings = torch.randn(1, num_agents, embed_dim, device=self.device, dtype=torch.float32)
        attn_mask = torch.zeros(num_agents, num_agents, dtype=torch.bool, device=self.device)
        num_heads = self.config['attention']['num_heads']
        final_mask = attn_mask.unsqueeze(0).repeat(1 * num_heads, 1, 1)

        with torch.no_grad():
            _, attention_weights = self.attention_router(
                query=agent_embeddings, key=agent_embeddings, value=agent_embeddings, attn_mask=final_mask
            )
        
        top_targets = torch.argmax(attention_weights, dim=-1)
        
        messages = []
        for i in range(num_agents):
            target_idx = top_targets[0, i].item()
            if target_idx != i:
                message_data = np.random.randn(10)
                message = {'type': 'info_sync', 'data': message_data, 'size': message_data.nbytes}
                messages.append(message)
        
        logger.info(f"Executed communication round with {len(messages)} messages.")
        return self.metrics_tracker.track_communication_overhead(messages)
        
    def execute_social_interactions(self, actions: List[int], action_results: List[Dict]):
        """Execute social interactions between agents based on their actions"""
        action_groups = {0: [], 1: [], 2: [], 3: []}
        for i, action in enumerate(actions):
            if action in action_groups:
                action_groups[action].append(i)
            
        collectors = action_groups.get(1, [])
        if len(collectors) >= 2:
            for i in range(0, len(collectors) - 1, 2):
                agent1_idx, agent2_idx = collectors[i], collectors[i+1]
                self.agents[agent1_idx].update_social_connections.remote(agent2_idx, 'cooperation', 0.1)
                self.agents[agent2_idx].update_social_connections.remote(agent1_idx, 'cooperation', 0.1)
                
        attackers = action_groups.get(2, [])
        if len(attackers) >= 2:
            for i in range(0, len(attackers) - 1, 2):
                agent1_idx, agent2_idx = attackers[i], attackers[i+1]
                agent1_success = action_results[agent1_idx]['success']
                agent2_success = action_results[agent2_idx]['success']
                if agent1_success and not agent2_success:
                    self.agents[agent1_idx].update_social_connections.remote(agent2_idx, 'competition', 0.05)
                    self.agents[agent2_idx].update_social_connections.remote(agent1_idx, 'competition', 0.1)
                elif agent2_success and not agent1_success:
                    self.agents[agent2_idx].update_social_connections.remote(agent1_idx, 'competition', 0.05)
                    self.agents[agent1_idx].update_social_connections.remote(agent2_idx, 'competition', 0.1)
                else:
                    self.agents[agent1_idx].update_social_connections.remote(agent2_idx, 'competition', 0.08)
                    self.agents[agent2_idx].update_social_connections.remote(agent1_idx, 'competition', 0.08)
                    
        logger.info(f"Executed social interactions: {len(collectors)} cooperators, {len(attackers)} competitors")

    def run_experiment(self) -> Dict:
        """Run complete experiment"""
        logger.info(f"Starting experiment with {len(self.agents)} agents")
        
        results = {
            'metrics': [],
            'config': self.config,
            'learning_logs': [],
            'start_time': time.time()
        }
        
        max_timesteps = self.config['environment']['max_timesteps']
        final_timestep = 0
        
        for t in tqdm(range(max_timesteps), desc="Running simulation"):
            final_timestep = t
            if not self.memory_manager.should_continue_training():
                logger.error(f"메모리 부족으로 타임스텝 {t}에서 시뮬레이션을 중단합니다.")
                break
                
            if t % self.memory_monitor_interval == 0:
                self.memory_manager.log_memory_usage(step=t)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            timestep_metrics = self.run_timestep(t)
            self.metrics_tracker.update(**timestep_metrics)
            
            if t > 0 and t % 100 == 0:
                learning_log = self.generate_learning_log(t, timestep_metrics)
                results['learning_logs'].append(learning_log)
                logger.info(f"[학습 로그 T={t:04d}] 성공률: {timestep_metrics.get('success_rate', 0):.3f}, "
                           f"평균 보상: {timestep_metrics.get('average_reward', 0):.3f}, "
                           f"엔트로피: {timestep_metrics.get('shannon_entropy', 0):.3f}")
            
            if t > 0 and t % self.config['monitoring']['log_frequency'] == 0:
                field_tensor = self.create_field_tensor()
                if field_tensor.nelement() > 0:
                    self.visualizer.plot_pheromone_field(field_tensor.cpu().numpy()[0], t, save=True)
                    metrics_history = self.metrics_tracker.get_metrics_history()
                    if metrics_history:
                        self.visualizer.create_training_progress_plot(metrics_history, t, save=True)
                    if hasattr(self.memory_manager, 'memory_history') and self.memory_manager.memory_history:
                        self.visualizer.create_memory_usage_plot(self.memory_manager.memory_history, t, save=True)
                    try:
                        agent_state_futures = [agent.get_state.remote() for agent in self.agents]
                        agent_states = ray.get(agent_state_futures)
                        if agent_states:
                            self.visualizer.plot_agent_states(agent_states, t, save=True)
                            agent_positions = np.array([state['position'] for state in agent_states])
                            social_connections = {i: state['social_connections'] for i, state in enumerate(agent_states)}
                            if any(social_connections.values()):
                                self.visualizer.plot_social_network(social_connections, agent_positions, t, save=True)
                    except Exception as e:
                        logger.warning(f"에이전트 상태 시각화 실패: {e}")
                logger.info(f"타임스텝 {t}: 종합 시각화 완료")
                
            if t > 0 and t % self.config['experiment']['save_interval'] == 0:
                self.save_checkpoint(t)
                
        results['summary'] = self.metrics_tracker.get_summary()
        
        final_summary = self.generate_training_summary(results, final_timestep + 1)
        results['training_summary'] = final_summary
        logger.info("훈련 완료 - 요약 보고서가 생성되었습니다.")
        
        summary_path = os.path.join(self.config['experiment']['log_dir'], 'training_summary.txt')
        self.save_training_summary_to_file(final_summary, summary_path)
        
        ray.shutdown()
        
        return results
        
    def save_checkpoint(self, timestep: int):
        """Save experiment checkpoint"""
        checkpoint = {
            'timestep': timestep,
            'diffusion_model': self.diffusion_model.state_dict(),
            'attention_router': self.attention_router.state_dict(),
            'metrics': self.metrics_tracker.metrics_history,
            'config': self.config
        }
        save_path = os.path.join(self.config['experiment']['log_dir'], f"checkpoint_t{timestep}.pt")
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint at timestep {timestep}")
    
    def generate_learning_log(self, timestep: int, metrics: Dict) -> Dict:
        """100 스텝마다 학습 결과 로그를 생성"""
        import psutil
        memory_info = psutil.virtual_memory()
        gpu_memory_info = None
        if torch.cuda.is_available():
            gpu_memory_info = {
                'allocated': torch.cuda.memory_allocated(self.device) / 1024**3,
                'reserved': torch.cuda.memory_reserved(self.device) / 1024**3,
                'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1024**3
            }
        
        field_stats = {
            'total_positions': len(self.pheromone_field.field),
            'active_positions': sum(1 for pos in self.pheromone_field.field.values() if pos),
            'field_density': len(self.pheromone_field.field) / (self.config['environment']['map_size'][0] * self.config['environment']['map_size'][1])
        }
        
        learning_log = {
            'timestep': timestep,
            'timestamp': time.time(),
            'performance_metrics': {
                'success_rate': metrics.get('success_rate', 0),
                'average_reward': metrics.get('average_reward', 0),
                'shannon_entropy': metrics.get('shannon_entropy', 0),
            },
            'system_metrics': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': memory_info.percent,
                'memory_available': memory_info.available / 1024**3,
                'gpu_memory': gpu_memory_info
            },
            'pheromone_field': field_stats,
            'agent_count': len(self.agents)
        }
        
        return learning_log
    
    def generate_training_summary(self, results: Dict, completed_timesteps: int) -> Dict:
        """훈련 완료 후 종합 요약 보고서 생성"""
        summary = {
            'experiment_info': {
                'total_timesteps': self.config['environment']['max_timesteps'],
                'completed_timesteps': completed_timesteps,
                'agent_count': len(self.agents),
                'start_time': results.get('start_time', time.time()),
                'end_time': time.time()
            },
            'performance_analysis': {},
            'learning_progression': {},
            'system_performance': {},
            'communication_overhead': {},
            'computation_load': {},
            'recommendations': []
        }
        
        metrics_summary = results.get('summary', {})
        
        if results.get('learning_logs'):
            logs = results['learning_logs']
            success_rates = [log['performance_metrics']['success_rate'] for log in logs]
            rewards = [log['performance_metrics']['average_reward'] for log in logs]
            entropies = [log['performance_metrics']['shannon_entropy'] for log in logs]
            
            summary['performance_analysis'] = {
                'success_rate': {'initial': success_rates[0], 'final': success_rates[-1], 'max': max(success_rates), 'average': np.mean(success_rates), 'improvement': success_rates[-1] - success_rates[0]} if success_rates else {},
                'reward': {'initial': rewards[0], 'final': rewards[-1], 'max': max(rewards), 'average': np.mean(rewards), 'improvement': rewards[-1] - rewards[0]} if rewards else {},
                'entropy': {'initial': entropies[0], 'final': entropies[-1], 'average': np.mean(entropies), 'stability': np.std(entropies)} if entropies else {}
            }
            
            if len(logs) > 1:
                mid_point = len(logs) // 2
                early_performance = np.mean(success_rates[:mid_point])
                late_performance = np.mean(success_rates[mid_point:])
                summary['learning_progression'] = {
                    'early_phase_performance': early_performance,
                    'late_phase_performance': late_performance,
                    'learning_trend': 'improving' if late_performance > early_performance else 'declining',
                    'convergence_indicator': np.std(success_rates[-5:]) if len(success_rates) >= 5 else float('inf')
                }
            
            cpu_usages = [log['system_metrics']['cpu_usage'] for log in logs]
            memory_usages = [log['system_metrics']['memory_usage'] for log in logs]
            summary['system_performance'] = {
                'average_cpu_usage': np.mean(cpu_usages), 'max_cpu_usage': max(cpu_usages),
                'average_memory_usage': np.mean(memory_usages), 'max_memory_usage': max(memory_usages),
                'resource_efficiency': 'high' if np.mean(cpu_usages) < 70 and np.mean(memory_usages) < 80 else 'moderate'
            }

        if 'communication_overhead' in metrics_summary:
            summary['communication_overhead'] = metrics_summary['communication_overhead']
        
        if 'computation_overhead' in metrics_summary and metrics_summary['computation_overhead']:
            # 안전하게 mean 키에 접근
            comp_overhead_data = metrics_summary['computation_overhead']
            if isinstance(comp_overhead_data, dict) and comp_overhead_data:
                # 각 서브키에 대한 데이터가 있는 경우에만 처리
                computation_load = {}
                for sub_key, stats in comp_overhead_data.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        # 히스토리에서 실제 값들을 찾아 평균 계산
                        values = [d.get(sub_key) for d in self.metrics_tracker.metrics_history.get('computation_overhead', []) 
                                 if isinstance(d, dict) and sub_key in d and d[sub_key] is not None]
                        if values:
                            computation_load[sub_key] = np.mean(values)
                
                if computation_load:
                    summary['computation_load'] = computation_load
                else:
                    summary['computation_load'] = comp_overhead_data

        recommendations = []
        if summary.get('performance_analysis') and summary['performance_analysis'].get('success_rate'):
            if summary['performance_analysis']['success_rate'].get('improvement', 0) > 0.1:
                recommendations.append("훈련이 효과적으로 진행되어 성공률이 크게 향상되었습니다.")
        if summary.get('system_performance', {}).get('max_memory_usage', 0) > 90:
            recommendations.append("메모리 사용량이 높습니다. 배치 크기 감소나 메모리 최적화를 권장합니다.")
        summary['recommendations'] = recommendations
        
        return summary
    
    def save_training_summary_to_file(self, summary: Dict, filepath: str):
        """요약 보고서를 텍스트 파일로 저장"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("4D 디지털 페로몬 MAS 훈련 요약 보고서\n")
            f.write("=" * 60 + "\n\n")
            
            exp_info = summary.get('experiment_info', {})
            f.write("🔬 실험 정보\n")
            f.write("-" * 30 + "\n")
            f.write(f"총 타임스텝: {exp_info.get('total_timesteps', 'N/A')}\n")
            f.write(f"완료된 타임스텝: {exp_info.get('completed_timesteps', 'N/A')}\n")
            f.write(f"에이전트 수: {exp_info.get('agent_count', 'N/A')}\n")
            if 'start_time' in exp_info and 'end_time' in exp_info:
                duration = exp_info['end_time'] - exp_info['start_time']
                f.write(f"실험 소요 시간: {duration/60:.1f}분\n")
            f.write("\n")
            
            perf = summary.get('performance_analysis', {})
            if perf:
                f.write("📊 성능 분석\n")
                f.write("-" * 30 + "\n")
                sr = perf.get('success_rate', {})
                f.write(f"성공률: 초기 {sr.get('initial', 0):.3f}, 최종 {sr.get('final', 0):.3f}, 평균 {sr.get('average', 0):.3f}, 향상도 {sr.get('improvement', 0):+.3f}\n")
                rw = perf.get('reward', {})
                f.write(f"보상: 초기 {rw.get('initial', 0):.3f}, 최종 {rw.get('final', 0):.3f}, 평균 {rw.get('average', 0):.3f}, 향상도 {rw.get('improvement', 0):+.3f}\n")
                en = perf.get('entropy', {})
                f.write(f"엔트로피: 초기 {en.get('initial', 0):.3f}, 최종 {en.get('final', 0):.3f}, 평균 {en.get('average', 0):.3f}, 안정성 {en.get('stability', 0):.3f}\n\n")

            comm_overhead = summary.get('communication_overhead', {})
            if comm_overhead:
                f.write("📡 통신 오버헤드\n")
                f.write("-" * 30 + "\n")
                f.write(f"평균 메시지 수/주기: {comm_overhead.get('mean', {}).get('total_messages', 0):.2f}\n")
                f.write(f"평균 데이터 크기/주기 (bytes): {comm_overhead.get('mean', {}).get('total_bytes', 0):.2f}\n")
                f.write(f"초당 평균 메시지 수: {comm_overhead.get('mean', {}).get('messages_per_second', 0):.2f}\n\n")

            comp_load = summary.get('computation_load', {})
            if comp_load:
                f.write("⚙️ 연산 부하 분석 (평균 시간/스텝, 초)\n")
                f.write("-" * 30 + "\n")
                for key, val in comp_load.items():
                    f.write(f"  - {key}: {val:.6f}\n")
                f.write("\n")

            sys_perf = summary.get('system_performance', {})
            if sys_perf:
                f.write("💻 시스템 성능\n")
                f.write("-" * 30 + "\n")
                f.write(f"평균 CPU: {sys_perf.get('average_cpu_usage', 0):.1f}%, 최대 CPU: {sys_perf.get('max_cpu_usage', 0):.1f}%\n")
                f.write(f"평균 메모리: {sys_perf.get('average_memory_usage', 0):.1f}%, 최대 메모리: {sys_perf.get('max_memory_usage', 0):.1f}%\n\n")

            recommendations = summary.get('recommendations', [])
            if recommendations:
                f.write("💡 추천 사항\n")
                f.write("-" * 30 + "\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")
            
            f.write("=" * 60 + "\n")
            f.write("보고서 생성 완료\n")
            f.write("=" * 60 + "\n")
        
        logger.info(f"훈련 요약 보고서가 저장되었습니다: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Run Digital Pheromone MAS Simulation")
    parser.add_argument('--config', type=str, default='config/experiment_config.yaml',
                        help='Path to the experiment configuration file.')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of times to run the experiment.')
    args = parser.parse_args()
    
    all_results = []
    for i in range(args.num_runs):
        logger.info(f"--- Starting Run {i+1}/{args.num_runs} ---")
        runner = ExperimentRunner(config_path=args.config)
        results = runner.run_experiment()
        all_results.append(results)
        
        run_save_path = os.path.join(runner.config['experiment']['log_dir'], f"run_{i+1}_results.pkl")
        
        if 'config' in results:
            results['config'] = dict(results['config'])
            
        with open(run_save_path, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Run {i+1} completed! Results saved to {run_save_path}")

    logger.info(f"All {args.num_runs} experiments completed!")
    
if __name__ == "__main__":
    main()
