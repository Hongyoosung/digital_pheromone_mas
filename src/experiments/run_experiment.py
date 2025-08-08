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
        
        # Phase 1: Agents perceive pheromones and decide actions (배치로 처리)
        field_dict = {pos: pheromones for pos, pheromones in self.pheromone_field.field.items()}
        
        # 배치별로 처리하여 메모리 사용량 감소
        encoded_pheromones_list = []
        actions = []
        
        for i in range(0, len(self.agents), self.batch_size):
            batch_agents = self.agents[i:i+self.batch_size]
            
            # 배치 단위로 지각 및 행동 결정
            perception_futures = [agent.perceive_pheromones.remote(field_dict) for agent in batch_agents]
            batch_encoded = ray.get(perception_futures)
            
            action_futures = [agent.decide_action.remote(encoded) for agent, encoded in zip(batch_agents, batch_encoded)]
            batch_actions = ray.get(action_futures)
            
            encoded_pheromones_list.extend(batch_encoded)
            actions.extend(batch_actions)
            
            # 배치 간 메모리 정리
            if i % (self.batch_size * 2) == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Phase 1.5: Execute actions and update agent states (배치별 처리)
        environment_state = {
            'field_density': len(self.pheromone_field.field),
            'timestep': t
        }
        
        action_results = []
        for i in range(0, len(self.agents), self.batch_size):
            batch_agents = self.agents[i:i+self.batch_size]
            batch_actions = actions[i:i+self.batch_size]
            
            execution_futures = [agent.execute_action.remote(action, environment_state) 
                               for agent, action in zip(batch_agents, batch_actions)]
            batch_results = ray.get(execution_futures)
            action_results.extend(batch_results)
        
        # Collect execution metrics
        successful_actions = sum(1 for result in action_results if result['success'])
        total_reward = sum(result['reward'] for result in action_results)
        timestep_metrics['success_rate'] = successful_actions / len(action_results)
        timestep_metrics['average_reward'] = total_reward / len(action_results)
        
        # Phase 2: Agents emit pheromones based on updated states (배치별 처리)
        new_pheromones = []
        for i in range(0, len(self.agents), self.batch_size):
            batch_agents = self.agents[i:i+self.batch_size]
            agent_futures = [agent.emit_pheromone.remote() for agent in batch_agents]
            batch_pheromones = ray.get(agent_futures)
            new_pheromones.extend(batch_pheromones)
            
        # Phase 3: Update pheromone field - 페로몬 강화 및 다중 위치 분비
        for i, pheromone in enumerate(new_pheromones):
            # 에이전트별 고정 위치 기반 분비 (더 현실적)
            agent_base_x = (i * 20) % self.config['environment']['map_size'][0]
            agent_base_y = (i * 15) % self.config['environment']['map_size'][1]
            
            # 메모리 효율성을 고려한 페로몬 분비 범위 최적화
            for dx in range(-2, 3):  # 7x7 -> 5x5 격자로 메모리 절약 및 안정성 향상
                for dy in range(-2, 3):
                    pos_x = max(0, min(self.config['environment']['map_size'][0]-1, agent_base_x + dx))
                    pos_y = max(0, min(self.config['environment']['map_size'][1]-1, agent_base_y + dy))
                    position = (pos_x, pos_y)
                    
                    # 중심에서 멀어질수록 약화된 페로몬 분비
                    # 안정적인 거리 가중치 설정
                    manhattan_distance = abs(dx) + abs(dy)
                    distance_weight = max(0.4, 1.0 - manhattan_distance * 0.12)  # 적절한 감쇠
                    
                    # 페로몬 생성량 대폭 증가
                    enhanced_pheromone = PheromoneVector(
                        # 수치 안정성을 고려한 적정 증폭 (overflow 방지)
                        behavior=np.clip(pheromone.behavior * distance_weight * 4.0, 0, 10.0),
                        emotion=np.clip(pheromone.emotion * distance_weight * 4.0, 0, 10.0),
                        social=np.clip(pheromone.social * distance_weight * 4.0, 0, 10.0),
                        context=np.clip(pheromone.context * distance_weight * 4.0, 0, 10.0),
                        timestamp=pheromone.timestamp,
                        agent_id=pheromone.agent_id
                    )
                    self.pheromone_field.deposit(position, enhanced_pheromone)
            
        # Phase 4: Apply diffusion and decay more gradually
        # 확산은 매 타임스텝이 아닌 주기적으로 적용
        # 메모리 사용량 모니터링 및 적응적 확산
        import psutil
        memory_usage = psutil.virtual_memory().percent
        
        if memory_usage < 80.0:  # 메모리 여유가 있을 때만 확산
            if t % 2 == 0:  # 2 타임스텝마다 확산 (메모리 절약)
                self.pheromone_field.diffuse(radius=4)  # 적절한 반경 (5->4)
        else:
            logger.warning(f"메모리 사용량 높음 ({memory_usage:.1f}%), 확산 건너뛰기")
            
        # 시계열 감쇠 모델 주기를 크게 완화 (페로몬 지속성 향상)
        field_tensor = self.create_field_tensor()
        # 메모리 상황에 따른 적응적 시계열 감쇠 주기
        decay_interval = 12 if memory_usage < 70.0 else 20  # 메모리 상황에 따른 주기 조정
        if field_tensor.nelement() > 0 and t % decay_interval == 0:
            try:
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):  # Mixed precision
                    # Reshape for diffusion model
                    field_tensor_reshaped = field_tensor.permute(0, 2, 3, 1).reshape(1, -1, field_tensor.shape[1])
                    
                    # Apply diffusion model
                    diffused_field_reshaped = self.diffusion_model(field_tensor_reshaped, t)
                    
                    # Reshape back
                    diffused_field = diffused_field_reshaped.reshape(
                        1, *self.config['environment']['map_size'], -1
                    ).permute(0, 3, 1, 2)
                    
                    # Convert back to float32 for field update if necessary
                    if diffused_field.dtype == torch.float16:
                        diffused_field = diffused_field.float()
                    
                    self.update_field_from_tensor(diffused_field)
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"GPU 메모리 부족으로 확산 모델 건너뜴: {e}")
                    torch.cuda.empty_cache()
                else:
                    raise e
            
        # 기본 감쇠 주기를 완화하여 페로몬 지속성 향상
        if t % 5 == 0:  # 5 타임스텝마다 적용 (2 -> 5로 완화)
            lifecycle_config = self.config['pheromone'].get('lifecycle', {})
            min_magnitude = lifecycle_config.get('min_magnitude_threshold', 0.01)
            max_lifetime = lifecycle_config.get('max_lifetime_seconds', 30.0)
            
            self.pheromone_field.decay_all(
                min_magnitude_threshold=min_magnitude,
                max_lifetime_seconds=max_lifetime
            )
        
        # Phase 5: Agent communication and social interactions (every N timesteps)
        if t > 0 and t % self.config['hyperparameters']['communication_period'][0] == 0:
            self.execute_communication_round()
            self.execute_social_interactions(actions, action_results)
            
        # Collect metrics (메모리 효율적으로)
        if field_tensor.nelement() > 0:
            # 엔트로피 계산 전 타입 변환 및 메모리 효율화
            field_numpy = field_tensor.float().cpu().numpy() if field_tensor.dtype == torch.float16 else field_tensor.cpu().numpy()
            timestep_metrics['shannon_entropy'] = self.metrics_tracker.compute_shannon_entropy(field_numpy)
            del field_numpy  # 메모리 정리
        # timestep_metrics['gpu_metrics'] = self.metrics_tracker.track_gpu_metrics() # Needs nvidia-smi
        
        return timestep_metrics
        
    def create_field_tensor(self) -> torch.Tensor:
        """Convert pheromone field to tensor with memory optimization"""
        H, W = self.config['environment']['map_size']
        p_dims = self.config['pheromone']['dimensions']
        dim_count = sum(p_dims.values())
        
        # 메모리 효율성을 위해 float16 사용 (GPU에서만)
        dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        field_tensor = torch.zeros(1, dim_count, H, W, device=self.device, dtype=dtype)
        
        # 배치 처리로 메모리 사용량 감소
        positions = list(self.pheromone_field.field.keys())
        batch_size = 50  # 한 번에 50개 위치씩 처리
        
        for i in range(0, len(positions), batch_size):
            batch_positions = positions[i:i+batch_size]
            
            for pos in batch_positions:
                pheromones = self.pheromone_field.field[pos]
                if pheromones:
                    x, y = pos
                    # Aggregate pheromones at position
                    aggregated = pheromones[0]
                    for p in pheromones[1:]:
                        aggregated = aggregated + p
                    # Convert to tensor for the field
                    tensor_vector = aggregated.to_tensor(device=self.device).to(dtype)
                        
                    # Place in tensor
                    field_tensor[0, :, x, y] = tensor_vector
                    
            # 배치별 메모리 정리
            if i % (batch_size * 4) == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        return field_tensor
        
    def update_field_from_tensor(self, tensor: torch.Tensor):
        """Update pheromone field from tensor"""
        tensor = tensor.cpu().numpy()[0]  # Remove batch dimension
        H, W = tensor.shape[1:]
        p_dims_config = self.config['pheromone']['dimensions']
        
        new_field = {}
        for x in range(H):
            for y in range(W):
                if np.any(tensor[:, x, y] > 1e-4): # 임계값 완화 (1e-3 -> 1e-4)
                    vector = tensor[:, x, y]
                    
                    # Deconstruct vector based on config
                    p_dims = list(p_dims_config.values())
                    sections = np.cumsum(p_dims)
                    
                    pheromone = PheromoneVector(
                        behavior=vector[0:sections[0]],
                        emotion=vector[sections[0]:sections[1]],
                        social=vector[sections[1]:sections[2]],
                        context=vector[sections[2]:sections[3]],
                        timestamp=time.time(),
                        agent_id=-1  # Field pheromone
                    )
                    new_field[(x, y)] = [pheromone]
                    
        self.pheromone_field.field = new_field
        
    def execute_communication_round(self):
        """Execute communication between agents"""
        num_agents = len(self.agents)
        embed_dim = self.config['attention']['embed_dim']
        
        # 1. Get agent embeddings (mock implementation)
        agent_embeddings = torch.randn(1, num_agents, embed_dim, device=self.device, dtype=torch.float32)

        # 2. Create adjacency matrix (who can talk to whom)
        # For simplicity, assume all agents can communicate with all others
        # A value of True in the mask means "do not attend"
        attn_mask = torch.zeros(num_agents, num_agents, dtype=torch.bool, device=self.device)
        
        # Repeat for batch and heads
        num_heads = self.config['attention']['num_heads']
        final_mask = attn_mask.unsqueeze(0).repeat(1 * num_heads, 1, 1)

        # 3. Compute routing decisions using the attention network
        # All agents attend to all other agents
        with torch.no_grad():  # Save memory during inference
            routing_output, attention_weights = self.attention_router(
                query=agent_embeddings, 
                key=agent_embeddings, 
                value=agent_embeddings,
                attn_mask=final_mask
            )
        
        # 4. Execute communication based on routing (mock)
        # For example, each agent communicates with the one it paid most attention to
        top_targets = torch.argmax(attention_weights, dim=-1)
        
        for i in range(num_agents):
            target_idx = top_targets[0, i].item()
            if target_idx != i:
                message = {'type': 'info_sync', 'data': np.random.randn(10)}
                # In a real scenario, you'd get the handle of the target agent
                # self.agents[i].communicate.remote(self.agents[target_idx], message)
        logger.info("Executed communication round.")
        
    def execute_social_interactions(self, actions: List[int], action_results: List[Dict]):
        """Execute social interactions between agents based on their actions"""
        num_agents = len(self.agents)
        
        # Group agents by action type for interactions
        action_groups = {0: [], 1: [], 2: [], 3: []}  # move, collect, attack, evade
        for i, action in enumerate(actions):
            action_groups[action].append(i)
            
        # Process cooperation (agents who collected resources)
        collectors = action_groups[1]
        if len(collectors) >= 2:
            # Randomly pair collectors for cooperation
            for i in range(0, len(collectors) - 1, 2):
                agent1_idx = collectors[i]
                agent2_idx = collectors[i + 1]
                
                # Both agents benefit from cooperation
                self.agents[agent1_idx].update_social_connections.remote(
                    agent2_idx, 'cooperation', 0.1
                )
                self.agents[agent2_idx].update_social_connections.remote(
                    agent1_idx, 'cooperation', 0.1
                )
                
        # Process competition (agents who attacked)
        attackers = action_groups[2]
        if len(attackers) >= 2:
            # Randomly pair attackers for competition
            for i in range(0, len(attackers) - 1, 2):
                agent1_idx = attackers[i]
                agent2_idx = attackers[i + 1]
                
                # Determine winner based on action results
                agent1_success = action_results[agent1_idx]['success']
                agent2_success = action_results[agent2_idx]['success']
                
                if agent1_success and not agent2_success:
                    # Agent 1 wins
                    self.agents[agent1_idx].update_social_connections.remote(
                        agent2_idx, 'competition', 0.05
                    )
                    self.agents[agent2_idx].update_social_connections.remote(
                        agent1_idx, 'competition', 0.1
                    )
                elif agent2_success and not agent1_success:
                    # Agent 2 wins
                    self.agents[agent2_idx].update_social_connections.remote(
                        agent1_idx, 'competition', 0.05
                    )
                    self.agents[agent1_idx].update_social_connections.remote(
                        agent2_idx, 'competition', 0.1
                    )
                else:
                    # Draw - mutual negative impact
                    self.agents[agent1_idx].update_social_connections.remote(
                        agent2_idx, 'competition', 0.08
                    )
                    self.agents[agent2_idx].update_social_connections.remote(
                        agent1_idx, 'competition', 0.08
                    )
                    
        logger.info(f"Executed social interactions: {len(collectors)} cooperators, {len(attackers)} competitors")

    def run_experiment(self) -> Dict:
        """Run complete experiment"""
        logger.info(f"Starting experiment with {len(self.agents)} agents")
        
        results = {
            'metrics': [],
            'config': self.config
        }
        
        max_timesteps = self.config['environment']['max_timesteps']
        
        for t in tqdm(range(max_timesteps), desc="Running simulation"):
            # 메모리 상태 확인 및 정리
            if not self.memory_manager.should_continue_training():
                logger.error(f"메모리 부족으로 타임스텝 {t}에서 시뮬레이션을 중단합니다.")
                break
                
            # 주기적인 메모리 모니터링 및 정리
            if t % self.memory_monitor_interval == 0:
                self.memory_manager.log_memory_usage(step=t)
                # GPU 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            timestep_metrics = self.run_timestep(t)
            self.metrics_tracker.update(**timestep_metrics)
            
            # Enhanced Visualization with training progress and agent states
            if t > 0 and t % self.config['monitoring']['log_frequency'] == 0:
                field_tensor = self.create_field_tensor()
                if field_tensor.nelement() > 0:
                    # 페로몬 필드 시각화 (개선된 버전)
                    self.visualizer.plot_pheromone_field(
                        field_tensor.cpu().numpy()[0], 
                        t, 
                        save=True
                    )
                    
                    # 훈련 진행상황 시각화 추가
                    metrics_history = self.metrics_tracker.get_metrics_history()
                    if metrics_history:
                        self.visualizer.create_training_progress_plot(
                            metrics_history, t, save=True
                        )
                    
                    # 메모리 사용량 시각화
                    if hasattr(self.memory_manager, 'memory_history') and self.memory_manager.memory_history:
                        self.visualizer.create_memory_usage_plot(
                            self.memory_manager.memory_history, t, save=True
                        )
                    
                    # 에이전트 상태 시각화
                    try:
                        # 에이전트로부터 실제 상태를 비동기적으로 가져옵니다.
                        agent_state_futures = [agent.get_state.remote() for agent in self.agents]
                        agent_states = ray.get(agent_state_futures)
                        
                        # 에이전트 상태 데이터가 비어있지 않은지 확인
                        if agent_states:
                            self.visualizer.plot_agent_states(agent_states, t, save=True)
                            
                            # 사회적 네트워크 시각화
                            # 위치 데이터를 numpy 배열로 변환
                            agent_positions = np.array([state['position'] for state in agent_states])
                            social_connections = {i: state['social_connections'] for i, state in enumerate(agent_states)}
                            
                            # 사회적 연결이 있는 경우에만 시각화
                            if any(social_connections.values()):
                                self.visualizer.plot_social_network(social_connections, agent_positions, t, save=True)
                        
                    except Exception as e:
                        logger.warning(f"에이전트 상태 시각화 실패: {e}")
                
                logger.info(f"타임스텝 {t}: 종합 시각화 완료")
                
            # Save checkpoint
            if t > 0 and t % self.config['experiment']['save_interval'] == 0:
                self.save_checkpoint(t)
                
        # Collect final metrics
        # agent_metrics = ray.get([agent.get_metrics.remote() for agent in self.agents])
        # results['agent_metrics'] = agent_metrics
        results['summary'] = self.metrics_tracker.get_summary()
        
        # Cleanup
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
        
        # wandb.Config 객체가 pickle로 저장되지 않는 문제를 해결하기 위해 dict로 변환
        if 'config' in results:
            results['config'] = dict(results['config'])
            
        with open(run_save_path, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Run {i+1} completed! Results saved to {run_save_path}")

    logger.info(f"All {args.num_runs} experiments completed!")
    
if __name__ == "__main__":
    main()
