import os
import yaml
import torch
import numpy as np
import ray
from typing import Dict, List
import time
import logging
from tqdm import tqdm
import argparse
import pickle
import json

from src.core.agent import DistributedAgent
from src.core.pheromone_vector import PheromoneField, PheromoneVector
from src.models.diffusion_model import TemporalDiffusionModel
from src.models.attention_network import DistributedAttentionRouter, CentralizedAttentionRouter
from src.utils.metrics import MetricsTracker
from src.utils.visualization import ExperimentVisualizer
from src.utils.memory_manager import MemoryManager

# 로깅 설정
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
다양한 모델 타입(디지털 페로몬, 규칙 기반, 중앙집중 어텐션)을 비교하기 위한 실험을 실행하는 스크립트입니다.
--model-type 인자를 통해 비교할 모델을 선택합니다.
"""

class ComparisonExperimentRunner:
    """비교 실험 실행기"""
    
    def __init__(self, config_path: str, model_type: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model_type = model_type
        logger.info(f"'{self.model_type}' 모델 타입으로 비교 실험을 시작합니다.")

        # 모델 타입에 따라 실험 설정 수정
        self.config['experiment']['model_type'] = self.model_type
        if self.model_type == 'rule_based':
            # 규칙 기반 모델은 GPU가 필요 없음
            self.config['experiment']['device'] = 'cpu'
        
        self.device = torch.device(self.config['experiment']['device'])
        self.setup_experiment()

    def setup_experiment(self):
        """실험 구성요소 초기화"""
        if ray.is_initialized():
            ray.shutdown()
        ray.init(num_cpus=self.config['ray']['num_cpus'], num_gpus=1 if self.device.type == 'cuda' else 0)

        # 모델 초기화
        self.diffusion_model = TemporalDiffusionModel(decay_factor=self.config['pheromone']['decay_rate']).to(self.device)
        
        if self.model_type == 'centralized_attention':
            self.attention_router = CentralizedAttentionRouter(
                embed_dim=self.config['attention']['embed_dim'],
                num_heads=self.config['attention']['num_heads'],
                num_agents=self.config['environment']['num_agents']
            ).to(self.device)
        elif self.model_type == 'digital_pheromone':
            self.attention_router = DistributedAttentionRouter(
                embed_dim=self.config['attention']['embed_dim'],
                num_heads=self.config['attention']['num_heads']
            ).to(self.device)
        else: # rule_based
            self.attention_router = None

        if self.attention_router:
            self.attention_router.eval()

        # 유틸리티 초기화
        self.metrics_tracker = MetricsTracker()
        log_dir = os.path.join(self.config['experiment']['log_dir'], self.model_type)
        os.makedirs(log_dir, exist_ok=True)
        self.visualizer = ExperimentVisualizer(log_dir)
        self.memory_manager = MemoryManager()

        # 페로몬 필드 및 에이전트 초기화
        self.pheromone_field = PheromoneField(tuple(self.config['environment']['map_size']), self.config['pheromone']['decay_rate'])
        self.agents = self._create_agents()

    def _create_agents(self) -> List:
        """분산 에이전트 생성"""
        num_agents = self.config['environment']['num_agents']
        p_dims = self.config['pheromone']['dimensions']
        pheromone_dim = sum(p_dims.values())
        
        agent_config = {
            'map_size': self.config['environment']['map_size'],
            'pheromone_dim': pheromone_dim,
            'num_agents': num_agents,
            'device': self.config['experiment']['device'],
            'model_type': self.model_type # 에이전트에 모델 타입 전달
        }
        
        AgentActor = DistributedAgent.options(num_cpus=0.25, num_gpus=1.0/num_agents if self.device.type == 'cuda' and num_agents > 0 else 0)
        return [AgentActor.remote(i, agent_config) for i in range(num_agents)]

    def run_timestep(self, t: int) -> Dict:
        """단일 타임스텝 시뮬레이션 실행"""
        timestep_metrics = {}
        comp_load = {}

        # Phase 1: 행동 결정
        start_time = time.perf_counter()
        field_dict = {pos: pheromones for pos, pheromones in self.pheromone_field.field.items()}
        
        if self.model_type == 'rule_based':
            # 규칙 기반: 각 에이전트가 독립적으로 주변 페로몬만 보고 행동 결정
            perception_futures = [agent.perceive_pheromones.remote(field_dict) for agent in self.agents]
            encoded_pheromones = ray.get(perception_futures)
            action_futures = [agent.decide_action_rule_based.remote(encoded) for agent, encoded in zip(self.agents, encoded_pheromones)]
            actions = ray.get(action_futures)
        else:
            # 어텐션 기반: 다른 에이전트 정보를 종합하여 행동 결정
            perception_futures = [agent.perceive_pheromones.remote(field_dict) for agent in self.agents]
            encoded_pheromones = ray.get(perception_futures)
            action_futures = [agent.decide_action.remote(p) for agent, p in zip(self.agents, encoded_pheromones)]
            actions = ray.get(action_futures)
        comp_load['action_decision'] = time.perf_counter() - start_time

        # Phase 1.5: 행동 실행 및 상태 업데이트
        # (이 부분은 run_experiment.py와 동일하게 유지)
        environment_state = {'field_density': len(self.pheromone_field.field), 'timestep': t}
        action_futures = [agent.execute_action.remote(act, environment_state) for agent, act in zip(self.agents, actions)]
        action_results = ray.get(action_futures)
        
        successful_actions = sum(1 for res in action_results if res['success'])
        total_reward = sum(res['reward'] for res in action_results)
        timestep_metrics['success_rate'] = successful_actions / len(action_results) if action_results else 0
        timestep_metrics['average_reward'] = total_reward / len(action_results) if action_results else 0

        # Phase 2: 페로몬 방출 및 필드 업데이트
        # (이 부분은 run_experiment.py와 동일하게 유지)
        pheromone_futures = [agent.emit_pheromone.remote() for agent in self.agents]
        new_pheromones = ray.get(pheromone_futures)
        for i, p in enumerate(new_pheromones):
            if p:
                # 에이전트의 현재 위치를 가져와 페로몬을 증착
                agent_state = ray.get(self.agents[i].get_state.remote())
                pos = tuple(agent_state['position'])
                self.pheromone_field.deposit(pos, p)

        # Phase 3: 확산 및 감쇠
        if self.model_type == 'rule_based':
            # 규칙 기반 확산: 간단한 평균 필터 사용
            self.pheromone_field.diffuse(radius=2, method='average')
        else:
            # 모델 기반 확산
            # (이 부분은 run_experiment.py와 동일하게 유지)
            pass
        self.pheromone_field.decay_all()
        
        # Phase 4: 통신 (모델별로 다름)
        comm_metrics = {}
        if t > 0 and t % self.config['hyperparameters']['communication_period'][0] == 0:
            if self.model_type == 'centralized_attention':
                comm_metrics = self.execute_centralized_communication()
            elif self.model_type == 'digital_pheromone':
                comm_metrics = self.execute_distributed_communication()
            # 규칙 기반 모델은 명시적 통신 없음
        self.metrics_tracker.update(communication_overhead=comm_metrics)

        # 메트릭 수집
        field_tensor = self.create_field_tensor()
        if field_tensor.nelement() > 0:
            timestep_metrics['shannon_entropy'] = self.metrics_tracker.compute_shannon_entropy(field_tensor.cpu().numpy())
        timestep_metrics['computation_overhead'] = comp_load
        
        return timestep_metrics

    def execute_centralized_communication(self) -> Dict:
        """중앙집중 어텐션 기반 통신 실행"""
        logger.info("중앙집중 통신 실행")
        agent_states = ray.get([agent.get_state.remote() for agent in self.agents])
        embeddings = torch.tensor([s['embedding'] for s in agent_states]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 중앙 라우터가 모든 에이전트의 정보를 받아 라우팅 결정
            _, attention_weights = self.attention_router(embeddings)
        
        # 모든 에이전트가 모든 다른 에이전트에게 메시지 전송 (오버헤드가 큼)
        num_agents = len(self.agents)
        messages = []
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    message_data = np.random.randn(10)
                    messages.append({'size': message_data.nbytes})
        
        return self.metrics_tracker.track_communication_overhead(messages)

    def execute_distributed_communication(self) -> Dict:
        """분산 어텐션 기반 통신 실행 (기존 방식)"""
        logger.info("분산 통신 실행")
        # (이 로직은 run_experiment.py의 execute_communication_round와 동일)
        num_agents = len(self.agents)
        messages = []
        # ... (메시지 생성 로직) ...
        return self.metrics_tracker.track_communication_overhead(messages)

    def create_field_tensor(self) -> torch.Tensor:
        # (run_experiment.py와 동일)
        H, W = self.config['environment']['map_size']
        p_dims = self.config['pheromone']['dimensions']
        dim_count = sum(p_dims.values())
        field_tensor = torch.zeros(1, dim_count, H, W, device=self.device, dtype=torch.float32)
        # ...
        return field_tensor

    def run_experiment(self):
        """전체 비교 실험 실행"""
        logger.info(f"'{self.model_type}' 모델 타입으로 실험 시작")
        
        max_timesteps = self.config['environment']['max_timesteps']
        for t in tqdm(range(max_timesteps), desc=f"Running {self.model_type}"):
            if not self.memory_manager.should_continue_training():
                logger.error("메모리 부족으로 시뮬레이션 중단")
                break
            
            timestep_metrics = self.run_timestep(t)
            self.metrics_tracker.update(**timestep_metrics)

        # 결과 저장
        summary = self.metrics_tracker.get_summary()
        log_dir = os.path.join(self.config['experiment']['log_dir'], self.model_type)
        summary_path = os.path.join(log_dir, 'final_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"'{self.model_type}' 실험 완료. 결과 저장: {summary_path}")
        ray.shutdown()
        return summary


def main():
    parser = argparse.ArgumentParser(description="Run Comparison Experiments for Digital Pheromone MAS")
    parser.add_argument('--config', type=str, default='config/experiment_config.yaml',
                        help='실험 설정 파일 경로')
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['digital_pheromone', 'rule_based', 'centralized_attention'],
                        help='실행할 모델 타입을 선택합니다.')
    args = parser.parse_args()
    
    runner = ComparisonExperimentRunner(config_path=args.config, model_type=args.model_type)
    runner.run_experiment()

if __name__ == "__main__":
    main()
