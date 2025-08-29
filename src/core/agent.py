import numpy as np
import torch
import torch.nn as nn
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import ray
from .pheromone_vector import PheromoneVector, PheromoneEncoder


"""
에이전트의 상태, 행동 결정, 페로몬 분비 및 통신을 담당하는 분산 에이전트 클래스입니다.
"""


@dataclass
class AgentState:
    """Agent state representation"""
    position: np.ndarray
    resources: float
    health: float
    action_history: List[int]
    emotion_state: np.ndarray
    social_connections: Dict[int, float]
    
@ray.remote(num_cpus=0.25, num_gpus=0.05)
class DistributedAgent:
    """Ray-based distributed agent"""
    def __init__(self, agent_id: int, config: Dict):
        self.agent_id = agent_id
        self.config = config
        
        # 페로몬 차원 설정 저장
        self.dimensions_config = config.get('dimensions_config', None)
        
        # Set device for this agent with memory optimization
        device_str = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device_str)
        
        # GPU 메모리 최적화 설정
        if self.device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(0.3)  # 각 에이전트는 30%만 사용
        
        self.state = self._initialize_state()
        self.pheromone_encoder = self._build_encoder().to(self.device)
        self.decision_network = self._build_decision_network().to(self.device)
        self.communication_buffer = []
        self.metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'computation_time': 0,
            'successful_actions': 0,
            'total_actions': 0,
            'cooperation_events': 0,
            'competition_events': 0,
            'action_timestamps': [],
            'action_results': [],
            'cooperation_history': [],
            'real_computation_times': []
        }
        
        # Initialize action tracking
        self.last_action_success = False
        self.last_action_reward = 0.0
        self.last_computation_time = 0.0
        
    def _initialize_state(self) -> AgentState:
        """Initialize agent state"""
        return AgentState(
            position=np.random.rand(2) * self.config['map_size'],
            resources=100.0,
            health=100.0,
            action_history=[],
            emotion_state=np.zeros(5),  # 5 emotions
            social_connections={}
        )
        
    def _build_encoder(self):
        """Build pheromone encoder network with reduced size"""
        # 페로몬 차원 계산: 설정에서 각 차원의 크기 합산
        if 'pheromone' in self.config and 'dimensions' in self.config['pheromone']:
            pheromone_dim = sum(self.config['pheromone']['dimensions'].values())
        else:
            pheromone_dim = self.config.get('pheromone_dim', 16)  # 기본값 16
        
        # 디버깅을 위한 로깅
        print(f"Agent {self.agent_id}: Expected pheromone input dim = {pheromone_dim}")
            
        return PheromoneEncoder(
            input_dim=pheromone_dim,
            hidden_dim=128,  # 크기 감소: 256 -> 128
            output_dim=64    # 출력 차원 감소: 128 -> 64
        )
        
    def _build_decision_network(self):
        """Build decision-making network with reduced complexity"""
        return nn.Sequential(
            nn.Linear(64 + 10, 128),  # 입력 차원 감소: 128->64, 은닉층 감소: 256->128
            nn.ReLU(),
            nn.Dropout(0.2),  # 드롭아웃 증가로 정규화 강화
            nn.Linear(128, 64),   # 중간 층 감소: 256->128, 128->64
            nn.ReLU(),
            nn.Linear(64, 4)   # 4 actions: move, collect, attack, evade
        )
        
    def perceive_pheromones(self, pheromone_field: Dict) -> torch.Tensor:
        """Perceive pheromones from environment"""
        local_pheromones = []
        x, y = int(self.state.position[0]), int(self.state.position[1])
        
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                pos = (x + dx, y + dy)
                if pos in pheromone_field:
                    local_pheromones.extend(pheromone_field[pos])
                    
        if local_pheromones:
            # Aggregate pheromones with memory optimization
            with torch.no_grad():  # 메모리 절약
                aggregated = self._aggregate_pheromones(local_pheromones)
                # Ensure aggregated tensor is on the correct device
                if aggregated.device != self.device:
                    aggregated = aggregated.to(self.device)
                
                # 차원 안전장치: 인코더의 예상 입력 차원과 맞지 않으면 조정
                expected_dim = self.pheromone_encoder.encoder[0].in_features
                if aggregated.shape[0] != expected_dim:
                    if aggregated.shape[0] < expected_dim:
                        # 패딩
                        padding = torch.zeros(expected_dim - aggregated.shape[0], device=self.device)
                        aggregated = torch.cat([aggregated, padding])
                    else:
                        # 잘라내기
                        aggregated = aggregated[:expected_dim]
                
                return self.pheromone_encoder(aggregated)
        else:
            return torch.zeros(64, device=self.device)  # 출력 차원 감소: 128->64
            
    def _aggregate_pheromones(self, pheromones: List) -> torch.Tensor:
        """Aggregate multiple pheromone vectors"""
        # 페로몬 차원 계산
        if 'pheromone' in self.config and 'dimensions' in self.config['pheromone']:
            pheromone_dim = sum(self.config['pheromone']['dimensions'].values())
        else:
            pheromone_dim = self.config.get('pheromone_dim', 16)
            
        if not pheromones:
            return torch.zeros(pheromone_dim, device=self.device)
            
        # 설정에서 dimensions_config를 가져옴
        dimensions_config = self.config.get('pheromone', {}).get('dimensions', None)
        vectors = [p.to_tensor(device=self.device, dimensions_config=dimensions_config) for p in pheromones]
        
        # 벡터 크기가 일치하지 않으면 가장 작은 크기로 맞춤
        if vectors:
            min_size = min(v.size(0) for v in vectors)
            vectors = [v[:min_size] for v in vectors]
            aggregated = torch.stack(vectors).mean(dim=0)
            
            # 차원 안전장치: 예상 차원과 다르면 조정
            if aggregated.size(0) != pheromone_dim:
                if aggregated.size(0) < pheromone_dim:
                    # 패딩
                    padding = torch.zeros(pheromone_dim - aggregated.size(0), device=self.device)
                    aggregated = torch.cat([aggregated, padding])
                else:
                    # 잘라내기
                    aggregated = aggregated[:pheromone_dim]
                    
            print(f"Agent {self.agent_id}: Aggregated pheromone tensor size = {aggregated.size()}, expected = {pheromone_dim}")
            return aggregated
        else:
            return torch.zeros(pheromone_dim, device=self.device)
        
    def decide_action(self, encoded_pheromones: torch.Tensor) -> int:
        """Decide next action based on pheromones and state"""
        state_vector = torch.tensor([
            self.state.position[0] / self.config['map_size'][0],
            self.state.position[1] / self.config['map_size'][1],
            self.state.resources / 100.0,
            self.state.health / 100.0,
            *self.state.emotion_state,
            len(self.state.social_connections) / 100.0
        ], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():  # 추론 시 메모리 절약
            combined = torch.cat([encoded_pheromones, state_vector])
            
            # 차원 안전장치: decision_network의 예상 입력 차원과 맞지 않으면 조정
            expected_dim = self.decision_network[0].in_features
            if combined.shape[0] != expected_dim:
                if combined.shape[0] < expected_dim:
                    # 패딩
                    padding = torch.zeros(expected_dim - combined.shape[0], device=self.device)
                    combined = torch.cat([combined, padding])
                else:
                    # 잘라내기
                    combined = combined[:expected_dim]
            
            action_logits = self.decision_network(combined)
            action_probs = torch.softmax(action_logits, dim=-1)
            
            # 탐험을 위한 엡실론-그리디 정책 추가
            epsilon = 0.3  # 30% 확률로 랜덤 행동
            if np.random.random() < epsilon:
                action = np.random.randint(0, 4)
            else:
                action = torch.multinomial(action_probs, 1).item()
            
            # 상태에 따른 행동 편향 추가
            if self.state.resources < 30:  # 자원이 부족하면 수집 행동 선호
                if np.random.random() < 0.4:
                    action = 1  # collect
            elif self.state.health < 50:  # 체력이 낮으면 회피 행동 선호
                if np.random.random() < 0.4:
                    action = 3  # evade
        
        # 액션 히스토리 길이 제한 (메모리 절약)
        if len(self.state.action_history) >= 20:  # 최대 20개만 보관
            self.state.action_history = self.state.action_history[-15:]
        self.state.action_history.append(action)
        
        return action
        
    def execute_action(self, action: int, environment_state: Dict = None) -> Dict:
        """Execute chosen action and update agent state"""
        start_time = time.perf_counter()
        action_result = {'success': False, 'reward': 0.0, 'info': {}}
        
        # 실제 행동 추적 시작
        self.metrics['total_actions'] += 1
        action_timestamp = time.time()
        
        # 0: move, 1: collect, 2: attack, 3: evade
        if action == 0:  # Move
            action_result = self._execute_move(environment_state)
        elif action == 1:  # Collect resources
            action_result = self._execute_collect(environment_state)
        elif action == 2:  # Attack/compete
            action_result = self._execute_attack(environment_state)
        elif action == 3:  # Evade/retreat
            action_result = self._execute_evade()
            
        # 실제 실행 시간 측정 완료
        execution_time = time.perf_counter() - start_time
        self.metrics['real_computation_times'].append(execution_time)
        
        # 행동 결과 추적
        if action_result['success']:
            self.metrics['successful_actions'] += 1
        
        # 행동 이력 기록
        self.metrics['action_timestamps'].append(action_timestamp)
        self.metrics['action_results'].append({
            'action': action,
            'success': action_result['success'],
            'reward': action_result['reward'],
            'execution_time': execution_time,
            'timestamp': action_timestamp
        })
        
        # Store last action results for state tracking
        self.last_action_success = action_result['success']
        self.last_action_reward = action_result['reward']
        self.last_computation_time = execution_time  # 실제 계산 시간 저장
        
        # Update emotional state based on action result
        self._update_emotional_state(action, action_result)
        
        # Update health over time
        self._update_health()
        
        return action_result
        
    def _execute_move(self, environment_state: Dict = None) -> Dict:
        """Execute movement action with improved mobility"""
        # 현재 위치에서 더 자연스러운 이동
        current_x, current_y = self.state.position
        
        # 이동 방향 결정 (더 다양한 패턴)
        if np.random.random() < 0.3:  # 30% 확률로 랜덤 이동
            # 랜덤 방향 이동
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(1, 3)
            new_x = current_x + distance * np.cos(angle)
            new_y = current_y + distance * np.sin(angle)
        else:
            # 목표 지점으로의 이동 (더 자연스러운 패턴)
            target_x = np.random.uniform(0, self.config['map_size'][0])
            target_y = np.random.uniform(0, self.config['map_size'][1])
            
            # 현재 위치에서 목표까지의 방향
            dx = target_x - current_x
            dy = target_y - current_y
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance > 0:
                # 최대 이동 거리 제한
                max_step = min(2.0, distance)
                new_x = current_x + (dx / distance) * max_step
                new_y = current_y + (dy / distance) * max_step
            else:
                new_x, new_y = current_x, current_y
        
        # 맵 경계 내로 제한
        new_x = np.clip(new_x, 0, self.config['map_size'][0] - 1)
        new_y = np.clip(new_y, 0, self.config['map_size'][1] - 1)
        
        # 위치 업데이트
        self.state.position = np.array([new_x, new_y])
        
        # 이동 비용 계산
        distance = np.sqrt((new_x - current_x)**2 + (new_y - current_y)**2)
        energy_cost = distance * 0.5  # 이동 거리에 비례한 에너지 소모
        self.state.resources = max(0, self.state.resources - energy_cost)
        
        return {
            'success': True,
            'reward': 0.1,  # Small positive reward for exploration
            'info': {'distance': distance, 'energy_cost': energy_cost}
        }
        
    def _execute_collect(self, environment_state: Dict = None) -> Dict:
        """Execute resource collection action"""
        # Simulate resource collection
        collected = np.random.exponential(5.0)  # Random resource amount
        
        # Success probability based on current position and health - 현실적으로 조정
        base_success = self.state.health / 100.0
        # 자원 경쟁도 추가 (다른 에이전트가 근처에 있으면 성공률 감소)
        resource_competition_factor = max(0.1, 1.0 - len(self.state.social_connections) * 0.05)
        success_prob = min(0.75, base_success * resource_competition_factor + 0.2)  # 0.9 -> 0.75로 감소
        
        if np.random.random() < success_prob:
            self.state.resources += collected
            return {
                'success': True,
                'reward': collected * 0.15,  # 보상 증가: 0.1 -> 0.15
                'info': {'collected': collected}
            }
        else:
            # Failed collection costs health - 페널티 감소
            self.state.health = max(0, self.state.health - 3.0)  # 5.0 -> 3.0
            return {
                'success': False,
                'reward': -0.3,  # 페널티 감소: -0.5 -> -0.3
                'info': {'health_lost': 3.0}
            }
            
    def _execute_attack(self, environment_state: Dict = None) -> Dict:
        """Execute attack/competition action"""
        # Attack success depends on health and resources - 현실적 조정
        attack_strength = (self.state.health + self.state.resources) / 200.0
        # 사회적 갈등 요소 추가 (부정적 연결이 있으면 공격 성공률 증가)
        negative_connections = sum(1 for strength in self.state.social_connections.values() if strength < 0)
        aggression_boost = negative_connections * 0.1
        success_prob = min(0.65, attack_strength + 0.15 + aggression_boost)  # 0.8 -> 0.65로 감소
        
        if np.random.random() < success_prob:
            # Successful attack - gain resources but lose health
            gained_resources = np.random.uniform(15, 35)  # 보상 증가: 10-30 -> 15-35
            health_cost = np.random.uniform(3, 12)  # 비용 감소: 5-15 -> 3-12
            
            self.state.resources += gained_resources
            self.state.health = max(0, self.state.health - health_cost)
            
            return {
                'success': True,
                'reward': gained_resources * 0.08,  # 보상 증가: 0.05 -> 0.08
                'info': {'gained_resources': gained_resources, 'health_cost': health_cost}
            }
        else:
            # Failed attack - health loss 감소
            health_lost = np.random.uniform(10, 20)  # 페널티 감소: 15-25 -> 10-20
            self.state.health = max(0, self.state.health - health_lost)
            
            return {
                'success': False,
                'reward': -0.7,  # 페널티 감소: -1.0 -> -0.7
                'info': {'health_lost': health_lost}
            }
            
    def _execute_evade(self) -> Dict:
        """Execute evasion/retreat action"""
        # Evasion preserves health but costs resources
        resource_cost = np.random.uniform(2, 8)
        self.state.resources = max(0, self.state.resources - resource_cost)
        
        # Small health recovery from avoiding conflict
        health_recovery = np.random.uniform(1, 5)
        self.state.health = min(100, self.state.health + health_recovery)
        
        return {
            'success': True,
            'reward': 0.05,  # Small positive reward for staying safe
            'info': {'resource_cost': resource_cost, 'health_recovery': health_recovery}
        }
        
    def _update_emotional_state(self, action: int, result: Dict):
        """Update emotional state based on action and result"""
        # Emotional dimensions: [joy, fear, anger, sadness, trust]
        emotion_delta = np.zeros(5)
        
        if result['success']:
            emotion_delta[0] += 0.1  # Joy increases with success
            emotion_delta[4] += 0.05  # Trust in own abilities
            emotion_delta[1] -= 0.05  # Fear decreases
            emotion_delta[3] -= 0.05  # Sadness decreases
        else:
            emotion_delta[0] -= 0.05  # Joy decreases with failure
            emotion_delta[1] += 0.1   # Fear increases
            emotion_delta[2] += 0.05  # Anger with failure
            emotion_delta[3] += 0.05  # Sadness with failure
            
        # Action-specific emotional updates
        if action == 2:  # Attack
            emotion_delta[2] += 0.1   # Anger from aggressive action
            emotion_delta[4] -= 0.05  # Trust in others decreases
        elif action == 3:  # Evade
            emotion_delta[1] += 0.05  # Fear from evasive action
            
        # Apply emotional update with decay
        self.state.emotion_state = np.clip(
            self.state.emotion_state * 0.95 + emotion_delta,
            0.0, 1.0
        )
        
    def _update_health(self):
        """Update health over time"""
        # Natural health recovery if well-fed
        if self.state.resources > 50:
            health_recovery = min(2.0, (self.state.resources - 50) * 0.02)
            self.state.health = min(100, self.state.health + health_recovery)
        elif self.state.resources < 20:
            # Health deteriorates when resources are low
            health_loss = (20 - self.state.resources) * 0.1
            self.state.health = max(0, self.state.health - health_loss)
            
    def update_social_connections(self, other_agent_id: int, interaction_type: str, strength: float = 0.1):
        """Update social connections based on interactions"""
        if other_agent_id not in self.state.social_connections:
            self.state.social_connections[other_agent_id] = 0.0
            
        # 실제 상호작용 이벤트 기록
        interaction_record = {
            'other_agent_id': other_agent_id,
            'interaction_type': interaction_type,
            'strength': strength,
            'timestamp': time.time()
        }
        
        # 현실적 경쟁/갈등 메커니즘 강화
        competition_probability = 0.3  # 30% 확률로 경쟁 상황 발생
        
        if interaction_type == 'cooperation':
            self.state.social_connections[other_agent_id] += strength * 1.5  # 3 -> 1.5로 조정
            self.metrics['cooperation_events'] += 1
            interaction_record['result'] = 'positive'
        elif interaction_type == 'competition':
            # 경쟁 페널티 강화
            penalty_multiplier = np.random.uniform(1.0, 2.5)  # 0.5 -> 1.0~2.5로 강화
            self.state.social_connections[other_agent_id] -= strength * penalty_multiplier
            self.metrics['competition_events'] += 1
            interaction_record['result'] = 'negative'
        elif interaction_type == 'communication':
            # 통신도 때로는 갈등을 야기할 수 있음
            if np.random.random() < competition_probability:
                # 통신 중 오해나 갈등 발생
                self.state.social_connections[other_agent_id] -= strength * 0.3
                interaction_record['result'] = 'conflict_in_communication'
                self.metrics['competition_events'] += 1
            else:
                self.state.social_connections[other_agent_id] += strength * 0.8
                interaction_record['result'] = 'positive_communication'
            
        # 협력 이력 기록
        self.metrics['cooperation_history'].append(interaction_record)
        
        # 연결 강도 제한
        self.state.social_connections[other_agent_id] = np.clip(
            self.state.social_connections[other_agent_id], -1.0, 1.0
        )
            
        # Clamp values
        self.state.social_connections[other_agent_id] = np.clip(
            self.state.social_connections[other_agent_id], -1.0, 1.0
        )
        
        # Remove very weak connections
        if abs(self.state.social_connections[other_agent_id]) < 0.01:
            del self.state.social_connections[other_agent_id]
        
    def emit_pheromone(self) -> PheromoneVector:
        """Emit pheromone based on current state"""
        # 설정에서 차원 정보 가져오기
        dimensions_config = self.config.get('pheromone', {}).get('dimensions', {
            'behavior': 4, 'emotion': 5, 'social': 10, 'context': 5
        })
        
        # 각 차원의 크기에 맞춰 벡터 생성
        behavior_size = dimensions_config.get('behavior', 4)
        emotion_size = dimensions_config.get('emotion', 5)
        social_size = dimensions_config.get('social', 10)
        context_size = dimensions_config.get('context', 5)
        
        # 기본 벡터 계산하고 크기에 맞춰 자르거나 패딩
        behavior_full = self._compute_behavior_vector()
        behavior = self._resize_vector(behavior_full, behavior_size)
        
        emotion_full = self.state.emotion_state
        emotion = self._resize_vector(emotion_full, emotion_size)
        
        social_full = self._compute_social_vector()
        social = self._resize_vector(social_full, social_size)
        
        context_full = self._compute_context_vector()
        context = self._resize_vector(context_full, context_size)
        
        return PheromoneVector(
            behavior=behavior,
            emotion=emotion,
            social=social,
            context=context,
            timestamp=time.time(),
            agent_id=self.agent_id
        )
    
    def _resize_vector(self, vector: np.ndarray, target_size: int) -> np.ndarray:
        """벡터를 목표 크기로 조정 (자르거나 패딩)"""
        if len(vector) == target_size:
            return vector
        elif len(vector) > target_size:
            return vector[:target_size]  # 자르기
        else:
            # 패딩
            padding = np.zeros(target_size - len(vector))
            return np.concatenate([vector, padding])
        
    def _compute_behavior_vector(self) -> np.ndarray:
        """Compute behavior dimension from action history"""
        if not self.state.action_history:
            # 초기값을 크게 증가하여 페로몬 활동도 향상
            # 수치 안정성을 고려한 적정 초기값 설정
            return np.ones(4) * 2.5  # 1.2 -> 2.5로 증가 (페로몬 강도 향상)
            
        recent_actions = self.state.action_history[-10:]
        behavior = np.zeros(4)
        for action in recent_actions:
            behavior[action] += 1
        # 정규화 후 최소값 대폭 증가
        normalized = behavior / len(recent_actions)
        # 안정적인 스케일링 및 overflow 방지
        scaled = normalized * 4.0 + 0.5  # 2.5 -> 4.0으로 증가
        return np.clip(scaled, 0.5, 5.0)  # 범위 확장: 0.2~3.0 -> 0.5~5.0
        
    def _compute_social_vector(self) -> np.ndarray:
        """Compute social relationship vector"""
        if not self.state.social_connections:
            # 기본 사회적 활동 레벨 대폭 증가
            # 사회적 차원 안정화
            social = np.random.rand(10) * 2.0 + 0.5  # 1.0 -> 2.0으로 증가, 0.3~1.3 -> 0.5~2.5
            return social
            
        connections = list(self.state.social_connections.values())[:10]
        social = np.zeros(10)
        social[:len(connections)] = connections
        # 최소값 보장 및 정규화 개선
        normalized = social / (np.max(social) + 1e-8)
        # 사회적 연결의 안정적 스케일링
        enhanced = normalized * 3.0 + 0.3  # 1.8 -> 3.0으로 증가
        return np.clip(enhanced, 0.3, 4.0)  # 범위 확장: 0.1~2.0 -> 0.3~4.0
        
    def _compute_context_vector(self) -> np.ndarray:
        """Compute environmental context vector"""
        context = np.array([
            self.state.position[0] / self.config['map_size'][0],
            self.state.position[1] / self.config['map_size'][1],
            self.state.resources / 100.0,
            self.state.health / 100.0,
            np.random.rand() * 1.2 + 0.5  # 환경적 요소 안정화 (0.3~1.1 -> 0.5~1.7)
        ])
        # 최소값 보장을 크게 증가하여 컨텍스트 정보 강화
        # 컨텍스트 차원 안정화 및 NaN 방지
        enhanced_context = context * 3.5 + 0.3  # 2.0 -> 3.5로 증가
        enhanced_context = np.nan_to_num(enhanced_context, nan=0.3, posinf=4.0, neginf=0.3)
        return np.clip(enhanced_context, 0.3, 4.5)  # 범위 확장: 0.1~2.5 -> 0.3~4.5
        
    def communicate(self, target_agent_id: int, message: Dict):
        """Send message to another agent"""
        message_size = len(str(message).encode('utf-8'))
        self.metrics['messages_sent'] += 1
        self.metrics['bytes_sent'] += message_size
        
        # Update social connections with realistic interaction
        interaction_success = np.random.random() < 0.7  # 70% 통신 성공률
        if interaction_success:
            self.update_social_connections(target_agent_id, 'communication', 0.1)
        else:
            # 통신 실패시 약간의 부정적 영향
            self.update_social_connections(target_agent_id, 'competition', 0.05)
        
        return message
        
    def receive_message(self, sender_id: int, message: Dict):
        """Receive message from another agent"""
        message_size = len(str(message).encode('utf-8'))
        self.metrics['messages_received'] += 1
        self.metrics['bytes_received'] += message_size
        
        self.communication_buffer.append({
            'sender': sender_id,
            'message': message,
            'timestamp': time.time()
        })
        
        # Update social connections with message understanding
        message_understanding = np.random.random() < 0.8  # 80% 메시지 이해율
        if message_understanding:
            self.update_social_connections(sender_id, 'communication', 0.08)
        else:
            # 메시지 오해시 부정적 영향
            self.update_social_connections(sender_id, 'competition', 0.03)
        
    def get_metrics(self) -> Dict:
        """Get agent metrics"""
        return self.metrics

    def get_state(self) -> Dict[str, Any]:
        """Returns the current state of the agent as a serializable dictionary."""
        return {
            "position": self.state.position.tolist(),
            "resources": self.state.resources,
            "health": self.state.health,
            "action_history": self.state.action_history,
            "emotion_state": self.state.emotion_state.tolist(),
            "social_connections": self.state.social_connections,
            "success": getattr(self, 'last_action_success', False),
            "reward": getattr(self, 'last_action_reward', 0.0),
            "messages_sent": self.metrics.get('messages_sent', 0),
            "messages_received": self.metrics.get('messages_received', 0),
            "bytes_sent": self.metrics.get('bytes_sent', 0),
            "bytes_received": self.metrics.get('bytes_received', 0),
            "computation_time": getattr(self, 'last_computation_time', 0.0),
            "communication_buffer": self.communication_buffer,
            # 실제 행동 메트릭 추가
            "successful_actions": self.metrics.get('successful_actions', 0),
            "total_actions": self.metrics.get('total_actions', 0),
            "cooperation_events": self.metrics.get('cooperation_events', 0),
            "competition_events": self.metrics.get('competition_events', 0),
            "real_computation_times": self.metrics.get('real_computation_times', []),
            "action_results": self.metrics.get('action_results', []),
            "cooperation_history": self.metrics.get('cooperation_history', [])
        }