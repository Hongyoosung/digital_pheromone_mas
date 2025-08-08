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
    
@ray.remote
class DistributedAgent:
    """Ray-based distributed agent"""
    def __init__(self, agent_id: int, config: Dict):
        self.agent_id = agent_id
        self.config = config
        
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
            'computation_time': 0
        }
        
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
        return PheromoneEncoder(
            input_dim=self.config['pheromone_dim'],
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
                return self.pheromone_encoder(aggregated)
        else:
            return torch.zeros(64, device=self.device)  # 출력 차원 감소: 128->64
            
    def _aggregate_pheromones(self, pheromones: List) -> torch.Tensor:
        """Aggregate multiple pheromone vectors"""
        if not pheromones:
            return torch.zeros(self.config['pheromone_dim'], device=self.device)
            
        vectors = [p.to_tensor(device=self.device) for p in pheromones]
        return torch.stack(vectors).mean(dim=0)
        
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
            action_logits = self.decision_network(combined)
            action_probs = torch.softmax(action_logits, dim=-1)
            
            action = torch.multinomial(action_probs, 1).item()
        
        # 액션 히스토리 길이 제한 (메모리 절약)
        if len(self.state.action_history) >= 20:  # 최대 20개만 보관
            self.state.action_history = self.state.action_history[-15:]
        self.state.action_history.append(action)
        
        return action
        
    def execute_action(self, action: int, environment_state: Dict = None) -> Dict:
        """Execute chosen action and update agent state"""
        action_result = {'success': False, 'reward': 0.0, 'info': {}}
        
        # 0: move, 1: collect, 2: attack, 3: evade
        if action == 0:  # Move
            action_result = self._execute_move()
        elif action == 1:  # Collect resources
            action_result = self._execute_collect(environment_state)
        elif action == 2:  # Attack/compete
            action_result = self._execute_attack(environment_state)
        elif action == 3:  # Evade/retreat
            action_result = self._execute_evade()
            
        # Update emotional state based on action result
        self._update_emotional_state(action, action_result)
        
        # Update health over time
        self._update_health()
        
        return action_result
        
    def _execute_move(self) -> Dict:
        """Execute movement action"""
        # Random movement within bounds
        dx = np.random.uniform(-2.0, 2.0)
        dy = np.random.uniform(-2.0, 2.0)
        
        new_x = np.clip(self.state.position[0] + dx, 0, self.config['map_size'][0] - 1)
        new_y = np.clip(self.state.position[1] + dy, 0, self.config['map_size'][1] - 1)
        
        old_pos = self.state.position.copy()
        self.state.position = np.array([new_x, new_y])
        
        # Movement costs energy
        distance = np.linalg.norm(self.state.position - old_pos)
        energy_cost = distance * 0.5
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
        
        # Success probability based on current position and health
        success_prob = min(0.8, self.state.health / 100.0 + 0.2)
        
        if np.random.random() < success_prob:
            self.state.resources += collected
            return {
                'success': True,
                'reward': collected * 0.1,
                'info': {'collected': collected}
            }
        else:
            # Failed collection costs health
            self.state.health = max(0, self.state.health - 5.0)
            return {
                'success': False,
                'reward': -0.5,
                'info': {'health_lost': 5.0}
            }
            
    def _execute_attack(self, environment_state: Dict = None) -> Dict:
        """Execute attack/competition action"""
        # Attack success depends on health and resources
        attack_strength = (self.state.health + self.state.resources) / 200.0
        success_prob = min(0.7, attack_strength)
        
        if np.random.random() < success_prob:
            # Successful attack - gain resources but lose health
            gained_resources = np.random.uniform(10, 30)
            health_cost = np.random.uniform(5, 15)
            
            self.state.resources += gained_resources
            self.state.health = max(0, self.state.health - health_cost)
            
            return {
                'success': True,
                'reward': gained_resources * 0.05,
                'info': {'gained_resources': gained_resources, 'health_cost': health_cost}
            }
        else:
            # Failed attack - significant health loss
            health_lost = np.random.uniform(15, 25)
            self.state.health = max(0, self.state.health - health_lost)
            
            return {
                'success': False,
                'reward': -1.0,
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
            
        if interaction_type == 'cooperation':
            self.state.social_connections[other_agent_id] += strength * 2
        elif interaction_type == 'competition':
            self.state.social_connections[other_agent_id] -= strength
        elif interaction_type == 'communication':
            self.state.social_connections[other_agent_id] += strength
            
        # Clamp values
        self.state.social_connections[other_agent_id] = np.clip(
            self.state.social_connections[other_agent_id], -1.0, 1.0
        )
        
        # Remove very weak connections
        if abs(self.state.social_connections[other_agent_id]) < 0.01:
            del self.state.social_connections[other_agent_id]
        
    def emit_pheromone(self) -> PheromoneVector:
        """Emit pheromone based on current state"""
        behavior = self._compute_behavior_vector()
        emotion = self.state.emotion_state
        social = self._compute_social_vector()
        context = self._compute_context_vector()
        
        return PheromoneVector(
            behavior=behavior,
            emotion=emotion,
            social=social,
            context=context,
            timestamp=time.time(),
            agent_id=self.agent_id
        )
        
    def _compute_behavior_vector(self) -> np.ndarray:
        """Compute behavior dimension from action history"""
        if not self.state.action_history:
            # 초기값을 더 크게 설정하여 시각화에서 감지 가능하게 함
            return np.ones(4) * 0.5
            
        recent_actions = self.state.action_history[-10:]
        behavior = np.zeros(4)
        for action in recent_actions:
            behavior[action] += 1
        # 정규화 후 최소값 보장
        normalized = behavior / len(recent_actions)
        return np.maximum(normalized, 0.1)  # 최소 0.1 이상 유지
        
    def _compute_social_vector(self) -> np.ndarray:
        """Compute social relationship vector"""
        if not self.state.social_connections:
            # 기본 사회적 활동 레벨 설정
            social = np.random.rand(10) * 0.2
            return social
            
        connections = list(self.state.social_connections.values())[:10]
        social = np.zeros(10)
        social[:len(connections)] = connections
        # 최소값 보장 및 정규화 개선
        normalized = social / (np.max(social) + 1e-8)
        return np.maximum(normalized, 0.05)  # 최소값 보장
        
    def _compute_context_vector(self) -> np.ndarray:
        """Compute environmental context vector"""
        context = np.array([
            self.state.position[0] / self.config['map_size'][0],
            self.state.position[1] / self.config['map_size'][1],
            self.state.resources / 100.0,
            self.state.health / 100.0,
            np.random.rand() * 0.3  # 환경적 요소 추가
        ])
        # 최소값 보장으로 컨텍스트 정보가 완전히 사라지지 않도록 함
        return np.maximum(context, 0.05)
        
    def communicate(self, target_agent_id: int, message: Dict):
        """Send message to another agent"""
        message_size = len(str(message).encode('utf-8'))
        self.metrics['messages_sent'] += 1
        self.metrics['bytes_sent'] += message_size
        
        # Update social connections
        if target_agent_id not in self.state.social_connections:
            self.state.social_connections[target_agent_id] = 0
        self.state.social_connections[target_agent_id] += 1
        
        return message
        
    def receive_message(self, sender_id: int, message: Dict):
        """Receive message from another agent"""
        message_size = len(str(message).encode('utf-8'))
        self.metrics['messages_received'] += 1
        self.metrics['bytes_received'] += message_size
        
        self.communication_buffer.append({
            'sender': sender_id,
            'message': message,
            'timestamp': ray.get_runtime_context().get_time()
        })
        
        # Update social connections
        if sender_id not in self.state.social_connections:
            self.state.social_connections[sender_id] = 0
        self.state.social_connections[sender_id] += 1
        
    def get_metrics(self) -> Dict:
        """Get agent metrics"""
        return self.metrics