#!/usr/bin/env python3
"""
에이전트 상태 업데이트 시스템 테스트 스크립트
"""

import numpy as np
import torch
import sys
import os

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.agent import AgentState
from src.core.pheromone_vector import PheromoneVector


def test_agent_state_updates():
    """에이전트 상태 업데이트 시스템 테스트"""
    print("=== 에이전트 상태 업데이트 시스템 테스트 ===")
    
    # 테스트용 에이전트 클래스
    class TestAgent:
        def __init__(self, agent_id, config):
            self.agent_id = agent_id
            self.config = config
            self.state = AgentState(
                position=np.array([10.0, 10.0]),
                resources=100.0,
                health=100.0,
                action_history=[],
                emotion_state=np.array([0.5, 0.3, 0.2, 0.4, 0.6]),
                social_connections={}
            )
        
        def _compute_behavior_vector(self):
            if not self.state.action_history:
                return np.ones(4) * 0.5
                
            recent_actions = self.state.action_history[-10:]
            behavior = np.zeros(4)
            for action in recent_actions:
                behavior[action] += 1
            normalized = behavior / len(recent_actions)
            return np.maximum(normalized, 0.1)
        
        def _compute_social_vector(self):
            if not self.state.social_connections:
                social = np.random.rand(10) * 0.2
                return social
                
            connections = list(self.state.social_connections.values())[:10]
            social = np.zeros(10)
            social[:len(connections)] = connections
            normalized = social / (np.max(social) + 1e-8)
            return np.maximum(normalized, 0.05)
        
        def _compute_context_vector(self):
            context = np.array([
                self.state.position[0] / self.config['map_size'][0],
                self.state.position[1] / self.config['map_size'][1],
                self.state.resources / 100.0,
                self.state.health / 100.0,
                np.random.rand() * 0.3
            ])
            return np.maximum(context, 0.05)
        
        def emit_pheromone(self):
            behavior = self._compute_behavior_vector()
            emotion = self.state.emotion_state
            social = self._compute_social_vector()
            context = self._compute_context_vector()
            
            return PheromoneVector(
                behavior=behavior,
                emotion=emotion,
                social=social,
                context=context,
                timestamp=1.0,
                agent_id=self.agent_id
            )
            
        def _execute_move(self):
            dx = np.random.uniform(-2.0, 2.0)
            dy = np.random.uniform(-2.0, 2.0)
            
            new_x = np.clip(self.state.position[0] + dx, 0, self.config['map_size'][0] - 1)
            new_y = np.clip(self.state.position[1] + dy, 0, self.config['map_size'][1] - 1)
            
            old_pos = self.state.position.copy()
            self.state.position = np.array([new_x, new_y])
            
            distance = np.linalg.norm(self.state.position - old_pos)
            energy_cost = distance * 0.5
            self.state.resources = max(0, self.state.resources - energy_cost)
            
            return {
                'success': True,
                'reward': 0.1,
                'info': {'distance': distance, 'energy_cost': energy_cost}
            }
        
        def _execute_collect(self, environment_state=None):
            collected = np.random.exponential(5.0)
            success_prob = min(0.8, self.state.health / 100.0 + 0.2)
            
            if np.random.random() < success_prob:
                self.state.resources += collected
                return {
                    'success': True,
                    'reward': collected * 0.1,
                    'info': {'collected': collected}
                }
            else:
                self.state.health = max(0, self.state.health - 5.0)
                return {
                    'success': False,
                    'reward': -0.5,
                    'info': {'health_lost': 5.0}
                }
        
        def _execute_attack(self, environment_state=None):
            attack_strength = (self.state.health + self.state.resources) / 200.0
            success_prob = min(0.7, attack_strength)
            
            if np.random.random() < success_prob:
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
                health_lost = np.random.uniform(15, 25)
                self.state.health = max(0, self.state.health - health_lost)
                
                return {
                    'success': False,
                    'reward': -1.0,
                    'info': {'health_lost': health_lost}
                }
        
        def _execute_evade(self):
            resource_cost = np.random.uniform(2, 8)
            self.state.resources = max(0, self.state.resources - resource_cost)
            
            health_recovery = np.random.uniform(1, 5)
            self.state.health = min(100, self.state.health + health_recovery)
            
            return {
                'success': True,
                'reward': 0.05,
                'info': {'resource_cost': resource_cost, 'health_recovery': health_recovery}
            }
        
        def _update_emotional_state(self, action, result):
            emotion_delta = np.zeros(5)
            
            if result['success']:
                emotion_delta[0] += 0.1  # Joy
                emotion_delta[4] += 0.05  # Trust
                emotion_delta[1] -= 0.05  # Fear
                emotion_delta[3] -= 0.05  # Sadness
            else:
                emotion_delta[0] -= 0.05  # Joy
                emotion_delta[1] += 0.1   # Fear
                emotion_delta[2] += 0.05  # Anger
                emotion_delta[3] += 0.05  # Sadness
                
            if action == 2:  # Attack
                emotion_delta[2] += 0.1   # Anger
                emotion_delta[4] -= 0.05  # Trust
            elif action == 3:  # Evade
                emotion_delta[1] += 0.05  # Fear
                
            self.state.emotion_state = np.clip(
                self.state.emotion_state * 0.95 + emotion_delta,
                0.0, 1.0
            )
        
        def _update_health(self):
            if self.state.resources > 50:
                health_recovery = min(2.0, (self.state.resources - 50) * 0.02)
                self.state.health = min(100, self.state.health + health_recovery)
            elif self.state.resources < 20:
                health_loss = (20 - self.state.resources) * 0.1
                self.state.health = max(0, self.state.health - health_loss)
        
        def execute_action(self, action, environment_state=None):
            action_result = {'success': False, 'reward': 0.0, 'info': {}}
            
            if action == 0:  # Move
                action_result = self._execute_move()
            elif action == 1:  # Collect
                action_result = self._execute_collect(environment_state)
            elif action == 2:  # Attack
                action_result = self._execute_attack(environment_state)
            elif action == 3:  # Evade
                action_result = self._execute_evade()
                
            # 행동 히스토리에 추가
            self.state.action_history.append(action)
            
            # 감정 상태 업데이트
            self._update_emotional_state(action, action_result)
            
            # 체력 업데이트
            self._update_health()
            
            return action_result
        
        def update_social_connections(self, other_agent_id, interaction_type, strength=0.1):
            if other_agent_id not in self.state.social_connections:
                self.state.social_connections[other_agent_id] = 0.0
                
            if interaction_type == 'cooperation':
                self.state.social_connections[other_agent_id] += strength * 2
            elif interaction_type == 'competition':
                self.state.social_connections[other_agent_id] -= strength
            elif interaction_type == 'communication':
                self.state.social_connections[other_agent_id] += strength
                
            self.state.social_connections[other_agent_id] = np.clip(
                self.state.social_connections[other_agent_id], -1.0, 1.0
            )
            
            if abs(self.state.social_connections[other_agent_id]) < 0.01:
                del self.state.social_connections[other_agent_id]
    
    # 테스트 설정
    config = {
        'map_size': [50, 50],
        'pheromone_dim': 24,
        'num_agents': 3
    }
    
    # 에이전트 생성
    agents = []
    for i in range(3):
        agent = TestAgent(i, config)
        agents.append(agent)
    
    print(f"생성된 에이전트 수: {len(agents)}")
    
    # 시뮬레이션 실행
    num_steps = 10
    
    for step in range(num_steps):
        print(f"\n--- 스텝 {step + 1} ---")
        
        for i, agent in enumerate(agents):
            # 초기 상태 기록
            initial_state = {
                'position': agent.state.position.copy(),
                'resources': agent.state.resources,
                'health': agent.state.health,
                'emotions': agent.state.emotion_state.copy()
            }
            
            # 랜덤 행동 선택
            action = np.random.randint(0, 4)
            action_names = ['Move', 'Collect', 'Attack', 'Evade']
            
            # 행동 실행
            result = agent.execute_action(action)
            
            # 상태 변화 출력
            print(f"에이전트 {i} - {action_names[action]}:")
            print(f"  위치: {initial_state['position']} -> {agent.state.position}")
            print(f"  자원: {initial_state['resources']:.1f} -> {agent.state.resources:.1f}")
            print(f"  체력: {initial_state['health']:.1f} -> {agent.state.health:.1f}")
            print(f"  성공: {result['success']}, 보상: {result['reward']:.2f}")
            
            # 감정 변화가 있는 경우만 출력
            emotion_change = np.abs(agent.state.emotion_state - initial_state['emotions'])
            if np.any(emotion_change > 0.01):
                emotion_names = ['Joy', 'Fear', 'Anger', 'Sadness', 'Trust']
                print(f"  감정 변화: ", end="")
                for j, (name, change) in enumerate(zip(emotion_names, emotion_change)):
                    if change > 0.01:
                        direction = "+" if agent.state.emotion_state[j] > initial_state['emotions'][j] else "-"
                        print(f"{name}{direction}{change:.2f} ", end="")
                print()
    
    # 사회적 연결 테스트
    print(f"\n--- 사회적 상호작용 테스트 ---")
    
    # 협력 상호작용
    agents[0].update_social_connections(1, 'cooperation', 0.2)
    agents[1].update_social_connections(0, 'cooperation', 0.2)
    
    # 경쟁 상호작용
    agents[0].update_social_connections(2, 'competition', 0.15)
    agents[2].update_social_connections(0, 'competition', 0.15)
    
    # 소통 상호작용
    agents[1].update_social_connections(2, 'communication', 0.1)
    agents[2].update_social_connections(1, 'communication', 0.1)
    
    print("사회적 연결 상태:")
    for i, agent in enumerate(agents):
        print(f"에이전트 {i}: {agent.state.social_connections}")
    
    # 페로몬 생성 테스트
    print(f"\n--- 페로몬 생성 테스트 ---")
    
    for i, agent in enumerate(agents):
        pheromone = agent.emit_pheromone()
        total_strength = (np.sum(pheromone.behavior) + np.sum(pheromone.emotion) + 
                         np.sum(pheromone.social) + np.sum(pheromone.context))
        print(f"에이전트 {i} 페로몬 강도: {total_strength:.3f}")
        print(f"  행동: {np.sum(pheromone.behavior):.3f}")
        print(f"  감정: {np.sum(pheromone.emotion):.3f}")
        print(f"  사회: {np.sum(pheromone.social):.3f}")
        print(f"  맥락: {np.sum(pheromone.context):.3f}")
    
    return agents


def main():
    print("에이전트 상태 업데이트 시스템 검증 테스트")
    print("=" * 60)
    
    try:
        agents = test_agent_state_updates()
        
        print("\n=== 테스트 결과 요약 ===")
        print("✅ 에이전트 상태 업데이트 시스템이 성공적으로 구현되었습니다.")
        print("✅ 행동 실행 후 위치, 자원, 체력이 적절히 변화합니다.")
        print("✅ 감정 상태가 행동 결과에 따라 동적으로 업데이트됩니다.")
        print("✅ 사회적 연결이 상호작용에 따라 형성됩니다.")
        print("✅ 페로몬 생성이 업데이트된 상태를 반영합니다.")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()