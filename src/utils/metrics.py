import numpy as np
import torch
from typing import List, Dict, Tuple
from scipy.stats import entropy
import time


"""
실험 성능을 측정하고 추적하기 위한 다양한 지표(엔트로피, 수렴 속도, 통신 오버헤드 등)를 관리합니다.
"""


class MetricsTracker:
    """Track and compute experiment metrics"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.metrics_history = {
            'shannon_entropy': [],
            'convergence_speed': [],
            'communication_overhead': [],
            'network_load': [],
            'gpu_utilization': [],
            'message_latency': [],
            'throughput': [],
            # 연구 계획서 명시 메트릭들
            'information_transfer_efficiency': [],
            'learning_convergence_epochs': [],
            'network_bandwidth_usage': [],
            'computation_overhead': [],
            'attention_entropy': [],
            'pheromone_diffusion_rate': [],
            'agent_cooperation_index': [],
            'social_network_density': [],
            'environmental_adaptation_score': []
        }
        self.start_time = time.time()
        
    def compute_shannon_entropy(self, pheromone_field: np.ndarray) -> float:
        """
        Compute Shannon entropy of pheromone distribution
        
        Args:
            pheromone_field: Pheromone field array
            
        Returns:
            Shannon entropy value
        """
        # Flatten and normalize to probability distribution
        flat_field = pheromone_field.flatten()
        flat_field = flat_field[flat_field > 0]  # Remove zeros
        
        if len(flat_field) == 0:
            return 0.0
            
        # Normalize to probability distribution
        prob_dist = flat_field / np.sum(flat_field)
        
        # Compute entropy
        return entropy(prob_dist, base=2)
        
    def track_convergence(self, loss_values: List[float]) -> float:
        """
        Track learning convergence speed
        
        Args:
            loss_values: List of loss values
            
        Returns:
            Convergence rate (epochs to 95% reduction)
        """
        if len(loss_values) < 2:
            return float('inf')
            
        initial_loss = loss_values[0]
        target_loss = initial_loss * 0.05  # 95% reduction
        
        for epoch, loss in enumerate(loss_values):
            if loss <= target_loss:
                return epoch
                
        return len(loss_values)
        
    def track_communication_overhead(self, messages: List[Dict]) -> Dict:
        """
        Track communication overhead metrics
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Communication overhead statistics
        """
        if not messages:
            return {'total_messages': 0, 'total_bytes': 0, 'avg_size': 0}
            
        total_messages = len(messages)
        total_bytes = sum(msg.get('size', 0) for msg in messages)
        avg_size = total_bytes / total_messages if total_messages > 0 else 0
        
        return {
            'total_messages': total_messages,
            'total_bytes': total_bytes,
            'avg_size': avg_size,
            'messages_per_second': total_messages / (time.time() - self.start_time)
        }
        
    def track_network_load(self, agent_metrics: List[Dict]) -> Dict:
        """
        Track network load from agent metrics
        
        Args:
            agent_metrics: List of metrics from each agent
            
        Returns:
            Network load statistics
        """
        if not agent_metrics:
            return {}
            
        total_sent = sum(m.get('bytes_sent', 0) for m in agent_metrics)
        total_received = sum(m.get('bytes_received', 0) for m in agent_metrics)
        total_computation = sum(m.get('computation_time', 0) for m in agent_metrics)
        
        elapsed_time = time.time() - self.start_time
        
        return {
            'bandwidth_usage': (total_sent + total_received) / elapsed_time,
            'avg_computation_time': total_computation / len(agent_metrics),
            'total_data_transferred': total_sent + total_received,
            'load_balance_ratio': np.std([m.get('computation_time', 0) for m in agent_metrics])
        }
        
    def track_gpu_metrics(self) -> Dict:
        """Track GPU utilization and memory"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            return {
                'gpu_utilization': util.gpu,
                'memory_utilization': util.memory,
                'memory_used': mem_info.used / 1024**3,  # GB
                'memory_total': mem_info.total / 1024**3  # GB
            }
        except:
            return {
                'gpu_utilization': 0,
                'memory_utilization': 0,
                'memory_used': 0,
                'memory_total': 0
            }
            
    def update(self, **kwargs):
        """Update metrics with new values"""
        for key, value in kwargs.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
                
    def get_summary(self) -> Dict:
        """Get summary statistics of all metrics"""
        summary = {}
        for key, values in self.metrics_history.items():
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'last': values[-1]
                }
        return summary
        
    def get_metrics_history(self) -> Dict:
        """Get the complete metrics history"""
        return self.metrics_history.copy()
        
    def compute_information_transfer_efficiency(self, sent_messages: List[Dict], successful_actions: int, total_actions: int) -> float:
        """
        연구 계획서 명시: 정보 전달 효율성 측정
        
        Args:
            sent_messages: 송신된 메시지 리스트
            successful_actions: 성공한 행동 수
            total_actions: 전체 행동 수
            
        Returns:
            정보 전달 효율성 점수 (0~1)
        """
        if total_actions == 0:
            return 0.0
            
        # 기본 성공률
        success_rate = successful_actions / total_actions
        
        # 메시지 효율성 (적은 메시지로 높은 성과)
        message_efficiency = 1.0 / (1.0 + len(sent_messages) / max(total_actions, 1))
        
        # 종합 효율성
        return 0.7 * success_rate + 0.3 * message_efficiency
        
    def compute_learning_convergence_epochs(self, loss_history: List[float], convergence_threshold: float = 0.01) -> int:
        """
        학습 수렴 속도 측정 (연구 계획서 명시)
        
        Args:
            loss_history: 손실 히스토리
            convergence_threshold: 수렴 임계값
            
        Returns:
            수렴까지 필요한 에포크 수
        """
        if len(loss_history) < 10:
            return len(loss_history)
            
        # 최근 10개 에포크의 변화율이 임계값 이하면 수렴
        for i in range(9, len(loss_history)):
            recent_losses = loss_history[i-9:i+1]
            if len(recent_losses) >= 10:
                change_rate = abs(recent_losses[-1] - recent_losses[0]) / max(recent_losses[0], 1e-8)
                if change_rate < convergence_threshold:
                    return i + 1
                    
        return len(loss_history)
        
    def compute_network_bandwidth_usage(self, agent_metrics: List[Dict], time_window: float) -> Dict:
        """
        네트워크 대역폭 사용량 분석 (연구 계획서 명시)
        
        Args:
            agent_metrics: 에이전트별 통신 메트릭
            time_window: 측정 시간 창 (초)
            
        Returns:
            네트워크 사용량 통계
        """
        total_bytes = sum(m.get('bytes_sent', 0) + m.get('bytes_received', 0) for m in agent_metrics)
        total_messages = sum(m.get('messages_sent', 0) + m.get('messages_received', 0) for m in agent_metrics)
        
        return {
            'bandwidth_mbps': (total_bytes * 8) / (time_window * 1_000_000),  # Mbps
            'message_rate': total_messages / time_window,  # messages/sec
            'avg_message_size': total_bytes / max(total_messages, 1),  # bytes
            'peak_bandwidth': max((m.get('bytes_sent', 0) + m.get('bytes_received', 0)) * 8 / time_window / 1_000_000 for m in agent_metrics)
        }
        
    def compute_attention_entropy(self, attention_weights: np.ndarray) -> float:
        """
        어텐션 가중치의 엔트로피 측정
        
        Args:
            attention_weights: 어텐션 가중치 행렬
            
        Returns:
            정규화된 엔트로피 값
        """
        if attention_weights.size == 0:
            return 0.0
            
        # 각 행(에이전트)별 어텐션 분포의 엔트로피
        entropies = []
        for row in attention_weights:
            if np.sum(row) > 0:
                prob_dist = row / np.sum(row)
                row_entropy = entropy(prob_dist, base=2)
                entropies.append(row_entropy)
                
        return np.mean(entropies) if entropies else 0.0
        
    def compute_pheromone_diffusion_rate(self, field_before: np.ndarray, field_after: np.ndarray) -> float:
        """
        페로몬 확산 속도 측정
        
        Args:
            field_before: 확산 전 페로몬 필드
            field_after: 확산 후 페로몬 필드
            
        Returns:
            확산 속도 지표
        """
        if field_before.size == 0 or field_after.size == 0:
            return 0.0
            
        # 공간적 분산의 변화
        before_variance = np.var(field_before)
        after_variance = np.var(field_after)
        
        # 활성 셀의 개수 변화
        active_before = np.sum(field_before > 0.01)
        active_after = np.sum(field_after > 0.01)
        
        # 확산 지표 (분산 증가 + 활성 영역 확장)
        variance_change = (after_variance - before_variance) / max(before_variance, 1e-8)
        spread_change = (active_after - active_before) / max(active_before, 1)
        
        return 0.6 * variance_change + 0.4 * spread_change
        
    def compute_agent_cooperation_index(self, social_connections: Dict) -> float:
        """
        에이전트 간 협력 지수 계산
        
        Args:
            social_connections: 에이전트 간 사회적 연결 정보
            
        Returns:
            협력 지수 (0~1)
        """
        if not social_connections:
            return 0.0
            
        total_connections = 0
        positive_connections = 0
        
        for agent_id, connections in social_connections.items():
            for other_id, strength in connections.items():
                total_connections += 1
                if strength > 0:
                    positive_connections += 1
                    
        return positive_connections / max(total_connections, 1)
        
    def compute_social_network_density(self, social_connections: Dict, num_agents: int) -> float:
        """
        사회 네트워크 밀도 계산
        
        Args:
            social_connections: 사회적 연결 정보
            num_agents: 총 에이전트 수
            
        Returns:
            네트워크 밀도 (0~1)
        """
        if num_agents <= 1:
            return 0.0
            
        max_possible_edges = num_agents * (num_agents - 1)
        actual_edges = 0
        
        for connections in social_connections.values():
            actual_edges += len(connections)
            
        return actual_edges / max_possible_edges
        
    def compute_environmental_adaptation_score(self, agent_states: List[Dict], environment_stats: Dict) -> float:
        """
        환경 적응 점수 계산
        
        Args:
            agent_states: 에이전트 상태 리스트
            environment_stats: 환경 통계 정보
            
        Returns:
            적응 점수 (0~1)
        """
        if not agent_states:
            return 0.0
            
        # 생존률 (체력 > 0)
        survival_rate = sum(1 for state in agent_states if state.get('health', 0) > 0) / len(agent_states)
        
        # 자원 효율성 (평균 자원 보유량)
        avg_resources = np.mean([state.get('resources', 0) for state in agent_states])
        max_possible_resources = environment_stats.get('total_resources', 1000) / len(agent_states)
        resource_efficiency = min(avg_resources / max_possible_resources, 1.0)
        
        # 환경적 위험 회피 (위험 지역 근처 에이전트 비율)
        danger_avoidance = 1.0  # 간단한 구현을 위해 기본값
        
        return 0.5 * survival_rate + 0.3 * resource_efficiency + 0.2 * danger_avoidance