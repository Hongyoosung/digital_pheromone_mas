import numpy as np
import torch
from typing import List, Dict, Tuple
from scipy.stats import entropy
import time
import ray
import psutil
import json
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


"""
실험 성능을 측정하고 추적하기 위한 다양한 지표(엔트로피, 수렴 속도, 통신 오버헤드 등)를 관리합니다.
"""


class MetricsTracker:
    """Track and compute experiment metrics"""
    
    def __init__(self):
        self.reset()
        self.ray_stats_history = []
        self.last_ray_stats = None
        self.process_stats = {
            'network_io': {'bytes_sent': 0, 'bytes_recv': 0},
            'cpu_percent': [],
            'memory_percent': []
        }
        
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
        # ray_stats_history를 reset에서도 재초기화
        self.ray_stats_history = []
        self.last_ray_stats = None
        self._init_ray_monitoring()
        
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
            
        # Check for nearly identical values to prevent scipy warning
        if len(flat_field) > 1:
            field_std = np.std(flat_field)
            field_mean = np.mean(flat_field)
            if field_std / (field_mean + 1e-12) < 1e-10:  # Nearly identical values
                # Add small random noise to prevent catastrophic cancellation
                noise = np.random.normal(0, field_mean * 1e-10, len(flat_field))
                flat_field = flat_field + noise
            
        # Normalize to probability distribution with numerical stability
        total_sum = np.sum(flat_field)
        if total_sum < 1e-12:
            return 0.0
            
        prob_dist = flat_field / total_sum
        
        # Ensure probability distribution sums to 1 and has no zeros
        prob_dist = np.clip(prob_dist, 1e-12, 1.0)
        prob_dist = prob_dist / np.sum(prob_dist)  # Renormalize
        
        # Compute entropy with error handling
        try:
            return entropy(prob_dist, base=2)
        except (RuntimeWarning, ValueError) as e:
            logger.warning(f"Entropy calculation warning: {e}")
            # Fallback to manual calculation
            return -np.sum(prob_dist * np.log2(prob_dist + 1e-12))
        
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
        
    def track_communication_overhead(self, messages: List[Dict], communication_metrics: Dict = None) -> Dict:
        """
        Track communication overhead metrics
        
        Args:
            messages: List of message dictionaries
            communication_metrics: Additional communication metrics
            
        Returns:
            Communication overhead statistics
        """
        if not messages:
            return {'total_messages': 0, 'total_bytes': 0, 'avg_size': 0, 'bandwidth_usage': 0}
            
        total_messages = len(messages)
        total_bytes = sum(msg.get('size', 0) for msg in messages)
        avg_size = total_bytes / total_messages if total_messages > 0 else 0
        
        result = {
            'total_messages': total_messages,
            'total_bytes': total_bytes,
            'avg_size': avg_size,
            'messages_per_second': total_messages / (time.time() - self.start_time)
        }
        
        # Ray 클러스터 통신 메트릭 추가
        ray_metrics = self.update_ray_metrics()
        result.update(ray_metrics)
        
        # 추가 통신 메트릭 병합
        if communication_metrics:
            result.update(communication_metrics)
            
        # 통신 성공률 계산 추가
        if communication_metrics:
            # 성공적으로 전송된 메시지 비율
            failed_messages = sum(1 for msg in messages if msg.get('status') == 'failed')
            total_attempted = len(messages)
            success_rate = (total_attempted - failed_messages) / total_attempted if total_attempted > 0 else 0.0
            result['success_rate'] = success_rate
        
        return result
        
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
            
        # 실제 데이터 기반 계산
        total_sent = sum(m.get('bytes_sent', 0) for m in agent_metrics)
        total_received = sum(m.get('bytes_received', 0) for m in agent_metrics)
        
        # 실제 연산 시간 사용 - 마이크로초 단위로 정밀도 개선
        real_computation_times = []
        for m in agent_metrics:
            if 'real_computation_times' in m and m['real_computation_times']:
                real_computation_times.extend(m['real_computation_times'])
        
        total_computation = sum(real_computation_times) if real_computation_times else 0
        # 마이크로초 단위로 변환하여 정밀도 향상
        avg_computation_ms = (total_computation / len(real_computation_times) * 1000) if real_computation_times else 0
        avg_computation_us = avg_computation_ms * 1000  # 마이크로초 단위
        
        elapsed_time = time.time() - self.start_time
        bandwidth_bps = (total_sent + total_received) / max(elapsed_time, 1)
        bandwidth_mbps = bandwidth_bps / (1024 * 1024)  # Convert to Mbps
        
        # 최대 대역폭 계산 개선 - 순간 대역폭 추적
        current_time = time.time()
        
        # 타임스텝별 대역폭 계산
        if not hasattr(self, '_bandwidth_history'):
            self._bandwidth_history = []
            self._last_measurement_time = current_time
            self._last_total_bytes = 0
        
        current_total_bytes = total_sent + total_received
        time_delta = current_time - self._last_measurement_time
        
        if time_delta > 0:
            # 순간 대역폭 계산 (이전 측정 이후의 증분)
            bytes_delta = current_total_bytes - self._last_total_bytes
            instantaneous_bps = bytes_delta / time_delta
            instantaneous_mbps = instantaneous_bps / (1024 * 1024)
            
            # 히스토리에 추가 (최근 10개 측정값만 유지)
            self._bandwidth_history.append((current_time, instantaneous_mbps))
            if len(self._bandwidth_history) > 10:
                self._bandwidth_history = self._bandwidth_history[-10:]
            
            # 피크 대역폭 계산
            peak_bandwidth = max(bw for _, bw in self._bandwidth_history) if self._bandwidth_history else 0
            
            # 업데이트
            self._last_measurement_time = current_time
            self._last_total_bytes = current_total_bytes
        else:
            peak_bandwidth = 0
        
        # 부하 균형 비율 개선 - 에이전트별 작업 분산도 측정
        if len(real_computation_times) > 1:
            # 표준편차를 평균으로 나누어 정규화된 분산도 계산
            mean_time = np.mean(real_computation_times)
            std_time = np.std(real_computation_times)
            load_balance_ratio = std_time / mean_time if mean_time > 0 else 0.0
            
            # 추가: 에이전트별 작업 분배 균등성 측정
            agent_workloads = []
            for m in agent_metrics:
                agent_total_time = sum(m.get('real_computation_times', []))
                agent_workloads.append(agent_total_time)
            
            if len(agent_workloads) > 1:
                workload_balance = np.std(agent_workloads) / max(np.mean(agent_workloads), 1e-8)
            else:
                workload_balance = 0.0
        else:
            load_balance_ratio = 0.0
            workload_balance = 0.0
        
        # Ray 클러스터 네트워크 부하 메트릭 추가
        ray_network_metrics = self.update_ray_metrics()
        
        base_metrics = {
            'bandwidth_usage': bandwidth_bps,
            'bandwidth_mbps': bandwidth_mbps,
            'peak_bandwidth': peak_bandwidth,
            'avg_computation_time_ms': avg_computation_ms,  # 밀리초 단위
            'avg_computation_time_us': avg_computation_us,   # 마이크로초 단위
            'total_data_transferred': total_sent + total_received,
            'load_balance_ratio': load_balance_ratio,
            'workload_balance_ratio': workload_balance,
            'real_computation_samples': len(real_computation_times),
            'instantaneous_bandwidth_mbps': instantaneous_mbps if 'instantaneous_mbps' in locals() else 0
        }
        
        # Ray 메트릭 병합 (접두사 추가로 구분)
        for key, value in ray_network_metrics.items():
            base_metrics[f'ray_{key}'] = value
            
        return base_metrics
        
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
                if isinstance(value, (list, tuple)):
                    self.metrics_history[key].extend(value)
                else:
                    self.metrics_history[key].append(value)
            else:
                # 새로운 메트릭 추가
                if isinstance(value, (list, tuple)):
                    self.metrics_history[key] = list(value)
                else:
                    self.metrics_history[key] = [value]
    
    def compute_pheromone_metrics(self, pheromone_field: np.ndarray) -> Dict:
        """페로몬 필드 관련 메트릭 계산"""
        if pheromone_field.size == 0:
            return {
                'pheromone_concentration_max': 0.0,
                'pheromone_concentration_mean': 0.0,
                'pheromone_concentration_std': 0.0,
                'active_cells': 0,
                'total_intensity': 0.0,
                'pheromone_diversity': 0.0
            }
        
        # 기본 통계
        max_concentration = np.max(pheromone_field)
        mean_concentration = np.mean(pheromone_field)
        std_concentration = np.std(pheromone_field)
        
        # 활성 셀 수 (임계값 이상)
        active_cells = np.sum(pheromone_field > 0.01)
        
        # 총 강도
        total_intensity = np.sum(pheromone_field)
        
        # 페로몬 다양성 (비제로 값들의 표준편차)
        non_zero_values = pheromone_field[pheromone_field > 0.01]
        diversity = np.std(non_zero_values) if len(non_zero_values) > 0 else 0.0
        
        return {
            'pheromone_concentration_max': float(max_concentration),
            'pheromone_concentration_mean': float(mean_concentration),
            'pheromone_concentration_std': float(std_concentration),
            'active_cells': int(active_cells),
            'total_intensity': float(total_intensity),
            'pheromone_diversity': float(diversity)
        }
        
    def get_summary(self) -> Dict:
        """Get summary statistics of all metrics"""
        summary = {}
        for key, values in self.metrics_history.items():
            if not values:
                continue

            # Check if the list contains dictionaries
            if isinstance(values[0], dict):
                summary[key] = {}
                # Aggregate keys from all dictionaries
                all_sub_keys = set(k for d in values for k in d.keys())

                for sub_key in all_sub_keys:
                    # Extract values for the sub_key, filtering out None
                    sub_values = [d.get(sub_key) for d in values if d.get(sub_key) is not None]
                    
                    if not sub_values:
                        continue

                    # Check if all values are numeric
                    if all(isinstance(v, (int, float)) for v in sub_values):
                        summary[key][sub_key] = {
                            'mean': np.mean(sub_values),
                            'std': np.std(sub_values),
                            'min': np.min(sub_values),
                            'max': np.max(sub_values),
                            'last': sub_values[-1]
                        }
                    else:
                        # For non-numeric, just take the last value
                        summary[key][sub_key] = {'last': sub_values[-1]}
            
            # Handle list of numbers
            elif isinstance(values[0], (int, float, np.generic)):
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'last': values[-1]
                }
        return summary
        
    def _init_ray_monitoring(self):
        """Initialize Ray cluster monitoring"""
        try:
            if ray.is_initialized():
                self._ray_dashboard_url = None
                self._ray_cluster_resources = None
                self._monitor_ray_cluster()
        except Exception as e:
            logger.warning(f"Ray 모니터링 초기화 실패: {e}")
            
    def _monitor_ray_cluster(self):
        """Monitor Ray cluster resources and network usage"""
        # ray_stats_history가 없으면 초기화
        if not hasattr(self, 'ray_stats_history'):
            self.ray_stats_history = []
            
        try:
            # Ray가 초기화되지 않은 경우 기본값 반환
            if not ray.is_initialized():
                logger.debug("Ray가 초기화되지 않아 기본값 사용")
                ray_stats = {
                    'cluster_resources': {},
                    'available_resources': {},
                    'task_summary': {"RUNNING": 0, "FINISHED": 0, "FAILED": 0},
                    'active_tasks': 0,
                    'network_stats': {'bytes_sent': 0, 'bytes_recv': 0, 'packets_sent': 0, 'packets_recv': 0},
                    'timestamp': time.time()
                }
                self.ray_stats_history.append(ray_stats)
                self.last_ray_stats = ray_stats
                return
                
            # Ray 클러스터 리소스 모니터링
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            # 태스크 상태 모니터링
            try:
                # Ray 2.0+ 지원
                if hasattr(ray, 'util') and hasattr(ray.util, 'state'):
                    task_summary = ray.util.state.summarize_tasks()
                    active_tasks = len([t for t in ray.util.state.list_tasks() if t.get('state') == 'RUNNING'])
                else:
                    # 기본값 사용
                    task_summary = {"RUNNING": 0, "FINISHED": 0, "FAILED": 0}
                    active_tasks = 0
            except Exception as e:
                logger.debug(f"Ray state API 사용 불가: {e}")
                task_summary = {"RUNNING": 0, "FINISHED": 0, "FAILED": 0}
                active_tasks = 0
            
            # 네트워크 통신 메트릭 수집
            network_stats = self._get_network_stats()
            
            ray_stats = {
                'cluster_resources': cluster_resources,
                'available_resources': available_resources,
                'task_summary': task_summary,
                'active_tasks': active_tasks,
                'network_stats': network_stats,
                'timestamp': time.time()
            }
            
            self.ray_stats_history.append(ray_stats)
            self.last_ray_stats = ray_stats
            
            # 최근 10개 기록만 유지
            if len(self.ray_stats_history) > 10:
                self.ray_stats_history = self.ray_stats_history[-10:]
                
        except Exception as e:
            logger.warning(f"Ray 클러스터 모니터링 오류: {e}")
            # 오류 발생 시 기본값으로 채우기
            if not hasattr(self, 'ray_stats_history') or len(self.ray_stats_history) == 0:
                default_stats = {
                    'cluster_resources': {},
                    'available_resources': {},
                    'task_summary': {"RUNNING": 0, "FINISHED": 0, "FAILED": 0},
                    'active_tasks': 0,
                    'network_stats': {'bytes_sent': 0, 'bytes_recv': 0, 'packets_sent': 0, 'packets_recv': 0},
                    'timestamp': time.time()
                }
                self.ray_stats_history.append(default_stats)
                self.last_ray_stats = default_stats
            
    def _get_network_stats(self) -> Dict:
        """Get system network statistics"""
        try:
            # psutil을 사용하여 네트워크 통계 수집
            net_io = psutil.net_io_counters()
            
            # 이전 측정값과의 차이 계산
            current_stats = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
            # Ray 특정 네트워크 통계 (가능한 경우)
            ray_network_stats = self._get_ray_network_stats()
            if ray_network_stats:
                current_stats.update(ray_network_stats)
            
            return current_stats
            
        except Exception as e:
            logger.warning(f"네트워크 통계 수집 오류: {e}")
            return {'bytes_sent': 0, 'bytes_recv': 0, 'packets_sent': 0, 'packets_recv': 0}
            
    def _get_ray_network_stats(self) -> Dict:
        """Get Ray-specific network statistics"""
        try:
            # Ray 노드 정보 가져오기
            nodes = ray.nodes()
            
            total_object_store_used = 0
            total_object_store_available = 0
            
            for node in nodes:
                if node.get('Alive', False):
                    resources = node.get('Resources', {})
                    object_store_used = node.get('ObjectStoreUsedMemory', 0)
                    object_store_available = node.get('ObjectStoreAvailableMemory', 0)
                    
                    total_object_store_used += object_store_used
                    total_object_store_available += object_store_available
            
            return {
                'ray_nodes': len([n for n in nodes if n.get('Alive', False)]),
                'object_store_used': total_object_store_used,
                'object_store_available': total_object_store_available,
                'object_store_utilization': total_object_store_used / max(total_object_store_used + total_object_store_available, 1)
            }
            
        except Exception as e:
            logger.warning(f"Ray 네트워크 통계 수집 오류: {e}")
            return {}
            
    def get_ray_communication_overhead(self) -> Dict:
        """Calculate Ray cluster communication overhead"""
        # ray_stats_history가 존재하지 않거나 충분한 데이터가 없는 경우 실제 측정 시도
        if not hasattr(self, 'ray_stats_history') or len(self.ray_stats_history) < 2:
            # 실시간 네트워크 통계 수집 시도
            current_net_stats = self._get_network_stats()
            if not hasattr(self, '_baseline_net_stats'):
                self._baseline_net_stats = current_net_stats
                return {
                    'communication_overhead': 0.0,
                    'network_load': 0.0,
                    'bandwidth_utilization': 0.0,
                    'task_throughput': 0.0
                }
            
            # 베이스라인 대비 네트워크 사용량 변화 계산
            bytes_diff = (current_net_stats.get('bytes_sent', 0) + current_net_stats.get('bytes_recv', 0)) - \
                        (self._baseline_net_stats.get('bytes_sent', 0) + self._baseline_net_stats.get('bytes_recv', 0))
            
            # 시간 경과 계산 (최소 1초로 제한)
            elapsed_time = max(time.time() - self.start_time, 1.0)
            
            # 통신 오버헤드 계산
            communication_rate = bytes_diff / elapsed_time
            network_load_mbps = (communication_rate * 8) / (1024 * 1024)
            
            return {
                'communication_overhead': float(max(communication_rate, 0.0)),
                'network_load': float(max(network_load_mbps, 0.0)),
                'bandwidth_utilization': float(min(network_load_mbps / 1024, 1.0)),  # 1 Gbps 기준
                'task_throughput': 0.0
            }
        
        try:
            current_stats = self.ray_stats_history[-1]
            previous_stats = self.ray_stats_history[-2]
            
            time_diff = current_stats['timestamp'] - previous_stats['timestamp']
            
            # 네트워크 사용량 차이 계산
            current_net = current_stats.get('network_stats', {})
            previous_net = previous_stats.get('network_stats', {})
            
            bytes_sent_diff = current_net.get('bytes_sent', 0) - previous_net.get('bytes_sent', 0)
            bytes_recv_diff = current_net.get('bytes_recv', 0) - previous_net.get('bytes_recv', 0)
            
            # 통신 오버헤드 계산 (bytes/second)
            communication_rate = (bytes_sent_diff + bytes_recv_diff) / max(time_diff, 1)
            
            # 네트워크 부하 계산 (Mbps)
            network_load_mbps = (communication_rate * 8) / (1024 * 1024)
            
            # 대역폭 사용률 (10 Gbps 기준)
            max_bandwidth_mbps = 10 * 1024  # 10 Gbps
            bandwidth_utilization = min(network_load_mbps / max_bandwidth_mbps, 1.0)
            
            # 태스크 처리량
            current_tasks = current_stats.get('task_summary', {}).get('FINISHED', 0)
            previous_tasks = previous_stats.get('task_summary', {}).get('FINISHED', 0)
            task_throughput = (current_tasks - previous_tasks) / max(time_diff, 1)
            
            return {
                'communication_overhead': float(communication_rate),
                'network_load': float(network_load_mbps),
                'bandwidth_utilization': float(bandwidth_utilization),
                'task_throughput': float(task_throughput),
                'bytes_per_second': float(communication_rate),
                'active_tasks': current_stats.get('active_tasks', 0)
            }
            
        except Exception as e:
            logger.warning(f"Ray 통신 오버헤드 계산 오류: {e}")
            # 오류 발생 시에도 실제 네트워크 통계를 반환하도록 개선
            try:
                current_net_stats = self._get_network_stats()
                if hasattr(self, '_baseline_net_stats'):
                    bytes_diff = (current_net_stats.get('bytes_sent', 0) + current_net_stats.get('bytes_recv', 0)) - \
                                (self._baseline_net_stats.get('bytes_sent', 0) + self._baseline_net_stats.get('bytes_recv', 0))
                    elapsed_time = max(time.time() - self.start_time, 1.0)
                    communication_rate = max(bytes_diff / elapsed_time, 0.0)
                    network_load_mbps = max((communication_rate * 8) / (1024 * 1024), 0.0)
                    
                    return {
                        'communication_overhead': float(communication_rate),
                        'network_load': float(network_load_mbps),
                        'bandwidth_utilization': float(min(network_load_mbps / 1024, 1.0)),
                        'task_throughput': 0.0
                    }
            except:
                pass
            
            return {
                'communication_overhead': 0.0,
                'network_load': 0.0,
                'bandwidth_utilization': 0.0,
                'task_throughput': 0.0
            }
            
    def update_ray_metrics(self):
        """Update Ray cluster metrics"""
        self._monitor_ray_cluster()
        return self.get_ray_communication_overhead()
        
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
        
    def compute_loss_improvement_metrics(self, current_loss: float, previous_losses: List[float]) -> Dict:
        """
        손실 개선 메트릭 계산
        
        Args:
            current_loss: 현재 손실값
            previous_losses: 이전 손실값들
            
        Returns:
            손실 개선 메트릭 딕셔너리
        """
        if not previous_losses or len(previous_losses) == 0:
            return {
                'loss_improvement': 0.0,
                'relative_improvement': 0.0,
                'improvement_rate': 0.0
            }
            
        recent_loss = previous_losses[-1] if previous_losses else current_loss
        
        # 절대적 개선
        loss_improvement = recent_loss - current_loss
        
        # 상대적 개선 (백분율)
        relative_improvement = (loss_improvement / max(abs(recent_loss), 1e-8)) * 100
        
        # 개선률 (최근 5개 스텝 평균 대비)
        if len(previous_losses) >= 5:
            recent_avg = np.mean(previous_losses[-5:])
            improvement_rate = (recent_avg - current_loss) / max(abs(recent_avg), 1e-8)
        else:
            improvement_rate = relative_improvement / 100
            
        return {
            'loss_improvement': float(loss_improvement),
            'relative_improvement': float(relative_improvement),
            'improvement_rate': float(improvement_rate)
        }
        
    def compute_pheromone_dynamics_metrics(self, current_field: np.ndarray, previous_field: np.ndarray = None) -> Dict:
        """
        페로몬 동역학 메트릭 계산 (감쇠율, 증착율 등)
        
        Args:
            current_field: 현재 페로몬 필드
            previous_field: 이전 페로몬 필드
            
        Returns:
            페로몬 동역학 메트릭 딕셔너리
        """
        metrics = {
            'pheromone_decay_rate': 0.0,
            'pheromone_deposit_rate': 0.0,
            'pheromone_evaporation_rate': 0.0,
            'diffusion_rate': 0.0
        }
        
        if previous_field is None or current_field.size == 0 or previous_field.size == 0:
            return metrics
            
        try:
            # 전체 농도 변화
            current_total = np.sum(current_field)
            previous_total = np.sum(previous_field)
            
            # 감쇠율 계산 (전체 농도 감소율)
            if previous_total > 0:
                decay_rate = (previous_total - current_total) / previous_total
                metrics['pheromone_decay_rate'] = max(0.0, float(decay_rate))
                
                # 증발률 (자연적 감소)
                evaporation_rate = decay_rate * 0.7  # 전체 감소의 70%를 증발로 가정
                metrics['pheromone_evaporation_rate'] = max(0.0, float(evaporation_rate))
            
            # 새로운 증착 감지 (농도가 증가한 영역)
            field_diff = current_field - previous_field
            deposit_areas = field_diff > 0
            if np.any(deposit_areas):
                deposit_amount = np.sum(field_diff[deposit_areas])
                metrics['pheromone_deposit_rate'] = float(deposit_amount)
            
            # 확산률 계산 (공간적 분산 변화)
            current_var = np.var(current_field)
            previous_var = np.var(previous_field)
            if previous_var > 0:
                diffusion_rate = (current_var - previous_var) / previous_var
                metrics['diffusion_rate'] = float(diffusion_rate)
                
        except Exception as e:
            logging.warning(f"페로몬 동역학 메트릭 계산 오류: {e}")
            
        return metrics
        
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
        cooperation_strength = 0.0
        
        for agent_id, connections in social_connections.items():
            for other_id, strength in connections.items():
                total_connections += 1
                # 강도 가중 협력 지수 (단순 양수/음수가 아닌)
                if strength > 0.1:  # 임계값 이상의 긍정적 연결만 협력으로 간주
                    cooperation_strength += min(strength, 1.0)  # 최대 1.0으로 제한
                    
        if total_connections == 0:
            return 0.0
            
        # 평균 협력 강도 반환
        return cooperation_strength / total_connections
        
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
        significant_edges = 0
        
        for connections in social_connections.values():
            for strength in connections.values():
                # 유의미한 연결만 카운트 (임계값 이상)
                if abs(strength) > 0.1:  # 0.1 이상의 강도를 가진 연결만 인정
                    significant_edges += 1
            
        return significant_edges / max_possible_edges
        
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
        
        # 환경적 위험 회피 개선 (실제 적응 능력 측정)
        # 체력과 자원의 변동성을 통한 적응력 측정
        health_values = [state.get('health', 0) for state in agent_states]
        resource_values = [state.get('resources', 0) for state in agent_states]
        
        # 적응력 지표: 낮은 변동성 = 높은 적응력
        health_stability = 1.0 - (np.std(health_values) / (np.mean(health_values) + 1e-8))
        resource_stability = 1.0 - (np.std(resource_values) / (np.mean(resource_values) + 1e-8))
        adaptation_stability = np.clip((health_stability + resource_stability) / 2, 0.0, 1.0)
        
        # 환경 압박 시뮬레이션 (자원 경쟁, 체력 손실 등)
        low_health_agents = sum(1 for state in agent_states if state.get('health', 0) < 0.3)
        low_resource_agents = sum(1 for state in agent_states if state.get('resources', 0) < 20)
        stress_factor = 1.0 - ((low_health_agents + low_resource_agents) / (len(agent_states) * 2))
        
        return 0.4 * survival_rate + 0.3 * resource_efficiency + 0.2 * adaptation_stability + 0.1 * stress_factor