import torch
import gc
import psutil
import os
from typing import Dict, Optional
import logging
import numpy as np

"""
CPU 메모리 사용량 모니터링 및 관리를 위한 유틸리티입니다.
시스템 메모리 부족 시 자동으로 정리 작업을 수행합니다.
"""

logger = logging.getLogger(__name__)


class MemoryManager:
    """메모리 사용량 모니터링 및 자동 관리 클래스"""
    
    def __init__(self, 
                 max_memory_percent: float = 85.0,
                 warning_memory_percent: float = 75.0,
                 cleanup_threshold_percent: float = 80.0):
        """
        Args:
            max_memory_percent: 최대 허용 메모리 사용률 (%)
            warning_memory_percent: 경고 메모리 사용률 (%)  
            cleanup_threshold_percent: 자동 정리 시작 임계값 (%)
        """
        self.max_memory_percent = max_memory_percent
        self.warning_memory_percent = warning_memory_percent
        self.cleanup_threshold_percent = cleanup_threshold_percent
        self.process = psutil.Process(os.getpid())
        self.memory_history = []
        
    def get_memory_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량 정보 반환"""
        memory_info = psutil.virtual_memory()
        process_memory = self.process.memory_info()
        
        usage_info = {
            'system_total_gb': memory_info.total / (1024**3),
            'system_used_gb': memory_info.used / (1024**3),
            'system_available_gb': memory_info.available / (1024**3),
            'system_percent': memory_info.percent,
            'process_rss_gb': process_memory.rss / (1024**3),  # 실제 사용 메모리
            'process_vms_gb': process_memory.vms / (1024**3),  # 가상 메모리
        }
        
        return usage_info
        
    def check_memory_status(self) -> str:
        """메모리 상태 확인"""
        usage = self.get_memory_usage()
        system_percent = usage['system_percent']
        
        if system_percent >= self.max_memory_percent:
            return "critical"
        elif system_percent >= self.cleanup_threshold_percent:
            return "high"  
        elif system_percent >= self.warning_memory_percent:
            return "warning"
        else:
            return "normal"
            
    def cleanup_memory(self, aggressive: bool = False):
        """메모리 정리 작업 수행"""
        logger.info("메모리 정리 작업 시작...")
        
        # Python 가비지 컬렉션 강제 실행
        collected = gc.collect()
        logger.info(f"가비지 컬렉션으로 {collected}개 객체 정리")
        
        # PyTorch 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU 캐시 정리 완료")
            
        # 공격적 정리 모드
        if aggressive:
            # 모든 세대의 가비지 컬렉션 실행
            for i in range(3):
                collected = gc.collect(i)
                logger.info(f"세대 {i} 가비지 컬렉션: {collected}개 객체 정리")
                
        # 정리 후 메모리 상태 확인
        usage_after = self.get_memory_usage()
        logger.info(f"정리 후 시스템 메모리 사용률: {usage_after['system_percent']:.1f}%")
        
    def should_continue_training(self) -> bool:
        """훈련 계속 진행 여부 판단"""
        status = self.check_memory_status()
        
        if status == "critical":
            logger.error("메모리 사용량이 임계치를 초과했습니다. 훈련을 중단합니다.")
            return False
        elif status == "high":
            logger.warning("메모리 사용량이 높습니다. 자동 정리를 수행합니다.")
            self.cleanup_memory()
            # 정리 후 재확인
            return self.check_memory_status() != "critical"
        
        return True
        
    def log_memory_usage(self, step: Optional[int] = None):
        """메모리 사용량 로깅"""
        usage = self.get_memory_usage()
        self.memory_history.append(usage)
        
        # 히스토리 길이 제한 (메모리 절약)
        if len(self.memory_history) > 1000:
            self.memory_history = self.memory_history[-500:]
            
        step_info = f"Step {step}: " if step is not None else ""
        logger.info(
            f"{step_info}Memory Usage - "
            f"System: {usage['system_percent']:.1f}% "
            f"({usage['system_used_gb']:.2f}GB/{usage['system_total_gb']:.2f}GB), "
            f"Process: {usage['process_rss_gb']:.2f}GB"
        )
        
    def get_memory_recommendations(self) -> Dict[str, str]:
        """메모리 최적화 권장사항 제공"""
        usage = self.get_memory_usage()
        recommendations = {}
        
        if usage['system_percent'] > 80:
            recommendations['high_usage'] = "배치 크기를 줄이거나 에이전트 수를 감소시키세요"
            
        if usage['process_rss_gb'] > 8:  # 8GB 이상
            recommendations['large_process'] = "텐서 크기를 최적화하고 중간 결과를 즉시 삭제하세요"
            
        if len(self.memory_history) > 10:
            # 메모리 사용량 증가 추세 분석
            recent_usage = [h['system_percent'] for h in self.memory_history[-10:]]
            if len(recent_usage) > 5 and recent_usage[-1] - recent_usage[0] > 10:
                recommendations['memory_leak'] = "메모리 누수 가능성이 있습니다. 변수 참조를 확인하세요"
                
        return recommendations


class BatchMemoryOptimizer:
    """배치 처리 시 메모리 최적화"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.original_batch_size = None
        self.adaptive_batch_size = None
        
    def optimize_batch_size(self, original_batch_size: int, data_size: int) -> int:
        """메모리 상황에 따른 적응적 배치 크기 조정"""
        self.original_batch_size = original_batch_size
        
        status = self.memory_manager.check_memory_status()
        
        if status == "critical":
            # 메모리 부족 시 배치 크기를 1/4로 감소
            self.adaptive_batch_size = max(1, original_batch_size // 4)
            logger.warning(f"메모리 부족으로 배치 크기를 {original_batch_size}에서 {self.adaptive_batch_size}로 조정")
            
        elif status == "high":
            # 메모리 사용량이 높을 때 배치 크기를 1/2로 감소
            self.adaptive_batch_size = max(1, original_batch_size // 2)
            logger.info(f"메모리 최적화를 위해 배치 크기를 {original_batch_size}에서 {self.adaptive_batch_size}로 조정")
            
        elif status == "warning":
            # 경고 상태에서는 3/4로 감소
            self.adaptive_batch_size = max(1, int(original_batch_size * 0.75))
            logger.info(f"예방적 메모리 관리로 배치 크기를 {original_batch_size}에서 {self.adaptive_batch_size}로 조정")
            
        else:
            self.adaptive_batch_size = original_batch_size
            
        return self.adaptive_batch_size
        
    def create_batches(self, data, batch_size: int):
        """메모리 효율적인 배치 생성기"""
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            yield batch
            
            # 배치 처리 후 메모리 정리
            if i % (batch_size * 4) == 0:  # 4배치마다 정리
                if not self.memory_manager.should_continue_training():
                    logger.error("메모리 부족으로 배치 처리를 중단합니다.")
                    break


def create_memory_efficient_tensor(shape, dtype=torch.float32, device='cpu'):
    """메모리 효율적인 텐서 생성"""
    try:
        return torch.zeros(shape, dtype=dtype, device=device)
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error(f"텐서 생성 실패 (크기: {shape}). 메모리가 부족합니다.")
            # GPU에서 실패하면 CPU로 폴백
            if device != 'cpu':
                logger.info("CPU 메모리로 폴백합니다.")
                return torch.zeros(shape, dtype=dtype, device='cpu')
        raise e


def memory_efficient_operation(func, *args, **kwargs):
    """메모리 효율적인 연산 실행 래퍼"""
    memory_manager = MemoryManager()
    
    # 연산 전 메모리 상태 확인
    if not memory_manager.should_continue_training():
        raise RuntimeError("메모리 부족으로 연산을 수행할 수 없습니다.")
        
    try:
        result = func(*args, **kwargs)
        return result
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("연산 중 메모리 부족 발생. 정리 후 재시도합니다.")
            memory_manager.cleanup_memory(aggressive=True)
            # 한 번 더 시도
            try:
                result = func(*args, **kwargs)
                return result
            except RuntimeError:
                logger.error("메모리 정리 후에도 연산 실패. 연산을 중단합니다.")
                raise
        raise e
    finally:
        # 연산 후 가벼운 정리
        gc.collect()