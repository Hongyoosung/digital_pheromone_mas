import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Union


"""
4D 페로몬 벡터의 각 차원을 정규화하는 유틸리티 클래스입니다.
RESEARCHPAPER.md 4.3절에 명시된 정규화 절차를 따릅니다.
"""


class PheromoneNormalizer:
    """
    RESEARCHPAPER.md 4.3절 기반 4D 페로몬 벡터 정규화 유틸리티.
    """
    
    # 사회관계 정규화를 위한 최대 교환 횟수 T (RESEARCHPAPER.md 5.3)
    MAX_SOCIAL_EXCHANGES = 50

    def normalize_behavior(self, behavior_vector: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        행동 차원을 softmax를 사용하여 정규화합니다. (RESEARCHPAPER.md 4.3)
        
        Args:
            behavior_vector: 원본 행동 점수 (raw scores)
            
        Returns:
            정규화된 행동 확률
        """
        if isinstance(behavior_vector, np.ndarray):
            exp_scores = np.exp(behavior_vector)
            return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        elif isinstance(behavior_vector, torch.Tensor):
            return F.softmax(behavior_vector, dim=-1)
        else:
            raise TypeError("Input must be a numpy array or a torch Tensor.")
            
    def normalize_emotion(self, emotion_vector: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        감정 차원을 min-max 정규화를 사용하여 [0, 1] 범위로 조정합니다.
        원본 값의 범위는 [-1, 1]로 가정합니다. (RESEARCHPAPER.md 4.3)
        
        Args:
            emotion_vector: 원본 감정 값 ([-1, 1] 범위)
            
        Returns:
            정규화된 감정 값 ([0, 1] 범위)
        """
        min_val, max_val = -1.0, 1.0
        
        if isinstance(emotion_vector, (np.ndarray, torch.Tensor)):
            return (emotion_vector - min_val) / (max_val - min_val)
        else:
            raise TypeError("Input must be a numpy array or a torch Tensor.")
            
    def normalize_social(self, social_vector: Union[np.ndarray, torch.Tensor], 
                         max_exchanges: int = MAX_SOCIAL_EXCHANGES) -> Union[np.ndarray, torch.Tensor]:
        """
        사회관계 차원을 최대 교환 횟수로 나누어 정규화합니다. (RESEARCHPAPER.md 4.3, 5.3)
        
        Args:
            social_vector: 원본 교환 횟수
            max_exchanges: 최대 교환 횟수 (기본값: 50)
            
        Returns:
            정규화된 사회관계 가중치 ([0, 1] 범위)
        """
        if max_exchanges <= 0:
            raise ValueError("max_exchanges must be positive.")

        if isinstance(social_vector, (np.ndarray, torch.Tensor)):
            # clip/clamp를 사용하여 0과 1 사이의 값을 보장
            normalized = social_vector / max_exchanges
            if isinstance(normalized, np.ndarray):
                return np.clip(normalized, 0, 1)
            else:
                return torch.clamp(normalized, 0, 1)
        else:
            raise TypeError("Input must be a numpy array or a torch Tensor.")
            
    def normalize_context(self, context_vector: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        환경맥락 차원을 각 피처별 min-max 스케일링을 사용하여 정규화합니다. (RESEARCHPAPER.md 4.3)
        
        Args:
            context_vector: 2D 벡터 (batch_size, num_features) 또는 1D 벡터 (num_features).
                            각 피처(열)별로 min-max 정규화를 수행합니다.
            
        Returns:
            정규화된 환경맥락 벡터
        """
        if isinstance(context_vector, (np.ndarray, torch.Tensor)):
            # 1D 벡터인 경우 2D로 변환하여 처리
            is_1d = False
            if context_vector.ndim == 1:
                is_1d = True
                # Reshape to (n, 1) to treat it as a single feature column
                if isinstance(context_vector, np.ndarray):
                    context_vector = context_vector.reshape(-1, 1)
                else:
                    context_vector = context_vector.view(-1, 1)

            if isinstance(context_vector, np.ndarray):
                min_val = np.min(context_vector, axis=0, keepdims=True)
                max_val = np.max(context_vector, axis=0, keepdims=True)
                denominator = max_val - min_val
                # np.where를 사용하여 분모가 0일 때 0을, 아닐 때 정규화된 값을 반환
                normalized = np.where(denominator == 0, 0, (context_vector - min_val) / denominator)
            else: # torch.Tensor
                min_val = torch.min(context_vector, dim=0, keepdim=True).values
                max_val = torch.max(context_vector, dim=0, keepdim=True).values
                denominator = max_val - min_val
                # torch.where를 사용하여 분모가 0일 때 0을, 아닐 때 정규화된 값을 반환
                normalized = torch.where(denominator == 0, 
                                         torch.zeros_like(context_vector), 
                                         (context_vector - min_val) / denominator)

            # 원래 1D 벡터였다면 다시 1D로 변환하여 반환
            if is_1d:
                if isinstance(normalized, np.ndarray):
                    return normalized.flatten()
                else:
                    return normalized.view(-1)
            return normalized
        else:
            raise TypeError("Input must be a numpy array or a torch Tensor.")

            
    def normalize_pheromone_vector(self, pheromone: Dict) -> Dict:
        """
        4D 페로몬 벡터 전체를 정규화합니다.
        
        Args:
            pheromone: 'behavior', 'emotion', 'social', 'context' 키를 포함하는 딕셔너리
            
        Returns:
            정규화된 페로몬 딕셔너리
        """
        return {
            'behavior': self.normalize_behavior(pheromone['behavior']),
            'emotion': self.normalize_emotion(pheromone['emotion']),
            'social': self.normalize_social(pheromone['social']),
            'context': self.normalize_context(pheromone['context'])
        }
