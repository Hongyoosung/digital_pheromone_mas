import torch
import torch.nn as nn

class TemporalDiffusionModel(nn.Module):
    """
    시계열 감쇠를 적용하는 확산 모델입니다.
    페로몬 벡터의 각 타임스텝별 중요도를 지수적으로 감소시킵니다.
    """
    def __init__(self, decay_factor: float = 0.9, **kwargs):
        """
        확산 모델을 초기화합니다.

        Args:
            decay_factor (float): 페로몬 강도가 시간의 흐름에 따라 감쇠하는 비율. 
                                  0과 1 사이의 값이어야 합니다.
        """
        super(TemporalDiffusionModel, self).__init__()
        if not 0.0 <= decay_factor <= 1.0:
            raise ValueError("decay_factor는 0과 1 사이의 값이어야 합니다.")
        self.decay_factor = decay_factor

    def forward(self, pheromone_vectors: torch.Tensor, t: int = None) -> torch.Tensor:
        """
        입력된 페로몬 벡터에 시간 감쇠를 적용합니다.

        Args:
            pheromone_vectors (torch.Tensor): (batch_size, sequence_length, vector_dim) 형태의
                                             페로몬 텐서.
            t (int, optional): 현재 타임스텝. 현재 구현에서는 사용되지 않지만,
                               향후 확장성을 위해 인터페이스를 유지합니다.

        Returns:
            torch.Tensor: 감쇠가 적용된 페로몬 텐서.
        """
        if pheromone_vectors.dim() != 3:
            raise ValueError("입력 텐서는 (batch_size, sequence_length, vector_dim) 형태여야 합니다.")

        seq_len = pheromone_vectors.size(1)
        
        # 감쇠 가중치 생성: [1, decay, decay^2, ..., decay^(n-1)]
        decay_weights = torch.pow(self.decay_factor, torch.arange(seq_len, device=pheromone_vectors.device))
        
        # (1, sequence_length, 1) 형태로 브로드캐스팅 가능하게 차원 확장
        decay_weights = decay_weights.view(1, seq_len, 1)
        
        # 각 시점의 페로몬 벡터에 감쇠 가중치 적용
        diffused_vectors = pheromone_vectors * decay_weights
        
        return diffused_vectors

if __name__ == '__main__':
    # 모델 사용 예시
    batch_size = 4
    sequence_length = 10
    vector_dim = 8
    
    # 모델 초기화 (감쇠 인자 0.9)
    diffusion_model = TemporalDiffusionModel(decay_factor=0.9)
    
    # 임의의 페로몬 데이터 생성
    pheromones = torch.ones(batch_size, sequence_length, vector_dim)
    print("Original Pheromones (first vector):\n", pheromones[0, :, 0])
    
    # 확산 모델 적용
    diffused_pheromones = diffusion_model(pheromones, t=0) # t 인자 추가
    
    print("\nDiffused Pheromones (first vector):\n", diffused_pheromones[0, :, 0])
    
    # 감쇠가 올바르게 적용되었는지 확인
    expected_decay = torch.pow(torch.tensor(0.9), torch.arange(10))
    assert torch.allclose(diffused_pheromones[0, :, 0], expected_decay)
    print("\n검증 통과: 확산 모델이 예상대로 작동합니다.")
