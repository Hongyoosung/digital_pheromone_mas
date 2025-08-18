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
        
        # 학습 가능한 파라미터들 추가
        self.learnable_decay = nn.Parameter(torch.tensor(decay_factor))
        self.temporal_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # 페로몬 벡터 처리용 레이어
        self.vector_projection = nn.Linear(64, 64)  # 64는 에이전트 인코더 출력 차원
        self.output_norm = nn.LayerNorm(64)

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

        batch_size, seq_len, vector_dim = pheromone_vectors.size()
        
        # 학습 가능한 감쇠 가중치 생성
        decay_factor = torch.clamp(self.learnable_decay, 0.1, 0.999)
        decay_weights = torch.pow(decay_factor, torch.arange(seq_len, device=pheromone_vectors.device))
        
        # 시간 인코딩을 통한 추가적인 시간적 특성 학습
        time_features = torch.arange(seq_len, dtype=torch.float32, device=pheromone_vectors.device).unsqueeze(-1) / seq_len
        temporal_weights = self.temporal_encoder(time_features).squeeze(-1)
        
        # 통합 가중치 계산
        combined_weights = decay_weights * temporal_weights
        
        # (1, sequence_length, 1) 형태로 브로드캐스팅 가능하게 차원 확장
        combined_weights = combined_weights.view(1, seq_len, 1)
        
        # 벡터 투영 및 정규화
        projected_vectors = self.vector_projection(pheromone_vectors)
        normalized_vectors = self.output_norm(projected_vectors)
        
        # 각 시점의 페로몬 벡터에 감쇠 가중치 적용
        diffused_vectors = normalized_vectors * combined_weights
        
        return diffused_vectors

if __name__ == '__main__':
    # 모델 사용 예시
    batch_size = 4
    sequence_length = 10
    vector_dim = 64  # 에이전트 인코더 출력 차원
    
    # 모델 초기화 (감쇠 인자 0.9)
    diffusion_model = TemporalDiffusionModel(decay_factor=0.9)
    
    # 임의의 페로몬 데이터 생성
    pheromones = torch.ones(batch_size, sequence_length, vector_dim)
    print("Original Pheromones shape:", pheromones.shape)
    
    # 확산 모델 적용
    diffused_pheromones = diffusion_model(pheromones, t=0)
    
    print("Diffused Pheromones shape:", diffused_pheromones.shape)
    print("Model parameters:", sum(p.numel() for p in diffusion_model.parameters()))
    print("Learnable parameters:", sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad))
