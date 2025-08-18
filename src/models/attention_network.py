import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DistributedAttentionRouter(nn.Module):
    """
    멀티-헤드 어텐션을 사용한 분산 라우팅 네트워크입니다.
    에이전트 간의 정보 흐름을 조절하기 위해 어텐션 가중치를 계산합니다.
    """
    def __init__(self, embed_dim: int, num_heads: int, **kwargs):
        """
        어텐션 네트워크를 초기화합니다.

        Args:
            embed_dim (int): 각 에이전트 벡터의 임베딩 차원.
            num_heads (int): 멀티-헤드 어텐션에서 사용할 헤드의 수. 
                             `embed_dim`은 `num_heads`로 나누어 떨어져야 합니다.
        """
        super(DistributedAttentionRouter, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim은 num_heads로 나누어 떨어져야 합니다.")
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # 추가 학습 가능한 레이어들
        self.input_projection = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.routing_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 학습을 위한 손실 추적
        self.training_metrics = {
            'routing_loss': [],
            'attention_entropy': [],
            'communication_efficiency': []
        }

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        입력된 쿼리, 키, 값에 대해 어텐션을 수행합니다.

        에이전트 자신을 쿼리로, 다른 에이전트들을 키/값으로 사용하여 
        정보의 중요도를 계산할 수 있습니다.

        Args:
            query (torch.Tensor): (batch_size, num_agents_query, embed_dim) 형태의 텐서.
            key (torch.Tensor): (batch_size, num_agents_kv, embed_dim) 형태의 텐서.
            value (torch.Tensor): (batch_size, num_agents_kv, embed_dim) 형태의 텐서.
            attn_mask (torch.Tensor, optional): 어텐션 가중치에 적용할 마스크.
                                                (batch_size * num_heads, num_agents_query, num_agents_kv)
                                                형태여야 합니다. 0인 위치에 어텐션을 적용하지 않습니다.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
            -attn_output (torch.Tensor): 어텐션이 적용된 출력 텐서. 
                                          (batch_size, num_agents_query, embed_dim)
            -attn_weights (torch.Tensor): 어텐션 가중치 텐서.
                                           (batch_size, num_agents_query, num_agents_kv)
        """
        # 입력 투영
        projected_query = self.input_projection(query)
        projected_key = self.input_projection(key)
        projected_value = self.input_projection(value)
        
        # 멀티헤드 어텐션 적용
        attn_output, attn_weights = self.attention(projected_query, projected_key, projected_value, attn_mask=attn_mask)
        
        # MLP 라우팅을 통한 추가 처리
        routing_output = self.routing_mlp(attn_output)
        
        # 출력 투영
        final_output = self.output_projection(routing_output)
        
        # 학습 메트릭 계산 (훈련 모드에서만)
        if self.training:
            self._update_training_metrics(attn_weights, final_output)
        
        return final_output, attn_weights
        
    def _update_training_metrics(self, attn_weights: torch.Tensor, output: torch.Tensor):
        """학습 메트릭 업데이트"""
        with torch.no_grad():
            # 어텐션 엔트로피 계산 (다양성 측정)
            entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1).mean()
            self.training_metrics['attention_entropy'].append(entropy.item())
            
            # 통신 효율성 (출력의 분산)
            output_variance = torch.var(output).item()
            self.training_metrics['communication_efficiency'].append(output_variance)
            
    def compute_routing_loss(self, attn_weights: torch.Tensor, target_connectivity: torch.Tensor = None) -> torch.Tensor:
        """라우팅 최적화를 위한 손실 함수 계산"""
        batch_size, num_agents_q, num_agents_kv = attn_weights.shape
        
        # 1. 어텐션 분산 손실 (과도한 집중 방지)
        attention_concentration_loss = torch.sum(attn_weights ** 2, dim=-1).mean()
        
        # 2. 통신 균형 손실 (모든 에이전트가 적절히 소통)
        agent_attention_sums = torch.sum(attn_weights, dim=1)  # 각 에이전트가 받는 전체 어텐션
        balance_target = torch.ones_like(agent_attention_sums) / num_agents_kv
        balance_loss = F.mse_loss(agent_attention_sums, balance_target)
        
        # 3. 목표 연결성 손실 (있는 경우)
        connectivity_loss = 0.0
        if target_connectivity is not None:
            connectivity_loss = F.mse_loss(attn_weights, target_connectivity)
            
        # 총 손실
        total_loss = 0.1 * attention_concentration_loss + 0.3 * balance_loss + 0.6 * connectivity_loss
        
        self.training_metrics['routing_loss'].append(total_loss.item())
        
        return total_loss
        
    def get_training_metrics(self) -> dict:
        """학습 메트릭 반환 및 초기화"""
        metrics = {key: np.mean(values) if values else 0.0 for key, values in self.training_metrics.items()}
        
        # 메트릭 초기화
        for key in self.training_metrics:
            self.training_metrics[key] = []
            
        return metrics
        
    def set_communication_topology(self, topology_type: str = 'full', **kwargs):
        """통신 토폴로지 설정"""
        self.topology_type = topology_type
        self.topology_kwargs = kwargs
        
    def generate_topology_mask(self, num_agents: int, device: torch.device) -> torch.Tensor:
        """지정된 토폴로지에 따른 어텐션 마스크 생성"""
        if self.topology_type == 'full':
            # 모든 에이전트 간 통신 허용
            return torch.zeros(1, num_agents, num_agents, dtype=torch.bool, device=device)
        elif self.topology_type == 'ring':
            # 링 토폴로지 (인접한 에이전트와만 통신)
            mask = torch.ones(1, num_agents, num_agents, dtype=torch.bool, device=device)
            for i in range(num_agents):
                # 자기 자신
                mask[0, i, i] = False
                # 이전 에이전트
                mask[0, i, (i - 1) % num_agents] = False
                # 다음 에이전트
                mask[0, i, (i + 1) % num_agents] = False
            return mask
        elif self.topology_type == 'random':
            # 랜덤 연결 (연결 확률 지정)
            connection_prob = kwargs.get('connection_prob', 0.3)
            mask = torch.rand(1, num_agents, num_agents, device=device) > connection_prob
            # 자기 자신과의 연결은 항상 허용
            for i in range(num_agents):
                mask[0, i, i] = False
            return mask
        else:
            raise ValueError(f"지원하지 않는 토폴로지 타입: {self.topology_type}")

if __name__ == '__main__':
    # 모델 사용 예시
    batch_size = 4
    num_agents = 10  # 전체 에이전트 수
    embed_dim = 64   # 에이전트 인코더 출력 차원
    num_heads = 8    # 어텐션 헤드 수

    # 모델 초기화
    attention_net = DistributedAttentionRouter(embed_dim=embed_dim, num_heads=num_heads)

    # 임의의 에이전트 페로몬 데이터 생성
    agent_pheromones = torch.randn(batch_size, num_agents, embed_dim)

    # 1. 마스크 없는 경우 (모든 에이전트가 서로 통신 가능)
    print("[1. No Masking]")
    attn_output, attn_weights = attention_net(agent_pheromones, agent_pheromones, agent_pheromones)
    print(f"Output shape: {attn_output.shape}")
    print(f"Weights shape: {attn_weights.shape}")
    print(f"Weights for agent 0 (sum={torch.sum(attn_weights[0, 0, :]):.2f}):\n{attn_weights[0, 0, :]}")


    # 2. 마스크 있는 경우 (에이전트 0은 짝수 에이전트와만 통신 가능)
    print("\n[2. With Masking]")
    mask = torch.zeros(batch_size, num_agents, num_agents, dtype=torch.bool)
    # 에이전트 0은 0, 2, 4, 6, 8번 에이전트와 통신 가능하도록 마스크 설정
    # 어텐션을 *허용할* 위치를 True로 설정
    mask[:, 0, ::2] = True 
    
    # MultiheadAttention은 마스킹될 위치(어텐션을 적용하지 않을 위치)를 True로 기대합니다.
    # 따라서 허용 마스크의 반전(~)을 사용합니다.
    final_mask = ~mask
    
    # 헤드 수만큼 마스크를 복제합니다.
    final_mask_for_attn = final_mask.repeat_interleave(num_heads, dim=0)
    
    attn_output_masked, attn_weights_masked = attention_net(
        agent_pheromones, agent_pheromones, agent_pheromones, attn_mask=final_mask_for_attn
    )

    print(f"Masked weights for agent 0 (sum={torch.sum(attn_weights_masked[0, 0, :]):.2f}):\n{attn_weights_masked[0, 0, :]}")
    
    # 마스킹된 위치(홀수 인덱스)의 가중치는 0이어야 함
    assert torch.all(attn_weights_masked[0, 0, 1::2] == 0)
    # 마스킹되지 않은 위치(짝수 인덱스)의 가중치는 0이 아니어야 함
    assert not torch.all(attn_weights_masked[0, 0, ::2] == 0)
    print("\n검증 통과: 어텐션 마스크가 올바르게 적용됩니다.")

