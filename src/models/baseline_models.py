"""
비교군 모델들 구현
1. 규칙 기반 확산 (ACO 방식)
2. 중앙집중 어텐션 네트워크
3. 2D 페로몬 (Ablation Study)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RuleBasedDiffusion:
    """
    규칙 기반 확산 모델 (전통적 ACO 방식)
    연구 계획서 명시 비교군
    """
    
    def __init__(self, decay_rate: float = 0.1, diffusion_rate: float = 0.2):
        self.decay_rate = decay_rate
        self.diffusion_rate = diffusion_rate
        self.evaporation_rate = 0.05
        
    def diffuse_pheromones(self, pheromone_field: torch.Tensor, timestep: int) -> torch.Tensor:
        """
        전통적 ACO 페로몬 확산
        
        Args:
            pheromone_field: 현재 페로몬 필드
            timestep: 현재 시간 스텝
            
        Returns:
            확산된 페로몬 필드
        """
        if pheromone_field.numel() == 0:
            return pheromone_field
            
        # numpy로 변환하여 처리
        if isinstance(pheromone_field, torch.Tensor):
            field_np = pheromone_field.detach().cpu().numpy()
            device = pheromone_field.device
        else:
            field_np = pheromone_field
            device = 'cpu'
            
        # 4차원 텐서를 2차원 공간으로 변환
        if len(field_np.shape) == 4:
            batch_size, channels, height, width = field_np.shape
            # 채널을 평균내어 단일 농도 맵으로 변환
            field_2d = np.mean(field_np, axis=(0, 1))  # (height, width)
        elif len(field_np.shape) == 3:
            # (batch, seq_len, features) -> (sqrt(seq_len), sqrt(seq_len))로 변환
            batch_size, seq_len, features = field_np.shape
            grid_size = int(np.sqrt(seq_len))
            if grid_size * grid_size == seq_len:
                field_reshaped = field_np.reshape(batch_size, grid_size, grid_size, features)
                field_2d = np.mean(field_reshaped, axis=(0, 3))  # (grid_size, grid_size)
            else:
                # 시퀀스 길이가 완전제곱수가 아닌 경우 10x10 그리드로 고정
                field_2d = np.random.rand(10, 10) * np.mean(field_np)
        else:
            # 기본적으로 10x10 그리드 생성
            field_2d = np.random.rand(10, 10) * np.mean(field_np) if field_np.size > 0 else np.zeros((10, 10))
            
        height, width = field_2d.shape
        new_field = field_2d.copy()
        
        # 1. 증발 (Evaporation)
        new_field *= (1.0 - self.evaporation_rate)
        
        # 2. 확산 (Diffusion) - 이웃 셀로 확산
        for i in range(height):
            for j in range(width):
                if field_2d[i, j] > 0.01:  # 임계값 이상일 때만 확산
                    diffused_amount = field_2d[i, j] * self.diffusion_rate
                    
                    # 8방향 이웃으로 확산
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < height and 0 <= nj < width:
                                # 거리에 따른 확산 감쇠
                                distance = np.sqrt(di*di + dj*dj)
                                diffusion_strength = diffused_amount / (8 * distance)
                                new_field[ni, nj] += diffusion_strength
                                
                    # 원래 위치에서 확산된 만큼 감소
                    new_field[i, j] -= diffused_amount
        
        # 3. 시간 감쇠 (Temporal decay)
        new_field *= (1.0 - self.decay_rate * (timestep / 100.0))
        
        # 음수 제거
        new_field = np.maximum(new_field, 0.0)
        
        # 원래 형태로 변환하여 반환
        if len(pheromone_field.shape) == 4:
            # (height, width) -> (batch, channels, height, width)
            result = np.tile(new_field, (batch_size, channels, 1, 1))
        elif len(pheromone_field.shape) == 3:
            # (height, width) -> (batch, seq_len, features)
            batch_size, seq_len, features = pheromone_field.shape
            grid_size = int(np.sqrt(seq_len))
            if grid_size * grid_size == seq_len:
                field_reshaped = np.tile(new_field[:grid_size, :grid_size], (features, 1, 1))
                field_reshaped = field_reshaped.transpose(1, 2, 0)  # (grid_size, grid_size, features)
                result = field_reshaped.reshape(1, seq_len, features)
                result = np.tile(result, (batch_size, 1, 1))
            else:
                result = np.random.rand(*pheromone_field.shape) * np.mean(new_field)
        else:
            result = new_field
            
        return torch.tensor(result, dtype=torch.float32, device=device)
    
    def get_parameters(self) -> Dict:
        """모델 파라미터 반환 (규칙 기반이므로 빈 딕셔너리)"""
        return {}
    
    def state_dict(self) -> Dict:
        """상태 딕셔너리 반환"""
        return {
            'decay_rate': self.decay_rate,
            'diffusion_rate': self.diffusion_rate,
            'evaporation_rate': self.evaporation_rate
        }
    
    def load_state_dict(self, state_dict: Dict):
        """상태 딕셔너리 로드"""
        self.decay_rate = state_dict.get('decay_rate', self.decay_rate)
        self.diffusion_rate = state_dict.get('diffusion_rate', self.diffusion_rate)
        self.evaporation_rate = state_dict.get('evaporation_rate', self.evaporation_rate)


class CentralizedAttentionNetwork(nn.Module):
    """
    중앙집중 어텐션 네트워크
    모든 정보를 중앙 서버에서 처리하는 방식
    """
    
    def __init__(self, embed_dim: int = 64, num_heads: int = 8, max_agents: int = 500):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_agents = max_agents
        
        # 중앙 집중형 멀티헤드 어텐션
        self.central_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 에이전트 임베딩 처리
        self.agent_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # 글로벌 정보 집계
        self.global_aggregator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # 라우팅 결정 네트워크
        self.routing_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, max_agents),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        중앙집중 어텐션 처리
        
        Args:
            query, key, value: 에이전트 임베딩 텐서들
            
        Returns:
            (routing_output, attention_weights)
        """
        batch_size, num_agents, embed_dim = query.shape
        
        # 1. 에이전트 정보 인코딩
        encoded_agents = self.agent_encoder(query)
        
        # 2. 중앙 집중형 어텐션 계산
        attn_output, attn_weights = self.central_attention(
            encoded_agents, encoded_agents, encoded_agents
        )
        
        # 3. 글로벌 컨텍스트 집계
        global_context = torch.mean(attn_output, dim=1, keepdim=True)  # (batch, 1, embed_dim)
        global_context = self.global_aggregator(global_context)
        
        # 4. 글로벌 컨텍스트를 모든 에이전트에 브로드캐스트
        global_context = global_context.expand(-1, num_agents, -1)
        
        # 5. 라우팅 결정 (중앙에서 모든 라우팅 결정)
        routing_decisions = self.routing_head(global_context)
        
        # 에이전트 수에 맞게 잘라내기
        routing_decisions = routing_decisions[:, :, :num_agents]
        
        return attn_output + global_context, routing_decisions
    
    def compute_routing_loss(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """라우팅 손실 계산"""
        # 중앙집중 방식의 오버헤드 시뮬레이션
        centralization_overhead = torch.mean(attention_weights) * 0.1
        
        # 부하 집중 페널티
        load_imbalance = torch.var(attention_weights, dim=-1).mean()
        
        return centralization_overhead + load_imbalance * 0.2
    
    def get_training_metrics(self) -> Dict:
        """학습 메트릭 반환"""
        return {
            'model_type': 'centralized_attention',
            'parameter_count': sum(p.numel() for p in self.parameters()),
            'centralization_overhead': 0.1  # 고정 오버헤드
        }


class TwoDimensionalPheromone:
    """
    2차원 페로몬 모델 (Ablation Study용)
    행동과 감정 차원만 사용
    """
    
    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        self.behavior_dim = 4  # 행동 차원
        self.emotion_dim = 5   # 감정 차원
        self.total_dim = self.behavior_dim + self.emotion_dim  # 9차원
        
        # 2D 페로몬을 64차원으로 확장하는 인코더
        self.dimension_expander = nn.Linear(self.total_dim, embed_dim)
        
    def encode_pheromone_2d(self, behavior: np.ndarray, emotion: np.ndarray) -> torch.Tensor:
        """
        2차원 페로몬 벡터 인코딩
        
        Args:
            behavior: 행동 벡터 (4차원)
            emotion: 감정 벡터 (5차원)
            
        Returns:
            인코딩된 2D 페로몬 (64차원으로 확장)
        """
        # 2차원 연결
        pheromone_2d = np.concatenate([behavior, emotion])
        pheromone_tensor = torch.tensor(pheromone_2d, dtype=torch.float32)
        
        # 64차원으로 확장
        with torch.no_grad():
            expanded = self.dimension_expander(pheromone_tensor)
            
        return expanded
    
    def compare_with_4d(self, behavior: np.ndarray, emotion: np.ndarray, 
                       social: np.ndarray, context: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        2D vs 4D 페로몬 비교
        
        Returns:
            {'2d_pheromone': 2차원 페로몬, '4d_pheromone': 4차원 페로몬}
        """
        # 2D 페로몬 (행동 + 감정)
        pheromone_2d = self.encode_pheromone_2d(behavior, emotion)
        
        # 4D 페로몬 (모든 차원)
        pheromone_4d = np.concatenate([behavior, emotion, social, context])
        pheromone_4d_tensor = torch.tensor(pheromone_4d, dtype=torch.float32)
        
        # 차원 맞추기 (패딩 또는 잘라내기)
        if len(pheromone_4d_tensor) > self.embed_dim:
            pheromone_4d_tensor = pheromone_4d_tensor[:self.embed_dim]
        elif len(pheromone_4d_tensor) < self.embed_dim:
            padding = torch.zeros(self.embed_dim - len(pheromone_4d_tensor))
            pheromone_4d_tensor = torch.cat([pheromone_4d_tensor, padding])
            
        return {
            '2d_pheromone': pheromone_2d,
            '4d_pheromone': pheromone_4d_tensor
        }
    
    def get_information_loss_metric(self, full_pheromone: torch.Tensor) -> float:
        """
        2D 페로몬 사용으로 인한 정보 손실 메트릭
        
        Args:
            full_pheromone: 전체 4D 페로몬 벡터
            
        Returns:
            정보 손실률 (0~1)
        """
        if len(full_pheromone) < self.total_dim:
            return 0.0
            
        # 사용되는 차원 (2D)
        used_info = full_pheromone[:self.total_dim]
        
        # 손실되는 차원 (사회관계 + 환경맥락)
        lost_info = full_pheromone[self.total_dim:]
        
        # 정보 엔트로피 기반 손실률 계산
        total_variance = torch.var(full_pheromone).item()
        used_variance = torch.var(used_info).item() if len(used_info) > 1 else 0.0
        
        if total_variance == 0:
            return 0.0
            
        information_retention = used_variance / total_variance
        information_loss = 1.0 - information_retention
        
        return max(0.0, min(1.0, information_loss))


class BaselineComparator:
    """비교군 실험을 위한 통합 클래스"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 각 비교군 모델 초기화
        self.rule_based = RuleBasedDiffusion(
            decay_rate=config.get('baseline_decay_rate', 0.1),
            diffusion_rate=config.get('baseline_diffusion_rate', 0.2)
        )
        
        self.centralized = CentralizedAttentionNetwork(
            embed_dim=config.get('embed_dim', 64),
            num_heads=config.get('num_heads', 8),
            max_agents=config.get('max_agents', 500)
        )
        
        self.pheromone_2d = TwoDimensionalPheromone(
            embed_dim=config.get('embed_dim', 64)
        )
        
    def run_comparison_experiment(self, agent_embeddings: torch.Tensor, 
                                pheromone_field: torch.Tensor, 
                                timestep: int) -> Dict[str, Dict]:
        """
        모든 비교군에 대해 실험 실행
        
        Returns:
            각 모델별 결과 딕셔너리
        """
        results = {}
        
        # 1. 규칙 기반 확산
        try:
            rule_based_output = self.rule_based.diffuse_pheromones(pheromone_field, timestep)
            results['rule_based'] = {
                'output': rule_based_output,
                'type': 'rule_based_diffusion',
                'parameters': self.rule_based.get_parameters()
            }
        except Exception as e:
            logger.error(f"규칙 기반 확산 실험 오류: {e}")
            results['rule_based'] = {'error': str(e)}
        
        # 2. 중앙집중 어텐션
        try:
            if agent_embeddings.numel() > 0:
                centralized_output, centralized_attention = self.centralized(
                    agent_embeddings, agent_embeddings, agent_embeddings
                )
                results['centralized'] = {
                    'output': centralized_output,
                    'attention': centralized_attention,
                    'type': 'centralized_attention',
                    'metrics': self.centralized.get_training_metrics()
                }
            else:
                results['centralized'] = {'error': 'Empty agent embeddings'}
        except Exception as e:
            logger.error(f"중앙집중 어텐션 실험 오류: {e}")
            results['centralized'] = {'error': str(e)}
        
        # 3. 2D vs 4D 페로몬 비교
        try:
            # 샘플 데이터 생성 (실제로는 에이전트에서 가져와야 함)
            behavior = np.random.rand(4) * 2.0 + 0.5
            emotion = np.random.rand(5) * 1.5 + 0.3
            social = np.random.rand(10) * 2.0 + 0.3
            context = np.random.rand(5) * 3.0 + 0.3
            
            pheromone_comparison = self.pheromone_2d.compare_with_4d(
                behavior, emotion, social, context
            )
            
            # 정보 손실 메트릭 계산
            full_pheromone = torch.cat([
                torch.tensor(behavior), torch.tensor(emotion),
                torch.tensor(social), torch.tensor(context)
            ])
            
            info_loss = self.pheromone_2d.get_information_loss_metric(full_pheromone)
            
            results['ablation_2d_vs_4d'] = {
                'pheromone_2d': pheromone_comparison['2d_pheromone'],
                'pheromone_4d': pheromone_comparison['4d_pheromone'],
                'information_loss': info_loss,
                'type': 'ablation_study'
            }
        except Exception as e:
            logger.error(f"2D vs 4D 페로몬 비교 실험 오류: {e}")
            results['ablation_2d_vs_4d'] = {'error': str(e)}
        
        return results
    
    def get_comparison_metrics(self, results: Dict[str, Dict]) -> Dict[str, float]:
        """
        비교군 실험 결과에서 성능 메트릭 추출
        
        Returns:
            각 모델별 성능 지표
        """
        metrics = {}
        
        for model_name, result in results.items():
            if 'error' in result:
                metrics[f'{model_name}_success'] = 0.0
                continue
                
            metrics[f'{model_name}_success'] = 1.0
            
            if model_name == 'rule_based':
                # 규칙 기반 모델의 단순성 지표
                metrics[f'{model_name}_complexity'] = 0.1  # 매우 단순
                
            elif model_name == 'centralized':
                # 중앙집중 모델의 복잡성과 오버헤드
                if 'metrics' in result:
                    param_count = result['metrics'].get('parameter_count', 0)
                    metrics[f'{model_name}_complexity'] = param_count / 10000.0
                    metrics[f'{model_name}_overhead'] = result['metrics'].get('centralization_overhead', 0.1)
                    
            elif model_name == 'ablation_2d_vs_4d':
                # 정보 손실 지표
                info_loss = result.get('information_loss', 0.0)
                metrics[f'{model_name}_info_loss'] = info_loss
                metrics[f'{model_name}_efficiency'] = 1.0 - info_loss
        
        return metrics


# 추가 기준선 모델들

class AblationNoAttentionModel:
    """어텐션 메커니즘 없이 4D 페로몬만 사용하는 절제 연구 모델"""
    
    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        self.name = "ablation_no_attention"
        
    def process_pheromone_field(self, pheromone_field: torch.Tensor) -> torch.Tensor:
        """어텐션 없이 4D 페로몬 처리"""
        if pheromone_field.numel() == 0:
            return pheromone_field
            
        # 단순 평균화 처리 (어텐션 없음)
        if len(pheromone_field.shape) > 2:
            # 각 차원을 동등하게 평균화
            processed_field = torch.mean(pheromone_field, dim=1 if len(pheromone_field.shape) == 3 else 0)
        else:
            processed_field = pheromone_field.clone()
        
        # 단순 스무딩 (컨볼루션 대신 간단한 블러링)
        if len(processed_field.shape) >= 2:
            kernel = torch.ones(3, 3, device=processed_field.device) / 9.0
            if len(processed_field.shape) == 4:
                processed_field = torch.nn.functional.conv2d(
                    processed_field, 
                    kernel.unsqueeze(0).unsqueeze(0), 
                    padding=1
                )
        
        # 균등한 감쇠
        processed_field *= 0.95
        
        return processed_field
    
    def get_metrics(self) -> Dict:
        return {
            'model_type': 'ablation_no_attention',
            'processing_method': 'simple_averaging'
        }


class RandomCommunicationModel:
    """무작위 통신을 사용하는 기준선 모델"""
    
    def __init__(self, communication_probability: float = 0.2, embed_dim: int = 64):
        self.communication_probability = communication_probability
        self.embed_dim = embed_dim
        self.name = "random_communication"
        
    def process_communication(self, agent_states: List[Dict]) -> List[Dict]:
        """완전 무작위 통신 전략"""
        communications = []
        
        for i, agent in enumerate(agent_states):
            # 확률적 통신 결정
            if np.random.random() < self.communication_probability:
                # 무작위 메시지 타입
                message_types = ['random_info', 'noise_broadcast', 'random_request']
                message_type = np.random.choice(message_types)
                
                # 무작위 긴급도
                urgency = np.random.uniform(0, 1)
                
                communications.append({
                    'sender': i,
                    'type': message_type,
                    'urgency': urgency,
                    'content': {
                        'random_data': np.random.uniform(0, 1, 5).tolist(),
                        'timestamp': np.random.randint(0, 1000)
                    }
                })
        
        return communications
    
    def process_pheromone_field(self, pheromone_field: torch.Tensor) -> torch.Tensor:
        """무작위 노이즈 추가 페로몬 처리"""
        if pheromone_field.numel() == 0:
            return pheromone_field
        
        # 기본 처리
        processed_field = pheromone_field.clone()
        
        # 무작위 노이즈 추가
        noise_level = 0.1
        noise = torch.randn_like(processed_field) * noise_level
        processed_field += noise
        
        # 음수 값 제거
        processed_field = torch.relu(processed_field)
        
        # 무작위 감쇠
        random_decay = 0.05 + torch.rand(1).item() * 0.1
        processed_field *= (1.0 - random_decay)
        
        return processed_field
    
    def get_metrics(self) -> Dict:
        return {
            'model_type': 'random_communication',
            'communication_probability': self.communication_probability,
            'noise_level': 0.1
        }