import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Any
import numpy as np
import logging
from collections import defaultdict

from src.models.attention_network import DistributedAttentionRouter
from src.models.diffusion_model import TemporalDiffusionModel

logger = logging.getLogger(__name__)


"""
분산 어텐션 라우터와 확산 모델의 학습을 담당하는 트레이너 클래스입니다.
"""


class PheromoneNetworkTrainer:
    """페로몬 네트워크 학습을 위한 트레이너"""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        
        # 모델 초기화
        self.attention_router = DistributedAttentionRouter(
            embed_dim=config['attention']['embed_dim'],
            num_heads=config['attention']['num_heads']
        ).to(device)
        
        self.diffusion_model = TemporalDiffusionModel(
            decay_factor=config['pheromone']['decay_rate']
        ).to(device)
        
        # 파라미터 확인 및 로깅
        attention_params = list(self.attention_router.parameters())
        diffusion_params = list(self.diffusion_model.parameters())
        
        logger.info(f"어텐션 라우터 파라미터 수: {sum(p.numel() for p in attention_params)}")
        logger.info(f"확산 모델 파라미터 수: {sum(p.numel() for p in diffusion_params)}")
        
        if not attention_params:
            logger.warning("어텐션 라우터에 학습 가능한 파라미터가 없습니다!")
        if not diffusion_params:
            logger.warning("확산 모델에 학습 가능한 파라미터가 없습니다!")
        
        # 옵티마이저 설정 (파라미터가 있는 경우에만)
        if attention_params:
            self.attention_optimizer = optim.Adam(
                self.attention_router.parameters(),
                lr=config['hyperparameters']['learning_rate'][0] * 0.5,  # 학습률 절반으로 감소
                weight_decay=5e-4  # 정규화 강화: 1e-5 -> 5e-4
            )
            logger.info("어텐션 옵티마이저 초기화 완료")
        else:
            self.attention_optimizer = None
            logger.warning("어텐션 옵티마이저를 초기화할 수 없습니다 (파라미터 없음)")
        
        if diffusion_params:
            self.diffusion_optimizer = optim.Adam(
                self.diffusion_model.parameters(),
                lr=config['hyperparameters']['learning_rate'][0] * 0.05,  # 확산 모델 학습률 더 감소: 0.1 -> 0.05
                weight_decay=3e-4  # 정규화 강화: 1e-5 -> 3e-4
            )
            logger.info("확산 옵티마이저 초기화 완료")
        else:
            self.diffusion_optimizer = None
            logger.warning("확산 옵티마이저를 초기화할 수 없습니다 (파라미터 없음)")
        
        # 학습률 스케줄러 (옵티마이저가 있는 경우에만)
        if self.attention_optimizer:
            self.attention_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.attention_optimizer, mode='min', factor=0.7, patience=15, verbose=True  # 더 빠른 학습률 감소
            )
        else:
            self.attention_scheduler = None
            
        if self.diffusion_optimizer:
            self.diffusion_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.diffusion_optimizer, mode='min', factor=0.7, patience=20, verbose=True  # 더 빠른 학습률 감소
            )
        else:
            self.diffusion_scheduler = None
            
        # 조기 종료를 위한 추가 속성들
        self.early_stopping_patience = 25
        self.early_stopping_counter = 0
        self.best_loss = float('inf')
        self.early_stopping_threshold = 1e-4
        
        # 학습 히스토리 추적
        self.training_history = defaultdict(list)
        self.epoch = 0
        self.previous_total_loss = None
        
        logger.info("페로몬 네트워크 트레이너 초기화 완료")
        
    def train_step(self, agent_embeddings: torch.Tensor, pheromone_field: torch.Tensor,
                   timestep: int, rewards: torch.Tensor) -> Dict[str, float]:
        """단일 학습 스텝 실행"""
        self.attention_router.train()
        self.diffusion_model.train()

        # 그래디언트 초기화 (옵티마이저가 있는 경우에만)
        if self.attention_optimizer:
            self.attention_optimizer.zero_grad()
        if self.diffusion_optimizer:
            self.diffusion_optimizer.zero_grad()

        losses = {}

        # 1. 어텐션 라우터 손실 계산
        # 에이전트 임베딩에 배치 차원 추가 (num_agents, embed_dim) -> (1, num_agents, embed_dim)
        if len(agent_embeddings.shape) == 2:
            agent_embeddings = agent_embeddings.unsqueeze(0)
        
        attention_loss = self._calculate_attention_loss(agent_embeddings, rewards, timestep)
        losses['attention_loss'] = attention_loss.item()

        # 2. 확산 모델 손실 계산
        diffusion_loss = self._calculate_diffusion_loss(pheromone_field, timestep)
        losses['diffusion_loss'] = diffusion_loss.item()

        # 3. 통합 손실 계산 (페로몬 신호와 어텐션의 일관성)
        consistency_loss = self._compute_consistency_loss(agent_embeddings, pheromone_field)
        losses['consistency_loss'] = consistency_loss.item()

        # NaN 체크 및 수정
        if np.isnan(attention_loss.item()) or np.isinf(attention_loss.item()):
            attention_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            losses['attention_loss'] = 0.0
            
        if np.isnan(diffusion_loss.item()) or np.isinf(diffusion_loss.item()):
            diffusion_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            losses['diffusion_loss'] = 0.0
            
        if np.isnan(consistency_loss.item()) or np.isinf(consistency_loss.item()):
            consistency_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            losses['consistency_loss'] = 0.0

        # 총 손실
        total_loss = attention_loss + diffusion_loss + 0.1 * consistency_loss
        losses['total_loss'] = total_loss.item()

        # 역전파 및 파라미터 업데이트 (옵티마이저가 있는 경우에만)
        if total_loss.requires_grad:
            total_loss.backward()
            
            if self.attention_optimizer:
                torch.nn.utils.clip_grad_norm_(self.attention_router.parameters(), 1.0)
                self.attention_optimizer.step()
                
            if self.diffusion_optimizer:
                torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), 1.0)
                self.diffusion_optimizer.step()

        # 학습 히스토리 업데이트
        for key, value in losses.items():
            self.training_history[key].append(value)
            
        # 학습률 추적 추가
        if self.attention_optimizer:
            current_lr_attention = self.attention_optimizer.param_groups[0]['lr']
            self.training_history['learning_rate_attention'].append(current_lr_attention)
            
        if self.diffusion_optimizer:
            current_lr_diffusion = self.diffusion_optimizer.param_groups[0]['lr']
            self.training_history['learning_rate_diffusion'].append(current_lr_diffusion)
            
        # 손실 개선 메트릭 계산 및 추가
        if self.previous_total_loss is not None:
            improvement_metrics = self._compute_loss_improvement_metrics(
                losses['total_loss'], 
                self.training_history['total_loss'][:-1]  # 현재 손실 제외
            )
            for key, value in improvement_metrics.items():
                self.training_history[key].append(value)
        
        self.previous_total_loss = losses['total_loss']
        
        # 조기 종료 체크
        self._check_early_stopping(losses['total_loss'])
        
        return losses
    
    def _check_early_stopping(self, current_loss: float) -> bool:
        """조기 종료 조건 확인"""
        if current_loss < self.best_loss - self.early_stopping_threshold:
            self.best_loss = current_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            
        if self.early_stopping_counter >= self.early_stopping_patience:
            logger.info(f"조기 종료 조건 만족: {self.early_stopping_patience} 에포크 동안 개선 없음")
            return True
        return False
    
    def should_stop_training(self) -> bool:
        """학습 중단 여부 확인"""
        return self.early_stopping_counter >= self.early_stopping_patience

    def _calculate_attention_loss(self, agent_embeddings: torch.Tensor, rewards: torch.Tensor, timestep: int) -> torch.Tensor:
        """어텐션 라우터 손실 계산"""
        try:
            # Forward pass
            routing_output, attn_weights = self.attention_router(
                agent_embeddings, agent_embeddings, agent_embeddings
            )

            # 라우팅 손실 계산
            routing_loss = self.attention_router.compute_routing_loss(attn_weights)

            # 보상 기반 손실 (어텐션이 높은 보상으로 이어지도록)
            if rewards is not None and rewards.numel() > 0:
                # 각 에이전트의 보상에 따른 어텐션 가중치 조정
                reward_normalized = torch.softmax(rewards, dim=0)
                attention_sum = torch.sum(attn_weights, dim=1)  # 각 에이전트가 받는 총 어텐션
                reward_loss = -torch.mean(reward_normalized * attention_sum.mean(dim=0))
                routing_loss += 0.3 * reward_loss

            return routing_loss
        except Exception as e:
            logger.error(f"어텐션 손실 계산 오류: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _calculate_diffusion_loss(self, pheromone_field: torch.Tensor, timestep: int) -> torch.Tensor:
        """확산 모델 손실 계산"""
        try:
            if pheromone_field.numel() == 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            original_shape = pheromone_field.shape
            logger.debug(f"확산 모델 입력 형태: {original_shape}")

            # 텐서 형태 검증 및 수정
            if len(pheromone_field.shape) == 4:
                # (batch, channels, height, width) -> (batch, sequence_length, vector_dim)
                batch_size, channels, height, width = pheromone_field.shape
                
                # 공간 차원을 시퀀스로 변환: (batch, channels, height, width) -> (batch, height*width, channels)
                spatial_size = height * width
                pheromone_field = pheromone_field.permute(0, 2, 3, 1)  # (batch, height, width, channels)
                pheromone_field = pheromone_field.contiguous().view(batch_size, spatial_size, channels)
                
                # 벡터 차원이 64가 되도록 조정
                if channels != 64:
                    adapter_key = f'_channel_adapter_{channels}_to_64'
                    if not hasattr(self, adapter_key):
                        setattr(self, adapter_key, nn.Linear(channels, 64).to(self.device))
                    # (batch, spatial_size, channels) -> (batch, spatial_size, 64)
                    adapter = getattr(self, adapter_key)
                    pheromone_field = adapter(pheromone_field)
                    
                logger.debug(f"변환된 형태: {pheromone_field.shape}")
                
            elif len(pheromone_field.shape) == 3:
                # 이미 올바른 형태: (batch, sequence_length, vector_dim)
                batch_size, seq_len, vector_dim = pheromone_field.shape
                if vector_dim != 64:
                    adapter_key = f'_channel_adapter_{vector_dim}_to_64'
                    if not hasattr(self, adapter_key):
                        setattr(self, adapter_key, nn.Linear(vector_dim, 64).to(self.device))
                    adapter = getattr(self, adapter_key)
                    pheromone_field = adapter(pheromone_field)
                    
            elif len(pheromone_field.shape) == 2:
                # (sequence_length, vector_dim) -> (1, sequence_length, vector_dim)
                pheromone_field = pheromone_field.unsqueeze(0)
                seq_len, vector_dim = pheromone_field.shape[1], pheromone_field.shape[2]
                if vector_dim != 64:
                    adapter_key = f'_channel_adapter_{vector_dim}_to_64'
                    if not hasattr(self, adapter_key):
                        setattr(self, adapter_key, nn.Linear(vector_dim, 64).to(self.device))
                    adapter = getattr(self, adapter_key)
                    pheromone_field = adapter(pheromone_field)
                    
            else:
                logger.warning(f"지원되지 않는 텐서 형태: {pheromone_field.shape}")
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            # 데이터 타입 통일 (float32로 변환)
            pheromone_field = pheromone_field.float()

            # 입력 검증
            if len(pheromone_field.shape) != 3:
                logger.error(f"확산 모델 입력 형태 오류: {pheromone_field.shape}, 예상: (batch, seq_len, 64)")
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            # 시계열 확산 적용
            diffused_field = self.diffusion_model(pheromone_field, timestep)

            # 확산 손실 계산 (안정성과 정보 보존의 균형)
            # 1. 과도한 감쇠 방지 (MSE 손실)
            preservation_loss = torch.mean((pheromone_field - diffused_field) ** 2)

            # 2. 공간적 일관성 (인접한 셀 간의 부드러운 전환)
            smoothness_loss = torch.tensor(0.0, device=self.device)
            if diffused_field.shape[1] > 1:  # sequence_length > 1
                # 시퀀스 내 인접 요소 간의 차이
                seq_diff = torch.abs(diffused_field[:, 1:, :] - diffused_field[:, :-1, :])
                smoothness_loss = torch.mean(seq_diff)

            # 3. 전체 정보량 보존 (엔트로피 유지)
            # 음수 값 처리를 위해 정규화 후 softmax 적용
            diffused_normalized = torch.softmax(diffused_field.view(-1, diffused_field.shape[-1]), dim=-1)
            pheromone_normalized = torch.softmax(pheromone_field.view(-1, pheromone_field.shape[-1]), dim=-1)
            
            field_entropy = -torch.sum(diffused_normalized * torch.log(diffused_normalized + 1e-8))
            target_entropy = -torch.sum(pheromone_normalized * torch.log(pheromone_normalized + 1e-8))
            entropy_loss = torch.abs(field_entropy - target_entropy)

            # 가중합으로 최종 손실 계산
            diffusion_loss = 0.5 * preservation_loss + 0.3 * smoothness_loss + 0.2 * entropy_loss

            logger.debug(f"확산 손실 - 보존: {preservation_loss:.4f}, 부드러움: {smoothness_loss:.4f}, 엔트로피: {entropy_loss:.4f}")

            return diffusion_loss
        except Exception as e:
            logger.error(f"확산 손실 계산 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
    def _compute_consistency_loss(self, agent_embeddings: torch.Tensor, pheromone_field: torch.Tensor) -> torch.Tensor:
        """어텐션과 페로몬 필드 간의 일관성 손실"""
        try:
            if pheromone_field.numel() == 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
                
            # 어텐션 가중치에서 파생된 공간적 영향력과 페로몬 분포의 일관성 확인
            _, attn_weights = self.attention_router(agent_embeddings, agent_embeddings, agent_embeddings)
            
            # 어텐션 가중치를 공간적 영향력으로 변환
            batch_size, num_agents = attn_weights.shape[0], attn_weights.shape[1]
            spatial_influence = torch.mean(attn_weights, dim=-1)  # 각 에이전트의 평균 영향력
            
            # 페로몬 필드의 전체적인 활성도
            if len(pheromone_field.shape) >= 3:
                # (batch, sequence_length, vector_dim) -> (batch, vector_dim)
                field_activity = torch.mean(pheromone_field, dim=1)
                # (batch, vector_dim) -> (batch,)
                field_activity = torch.mean(field_activity, dim=1)
            else:
                field_activity = torch.mean(pheromone_field, dim=-1)
                
            # 차원 맞추기
            if len(spatial_influence.shape) > len(field_activity.shape):
                field_activity = field_activity.unsqueeze(0).expand(spatial_influence.shape[0], -1)
            elif len(field_activity.shape) > len(spatial_influence.shape):
                spatial_influence = spatial_influence.unsqueeze(-1).expand(-1, -1, field_activity.shape[-1])
                spatial_influence = torch.mean(spatial_influence, dim=-1)
                
            # 일관성 손실 계산
            if spatial_influence.shape == field_activity.shape:
                consistency_loss = torch.mean((spatial_influence - field_activity) ** 2)
            else:
                # 크기가 맞지 않으면 0 손실
                consistency_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
            return consistency_loss
        except Exception as e:
            logger.error(f"일관성 손실 계산 오류: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
    def _compute_loss_improvement_metrics(self, current_loss: float, previous_losses: List[float]) -> Dict[str, float]:
        """손실 개선 메트릭 계산"""
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
        
    def evaluate(self, agent_embeddings: torch.Tensor, pheromone_field: torch.Tensor, timestep: int) -> Dict[str, float]:
        """모델 평가"""
        self.attention_router.eval()
        self.diffusion_model.eval()
        
        with torch.no_grad():
            metrics = {}
            
            # 어텐션 메트릭
            routing_output, attn_weights = self.attention_router(
                agent_embeddings, agent_embeddings, agent_embeddings
            )
            
            attention_metrics = self.attention_router.get_training_metrics()
            metrics.update(attention_metrics)
            
            # 어텐션 분포 분석
            attention_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1).mean()
            attention_sparsity = (attn_weights < 0.01).float().mean()  # 희소성
            
            metrics['attention_entropy'] = attention_entropy.item()
            metrics['attention_sparsity'] = attention_sparsity.item()
            
            # 페로몬 필드 메트릭
            if pheromone_field.numel() > 0:
                # 페로몬 필드 형태 변환 (확산 모델과 동일한 로직)
                processed_field = pheromone_field
                if len(pheromone_field.shape) == 4:
                    batch_size, channels, height, width = pheromone_field.shape
                    spatial_size = height * width
                    processed_field = pheromone_field.permute(0, 2, 3, 1).contiguous().view(batch_size, spatial_size, channels)
                    if channels != 64:
                        adapter_key = f'_channel_adapter_{channels}_to_64'
                        if not hasattr(self, adapter_key):
                            setattr(self, adapter_key, nn.Linear(channels, 64).to(self.device))
                        adapter = getattr(self, adapter_key)
                        processed_field = adapter(processed_field)
                elif len(pheromone_field.shape) == 2:
                    processed_field = pheromone_field.unsqueeze(0)
                    
                if len(processed_field.shape) == 3:
                    diffused_field = self.diffusion_model(processed_field, timestep)
                else:
                    diffused_field = processed_field  # 변환 실패시 원본 사용
                
                field_energy = torch.sum(diffused_field).item()
                field_variance = torch.var(diffused_field).item()
                field_sparsity = (diffused_field < 0.01).float().mean().item()
                
                metrics['field_energy'] = field_energy
                metrics['field_variance'] = field_variance
                metrics['field_sparsity'] = field_sparsity
                
        return metrics
        
    def update_learning_rates(self, attention_loss: float, diffusion_loss: float):
        """학습률 스케줄러 업데이트"""
        if self.attention_scheduler:
            self.attention_scheduler.step(attention_loss)
        if self.diffusion_scheduler:
            self.diffusion_scheduler.step(diffusion_loss)
        
    def save_checkpoint(self, filepath: str):
        """모델 체크포인트 저장"""
        checkpoint = {
            'epoch': self.epoch,
            'attention_router_state_dict': self.attention_router.state_dict(),
            'diffusion_model_state_dict': self.diffusion_model.state_dict(),
            'training_history': dict(self.training_history),
            'config': self.config
        }
        
        # 옵티마이저 상태 저장 (있는 경우에만)
        if self.attention_optimizer:
            checkpoint['attention_optimizer_state_dict'] = self.attention_optimizer.state_dict()
        if self.diffusion_optimizer:
            checkpoint['diffusion_optimizer_state_dict'] = self.diffusion_optimizer.state_dict()
            
        # 스케줄러 상태 저장 (있는 경우에만)
        if self.attention_scheduler:
            checkpoint['attention_scheduler_state_dict'] = self.attention_scheduler.state_dict()
        if self.diffusion_scheduler:
            checkpoint['diffusion_scheduler_state_dict'] = self.diffusion_scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"체크포인트 저장: {filepath}")
        
    def load_checkpoint(self, filepath: str):
        """모델 체크포인트 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.attention_router.load_state_dict(checkpoint['attention_router_state_dict'])
        self.diffusion_model.load_state_dict(checkpoint['diffusion_model_state_dict'])
        if self.attention_optimizer:
            self.attention_optimizer.load_state_dict(checkpoint['attention_optimizer_state_dict'])
        if self.diffusion_optimizer:
            self.diffusion_optimizer.load_state_dict(checkpoint['diffusion_optimizer_state_dict'])
        if self.attention_scheduler:
            self.attention_scheduler.load_state_dict(checkpoint['attention_scheduler_state_dict'])
        if self.diffusion_scheduler:
            self.diffusion_scheduler.load_state_dict(checkpoint['diffusion_scheduler_state_dict'])
        self.training_history = defaultdict(list, checkpoint['training_history'])
        
        logger.info(f"체크포인트 로드: {filepath}, 에포크: {self.epoch}")
        
    def get_training_summary(self) -> Dict[str, Any]:
        """학습 요약 정보 반환"""
        summary = {
            'epoch': self.epoch,
            'total_steps': len(self.training_history.get('total_loss', [])),
            'current_lr_attention': self.attention_optimizer.param_groups[0]['lr'] if self.attention_optimizer else 0,
            'current_lr_diffusion': self.diffusion_optimizer.param_groups[0]['lr'] if self.diffusion_optimizer else 0
        }
        
        # 최근 손실 평균
        for loss_type in ['attention_loss', 'diffusion_loss', 'consistency_loss', 'total_loss']:
            if loss_type in self.training_history and self.training_history[loss_type]:
                recent_losses = self.training_history[loss_type][-50:]  # 최근 50스텝
                summary[f'recent_{loss_type}'] = np.mean(recent_losses)
                
        return summary