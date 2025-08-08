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
        
        # 옵티마이저 설정
        self.attention_optimizer = optim.Adam(
            self.attention_router.parameters(),
            lr=config['hyperparameters']['learning_rate'][0],
            weight_decay=1e-5
        )
        
        self.diffusion_optimizer = optim.Adam(
            self.diffusion_model.parameters(),
            lr=config['hyperparameters']['learning_rate'][0] * 0.1,  # 확산 모델은 더 천천히 학습
            weight_decay=1e-5
        )
        
        # 학습률 스케줄러
        self.attention_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.attention_optimizer, mode='min', factor=0.5, patience=20, verbose=True
        )
        
        self.diffusion_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.diffusion_optimizer, mode='min', factor=0.5, patience=30, verbose=True
        )
        
        # 학습 히스토리 추적
        self.training_history = defaultdict(list)
        self.epoch = 0
        
        logger.info("페로몬 네트워크 트레이너 초기화 완료")
        
    def train_step(self, agent_embeddings: torch.Tensor, pheromone_field: torch.Tensor, 
                   timestep: int, rewards: torch.Tensor) -> Dict[str, float]:
        """단일 학습 스텝 실행"""
        self.attention_router.train()
        self.diffusion_model.train()
        
        losses = {}
        
        # 1. 어텐션 라우터 학습
        attention_loss = self._train_attention_router(agent_embeddings, rewards, timestep)
        losses['attention_loss'] = attention_loss.item()
        
        # 2. 확산 모델 학습
        diffusion_loss = self._train_diffusion_model(pheromone_field, timestep)
        losses['diffusion_loss'] = diffusion_loss.item()
        
        # 3. 통합 손실 계산 (페로몬 신호와 어텐션의 일관성)
        consistency_loss = self._compute_consistency_loss(agent_embeddings, pheromone_field)
        losses['consistency_loss'] = consistency_loss.item()
        
        # 총 손실
        total_loss = attention_loss + diffusion_loss + 0.1 * consistency_loss
        losses['total_loss'] = total_loss.item()
        
        # 학습 히스토리 업데이트
        for key, value in losses.items():
            self.training_history[key].append(value)
            
        return losses
        
    def _train_attention_router(self, agent_embeddings: torch.Tensor, rewards: torch.Tensor, timestep: int) -> torch.Tensor:
        """어텐션 라우터 학습"""
        self.attention_optimizer.zero_grad()
        
        # Forward pass
        routing_output, attn_weights = self.attention_router(
            agent_embeddings, agent_embeddings, agent_embeddings
        )
        
        # 라우팅 손실 계산
        routing_loss = self.attention_router.compute_routing_loss(attn_weights)
        
        # 보상 기반 손실 (어텐션이 높은 보상으로 이어지도록)
        if rewards is not None:
            # 각 에이전트의 보상에 따른 어텐션 가중치 조정
            reward_normalized = torch.softmax(rewards, dim=0)
            attention_sum = torch.sum(attn_weights, dim=1)  # 각 에이전트가 받는 총 어텐션
            reward_loss = -torch.mean(reward_normalized * attention_sum.mean(dim=0))
            routing_loss += 0.3 * reward_loss
            
        # Backward pass
        routing_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.attention_router.parameters(), 1.0)
        self.attention_optimizer.step()
        
        return routing_loss
        
    def _train_diffusion_model(self, pheromone_field: torch.Tensor, timestep: int) -> torch.Tensor:
        """확산 모델 학습"""
        self.diffusion_optimizer.zero_grad()
        
        if pheromone_field.numel() == 0:
            return torch.tensor(0.0, device=self.device)
            
        # 시계열 확산 적용
        diffused_field = self.diffusion_model(pheromone_field, timestep)
        
        # 확산 손실 계산 (안정성과 정보 보존의 균형)
        # 1. 과도한 감쇠 방지
        preservation_loss = torch.mean((pheromone_field - diffused_field) ** 2)
        
        # 2. 공간적 일관성 (인접한 셀 간의 부드러운 전환)
        if len(pheromone_field.shape) >= 4:  # (batch, channels, height, width)
            # 수평 차이
            h_diff = torch.abs(diffused_field[:, :, 1:, :] - diffused_field[:, :, :-1, :])
            # 수직 차이
            v_diff = torch.abs(diffused_field[:, :, :, 1:] - diffused_field[:, :, :, :-1])
            smoothness_loss = torch.mean(h_diff) + torch.mean(v_diff)
        else:
            smoothness_loss = torch.tensor(0.0, device=self.device)
            
        # 3. 전체 정보량 보존 (엔트로피 유지)
        field_entropy = -torch.sum(diffused_field * torch.log(diffused_field + 1e-8))
        target_entropy = -torch.sum(pheromone_field * torch.log(pheromone_field + 1e-8))
        entropy_loss = torch.abs(field_entropy - target_entropy)
        
        diffusion_loss = 0.5 * preservation_loss + 0.3 * smoothness_loss + 0.2 * entropy_loss
        
        # Backward pass
        diffusion_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), 1.0)
        self.diffusion_optimizer.step()
        
        return diffusion_loss
        
    def _compute_consistency_loss(self, agent_embeddings: torch.Tensor, pheromone_field: torch.Tensor) -> torch.Tensor:
        """어텐션과 페로몬 필드 간의 일관성 손실"""
        if pheromone_field.numel() == 0:
            return torch.tensor(0.0, device=self.device)
            
        # 어텐션 가중치에서 파생된 공간적 영향력과 페로몬 분포의 일관성 확인
        _, attn_weights = self.attention_router(agent_embeddings, agent_embeddings, agent_embeddings)
        
        # 어텐션 가중치를 공간적 영향력으로 변환
        batch_size, num_agents = attn_weights.shape[0], attn_weights.shape[1]
        spatial_influence = torch.mean(attn_weights, dim=-1)  # 각 에이전트의 평균 영향력
        
        # 페로몬 필드의 전체적인 활성도
        if len(pheromone_field.shape) >= 3:
            field_activity = torch.mean(pheromone_field, dim=(1, 2))
            if field_activity.shape[0] == 1 and len(field_activity.shape) >= 2:
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
            consistency_loss = torch.tensor(0.0, device=self.device)
            
        return consistency_loss
        
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
                diffused_field = self.diffusion_model(pheromone_field, timestep)
                
                field_energy = torch.sum(diffused_field).item()
                field_variance = torch.var(diffused_field).item()
                field_sparsity = (diffused_field < 0.01).float().mean().item()
                
                metrics['field_energy'] = field_energy
                metrics['field_variance'] = field_variance
                metrics['field_sparsity'] = field_sparsity
                
        return metrics
        
    def update_learning_rates(self, attention_loss: float, diffusion_loss: float):
        """학습률 스케줄러 업데이트"""
        self.attention_scheduler.step(attention_loss)
        self.diffusion_scheduler.step(diffusion_loss)
        
    def save_checkpoint(self, filepath: str):
        """모델 체크포인트 저장"""
        checkpoint = {
            'epoch': self.epoch,
            'attention_router_state_dict': self.attention_router.state_dict(),
            'diffusion_model_state_dict': self.diffusion_model.state_dict(),
            'attention_optimizer_state_dict': self.attention_optimizer.state_dict(),
            'diffusion_optimizer_state_dict': self.diffusion_optimizer.state_dict(),
            'attention_scheduler_state_dict': self.attention_scheduler.state_dict(),
            'diffusion_scheduler_state_dict': self.diffusion_scheduler.state_dict(),
            'training_history': dict(self.training_history),
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"체크포인트 저장: {filepath}")
        
    def load_checkpoint(self, filepath: str):
        """모델 체크포인트 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.attention_router.load_state_dict(checkpoint['attention_router_state_dict'])
        self.diffusion_model.load_state_dict(checkpoint['diffusion_model_state_dict'])
        self.attention_optimizer.load_state_dict(checkpoint['attention_optimizer_state_dict'])
        self.diffusion_optimizer.load_state_dict(checkpoint['diffusion_optimizer_state_dict'])
        self.attention_scheduler.load_state_dict(checkpoint['attention_scheduler_state_dict'])
        self.diffusion_scheduler.load_state_dict(checkpoint['diffusion_scheduler_state_dict'])
        self.training_history = defaultdict(list, checkpoint['training_history'])
        
        logger.info(f"체크포인트 로드: {filepath}, 에포크: {self.epoch}")
        
    def get_training_summary(self) -> Dict[str, Any]:
        """학습 요약 정보 반환"""
        summary = {
            'epoch': self.epoch,
            'total_steps': len(self.training_history.get('total_loss', [])),
            'current_lr_attention': self.attention_optimizer.param_groups[0]['lr'],
            'current_lr_diffusion': self.diffusion_optimizer.param_groups[0]['lr']
        }
        
        # 최근 손실 평균
        for loss_type in ['attention_loss', 'diffusion_loss', 'consistency_loss', 'total_loss']:
            if loss_type in self.training_history and self.training_history[loss_type]:
                recent_losses = self.training_history[loss_type][-50:]  # 최근 50스텝
                summary[f'recent_{loss_type}'] = np.mean(recent_losses)
                
        return summary