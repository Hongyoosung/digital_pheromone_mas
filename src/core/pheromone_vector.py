import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


"""
4D 페로몬 벡터의 자료구조와 공간 내 페로몬 필드를 관리하는 클래스를 정의합니다.
"""


@dataclass
class PheromoneVector:
    """4D Digital Pheromone Vector"""
    behavior: np.ndarray  # Action probabilities
    emotion: np.ndarray   # Emotional states
    social: np.ndarray    # Social relationships
    context: np.ndarray   # Environmental context
    timestamp: float
    agent_id: int

    def get_total_magnitude(self) -> float:
        """페로몬 벡터의 전체 강도를 계산합니다."""
        if self.behavior.size == 0 and self.emotion.size == 0 and self.social.size == 0 and self.context.size == 0:
            return 0.0
        return np.linalg.norm(
            np.concatenate([self.behavior, self.emotion, self.social, self.context])
        )
    
    def to_tensor(self, device='cuda'):
        """Convert to PyTorch tensor"""
        return torch.tensor(
            np.concatenate([
                self.behavior, 
                self.emotion, 
                self.social, 
                self.context
            ]),
            dtype=torch.float32,
            device=device
        )
    
    def decay(self, rate: float):
        """Apply temporal decay with numerical stability"""
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"감쇠율은 0과 1 사이여야 함: {rate}")
        
        current_magnitude = self.get_total_magnitude()
        
        if current_magnitude < 0.05:
            effective_rate = max(rate, 0.995)
        elif current_magnitude > 5.0:
            effective_rate = min(rate, 0.92)
        else:
            effective_rate = rate
        
        min_threshold = 1e-6
        
        self.behavior = np.maximum(self.behavior * effective_rate, min_threshold)
        self.emotion = np.maximum(self.emotion * effective_rate, min_threshold)
        self.social = np.maximum(self.social * effective_rate, min_threshold)
        self.context = np.maximum(self.context * effective_rate, min_threshold)
        
        cleanup_threshold = min_threshold * 100
        self.behavior[self.behavior < cleanup_threshold] = 0
        self.emotion[self.emotion < cleanup_threshold] = 0
        self.social[self.social < cleanup_threshold] = 0
        self.context[self.context < cleanup_threshold] = 0
        
    def __add__(self, other):
        """Vector addition for pheromone aggregation"""
        return PheromoneVector(
            behavior=self.behavior + other.behavior,
            emotion=self.emotion + other.emotion,
            social=self.social + other.social,
            context=self.context + other.context,
            timestamp=max(self.timestamp, other.timestamp),
            agent_id=self.agent_id
        )

    @classmethod
    def zeros(cls, p_dims: Dict[str, int]):
        """Create a PheromoneVector filled with zeros."""
        return cls(
            behavior=np.zeros(p_dims['behavior']),
            emotion=np.zeros(p_dims['emotion']),
            social=np.zeros(p_dims['social']),
            context=np.zeros(p_dims['context']),
            timestamp=0.0,
            agent_id=-1
        )

class PheromoneEncoder(nn.Module):
    """Neural encoder for pheromone vectors"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

class PheromoneField:
    """Spatial pheromone field management"""
    def __init__(self, grid_size: Tuple[int, int], decay_rate: float = 0.98):
        self.grid_size = grid_size
        self.decay_rate = decay_rate
        self.field: Dict[Tuple[int, int], List[PheromoneVector]] = {}
        
        # 초기 페로몬 필드 생성 - 빈 필드 문제 해결
        self._initialize_field()
        
    def _initialize_field(self):
        """초기 페로몬 필드를 랜덤하게 생성하여 활성화"""
        logger.info(f"페로몬 필드 초기화: {self.grid_size}")
        
        # 맵의 10% 위치에 초기 페로몬 생성
        num_initial_pheromones = max(1, int(self.grid_size[0] * self.grid_size[1] * 0.1))
        
        for _ in range(num_initial_pheromones):
            x = np.random.randint(0, self.grid_size[0])
            y = np.random.randint(0, self.grid_size[1])
            position = (x, y)
            
            # 초기 페로몬 벡터 생성
            initial_pheromone = PheromoneVector(
                behavior=np.random.rand(4) * 2.0 + 0.5,  # 0.5~2.5 범위
                emotion=np.random.rand(5) * 1.5 + 0.3,   # 0.3~1.8 범위
                social=np.random.rand(10) * 1.2 + 0.2,   # 0.2~1.4 범위
                context=np.random.rand(5) * 1.8 + 0.4,   # 0.4~2.2 범위
                timestamp=time.time(),
                agent_id=-1  # 시스템 생성 페로몬
            )
            
            self.field[position] = [initial_pheromone]
        
        logger.info(f"초기 페로몬 {len(self.field)}개 생성 완료")
        
    def deposit(self, position: Tuple[int, int], pheromone: PheromoneVector):
        """Deposit pheromone at position with enhanced logic"""
        if position not in self.field:
            self.field[position] = []
        
        # 페로몬 강도 증가 (기존 문제 해결)
        enhanced_pheromone = PheromoneVector(
            behavior=np.clip(pheromone.behavior * 3.0, 0.1, 5.0),  # 강도 증가
            emotion=np.clip(pheromone.emotion * 2.5, 0.1, 4.0),
            social=np.clip(pheromone.social * 2.8, 0.1, 4.5),
            context=np.clip(pheromone.context * 2.2, 0.1, 3.5),
            timestamp=pheromone.timestamp,
            agent_id=pheromone.agent_id
        )
        
        self.field[position].append(enhanced_pheromone)
        
        # 디버깅을 위한 로깅 (필요시 주석 처리)
        if len(self.field[position]) % 10 == 0:
            logger.debug(f"위치 {position}에 페로몬 {len(self.field[position])}개 누적")
        
    def diffuse(self, radius: int = 5, device: str = 'cuda'):
        """Diffuse pheromones using GPU-accelerated convolution."""
        if not self.field or 'cuda' not in str(device) or not torch.cuda.is_available():
            self._diffuse_cpu(radius)
            return

        try:
            first_pheromone = self.field[next(iter(self.field))][0]
            p_dims = {
                'behavior': first_pheromone.behavior.shape[0],
                'emotion': first_pheromone.emotion.shape[0],
                'social': first_pheromone.social.shape[0],
                'context': first_pheromone.context.shape[0],
            }
            total_dim = sum(p_dims.values())
            
            field_tensor = self.get_field_as_tensor(p_dims, device=device)

            kernel_size = radius * 2 + 1
            # 커널의 데이터 타입을 필드 텐서와 일치시킴 (float16)
            kernel = self._create_gaussian_kernel(kernel_size, sigma=radius / 2).to(device, dtype=field_tensor.dtype)
            kernel = kernel.repeat(total_dim, 1, 1, 1)

            with torch.no_grad():
                diffused_tensor = torch.nn.functional.conv2d(
                    field_tensor,
                    kernel,
                    padding='same',
                    groups=total_dim
                )

            self._update_field_from_tensor(diffused_tensor, p_dims)
        except Exception as e:
            logger.error(f"GPU diffusion failed: {e}. Falling back to CPU.")
            self._diffuse_cpu(radius)

    def _create_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Creates a 2D Gaussian kernel."""
        ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    def _update_field_from_tensor(self, tensor: torch.Tensor, p_dims: Dict):
        """Update pheromone field from a tensor."""
        tensor_np = tensor.squeeze(0).cpu().numpy()
        new_field = {}
        
        non_zero_indices = np.argwhere(tensor_np.sum(axis=0) > 1e-5)

        p_dims_values = list(p_dims.values())
        sections = np.cumsum(p_dims_values)

        for x, y in non_zero_indices:
            vector = tensor_np[:, x, y]
            
            pheromone = PheromoneVector(
                behavior=vector[0:sections[0]],
                emotion=vector[sections[0]:sections[1]],
                social=vector[sections[1]:sections[2]],
                context=vector[sections[2]:sections[3]],
                timestamp=time.time(),
                agent_id=-1
            )
            new_field[(x, y)] = [pheromone]
        self.field = new_field

    def _diffuse_cpu(self, radius: int = 5):
        """Fallback CPU-based diffusion for compatibility."""
        new_field = {}
        for pos, pheromones in self.field.items():
            x, y = pos
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]:
                        distance = np.sqrt(dx**2 + dy**2)
                        if distance <= radius:
                            weight = max(0.3, 1.2 / (1.0 + distance * 0.3))
                            new_pos = (new_x, new_y)
                            if new_pos not in new_field:
                                new_field[new_pos] = []
                            for p in pheromones:
                                diffused = PheromoneVector(
                                    behavior=np.clip(p.behavior * weight, 0, 5.0),
                                    emotion=np.clip(p.emotion * weight, 0, 5.0),
                                    social=np.clip(p.social * weight, 0, 5.0),
                                    context=np.clip(p.context * weight, 0, 5.0),
                                    timestamp=p.timestamp,
                                    agent_id=p.agent_id
                                )
                                new_field[new_pos].append(diffused)
        self.field = new_field
        
    def decay_all(self, min_magnitude_threshold: float, max_lifetime_seconds: float):
        """
        모든 페로몬에 감쇠를 적용하고, 약하거나 오래된 페로몬을 제거합니다.
        """
        current_time = time.time()
        empty_positions = []
        
        # 감쇠율을 더 보수적으로 조정 (기존 문제 해결)
        effective_decay_rate = min(0.99, self.decay_rate)  # 최대 1% 감쇠
        
        for pos, pheromones in list(self.field.items()):
            surviving_pheromones = []
            for p in pheromones:
                p.decay(effective_decay_rate)
                
                # 임계값을 낮춰서 더 많은 페로몬이 유지되도록 함
                is_strong_enough = p.get_total_magnitude() > min_magnitude_threshold * 0.5
                is_not_expired = (current_time - p.timestamp) < max_lifetime_seconds * 2.0  # 수명 2배 연장
                
                if is_strong_enough and is_not_expired:
                    surviving_pheromones.append(p)
            
            if surviving_pheromones:
                self.field[pos] = surviving_pheromones
            else:
                empty_positions.append(pos)
        
        for pos in empty_positions:
            del self.field[pos]
            
        # 디버깅을 위한 로깅
        if len(self.field) % 50 == 0:
            logger.debug(f"페로몬 감쇠 후 남은 위치: {len(self.field)}개")
                
    def get_pheromones(self, position: Tuple[int, int]) -> List[PheromoneVector]:
        """Get pheromones at position"""
        return self.field.get(position, [])

    def get_field_as_tensor(self, p_dims: Dict, device='cuda') -> torch.Tensor:
        """Convert the entire field to a tensor"""
        total_dim = sum(p_dims.values())
        field_tensor = torch.zeros(
            (1, total_dim, self.grid_size[0], self.grid_size[1]),
            device=device,
            dtype=torch.float16
        )
        
        for pos, pheromones in self.field.items():
            if pheromones:
                # Aggregate by summing up all vectors at the position
                aggregated_vector = pheromones[0]
                for p in pheromones[1:]:
                    aggregated_vector += p
                
                field_tensor[0, :, pos[0], pos[1]] = aggregated_vector.to_tensor(device).half()
                
        return field_tensor
