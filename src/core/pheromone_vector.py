import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

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
    
    def to_tensor(self, device='cuda'):
        """Convert to PyTorch tensor"""
        # Handle both string and torch.device objects
        if isinstance(device, torch.device):
            device_str = device
        else:
            device_str = device
            
        return torch.tensor(
            np.concatenate([
                self.behavior, 
                self.emotion, 
                self.social, 
                self.context
            ]),
            dtype=torch.float32,
            device=device_str
        )
    
    def decay(self, rate: float):
        """Apply temporal decay"""
        self.behavior *= rate
        self.emotion *= rate
        self.social *= rate
        self.context *= rate
        
        # 최소 임계값 유지하여 완전히 사라지지 않도록 함
        min_threshold = 0.01
        self.behavior = np.maximum(self.behavior, min_threshold)
        self.emotion = np.maximum(self.emotion, min_threshold)
        self.social = np.maximum(self.social, min_threshold)
        self.context = np.maximum(self.context, min_threshold)
        
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
        self.decay_rate = decay_rate  # 감쇠율을 늦춤 (0.95 -> 0.98)
        self.field = {}  # Position -> List[PheromoneVector]
        
    def deposit(self, position: Tuple[int, int], pheromone: PheromoneVector):
        """Deposit pheromone at position"""
        if position not in self.field:
            self.field[position] = []
        self.field[position].append(pheromone)
        
    def diffuse(self, radius: int = 5):
        """Diffuse pheromones to neighboring cells"""
        new_field = {}
        for pos, pheromones in self.field.items():
            x, y = pos
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]:
                        distance = np.sqrt(dx**2 + dy**2)
                        if distance <= radius:
                            # 개선된 가중치 함수: 더 넓은 확산과 강한 중심 유지
                            weight = max(0.1, 1.0 / (1.0 + distance * 0.5))
                            new_pos = (new_x, new_y)
                            if new_pos not in new_field:
                                new_field[new_pos] = []
                            for p in pheromones:
                                diffused = PheromoneVector(
                                    behavior=p.behavior * weight,
                                    emotion=p.emotion * weight,
                                    social=p.social * weight,
                                    context=p.context * weight,
                                    timestamp=p.timestamp,
                                    agent_id=p.agent_id
                                )
                                new_field[new_pos].append(diffused)
        self.field = new_field
        
    def decay_all(self):
        """Apply decay to all pheromones"""
        for pheromones in self.field.values():
            for p in pheromones:
                p.decay(self.decay_rate)
                
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
        return np.linalg.norm(
            np.concatenate([self.behavior, self.emotion, self.social, self.context])
        )
    
    def to_tensor(self, device='cuda'):
        """Convert to PyTorch tensor"""
        # Handle both string and torch.device objects
        if isinstance(device, torch.device):
            device_str = device
        else:
            device_str = device
            
        return torch.tensor(
            np.concatenate([
                self.behavior, 
                self.emotion, 
                self.social, 
                self.context
            ]),
            dtype=torch.float32,
            device=device_str
        )
    
    def decay(self, rate: float):
        """Apply temporal decay"""
        self.behavior *= rate
        self.emotion *= rate
        self.social *= rate
        self.context *= rate
        
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
        self.field = {}  # Position -> List[PheromoneVector]
        
    def deposit(self, position: Tuple[int, int], pheromone: PheromoneVector):
        """Deposit pheromone at position"""
        if position not in self.field:
            self.field[position] = []
        self.field[position].append(pheromone)
        
    def diffuse(self, radius: int = 5):
        """Diffuse pheromones to neighboring cells"""
        new_field = {}
        for pos, pheromones in self.field.items():
            x, y = pos
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]:
                        distance = np.sqrt(dx**2 + dy**2)
                        if distance <= radius:
                            weight = max(0.1, 1.0 / (1.0 + distance * 0.5))
                            new_pos = (new_x, new_y)
                            if new_pos not in new_field:
                                new_field[new_pos] = []
                            for p in pheromones:
                                diffused = PheromoneVector(
                                    behavior=p.behavior * weight,
                                    emotion=p.emotion * weight,
                                    social=p.social * weight,
                                    context=p.context * weight,
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
        
        for pos, pheromones in self.field.items():
            surviving_pheromones = []
            for p in pheromones:
                p.decay(self.decay_rate)
                
                is_strong_enough = p.get_total_magnitude() > min_magnitude_threshold
                is_not_expired = (current_time - p.timestamp) < max_lifetime_seconds
                
                if is_strong_enough and is_not_expired:
                    surviving_pheromones.append(p)
            
            if surviving_pheromones:
                self.field[pos] = surviving_pheromones
            else:
                empty_positions.append(pos)
        
        for pos in empty_positions:
            del self.field[pos]
                
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
                aggregated_vector = sum(pheromones[1:], start=pheromones[0])
                field_tensor[0, :, pos[0], pos[1]] = aggregated_vector.to_tensor(device).half()
                
        return field_tensor