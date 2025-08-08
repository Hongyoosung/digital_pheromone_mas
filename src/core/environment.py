import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


"""
RTS 게임 시나리오를 위한 환경 시뮬레이션 시스템입니다.
자원 분포, 위험 요소, 맵 특성 등을 관리합니다.
"""


@dataclass
class ResourceNode:
    """환경 내 자원 노드"""
    position: Tuple[int, int]
    resource_type: str  # 'mineral', 'energy', 'rare'
    current_amount: float
    max_amount: float
    regeneration_rate: float
    
    def extract(self, amount: float) -> float:
        """자원 채굴"""
        extracted = min(amount, self.current_amount)
        self.current_amount -= extracted
        return extracted
        
    def regenerate(self):
        """자원 재생"""
        if self.current_amount < self.max_amount:
            self.current_amount = min(
                self.max_amount, 
                self.current_amount + self.regeneration_rate
            )


@dataclass
class HazardZone:
    """위험 지역"""
    position: Tuple[int, int]
    radius: float
    damage_rate: float
    zone_type: str  # 'poison', 'energy_drain', 'explosion'
    
    def get_damage_at_position(self, pos: Tuple[int, int]) -> float:
        """해당 위치에서의 데미지 계산"""
        distance = np.sqrt((pos[0] - self.position[0])**2 + (pos[1] - self.position[1])**2)
        if distance <= self.radius:
            return self.damage_rate * (1.0 - distance / self.radius)
        return 0.0


class GameEnvironment:
    """RTS 게임 환경 시뮬레이터"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.map_size = tuple(config['environment']['map_size'])
        self.timestep = 0
        
        # 환경 구성 요소 초기화
        self.resource_nodes: List[ResourceNode] = []
        self.hazard_zones: List[HazardZone] = []
        self.terrain_map = self._generate_terrain()
        self.visibility_map = self._generate_visibility_map()
        
        self._initialize_resources()
        self._initialize_hazards()
        
        logger.info(f"게임 환경 초기화 완료: {self.map_size}, 자원 노드: {len(self.resource_nodes)}, 위험 지역: {len(self.hazard_zones)}")
        
    def _generate_terrain(self) -> np.ndarray:
        """지형 맵 생성 (0: 평지, 1: 산, 2: 물)"""
        terrain = np.random.choice(
            [0, 1, 2], 
            size=self.map_size, 
            p=[0.7, 0.2, 0.1]  # 평지 70%, 산 20%, 물 10%
        )
        return terrain
        
    def _generate_visibility_map(self) -> np.ndarray:
        """시야 제한 맵 생성 (산과 물은 시야를 방해)"""
        visibility = np.ones(self.map_size)
        
        # 산과 물 주변은 시야 제한
        for x in range(self.map_size[0]):
            for y in range(self.map_size[1]):
                if self.terrain_map[x, y] in [1, 2]:  # 산 또는 물
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.map_size[0] and 0 <= ny < self.map_size[1]:
                                distance = np.sqrt(dx**2 + dy**2)
                                if distance <= 2:
                                    visibility[nx, ny] *= (1.0 - 0.3 * (1.0 - distance/2.0))
                                    
        return visibility
        
    def _initialize_resources(self):
        """자원 노드 초기화"""
        num_resources = int(self.map_size[0] * self.map_size[1] * 0.05)  # 맵 크기의 5%
        
        for _ in range(num_resources):
            # 평지에만 자원 생성
            while True:
                x = np.random.randint(0, self.map_size[0])
                y = np.random.randint(0, self.map_size[1])
                if self.terrain_map[x, y] == 0:  # 평지
                    break
                    
            resource_type = np.random.choice(['mineral', 'energy', 'rare'], p=[0.6, 0.3, 0.1])
            
            if resource_type == 'mineral':
                max_amount = np.random.uniform(100, 300)
                regen_rate = np.random.uniform(1, 3)
            elif resource_type == 'energy':
                max_amount = np.random.uniform(50, 150)
                regen_rate = np.random.uniform(2, 5)
            else:  # rare
                max_amount = np.random.uniform(20, 80)
                regen_rate = np.random.uniform(0.5, 1.5)
                
            resource = ResourceNode(
                position=(x, y),
                resource_type=resource_type,
                current_amount=max_amount,
                max_amount=max_amount,
                regeneration_rate=regen_rate
            )
            self.resource_nodes.append(resource)
            
    def _initialize_hazards(self):
        """위험 지역 초기화"""
        num_hazards = int(self.map_size[0] * self.map_size[1] * 0.02)  # 맵 크기의 2%
        
        for _ in range(num_hazards):
            x = np.random.randint(5, self.map_size[0] - 5)
            y = np.random.randint(5, self.map_size[1] - 5)
            
            zone_type = np.random.choice(['poison', 'energy_drain', 'explosion'])
            
            if zone_type == 'poison':
                radius = np.random.uniform(3, 8)
                damage_rate = np.random.uniform(2, 8)
            elif zone_type == 'energy_drain':
                radius = np.random.uniform(5, 12)
                damage_rate = np.random.uniform(1, 4)
            else:  # explosion
                radius = np.random.uniform(2, 5)
                damage_rate = np.random.uniform(10, 25)
                
            hazard = HazardZone(
                position=(x, y),
                radius=radius,
                damage_rate=damage_rate,
                zone_type=zone_type
            )
            self.hazard_zones.append(hazard)
            
    def get_local_environment(self, position: Tuple[float, float], perception_radius: float = 10.0) -> Dict:
        """에이전트 주변 환경 정보 수집"""
        x, y = int(position[0]), int(position[1])
        
        # 인식 범위 내 자원 탐지
        nearby_resources = []
        for resource in self.resource_nodes:
            rx, ry = resource.position
            distance = np.sqrt((x - rx)**2 + (y - ry)**2)
            if distance <= perception_radius and resource.current_amount > 0:
                # 시야 제한 적용
                visibility = self.visibility_map[rx, ry]
                if np.random.random() < visibility:
                    nearby_resources.append({
                        'position': resource.position,
                        'type': resource.resource_type,
                        'amount': resource.current_amount,
                        'distance': distance
                    })
                    
        # 인식 범위 내 위험 지역 탐지
        nearby_hazards = []
        for hazard in self.hazard_zones:
            hx, hy = hazard.position
            distance = np.sqrt((x - hx)**2 + (y - hy)**2)
            if distance <= perception_radius + hazard.radius:
                nearby_hazards.append({
                    'position': hazard.position,
                    'type': hazard.zone_type,
                    'radius': hazard.radius,
                    'damage_rate': hazard.damage_rate,
                    'distance': distance
                })
                
        # 지형 정보
        terrain_info = []
        for dx in range(-int(perception_radius), int(perception_radius) + 1):
            for dy in range(-int(perception_radius), int(perception_radius) + 1):
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.map_size[0] and 0 <= ny < self.map_size[1] and 
                    np.sqrt(dx**2 + dy**2) <= perception_radius):
                    terrain_info.append({
                        'position': (nx, ny),
                        'terrain_type': int(self.terrain_map[nx, ny]),
                        'visibility': self.visibility_map[nx, ny]
                    })
                    
        return {
            'resources': nearby_resources,
            'hazards': nearby_hazards,
            'terrain': terrain_info,
            'timestep': self.timestep
        }
        
    def attempt_resource_extraction(self, position: Tuple[float, float], extraction_amount: float) -> Dict:
        """자원 채굴 시도"""
        x, y = int(position[0]), int(position[1])
        
        # 해당 위치에 자원 노드가 있는지 확인
        for resource in self.resource_nodes:
            rx, ry = resource.position
            distance = np.sqrt((x - rx)**2 + (y - ry)**2)
            
            if distance <= 2.0:  # 채굴 가능 거리
                extracted = resource.extract(extraction_amount)
                if extracted > 0:
                    return {
                        'success': True,
                        'extracted': extracted,
                        'resource_type': resource.resource_type,
                        'remaining': resource.current_amount
                    }
                    
        return {
            'success': False,
            'extracted': 0.0,
            'resource_type': None,
            'remaining': 0.0
        }
        
    def get_environmental_damage(self, position: Tuple[float, float]) -> float:
        """해당 위치에서의 환경 데미지 계산"""
        total_damage = 0.0
        
        for hazard in self.hazard_zones:
            damage = hazard.get_damage_at_position((int(position[0]), int(position[1])))
            total_damage += damage
            
        return total_damage
        
    def is_position_accessible(self, position: Tuple[float, float]) -> bool:
        """해당 위치가 접근 가능한지 확인"""
        x, y = int(position[0]), int(position[1])
        
        if not (0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]):
            return False
            
        # 물은 접근 불가
        if self.terrain_map[x, y] == 2:
            return False
            
        return True
        
    def get_movement_cost(self, from_pos: Tuple[float, float], to_pos: Tuple[float, float]) -> float:
        """이동 비용 계산"""
        if not self.is_position_accessible(to_pos):
            return float('inf')
            
        distance = np.sqrt((to_pos[0] - from_pos[0])**2 + (to_pos[1] - from_pos[1])**2)
        
        # 지형에 따른 이동 비용 조정
        tx, ty = int(to_pos[0]), int(to_pos[1])
        terrain_type = self.terrain_map[tx, ty]
        
        if terrain_type == 0:  # 평지
            terrain_multiplier = 1.0
        elif terrain_type == 1:  # 산
            terrain_multiplier = 2.0
        else:  # 물 (접근 불가이므로 여기 오면 안됨)
            terrain_multiplier = float('inf')
            
        return distance * terrain_multiplier
        
    def update(self):
        """환경 업데이트 (매 타임스텝마다 호출)"""
        self.timestep += 1
        
        # 자원 재생
        for resource in self.resource_nodes:
            resource.regenerate()
            
        # 주기적으로 새로운 위험 지역 생성 (낮은 확률)
        if self.timestep % 100 == 0 and np.random.random() < 0.1:
            self._add_random_hazard()
            
    def _add_random_hazard(self):
        """무작위 위험 지역 추가"""
        x = np.random.randint(0, self.map_size[0])
        y = np.random.randint(0, self.map_size[1])
        
        hazard = HazardZone(
            position=(x, y),
            radius=np.random.uniform(2, 6),
            damage_rate=np.random.uniform(1, 5),
            zone_type='poison'
        )
        
        self.hazard_zones.append(hazard)
        logger.info(f"새로운 위험 지역 생성: {hazard.position}")
        
    def get_global_statistics(self) -> Dict:
        """전체 환경 통계"""
        total_resources = sum(r.current_amount for r in self.resource_nodes)
        resource_types = {}
        for r in self.resource_nodes:
            if r.resource_type not in resource_types:
                resource_types[r.resource_type] = {'count': 0, 'total_amount': 0.0}
            resource_types[r.resource_type]['count'] += 1
            resource_types[r.resource_type]['total_amount'] += r.current_amount
            
        return {
            'timestep': self.timestep,
            'total_resources': total_resources,
            'resource_nodes': len(self.resource_nodes),
            'hazard_zones': len(self.hazard_zones),
            'resource_breakdown': resource_types,
            'map_size': self.map_size
        }