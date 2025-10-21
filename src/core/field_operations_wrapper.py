"""
Python wrapper for C++ SIMD-optimized field operations with automatic fallback.

This module provides a transparent interface to field operations that:
1. Uses C++ SIMD (AVX2) implementation if available (2-5x faster)
2. Falls back to pure Python if C++ module not available
3. Collects performance metrics for both implementations
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple

# Try to import C++ accelerators
try:
    from src.core.cpp_accelerators import (
        PheromoneFieldCPP,
        PheromoneVector4D,
        FieldMetrics
    )
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("Warning: C++ field operations not available, using pure Python fallback")


class PheromoneVector4DPython:
    """Pure Python implementation of PheromoneVector4D for fallback"""

    def __init__(self):
        self.behavior = np.zeros(4, dtype=np.float32)
        self.emotion = np.zeros(5, dtype=np.float32)
        self.social = np.zeros(10, dtype=np.float32)
        self.context = np.zeros(5, dtype=np.float32)
        self.timestamp = 0.0
        self.agent_id = -1

    def magnitude(self) -> float:
        """Calculate Euclidean magnitude of all components"""
        total = (np.sum(self.behavior ** 2) +
                np.sum(self.emotion ** 2) +
                np.sum(self.social ** 2) +
                np.sum(self.context ** 2))
        return np.sqrt(total)

    def decay(self, rate: float):
        """Apply decay to all components"""
        self.behavior *= rate
        self.emotion *= rate
        self.social *= rate
        self.context *= rate

        # Apply minimum threshold
        min_threshold = 1e-6
        self.behavior = np.maximum(self.behavior, min_threshold)
        self.emotion = np.maximum(self.emotion, min_threshold)
        self.social = np.maximum(self.social, min_threshold)
        self.context = np.maximum(self.context, min_threshold)

    def __add__(self, other):
        """Vector addition"""
        result = PheromoneVector4DPython()
        result.behavior = self.behavior + other.behavior
        result.emotion = self.emotion + other.emotion
        result.social = self.social + other.social
        result.context = self.context + other.context
        result.timestamp = self.timestamp
        result.agent_id = self.agent_id
        return result


class FieldMetricsPython:
    """Pure Python metrics structure"""

    def __init__(self):
        self.decay_time_ms = 0.0
        self.aggregation_time_ms = 0.0
        self.diffusion_time_ms = 0.0
        self.num_positions = 0
        self.num_pheromones = 0


class PheromoneFieldPython:
    """Pure Python implementation of PheromoneField for fallback"""

    def __init__(self, width: int, height: int, decay_rate: float):
        self.width = width
        self.height = height
        self.decay_rate = decay_rate
        self.field: Dict[Tuple[int, int], List[PheromoneVector4DPython]] = {}
        self.last_metrics = FieldMetricsPython()

    def add_pheromone(self, x: int, y: int, pheromone: PheromoneVector4DPython):
        """Add pheromone at position"""
        key = (x, y)
        if key not in self.field:
            self.field[key] = []
        self.field[key].append(pheromone)

    def get_pheromones_at(self, x: int, y: int) -> List[PheromoneVector4DPython]:
        """Get all pheromones at position"""
        return self.field.get((x, y), [])

    def clear(self):
        """Clear all pheromones"""
        self.field.clear()

    def size(self) -> int:
        """Get number of occupied positions"""
        return len(self.field)

    def decay_all_parallel(self, min_magnitude: float, max_lifetime_seconds: float, num_threads: int = 8):
        """Apply decay to all pheromones (Python implementation is single-threaded)"""
        start = time.perf_counter()
        current_time = time.time()

        positions_to_remove = []
        total_pheromones = 0

        for pos, pheromones in self.field.items():
            survivors = []
            for p in pheromones:
                p.decay(self.decay_rate)

                strong_enough = p.magnitude() > min_magnitude * 0.5
                not_expired = (current_time - p.timestamp) < max_lifetime_seconds * 2.0

                if strong_enough and not_expired:
                    survivors.append(p)

            if survivors:
                self.field[pos] = survivors
                total_pheromones += len(survivors)
            else:
                positions_to_remove.append(pos)

        # Remove empty positions
        for pos in positions_to_remove:
            del self.field[pos]

        self.last_metrics.decay_time_ms = (time.perf_counter() - start) * 1000
        self.last_metrics.num_positions = len(self.field)
        self.last_metrics.num_pheromones = total_pheromones

    def aggregate_pheromones_simd(self, pheromones_by_position: List[List[PheromoneVector4DPython]]) -> List[PheromoneVector4DPython]:
        """Aggregate pheromones (Python uses numpy vectorization)"""
        start = time.perf_counter()

        results = []
        for pheromones in pheromones_by_position:
            if not pheromones:
                results.append(PheromoneVector4DPython())
                continue

            result = pheromones[0]
            for p in pheromones[1:]:
                result = result + p
            results.append(result)

        self.last_metrics.aggregation_time_ms = (time.perf_counter() - start) * 1000
        return results

    def diffuse_parallel(self, radius: int, num_threads: int = 8):
        """Apply diffusion (Python is single-threaded)"""
        start = time.perf_counter()

        diffused_field = {}

        for (x, y), pheromones in list(self.field.items()):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue

                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        dist = np.sqrt(dx*dx + dy*dy)
                        diffusion_factor = 0.1 / (1.0 + dist)

                        for p in pheromones:
                            diffused = PheromoneVector4DPython()
                            diffused.behavior = p.behavior.copy()
                            diffused.emotion = p.emotion.copy()
                            diffused.social = p.social.copy()
                            diffused.context = p.context.copy()
                            diffused.decay(1.0 - diffusion_factor)
                            diffused.timestamp = p.timestamp
                            diffused.agent_id = p.agent_id

                            key = (nx, ny)
                            if key not in diffused_field:
                                diffused_field[key] = []
                            diffused_field[key].append(diffused)

        # Merge diffused pheromones
        for pos, diffused_pheromones in diffused_field.items():
            if pos in self.field:
                self.field[pos].extend(diffused_pheromones)
            else:
                self.field[pos] = diffused_pheromones

        self.last_metrics.diffusion_time_ms = (time.perf_counter() - start) * 1000

    def get_last_metrics(self) -> FieldMetricsPython:
        """Get last operation metrics"""
        return self.last_metrics


class FieldOperationsWrapper:
    """
    Wrapper that automatically selects C++ or Python implementation.

    Provides a unified interface with automatic backend selection and
    performance metrics collection.
    """

    def __init__(self, width: int, height: int, decay_rate: float, force_python: bool = False):
        """
        Initialize field operations wrapper.

        Args:
            width: Field width in cells
            height: Field height in cells
            decay_rate: Decay factor per timestep (0.0 to 1.0)
            force_python: Force Python implementation even if C++ available
        """
        self.width = width
        self.height = height
        self.decay_rate = decay_rate

        if CPP_AVAILABLE and not force_python:
            self.field = PheromoneFieldCPP(width, height, decay_rate)
            self.backend = 'cpp'
            print(f"FieldOperations: Using C++ SIMD backend (AVX2)")
        else:
            self.field = PheromoneFieldPython(width, height, decay_rate)
            self.backend = 'python'
            print(f"FieldOperations: Using Python fallback backend")

        self.Vector4D = PheromoneVector4D if self.backend == 'cpp' else PheromoneVector4DPython

    def create_vector(self) -> 'PheromoneVector4D':
        """Create a new pheromone vector"""
        return self.Vector4D()

    def add_pheromone(self, x: int, y: int, pheromone):
        """Add pheromone at position"""
        return self.field.add_pheromone(x, y, pheromone)

    def get_pheromones_at(self, x: int, y: int):
        """Get pheromones at position"""
        return self.field.get_pheromones_at(x, y)

    def decay_all_parallel(self, min_magnitude: float, max_lifetime_seconds: float, num_threads: int = 8):
        """Apply parallel decay"""
        return self.field.decay_all_parallel(min_magnitude, max_lifetime_seconds, num_threads)

    def aggregate_pheromones_simd(self, pheromones_by_position):
        """Aggregate pheromones using SIMD"""
        return self.field.aggregate_pheromones_simd(pheromones_by_position)

    def diffuse_parallel(self, radius: int, num_threads: int = 8):
        """Apply parallel diffusion"""
        return self.field.diffuse_parallel(radius, num_threads)

    def clear(self):
        """Clear all pheromones"""
        return self.field.clear()

    def size(self) -> int:
        """Get number of occupied positions"""
        return self.field.size()

    def get_last_metrics(self):
        """Get performance metrics"""
        return self.field.get_last_metrics()

    def get_backend_info(self) -> Dict[str, any]:
        """Get information about the current backend"""
        return {
            'backend': self.backend,
            'cpp_available': CPP_AVAILABLE,
            'simd_enabled': self.backend == 'cpp',
            'multithreading': self.backend == 'cpp',
        }


# Convenience function for creating wrapper
def create_field_operations(width: int, height: int, decay_rate: float,
                           force_python: bool = False) -> FieldOperationsWrapper:
    """
    Create a field operations wrapper with automatic backend selection.

    Args:
        width: Field width in cells
        height: Field height in cells
        decay_rate: Decay factor per timestep (0.0 to 1.0)
        force_python: Force Python implementation for testing

    Returns:
        FieldOperationsWrapper instance
    """
    return FieldOperationsWrapper(width, height, decay_rate, force_python)
