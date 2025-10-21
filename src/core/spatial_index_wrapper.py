"""
Spatial Index Wrapper with Automatic Fallback
Phase 3: R-tree Spatial Indexing for Efficient Spatial Queries

Performance Target: 3-10x speedup for spatial queries
Batch Target: 10x speedup for 50 agent queries

Features:
- R-tree spatial indexing with O(log n) query time
- Thread-safe operations
- Batch processing with multi-threading
- Automatic fallback to Python implementation
"""

import time
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Try to import C++ accelerators
_CPP_AVAILABLE = False
try:
    from src.core import cpp_accelerators
    _CPP_AVAILABLE = True
    print("✓ C++ Spatial Index loaded successfully")
except ImportError as e:
    print(f"⚠ C++ Spatial Index not available, using Python fallback: {e}")


@dataclass
class ResourcePoint:
    """Resource point in 2D space with associated metadata"""
    x: float
    y: float
    resource_id: int
    value: float

    def __repr__(self):
        return f"ResourcePoint(x={self.x}, y={self.y}, id={self.resource_id}, value={self.value})"


class SpatialIndexPython:
    """
    Pure Python spatial index fallback using linear search
    Much slower than R-tree, but works without C++ backend
    """

    def __init__(self):
        self.resources: List[ResourcePoint] = []

    def insert(self, x: float, y: float, resource_id: int, value: float):
        """Insert a resource into the index"""
        self.resources.append(ResourcePoint(x, y, resource_id, value))

    def remove(self, resource_id: int):
        """Remove a resource by ID"""
        self.resources = [r for r in self.resources if r.resource_id != resource_id]

    def clear(self):
        """Clear all resources"""
        self.resources.clear()

    def query_radius(self, x: float, y: float, radius: float) -> List[ResourcePoint]:
        """Find all resources within a radius - O(n) linear search"""
        results = []
        radius_sq = radius * radius

        for resource in self.resources:
            dx = resource.x - x
            dy = resource.y - y
            dist_sq = dx * dx + dy * dy

            if dist_sq <= radius_sq:
                results.append(resource)

        return results

    def query_knn(self, x: float, y: float, k: int) -> List[ResourcePoint]:
        """Find k nearest neighbors - O(n log n) with sorting"""
        # Calculate distances
        distances = []
        for resource in self.resources:
            dx = resource.x - x
            dy = resource.y - y
            dist_sq = dx * dx + dy * dy
            distances.append((dist_sq, resource))

        # Sort by distance and take k nearest
        distances.sort(key=lambda item: item[0])
        return [resource for _, resource in distances[:k]]

    def insert_batch(self, resources: List[ResourcePoint]):
        """Insert multiple resources at once"""
        self.resources.extend(resources)

    def query_radius_batch(
        self,
        positions: List[Tuple[float, float]],
        radius: float,
        num_threads: int = 8
    ) -> List[List[ResourcePoint]]:
        """Perform radius queries for multiple positions (single-threaded in Python)"""
        results = []
        for x, y in positions:
            results.append(self.query_radius(x, y, radius))
        return results

    def query_knn_batch(
        self,
        positions: List[Tuple[float, float]],
        k: int,
        num_threads: int = 8
    ) -> List[List[ResourcePoint]]:
        """Perform KNN queries for multiple positions (single-threaded in Python)"""
        results = []
        for x, y in positions:
            results.append(self.query_knn(x, y, k))
        return results

    def size(self) -> int:
        """Get the number of resources"""
        return len(self.resources)

    def empty(self) -> bool:
        """Check if the index is empty"""
        return len(self.resources) == 0

    def __len__(self):
        return self.size()

    def __bool__(self):
        return not self.empty()


class SpatialIndexWrapper:
    """
    Wrapper for spatial index with automatic C++/Python fallback
    Provides unified interface regardless of backend
    """

    def __init__(self, use_cpp: Optional[bool] = None):
        """
        Initialize spatial index with automatic backend selection

        Args:
            use_cpp: Force C++ (True) or Python (False), None for auto-detect
        """
        self.backend = "python"
        self.metrics = {
            "insert_time_ms": 0.0,
            "query_time_ms": 0.0,
            "num_queries": 0,
            "backend": "python"
        }

        # Determine backend
        if use_cpp is None:
            use_cpp = _CPP_AVAILABLE
        elif use_cpp and not _CPP_AVAILABLE:
            print("⚠ C++ requested but not available, falling back to Python")
            use_cpp = False

        # Create appropriate backend
        if use_cpp:
            try:
                self.index = cpp_accelerators.SpatialIndex()
                self.backend = "cpp"
                self.metrics["backend"] = "cpp"
                print("✓ Using C++ R-tree spatial index")
            except Exception as e:
                print(f"⚠ Failed to create C++ index, using Python: {e}")
                self.index = SpatialIndexPython()
        else:
            self.index = SpatialIndexPython()
            print("Using Python spatial index (linear search)")

    def insert(self, x: float, y: float, resource_id: int, value: float):
        """Insert a resource into the spatial index"""
        start = time.perf_counter()
        self.index.insert(x, y, resource_id, value)
        self.metrics["insert_time_ms"] += (time.perf_counter() - start) * 1000

    def remove(self, resource_id: int):
        """Remove a resource by ID"""
        self.index.remove(resource_id)

    def clear(self):
        """Clear all resources from the index"""
        self.index.clear()

    def query_radius(self, x: float, y: float, radius: float) -> List:
        """Find all resources within a radius"""
        start = time.perf_counter()
        results = self.index.query_radius(x, y, radius)
        elapsed = (time.perf_counter() - start) * 1000

        self.metrics["query_time_ms"] += elapsed
        self.metrics["num_queries"] += 1

        # Convert C++ ResourcePoint to Python if needed
        if self.backend == "cpp":
            return [ResourcePoint(r.x, r.y, r.resource_id, r.value) for r in results]
        return results

    def query_knn(self, x: float, y: float, k: int) -> List:
        """Find k nearest neighbors"""
        start = time.perf_counter()
        results = self.index.query_knn(x, y, k)
        elapsed = (time.perf_counter() - start) * 1000

        self.metrics["query_time_ms"] += elapsed
        self.metrics["num_queries"] += 1

        # Convert C++ ResourcePoint to Python if needed
        if self.backend == "cpp":
            return [ResourcePoint(r.x, r.y, r.resource_id, r.value) for r in results]
        return results

    def insert_batch(self, resources: List[ResourcePoint]):
        """Insert multiple resources at once"""
        start = time.perf_counter()

        if self.backend == "cpp":
            # Convert Python ResourcePoint to C++ ResourcePoint
            cpp_resources = [
                cpp_accelerators.ResourcePoint(r.x, r.y, r.resource_id, r.value)
                for r in resources
            ]
            self.index.insert_batch(cpp_resources)
        else:
            self.index.insert_batch(resources)

        self.metrics["insert_time_ms"] += (time.perf_counter() - start) * 1000

    def query_radius_batch(
        self,
        positions: List[Tuple[float, float]],
        radius: float,
        num_threads: int = 8
    ) -> List[List]:
        """Perform radius queries for multiple positions in parallel"""
        start = time.perf_counter()
        results = self.index.query_radius_batch(positions, radius, num_threads)
        elapsed = (time.perf_counter() - start) * 1000

        self.metrics["query_time_ms"] += elapsed
        self.metrics["num_queries"] += len(positions)

        # Convert C++ ResourcePoint to Python if needed
        if self.backend == "cpp":
            return [
                [ResourcePoint(r.x, r.y, r.resource_id, r.value) for r in result_set]
                for result_set in results
            ]
        return results

    def query_knn_batch(
        self,
        positions: List[Tuple[float, float]],
        k: int,
        num_threads: int = 8
    ) -> List[List]:
        """Perform KNN queries for multiple positions in parallel"""
        start = time.perf_counter()
        results = self.index.query_knn_batch(positions, k, num_threads)
        elapsed = (time.perf_counter() - start) * 1000

        self.metrics["query_time_ms"] += elapsed
        self.metrics["num_queries"] += len(positions)

        # Convert C++ ResourcePoint to Python if needed
        if self.backend == "cpp":
            return [
                [ResourcePoint(r.x, r.y, r.resource_id, r.value) for r in result_set]
                for result_set in results
            ]
        return results

    def size(self) -> int:
        """Get the number of resources in the index"""
        return self.index.size()

    def empty(self) -> bool:
        """Check if the index is empty"""
        return self.index.empty()

    def get_metrics(self) -> dict:
        """Get performance metrics"""
        return self.metrics.copy()

    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics["insert_time_ms"] = 0.0
        self.metrics["query_time_ms"] = 0.0
        self.metrics["num_queries"] = 0

    def __len__(self):
        return self.size()

    def __bool__(self):
        return not self.empty()

    def __repr__(self):
        return (f"SpatialIndexWrapper(backend={self.backend}, "
                f"size={self.size()}, "
                f"queries={self.metrics['num_queries']})")


# Convenience function for quick usage
def create_spatial_index(use_cpp: Optional[bool] = None) -> SpatialIndexWrapper:
    """
    Create a spatial index with automatic backend selection

    Args:
        use_cpp: Force C++ (True) or Python (False), None for auto-detect

    Returns:
        SpatialIndexWrapper instance
    """
    return SpatialIndexWrapper(use_cpp=use_cpp)
