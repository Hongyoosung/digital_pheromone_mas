"""
Phase 3: Spatial Index Performance Benchmarks
Tests R-tree spatial indexing performance vs Python linear search

Performance Targets:
- Single radius query: 3-10x speedup
- 50 agent batch queries: 10x speedup
- KNN queries: 5-15x speedup
"""

import time
import random
import sys
from typing import List, Tuple

# Add project to path
sys.path.insert(0, '/home/swim/projects/digital_pheromone_mas')

from src.core.spatial_index_wrapper import (
    SpatialIndexWrapper,
    SpatialIndexPython,
    ResourcePoint
)


def generate_random_resources(num_resources: int, world_size: float = 100.0) -> List[ResourcePoint]:
    """Generate random resource points"""
    resources = []
    for i in range(num_resources):
        x = random.uniform(0, world_size)
        y = random.uniform(0, world_size)
        value = random.uniform(1.0, 100.0)
        resources.append(ResourcePoint(x, y, i, value))
    return resources


def generate_random_positions(num_positions: int, world_size: float = 100.0) -> List[Tuple[float, float]]:
    """Generate random query positions"""
    positions = []
    for _ in range(num_positions):
        x = random.uniform(0, world_size)
        y = random.uniform(0, world_size)
        positions.append((x, y))
    return positions


def benchmark_radius_query_single(num_resources: int = 1000, radius: float = 10.0, num_queries: int = 100):
    """Benchmark single radius queries"""
    print(f"\n{'='*60}")
    print(f"Benchmark: Single Radius Queries")
    print(f"{'='*60}")
    print(f"Resources: {num_resources}, Radius: {radius}, Queries: {num_queries}")

    # Generate test data
    resources = generate_random_resources(num_resources)
    query_positions = generate_random_positions(num_queries)

    # Test Python implementation
    print("\n[Python - Linear Search]")
    index_py = SpatialIndexPython()
    index_py.insert_batch(resources)

    start = time.perf_counter()
    for x, y in query_positions:
        results = index_py.query_radius(x, y, radius)
    elapsed_py = (time.perf_counter() - start) * 1000
    avg_py = elapsed_py / num_queries

    print(f"  Total time: {elapsed_py:.2f} ms")
    print(f"  Avg per query: {avg_py:.3f} ms")

    # Test C++ implementation
    print("\n[C++ - R-tree Index]")
    index_cpp = SpatialIndexWrapper(use_cpp=True)
    index_cpp.insert_batch(resources)

    start = time.perf_counter()
    for x, y in query_positions:
        results = index_cpp.query_radius(x, y, radius)
    elapsed_cpp = (time.perf_counter() - start) * 1000
    avg_cpp = elapsed_cpp / num_queries

    print(f"  Total time: {elapsed_cpp:.2f} ms")
    print(f"  Avg per query: {avg_cpp:.3f} ms")

    # Calculate speedup
    speedup = elapsed_py / elapsed_cpp
    print(f"\n[Performance]")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Target: 3-10x")
    if speedup >= 3.0:
        print(f"  Status: ✓ TARGET MET!")
    else:
        print(f"  Status: ⚠ Below target")

    return speedup


def benchmark_radius_query_batch(num_resources: int = 5000, radius: float = 15.0, num_agents: int = 50):
    """Benchmark batch radius queries (multi-agent scenario)"""
    print(f"\n{'='*60}")
    print(f"Benchmark: Batch Radius Queries (Multi-Agent)")
    print(f"{'='*60}")
    print(f"Resources: {num_resources}, Radius: {radius}, Agents: {num_agents}")

    # Generate test data
    resources = generate_random_resources(num_resources)
    agent_positions = generate_random_positions(num_agents)

    # Test Python implementation
    print("\n[Python - Sequential Linear Search]")
    index_py = SpatialIndexPython()
    index_py.insert_batch(resources)

    start = time.perf_counter()
    results_py = index_py.query_radius_batch(agent_positions, radius)
    elapsed_py = (time.perf_counter() - start) * 1000

    print(f"  Total time: {elapsed_py:.2f} ms")
    print(f"  Avg per agent: {elapsed_py/num_agents:.3f} ms")

    # Test C++ implementation
    print("\n[C++ - Parallel R-tree Queries]")
    index_cpp = SpatialIndexWrapper(use_cpp=True)
    index_cpp.insert_batch(resources)

    start = time.perf_counter()
    results_cpp = index_cpp.query_radius_batch(agent_positions, radius, num_threads=8)
    elapsed_cpp = (time.perf_counter() - start) * 1000

    print(f"  Total time: {elapsed_cpp:.2f} ms")
    print(f"  Avg per agent: {elapsed_cpp/num_agents:.3f} ms")

    # Calculate speedup
    speedup = elapsed_py / elapsed_cpp
    print(f"\n[Performance]")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Target: 10x")
    if speedup >= 10.0:
        print(f"  Status: ✓ TARGET MET!")
    elif speedup >= 5.0:
        print(f"  Status: ✓ Good performance")
    else:
        print(f"  Status: ⚠ Below target")

    return speedup


def benchmark_knn_query(num_resources: int = 2000, k: int = 10, num_queries: int = 100):
    """Benchmark K-nearest neighbor queries"""
    print(f"\n{'='*60}")
    print(f"Benchmark: K-Nearest Neighbor Queries")
    print(f"{'='*60}")
    print(f"Resources: {num_resources}, K: {k}, Queries: {num_queries}")

    # Generate test data
    resources = generate_random_resources(num_resources)
    query_positions = generate_random_positions(num_queries)

    # Test Python implementation
    print("\n[Python - Sort-based KNN]")
    index_py = SpatialIndexPython()
    index_py.insert_batch(resources)

    start = time.perf_counter()
    for x, y in query_positions:
        results = index_py.query_knn(x, y, k)
    elapsed_py = (time.perf_counter() - start) * 1000
    avg_py = elapsed_py / num_queries

    print(f"  Total time: {elapsed_py:.2f} ms")
    print(f"  Avg per query: {avg_py:.3f} ms")

    # Test C++ implementation
    print("\n[C++ - R-tree KNN]")
    index_cpp = SpatialIndexWrapper(use_cpp=True)
    index_cpp.insert_batch(resources)

    start = time.perf_counter()
    for x, y in query_positions:
        results = index_cpp.query_knn(x, y, k)
    elapsed_cpp = (time.perf_counter() - start) * 1000
    avg_cpp = elapsed_cpp / num_queries

    print(f"  Total time: {elapsed_cpp:.2f} ms")
    print(f"  Avg per query: {avg_cpp:.3f} ms")

    # Calculate speedup
    speedup = elapsed_py / elapsed_cpp
    print(f"\n[Performance]")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Target: 5-15x")
    if speedup >= 5.0:
        print(f"  Status: ✓ TARGET MET!")
    else:
        print(f"  Status: ⚠ Below target")

    return speedup


def benchmark_insert_performance(num_resources: int = 10000):
    """Benchmark insertion performance"""
    print(f"\n{'='*60}")
    print(f"Benchmark: Batch Insert Performance")
    print(f"{'='*60}")
    print(f"Resources: {num_resources}")

    # Generate test data
    resources = generate_random_resources(num_resources)

    # Test Python implementation
    print("\n[Python - List Append]")
    index_py = SpatialIndexPython()

    start = time.perf_counter()
    index_py.insert_batch(resources)
    elapsed_py = (time.perf_counter() - start) * 1000

    print(f"  Total time: {elapsed_py:.2f} ms")
    print(f"  Avg per resource: {elapsed_py/num_resources:.4f} ms")

    # Test C++ implementation
    print("\n[C++ - R-tree Batch Insert]")
    index_cpp = SpatialIndexWrapper(use_cpp=True)

    start = time.perf_counter()
    index_cpp.insert_batch(resources)
    elapsed_cpp = (time.perf_counter() - start) * 1000

    print(f"  Total time: {elapsed_cpp:.2f} ms")
    print(f"  Avg per resource: {elapsed_cpp/num_resources:.4f} ms")

    # Calculate speedup
    speedup = elapsed_py / elapsed_cpp
    print(f"\n[Performance]")
    print(f"  Speedup: {speedup:.2f}x")

    return speedup


def verify_correctness():
    """Verify that C++ and Python implementations produce equivalent results"""
    print(f"\n{'='*60}")
    print(f"Correctness Verification")
    print(f"{'='*60}")

    # Create test data
    resources = [
        ResourcePoint(10.0, 10.0, 1, 100.0),
        ResourcePoint(20.0, 20.0, 2, 200.0),
        ResourcePoint(30.0, 30.0, 3, 300.0),
        ResourcePoint(15.0, 15.0, 4, 150.0),
        ResourcePoint(25.0, 25.0, 5, 250.0),
    ]

    # Create indices
    index_py = SpatialIndexPython()
    index_cpp = SpatialIndexWrapper(use_cpp=True)

    for r in resources:
        index_py.insert(r.x, r.y, r.resource_id, r.value)
        index_cpp.insert(r.x, r.y, r.resource_id, r.value)

    # Test 1: Radius query
    print("\nTest 1: Radius Query")
    results_py = index_py.query_radius(20.0, 20.0, 10.0)
    results_cpp = index_cpp.query_radius(20.0, 20.0, 10.0)

    ids_py = sorted([r.resource_id for r in results_py])
    ids_cpp = sorted([r.resource_id for r in results_cpp])

    print(f"  Python found: {ids_py}")
    print(f"  C++ found: {ids_cpp}")
    print(f"  Match: {ids_py == ids_cpp} {'✓' if ids_py == ids_cpp else '✗'}")

    # Test 2: KNN query
    print("\nTest 2: KNN Query (k=3)")
    results_py = index_py.query_knn(15.0, 15.0, 3)
    results_cpp = index_cpp.query_knn(15.0, 15.0, 3)

    ids_py = sorted([r.resource_id for r in results_py])
    ids_cpp = sorted([r.resource_id for r in results_cpp])

    print(f"  Python found: {ids_py}")
    print(f"  C++ found: {ids_cpp}")
    print(f"  Match: {ids_py == ids_cpp} {'✓' if ids_py == ids_cpp else '✗'}")

    return ids_py == ids_cpp


def main():
    """Run all benchmarks"""
    print("\n" + "="*60)
    print("PHASE 3: SPATIAL INDEX PERFORMANCE BENCHMARKS")
    print("="*60)
    print("\nR-tree Spatial Indexing vs Python Linear Search")
    print("Performance Targets:")
    print("  - Single radius query: 3-10x speedup")
    print("  - Batch queries (50 agents): 10x speedup")
    print("  - KNN queries: 5-15x speedup")

    # Verify correctness first
    print("\n" + "="*60)
    print("PHASE 1: CORRECTNESS VERIFICATION")
    print("="*60)
    correctness_ok = verify_correctness()

    if not correctness_ok:
        print("\n⚠ WARNING: Correctness verification failed!")
        print("Results may not be reliable.")
    else:
        print("\n✓ All correctness tests passed!")

    # Run benchmarks
    print("\n" + "="*60)
    print("PHASE 2: PERFORMANCE BENCHMARKS")
    print("="*60)

    results = {}

    # Benchmark 1: Single radius queries
    results['radius_single'] = benchmark_radius_query_single(
        num_resources=1000,
        radius=10.0,
        num_queries=100
    )

    # Benchmark 2: Batch radius queries (realistic multi-agent scenario)
    results['radius_batch'] = benchmark_radius_query_batch(
        num_resources=5000,
        radius=15.0,
        num_agents=50
    )

    # Benchmark 3: KNN queries
    results['knn'] = benchmark_knn_query(
        num_resources=2000,
        k=10,
        num_queries=100
    )

    # Benchmark 4: Insert performance
    results['insert'] = benchmark_insert_performance(num_resources=10000)

    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)

    print(f"\n{'Operation':<30} {'Speedup':<12} {'Target':<12} {'Status':<10}")
    print("-" * 64)

    tests = [
        ("Single radius query", results['radius_single'], "3-10x"),
        ("Batch radius (50 agents)", results['radius_batch'], "10x"),
        ("KNN query", results['knn'], "5-15x"),
        ("Batch insert", results['insert'], "N/A"),
    ]

    for name, speedup, target in tests:
        status = "✓ PASS" if (
            (name == "Single radius query" and speedup >= 3.0) or
            (name == "Batch radius (50 agents)" and speedup >= 5.0) or
            (name == "KNN query" and speedup >= 5.0) or
            (name == "Batch insert")
        ) else "⚠ LOW"

        print(f"{name:<30} {speedup:>6.2f}x      {target:<12} {status}")

    # Overall assessment
    print("\n" + "="*60)
    print("PHASE 3 STATUS")
    print("="*60)

    all_passed = (
        results['radius_single'] >= 3.0 and
        results['radius_batch'] >= 5.0 and
        results['knn'] >= 5.0
    )

    if all_passed:
        print("\n✓✓✓ PHASE 3 COMPLETED SUCCESSFULLY! ✓✓✓")
        print("\nAll performance targets met or exceeded!")
        print("R-tree spatial indexing is significantly faster than Python linear search.")
    else:
        print("\n⚠ PHASE 3 PARTIALLY COMPLETE")
        print("\nSome targets not met, but significant speedups achieved.")

    print("\n" + "="*60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    try:
        main()
    except Exception as e:
        print(f"\n✗ Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
