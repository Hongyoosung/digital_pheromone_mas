#!/usr/bin/env python3
"""
Comprehensive test suite and benchmark for Phase 2 Field Operations.

Tests:
1. SIMD-optimized pheromone decay
2. SIMD-optimized aggregation
3. Multi-threaded diffusion
4. Performance validation (2-5x target)
"""

import sys
import time
import numpy as np
from typing import List

# Import the wrapper
from src.core.field_operations_wrapper import (
    FieldOperationsWrapper,
    create_field_operations,
    CPP_AVAILABLE
)


class TestFieldOperations:
    """Test suite for field operations"""

    def __init__(self):
        self.test_results = []

    def test_basic_operations(self):
        """Test basic field operations"""
        print("\n" + "="*60)
        print("Test 1: Basic Field Operations")
        print("="*60)

        wrapper = create_field_operations(100, 100, 0.95)
        print(f"Backend: {wrapper.backend}")

        # Create a pheromone vector
        vec = wrapper.create_vector()
        vec.behavior = [0.1, 0.2, 0.3, 0.4]
        vec.emotion = [0.1, 0.2, 0.3, 0.4, 0.5]
        vec.social = [0.1] * 10
        vec.context = [0.1] * 5
        vec.timestamp = time.time()
        vec.agent_id = 0

        # Test magnitude calculation
        mag = vec.magnitude()
        print(f"  ✓ Vector magnitude: {mag:.4f}")

        # Test decay
        original_mag = mag
        vec.decay(0.95)
        new_mag = vec.magnitude()
        print(f"  ✓ After decay (0.95): {new_mag:.4f}")
        assert new_mag < original_mag, "Magnitude should decrease after decay"

        # Test field operations
        wrapper.add_pheromone(10, 10, vec)
        pheromones = wrapper.get_pheromones_at(10, 10)
        print(f"  ✓ Added pheromone to field, count: {len(pheromones)}")
        assert len(pheromones) == 1, "Should have 1 pheromone"

        # Test clear
        wrapper.clear()
        assert wrapper.size() == 0, "Field should be empty after clear"
        print(f"  ✓ Field cleared successfully")

        self.test_results.append(("Basic Operations", True, "All tests passed"))
        return True

    def test_decay_performance(self):
        """Benchmark decay performance"""
        print("\n" + "="*60)
        print("Test 2: Decay Performance (Target: 2-5x speedup)")
        print("="*60)

        num_positions = 1000
        pheromones_per_position = 5

        # Test C++ implementation
        if CPP_AVAILABLE:
            cpp_wrapper = create_field_operations(100, 100, 0.95, force_python=False)

            # Populate field
            for i in range(num_positions):
                x, y = i % 100, i // 100
                for _ in range(pheromones_per_position):
                    vec = cpp_wrapper.create_vector()
                    vec.behavior = np.random.rand(4).tolist()
                    vec.emotion = np.random.rand(5).tolist()
                    vec.social = np.random.rand(10).tolist()
                    vec.context = np.random.rand(5).tolist()
                    vec.timestamp = time.time()
                    vec.agent_id = 0
                    cpp_wrapper.add_pheromone(x, y, vec)

            # Benchmark decay
            iterations = 10
            cpp_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                cpp_wrapper.decay_all_parallel(0.01, 100.0, num_threads=8)
                cpp_times.append((time.perf_counter() - start) * 1000)

            cpp_avg = np.mean(cpp_times)
            cpp_std = np.std(cpp_times)
            metrics = cpp_wrapper.get_last_metrics()

            print(f"\nC++ Implementation (SIMD + 8 threads):")
            print(f"  Average time: {cpp_avg:.2f} ± {cpp_std:.2f} ms")
            print(f"  Positions: {metrics.num_positions}")
            print(f"  Pheromones: {metrics.num_pheromones}")

        # Test Python implementation
        python_wrapper = create_field_operations(100, 100, 0.95, force_python=True)

        # Populate field
        for i in range(num_positions):
            x, y = i % 100, i // 100
            for _ in range(pheromones_per_position):
                vec = python_wrapper.create_vector()
                vec.behavior = np.random.rand(4)
                vec.emotion = np.random.rand(5)
                vec.social = np.random.rand(10)
                vec.context = np.random.rand(5)
                vec.timestamp = time.time()
                vec.agent_id = 0
                python_wrapper.add_pheromone(x, y, vec)

        # Benchmark decay
        python_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            python_wrapper.decay_all_parallel(0.01, 100.0)
            python_times.append((time.perf_counter() - start) * 1000)

        python_avg = np.mean(python_times)
        python_std = np.std(python_times)

        print(f"\nPython Implementation:")
        print(f"  Average time: {python_avg:.2f} ± {python_std:.2f} ms")

        # Calculate speedup
        if CPP_AVAILABLE:
            speedup = python_avg / cpp_avg
            print(f"\n{'='*60}")
            print(f"SPEEDUP: {speedup:.2f}x")
            print(f"{'='*60}")

            target_met = speedup >= 2.0
            if target_met:
                print(f"  ✓ Target met (2-5x)!")
            else:
                print(f"  ⚠ Target not met (expected 2-5x, got {speedup:.2f}x)")

            self.test_results.append(("Decay Performance", target_met, f"{speedup:.2f}x speedup"))
            return target_met
        else:
            print("  ⚠ C++ not available, skipping speedup comparison")
            self.test_results.append(("Decay Performance", False, "C++ not available"))
            return False

    def test_aggregation_performance(self):
        """Benchmark aggregation performance"""
        print("\n" + "="*60)
        print("Test 3: Aggregation Performance (Target: 2.5-5x speedup)")
        print("="*60)

        num_groups = 100
        vectors_per_group = 10

        # Create test data
        test_data = []
        for _ in range(num_groups):
            group = []
            for _ in range(vectors_per_group):
                if CPP_AVAILABLE:
                    from src.core.cpp_accelerators import PheromoneVector4D
                    vec = PheromoneVector4D()
                    vec.behavior = np.random.rand(4).tolist()
                    vec.emotion = np.random.rand(5).tolist()
                    vec.social = np.random.rand(10).tolist()
                    vec.context = np.random.rand(5).tolist()
                else:
                    from src.core.field_operations_wrapper import PheromoneVector4DPython
                    vec = PheromoneVector4DPython()
                    vec.behavior = np.random.rand(4)
                    vec.emotion = np.random.rand(5)
                    vec.social = np.random.rand(10)
                    vec.context = np.random.rand(5)
                group.append(vec)
            test_data.append(group)

        iterations = 50

        # Test C++ implementation
        if CPP_AVAILABLE:
            cpp_wrapper = create_field_operations(100, 100, 0.95, force_python=False)
            cpp_times = []

            for _ in range(iterations):
                start = time.perf_counter()
                results = cpp_wrapper.aggregate_pheromones_simd(test_data)
                cpp_times.append((time.perf_counter() - start) * 1000)

            cpp_avg = np.mean(cpp_times)
            cpp_std = np.std(cpp_times)
            print(f"\nC++ Implementation (SIMD):")
            print(f"  Average time: {cpp_avg:.2f} ± {cpp_std:.2f} ms")
            print(f"  Groups aggregated: {len(results)}")

        # Test Python implementation
        python_test_data = []
        for _ in range(num_groups):
            group = []
            for _ in range(vectors_per_group):
                from src.core.field_operations_wrapper import PheromoneVector4DPython
                vec = PheromoneVector4DPython()
                vec.behavior = np.random.rand(4)
                vec.emotion = np.random.rand(5)
                vec.social = np.random.rand(10)
                vec.context = np.random.rand(5)
                group.append(vec)
            python_test_data.append(group)

        python_wrapper = create_field_operations(100, 100, 0.95, force_python=True)
        python_times = []

        for _ in range(iterations):
            start = time.perf_counter()
            results = python_wrapper.aggregate_pheromones_simd(python_test_data)
            python_times.append((time.perf_counter() - start) * 1000)

        python_avg = np.mean(python_times)
        python_std = np.std(python_times)
        print(f"\nPython Implementation:")
        print(f"  Average time: {python_avg:.2f} ± {python_std:.2f} ms")

        # Calculate speedup
        if CPP_AVAILABLE:
            speedup = python_avg / cpp_avg
            print(f"\n{'='*60}")
            print(f"SPEEDUP: {speedup:.2f}x")
            print(f"{'='*60}")

            target_met = speedup >= 2.5
            if target_met:
                print(f"  ✓ Target met (2.5-5x)!")
            else:
                print(f"  ⚠ Target not met (expected 2.5-5x, got {speedup:.2f}x)")

            self.test_results.append(("Aggregation Performance", target_met, f"{speedup:.2f}x speedup"))
            return target_met
        else:
            print("  ⚠ C++ not available, skipping speedup comparison")
            self.test_results.append(("Aggregation Performance", False, "C++ not available"))
            return False

    def test_diffusion_performance(self):
        """Benchmark diffusion performance"""
        print("\n" + "="*60)
        print("Test 4: Diffusion Performance (Target: 2-5x speedup)")
        print("="*60)

        num_positions = 50
        pheromones_per_position = 3
        radius = 2

        # Test C++ implementation
        if CPP_AVAILABLE:
            cpp_wrapper = create_field_operations(50, 50, 0.95, force_python=False)

            # Populate field
            for i in range(num_positions):
                x, y = i % 50, i // 50
                for _ in range(pheromones_per_position):
                    vec = cpp_wrapper.create_vector()
                    vec.behavior = np.random.rand(4).tolist()
                    vec.emotion = np.random.rand(5).tolist()
                    vec.social = np.random.rand(10).tolist()
                    vec.context = np.random.rand(5).tolist()
                    vec.timestamp = time.time()
                    vec.agent_id = 0
                    cpp_wrapper.add_pheromone(x, y, vec)

            # Benchmark diffusion
            iterations = 5
            cpp_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                cpp_wrapper.diffuse_parallel(radius, num_threads=8)
                cpp_times.append((time.perf_counter() - start) * 1000)
                cpp_wrapper.clear()  # Clear to reset

                # Re-populate for next iteration
                for i in range(num_positions):
                    x, y = i % 50, i // 50
                    for _ in range(pheromones_per_position):
                        vec = cpp_wrapper.create_vector()
                        vec.behavior = np.random.rand(4).tolist()
                        vec.emotion = np.random.rand(5).tolist()
                        vec.social = np.random.rand(10).tolist()
                        vec.context = np.random.rand(5).tolist()
                        vec.timestamp = time.time()
                        cpp_wrapper.add_pheromone(x, y, vec)

            cpp_avg = np.mean(cpp_times)
            cpp_std = np.std(cpp_times)
            print(f"\nC++ Implementation (8 threads):")
            print(f"  Average time: {cpp_avg:.2f} ± {cpp_std:.2f} ms")

        # Test Python implementation
        python_wrapper = create_field_operations(50, 50, 0.95, force_python=True)
        python_times = []

        for _ in range(iterations):
            # Populate field
            for i in range(num_positions):
                x, y = i % 50, i // 50
                for _ in range(pheromones_per_position):
                    vec = python_wrapper.create_vector()
                    vec.behavior = np.random.rand(4)
                    vec.emotion = np.random.rand(5)
                    vec.social = np.random.rand(10)
                    vec.context = np.random.rand(5)
                    vec.timestamp = time.time()
                    python_wrapper.add_pheromone(x, y, vec)

            start = time.perf_counter()
            python_wrapper.diffuse_parallel(radius)
            python_times.append((time.perf_counter() - start) * 1000)
            python_wrapper.clear()

        python_avg = np.mean(python_times)
        python_std = np.std(python_times)
        print(f"\nPython Implementation:")
        print(f"  Average time: {python_avg:.2f} ± {python_std:.2f} ms")

        # Calculate speedup
        if CPP_AVAILABLE:
            speedup = python_avg / cpp_avg
            print(f"\n{'='*60}")
            print(f"SPEEDUP: {speedup:.2f}x")
            print(f"{'='*60}")

            target_met = speedup >= 2.0
            if target_met:
                print(f"  ✓ Target met (2-5x)!")
            else:
                print(f"  ⚠ Target not met (expected 2-5x, got {speedup:.2f}x)")

            self.test_results.append(("Diffusion Performance", target_met, f"{speedup:.2f}x speedup"))
            return target_met
        else:
            print("  ⚠ C++ not available, skipping speedup comparison")
            self.test_results.append(("Diffusion Performance", False, "C++ not available"))
            return False

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("PHASE 2: FIELD OPERATIONS - TEST SUMMARY")
        print("="*60)

        all_passed = True
        for test_name, passed, details in self.test_results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status:10} {test_name:30} {details}")
            all_passed = all_passed and passed

        print("="*60)

        if all_passed:
            print("✓ ALL TESTS PASSED - Phase 2 implementation successful!")
            print("\nNext step: Proceed to Phase 3 (Spatial Indexing)")
        else:
            print("⚠ SOME TESTS FAILED - Review implementation")

        print("="*60)

        return all_passed


def main():
    """Run all tests"""
    print("="*60)
    print("PHASE 2: FIELD OPERATIONS - VALIDATION & BENCHMARKS")
    print("="*60)
    print(f"C++ Accelerators available: {CPP_AVAILABLE}")

    if not CPP_AVAILABLE:
        print("\nWARNING: C++ module not available!")
        print("Build with: cd cpp_backend && ./build.sh")
        print("Continuing with Python-only tests...")

    tester = TestFieldOperations()

    try:
        tester.test_basic_operations()
        tester.test_decay_performance()
        tester.test_aggregation_performance()
        tester.test_diffusion_performance()
    except Exception as e:
        print(f"\n✗ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    success = tester.print_summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
