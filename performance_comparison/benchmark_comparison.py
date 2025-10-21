#!/usr/bin/env python3
"""
Benchmark Comparison Script: Python vs C++ Implementation
===========================================================

This script compares performance metrics before and after C++ optimization.
It can be used to:
1. Establish Python baseline performance
2. Compare C++ optimized version against baseline
3. Generate comparative visualizations
4. Validate speedup claims

Usage:
    # Establish baseline (before C++ implementation)
    python benchmark_comparison.py --mode baseline --output baseline_results.json

    # Benchmark C++ version (after implementation)
    python benchmark_comparison.py --mode cpp --output cpp_results.json

    # Compare results
    python benchmark_comparison.py --mode compare --baseline baseline_results.json --cpp cpp_results.json
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import C++ module
try:
    from src.core.cpp_accelerators import MessageCodecWrapper, CPP_AVAILABLE
except ImportError:
    CPP_AVAILABLE = False
    logger.warning("C++ accelerators not available")


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    implementation: str  # 'python' or 'cpp'
    component: str
    num_samples: int
    avg_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput: float  # operations per second
    timestamp: str

    def to_dict(self):
        return asdict(self)


class ComponentBenchmark:
    """Base class for component benchmarks"""

    def __init__(self, name: str):
        self.name = name
        self.results = []

    def run(self, num_samples: int, num_iterations: int) -> BenchmarkResult:
        raise NotImplementedError

    def save_results(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)

    def load_results(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.results = [BenchmarkResult(**r) for r in data]


class MessageCodecBenchmark(ComponentBenchmark):
    """Benchmark message serialization/deserialization"""

    def __init__(self):
        super().__init__("Message Codec")

    def _create_sample_message(self, agent_id: int) -> Dict:
        """Create a realistic 4D pheromone message"""
        return {
            'type': 'comprehensive_pheromone_exchange',
            'sender_id': agent_id,
            'timestamp': time.time(),
            'pheromone_data': {
                'behavior': [float(x) for x in np.random.rand(4)],
                'emotion': [float(x) for x in np.random.rand(5)],
                'social_relations': {str(i): float(np.random.rand()) for i in range(10)},
                'environmental_context': {
                    'position': [float(np.random.rand() * 100), float(np.random.rand() * 100)],
                    'local_resources': float(np.random.rand() * 100),
                    'danger_level': float(np.random.rand()),
                    'exploration_map': [[float(x), float(y)] for x, y in np.random.rand(20, 2)],
                    'territory_info': {str(i): float(np.random.rand()) for i in range(10)}
                }
            },
            'agent_status': {
                'health': float(np.random.rand()),
                'energy': float(np.random.rand()),
                'recent_actions': ['move', 'collect', 'move'],
                'cooperation_history': {str(i): float(np.random.rand()) for i in range(5)},
                'learning_state': {'loss': 0.5, 'reward': 10.0}
            },
            'metadata': {
                'protocol_version': '4D_PHEROMONE_V1.2',
                'compression_method': 'none',
                'priority': 'high',
                'expected_response': True,
                'routing_path': [agent_id, (agent_id + 1) % 50],
                'security_token': f"token_{agent_id}_{time.time()}"
            }
        }

    def benchmark_python(self, num_messages: int = 200, num_iterations: int = 50) -> BenchmarkResult:
        """Benchmark Python JSON encoding"""
        import json

        messages = [self._create_sample_message(i) for i in range(num_messages)]
        times = []

        for _ in range(num_iterations):
            start = time.perf_counter()
            encoded = [json.dumps(msg, ensure_ascii=False, default=str) for msg in messages]
            times.append((time.perf_counter() - start) * 1000)  # ms

        result = BenchmarkResult(
            implementation='python',
            component='message_codec',
            num_samples=num_messages,
            avg_time_ms=np.mean(times),
            std_time_ms=np.std(times),
            min_time_ms=np.min(times),
            max_time_ms=np.max(times),
            throughput=num_messages / (np.mean(times) / 1000),  # messages/sec
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )

        self.results.append(result)
        return result

    def benchmark_cpp(self, num_messages: int = 200, num_iterations: int = 50) -> BenchmarkResult:
        """Benchmark C++ encoding"""
        if not CPP_AVAILABLE:
            logger.error("C++ module not available")
            return None

        codec = MessageCodecWrapper(num_threads=8)
        messages = [self._create_sample_message(i) for i in range(num_messages)]
        times = []

        for _ in range(num_iterations):
            start = time.perf_counter()
            encoded = codec.encode_batch(messages)
            times.append((time.perf_counter() - start) * 1000)  # ms

        result = BenchmarkResult(
            implementation='cpp',
            component='message_codec',
            num_samples=num_messages,
            avg_time_ms=np.mean(times),
            std_time_ms=np.std(times),
            min_time_ms=np.min(times),
            max_time_ms=np.max(times),
            throughput=num_messages / (np.mean(times) / 1000),
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )

        self.results.append(result)
        return result


class PheromoneFieldBenchmark(ComponentBenchmark):
    """Benchmark pheromone field operations"""

    def __init__(self):
        super().__init__("Pheromone Field")

    def benchmark_python_decay(self, grid_size: Tuple[int, int] = (100, 100),
                               num_iterations: int = 50) -> BenchmarkResult:
        """Benchmark Python decay implementation"""
        from src.core.pheromone_vector import PheromoneField, PheromoneVector

        p_dims = {'behavior': 4, 'emotion': 5, 'social': 10, 'context': 5}
        field = PheromoneField(grid_size, decay_rate=0.95, p_dims=p_dims)

        # Populate field
        num_deposits = int(grid_size[0] * grid_size[1] * 0.1)
        for _ in range(num_deposits):
            pos = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))
            pheromone = PheromoneVector(
                behavior=np.random.rand(4),
                emotion=np.random.rand(5),
                social=np.random.rand(10),
                context=np.random.rand(5),
                timestamp=time.time(),
                agent_id=0
            )
            field.deposit(pos, pheromone)

        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            field.decay_all(min_magnitude_threshold=0.01, max_lifetime_seconds=100)
            times.append((time.perf_counter() - start) * 1000)

        result = BenchmarkResult(
            implementation='python',
            component='field_decay',
            num_samples=len(field.field),
            avg_time_ms=np.mean(times),
            std_time_ms=np.std(times),
            min_time_ms=np.min(times),
            max_time_ms=np.max(times),
            throughput=len(field.field) / (np.mean(times) / 1000),
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )

        self.results.append(result)
        return result

    def benchmark_cpp_decay(self, grid_size: Tuple[int, int] = (100, 100),
                           num_iterations: int = 50) -> BenchmarkResult:
        """Benchmark C++ decay implementation"""
        if not CPP_AVAILABLE:
            logger.error("C++ module not available")
            return None

        # TODO: Implement after C++ module is ready
        logger.warning("C++ field operations not yet implemented")
        return None


class SpatialQueryBenchmark(ComponentBenchmark):
    """Benchmark spatial queries"""

    def __init__(self):
        super().__init__("Spatial Queries")

    def benchmark_python_linear_search(self, num_resources: int = 500,
                                      num_queries: int = 1000) -> BenchmarkResult:
        """Benchmark Python linear search"""
        resources = [(np.random.randint(0, 100), np.random.randint(0, 100))
                    for _ in range(num_resources)]

        times = []
        for _ in range(num_queries):
            agent_pos = (np.random.randint(0, 100), np.random.randint(0, 100))
            search_radius = 10

            start = time.perf_counter()
            nearby_resources = []
            for res_pos in resources:
                distance = np.sqrt((agent_pos[0] - res_pos[0])**2 +
                                 (agent_pos[1] - res_pos[1])**2)
                if distance <= search_radius:
                    nearby_resources.append(res_pos)
            times.append((time.perf_counter() - start) * 1000)

        result = BenchmarkResult(
            implementation='python',
            component='spatial_query',
            num_samples=num_queries,
            avg_time_ms=np.mean(times),
            std_time_ms=np.std(times),
            min_time_ms=np.min(times),
            max_time_ms=np.max(times),
            throughput=num_queries / (np.mean(times) / 1000),
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )

        self.results.append(result)
        return result

    def benchmark_cpp_rtree(self, num_resources: int = 500,
                           num_queries: int = 1000) -> BenchmarkResult:
        """Benchmark C++ R-tree"""
        if not CPP_AVAILABLE:
            logger.error("C++ module not available")
            return None

        # TODO: Implement after C++ module is ready
        logger.warning("C++ spatial index not yet implemented")
        return None


class BenchmarkComparator:
    """Compare baseline vs optimized results"""

    def __init__(self, baseline_file: str, cpp_file: str = None):
        self.baseline_results = self._load_results(baseline_file)
        self.cpp_results = self._load_results(cpp_file) if cpp_file else None

    def _load_results(self, filepath: str) -> List[BenchmarkResult]:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return [BenchmarkResult(**r) for r in data]

    def calculate_speedup(self) -> Dict[str, float]:
        """Calculate speedup for each component"""
        speedups = {}

        baseline_by_component = {r.component: r for r in self.baseline_results}

        if self.cpp_results:
            cpp_by_component = {r.component: r for r in self.cpp_results}

            for component in baseline_by_component:
                if component in cpp_by_component:
                    baseline_time = baseline_by_component[component].avg_time_ms
                    cpp_time = cpp_by_component[component].avg_time_ms
                    speedups[component] = baseline_time / cpp_time

        return speedups

    def generate_comparison_report(self, output_file: str = "BENCHMARK_COMPARISON.md"):
        """Generate detailed comparison report"""
        with open(output_file, 'w') as f:
            f.write("# Performance Benchmark Comparison Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Baseline Results (Python)\n\n")
            f.write("| Component | Samples | Avg Time (ms) | Std Dev | Throughput (ops/s) |\n")
            f.write("|-----------|---------|---------------|---------|-------------------|\n")
            for r in self.baseline_results:
                f.write(f"| {r.component} | {r.num_samples} | {r.avg_time_ms:.2f} | "
                       f"{r.std_time_ms:.2f} | {r.throughput:.0f} |\n")

            if self.cpp_results:
                f.write("\n## Optimized Results (C++)\n\n")
                f.write("| Component | Samples | Avg Time (ms) | Std Dev | Throughput (ops/s) |\n")
                f.write("|-----------|---------|---------------|---------|-------------------|\n")
                for r in self.cpp_results:
                    f.write(f"| {r.component} | {r.num_samples} | {r.avg_time_ms:.2f} | "
                           f"{r.std_time_ms:.2f} | {r.throughput:.0f} |\n")

                speedups = self.calculate_speedup()
                f.write("\n## Speedup Analysis\n\n")
                f.write("| Component | Baseline (ms) | C++ (ms) | Speedup |\n")
                f.write("|-----------|---------------|----------|----------|\n")

                baseline_by_component = {r.component: r for r in self.baseline_results}
                cpp_by_component = {r.component: r for r in self.cpp_results}

                total_baseline = 0
                total_cpp = 0

                for component in speedups:
                    baseline_time = baseline_by_component[component].avg_time_ms
                    cpp_time = cpp_by_component[component].avg_time_ms
                    speedup = speedups[component]

                    total_baseline += baseline_time
                    total_cpp += cpp_time

                    f.write(f"| {component} | {baseline_time:.2f} | {cpp_time:.2f} | "
                           f"**{speedup:.2f}x** |\n")

                overall_speedup = total_baseline / total_cpp if total_cpp > 0 else 0
                f.write(f"| **Overall** | {total_baseline:.2f} | {total_cpp:.2f} | "
                       f"**{overall_speedup:.2f}x** |\n")

                # Performance summary
                f.write("\n## Performance Summary\n\n")
                f.write(f"- **Total baseline time**: {total_baseline:.2f} ms\n")
                f.write(f"- **Total optimized time**: {total_cpp:.2f} ms\n")
                f.write(f"- **Overall speedup**: {overall_speedup:.2f}x\n")
                f.write(f"- **Time saved per iteration**: {total_baseline - total_cpp:.2f} ms\n")

                # Estimate full simulation improvement
                timesteps = 1000
                time_saved_per_timestep = total_baseline - total_cpp
                total_time_saved = (time_saved_per_timestep * timesteps) / 1000  # seconds

                f.write(f"\n### Estimated Full Simulation Impact (1000 timesteps)\n\n")
                f.write(f"- **Time saved**: {total_time_saved:.1f} seconds\n")
                f.write(f"- **Baseline simulation time**: {total_baseline * timesteps / 1000:.1f} seconds\n")
                f.write(f"- **Optimized simulation time**: {total_cpp * timesteps / 1000:.1f} seconds\n")

        logger.info(f"Comparison report saved to {output_file}")

    def visualize_comparison(self, output_file: str = "benchmark_comparison.png"):
        """Generate visual comparison"""
        if not self.cpp_results:
            logger.warning("No C++ results to compare")
            return

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        baseline_by_component = {r.component: r for r in self.baseline_results}
        cpp_by_component = {r.component: r for r in self.cpp_results}

        components = list(baseline_by_component.keys())

        # 1. Execution time comparison
        ax1 = axes[0, 0]
        baseline_times = [baseline_by_component[c].avg_time_ms for c in components]
        cpp_times = [cpp_by_component[c].avg_time_ms if c in cpp_by_component else 0
                    for c in components]

        x = np.arange(len(components))
        width = 0.35
        ax1.bar(x - width/2, baseline_times, width, label='Python Baseline', color='#e74c3c', alpha=0.7)
        ax1.bar(x + width/2, cpp_times, width, label='C++ Optimized', color='#2ecc71', alpha=0.7)
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Execution Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([c.replace('_', '\n') for c in components], fontsize=9)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Speedup chart
        ax2 = axes[0, 1]
        speedups = self.calculate_speedup()
        speedup_values = [speedups.get(c, 0) for c in components]
        colors = ['#2ecc71' if s >= 2 else '#f39c12' if s >= 1.5 else '#e74c3c'
                 for s in speedup_values]

        bars = ax2.bar(components, speedup_values, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='No improvement')
        ax2.set_ylabel('Speedup (x)')
        ax2.set_title('Speedup Analysis')
        ax2.set_xticklabels([c.replace('_', '\n') for c in components], fontsize=9)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add speedup labels on bars
        for bar, speedup in zip(bars, speedup_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{speedup:.2f}x', ha='center', va='bottom', fontweight='bold')

        # 3. Throughput comparison
        ax3 = axes[1, 0]
        baseline_throughput = [baseline_by_component[c].throughput for c in components]
        cpp_throughput = [cpp_by_component[c].throughput if c in cpp_by_component else 0
                         for c in components]

        ax3.bar(x - width/2, baseline_throughput, width, label='Python Baseline',
               color='#e74c3c', alpha=0.7)
        ax3.bar(x + width/2, cpp_throughput, width, label='C++ Optimized',
               color='#2ecc71', alpha=0.7)
        ax3.set_ylabel('Throughput (ops/sec)')
        ax3.set_title('Throughput Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels([c.replace('_', '\n') for c in components], fontsize=9)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Full simulation time projection
        ax4 = axes[1, 1]
        timesteps = np.array([100, 500, 1000, 5000])

        total_baseline_per_step = sum(baseline_by_component[c].avg_time_ms for c in components)
        total_cpp_per_step = sum(cpp_by_component[c].avg_time_ms if c in cpp_by_component else 0
                                for c in components)

        baseline_total = timesteps * total_baseline_per_step / 1000  # seconds
        cpp_total = timesteps * total_cpp_per_step / 1000

        ax4.plot(timesteps, baseline_total, 'o-', linewidth=2, markersize=8,
                label='Python Baseline', color='#e74c3c')
        ax4.plot(timesteps, cpp_total, 'o-', linewidth=2, markersize=8,
                label='C++ Optimized', color='#2ecc71')
        ax4.fill_between(timesteps, baseline_total, cpp_total, alpha=0.2, color='green',
                         label='Time Saved')
        ax4.set_xlabel('Timesteps')
        ax4.set_ylabel('Total Time (seconds)')
        ax4.set_title('Full Simulation Time Projection')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison visualization saved to {output_file}")
        plt.close()


def run_baseline_benchmarks(output_file: str = "baseline_results.json"):
    """Run all baseline benchmarks"""
    logger.info("Running baseline benchmarks...")

    all_results = []

    # Message codec benchmark
    logger.info("Benchmarking message codec...")
    codec_bench = MessageCodecBenchmark()
    result = codec_bench.benchmark_python(num_messages=200, num_iterations=50)
    all_results.append(result)
    logger.info(f"  Python: {result.avg_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")

    # Field operations benchmark
    logger.info("Benchmarking pheromone field decay...")
    field_bench = PheromoneFieldBenchmark()
    result = field_bench.benchmark_python_decay(num_iterations=50)
    all_results.append(result)
    logger.info(f"  Python: {result.avg_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")

    # Spatial query benchmark
    logger.info("Benchmarking spatial queries...")
    spatial_bench = SpatialQueryBenchmark()
    result = spatial_bench.benchmark_python_linear_search(num_queries=1000)
    all_results.append(result)
    logger.info(f"  Python: {result.avg_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")

    # Save results
    with open(output_file, 'w') as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)

    logger.info(f"Baseline results saved to {output_file}")


def run_cpp_benchmarks(output_file: str = "cpp_results.json"):
    """Run C++ benchmarks"""
    if not CPP_AVAILABLE:
        logger.error("C++ module not available. Please build and install the C++ backend first.")
        return

    logger.info("Running C++ benchmarks...")

    all_results = []

    # Message codec benchmark
    logger.info("Benchmarking C++ message codec...")
    codec_bench = MessageCodecBenchmark()
    result = codec_bench.benchmark_cpp(num_messages=200, num_iterations=50)
    if result:
        all_results.append(result)
        logger.info(f"  C++: {result.avg_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")

    # TODO: Add field and spatial benchmarks when C++ modules are ready

    # Save results
    with open(output_file, 'w') as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)

    logger.info(f"C++ results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark comparison script')
    parser.add_argument('--mode', choices=['baseline', 'cpp', 'compare'],
                       default='baseline', help='Benchmark mode')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--baseline', type=str, help='Baseline results file for comparison')
    parser.add_argument('--cpp', type=str, help='C++ results file for comparison')

    args = parser.parse_args()

    if args.mode == 'baseline':
        run_baseline_benchmarks(args.output)

    elif args.mode == 'cpp':
        run_cpp_benchmarks(args.output)

    elif args.mode == 'compare':
        if not args.baseline:
            logger.error("--baseline file required for comparison mode")
            return

        comparator = BenchmarkComparator(args.baseline, args.cpp)
        comparator.generate_comparison_report()
        comparator.visualize_comparison()

        logger.info("Comparison complete!")


if __name__ == '__main__':
    main()
