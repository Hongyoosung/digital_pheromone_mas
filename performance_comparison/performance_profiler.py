#!/usr/bin/env python3
"""
Performance Profiling and Visualization Script for Digital Pheromone MAS
=========================================================================

This script measures and visualizes performance bottlenecks to identify
C++ optimization opportunities. It profiles:
1. Communication serialization/deserialization overhead
2. Pheromone field operations (diffusion, decay, aggregation)
3. Spatial queries and environment operations
4. Overall simulation performance

Usage:
    python performance_profiler.py --config config/experiment_config.yaml --output profiling_results/
"""

import os
import sys
import time
import json
import pickle
import argparse
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import cProfile
import pstats
from io import StringIO

import numpy as np
import torch
import ray
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# Import project modules
from src.core.pheromone_vector import PheromoneField, PheromoneVector
from src.core.agent import DistributedAgent
from src.experiments.run_experiment import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Ensure output to console
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    # Communication metrics
    message_serialization_time: List[float] = None
    message_deserialization_time: List[float] = None
    message_sizes: List[int] = None
    communication_round_time: List[float] = None

    # Pheromone field metrics
    field_diffusion_time: List[float] = None
    field_decay_time: List[float] = None
    pheromone_aggregation_time: List[float] = None
    pheromone_deposit_time: List[float] = None

    # Spatial query metrics
    environment_query_time: List[float] = None
    resource_search_time: List[float] = None

    # Agent metrics
    perception_time: List[float] = None
    decision_time: List[float] = None
    action_execution_time: List[float] = None

    # Network training metrics
    attention_forward_time: List[float] = None
    diffusion_forward_time: List[float] = None
    network_training_time: List[float] = None

    # System metrics
    cpu_usage: List[float] = None
    memory_usage: List[float] = None
    gpu_memory_usage: List[float] = None

    def __post_init__(self):
        """Initialize all lists"""
        for field_name in self.__dataclass_fields__:
            if getattr(self, field_name) is None:
                setattr(self, field_name, [])


class PerformanceProfiler:
    """Main performance profiler class"""

    def __init__(self, config_path: str, output_dir: str):
        self.config_path = config_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.metrics = PerformanceMetrics()
        self.bottleneck_analysis = {}

    def profile_communication_overhead(self, num_samples: int = 100) -> Dict:
        """Profile message serialization/deserialization performance"""
        logger.info("=" * 80)
        logger.info(f"[1/4] Profiling communication overhead ({num_samples} samples)...")
        logger.info("=" * 80)

        # Create sample pheromone data
        sample_message = {
            'type': 'comprehensive_pheromone_exchange',
            'sender_id': 0,
            'timestamp': time.time(),
            'pheromone_data': {
                'behavior': [float(x) for x in np.random.rand(4)],
                'emotion': [float(x) for x in np.random.rand(5)],
                'social_relations': {str(i): float(np.random.rand()) for i in range(10)},
                'environmental_context': {
                    'position': [10.0, 20.0],
                    'local_resources': 50.0,
                    'danger_level': 0.3,
                    'exploration_map': [[float(x), float(y)] for x, y in np.random.rand(20, 2)],
                    'territory_info': {str(i): float(np.random.rand()) for i in range(10)}
                }
            },
            'agent_status': {
                'health': 0.8,
                'energy': 0.9,
                'recent_actions': ['move', 'collect', 'move'],
                'cooperation_history': {str(i): float(np.random.rand()) for i in range(5)},
                'learning_state': {'loss': 0.5, 'reward': 10.0}
            },
            'metadata': {
                'protocol_version': '4D_PHEROMONE_V1.2',
                'compression_method': 'none',
                'priority': 'high',
                'expected_response': True,
                'routing_path': [0, 1],
                'security_token': f"token_0_1_{time.time()}"
            }
        }

        serialization_times = []
        deserialization_times = []
        message_sizes = []

        logger.info("Testing message serialization/deserialization...")
        for i in range(num_samples):
            if i % 100 == 0 and i > 0:
                logger.info(f"  Progress: {i}/{num_samples} samples processed...")

            # Measure serialization
            start = time.perf_counter()
            serialized = json.dumps(sample_message, ensure_ascii=False, default=str)
            ser_time = time.perf_counter() - start
            serialization_times.append(ser_time)

            # Measure size
            message_size = len(serialized.encode('utf-8'))
            message_sizes.append(message_size)

            # Measure deserialization
            start = time.perf_counter()
            deserialized = json.loads(serialized)
            deser_time = time.perf_counter() - start
            deserialization_times.append(deser_time)

        self.metrics.message_serialization_time.extend(serialization_times)
        self.metrics.message_deserialization_time.extend(deserialization_times)
        self.metrics.message_sizes.extend(message_sizes)

        results = {
            'avg_serialization_time_ms': np.mean(serialization_times) * 1000,
            'avg_deserialization_time_ms': np.mean(deserialization_times) * 1000,
            'avg_message_size_bytes': np.mean(message_sizes),
            'total_overhead_per_message_ms': (np.mean(serialization_times) + np.mean(deserialization_times)) * 1000
        }

        logger.info(f"✓ Communication overhead profiling complete:")
        logger.info(f"  - Avg serialization: {results['avg_serialization_time_ms']:.4f} ms")
        logger.info(f"  - Avg deserialization: {results['avg_deserialization_time_ms']:.4f} ms")
        logger.info(f"  - Avg message size: {results['avg_message_size_bytes']:.0f} bytes")
        logger.info(f"  - Total overhead: {results['total_overhead_per_message_ms']:.4f} ms/message")
        logger.info("")

        return results

    def profile_pheromone_field_operations(self, grid_size: Tuple[int, int] = (100, 100),
                                          num_iterations: int = 50) -> Dict:
        """Profile pheromone field operations (critical bottleneck)"""
        logger.info("=" * 80)
        logger.info(f"[2/4] Profiling pheromone field operations on {grid_size} grid...")
        logger.info("=" * 80)

        p_dims = {'behavior': 4, 'emotion': 5, 'social': 10, 'context': 5}
        field = PheromoneField(grid_size, decay_rate=0.95, p_dims=p_dims)

        # Deposit test pheromones
        num_deposits = int(grid_size[0] * grid_size[1] * 0.1)  # 10% coverage
        deposit_times = []

        logger.info(f"Depositing {num_deposits} test pheromones...")
        for i in range(num_deposits):
            if i % 500 == 0 and i > 0:
                logger.info(f"  Progress: {i}/{num_deposits} deposits...")

            pos = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))
            pheromone = PheromoneVector(
                behavior=np.random.rand(4),
                emotion=np.random.rand(5),
                social=np.random.rand(10),
                context=np.random.rand(5),
                timestamp=time.time(),
                agent_id=0
            )

            start = time.perf_counter()
            field.deposit(pos, pheromone)
            deposit_times.append(time.perf_counter() - start)

        self.metrics.pheromone_deposit_time.extend(deposit_times)
        logger.info(f"✓ Deposit phase complete: avg {np.mean(deposit_times)*1e6:.2f} μs per deposit")

        # Test diffusion (GPU vs CPU)
        diffusion_times_gpu = []
        diffusion_times_cpu = []

        logger.info(f"Testing diffusion operations ({num_iterations} iterations)...")
        for i in range(num_iterations):
            if i % 10 == 0 and i > 0:
                logger.info(f"  Diffusion progress: {i}/{num_iterations}...")
            # GPU diffusion
            start = time.perf_counter()
            field.diffuse(radius=2, device='cuda')
            diffusion_times_gpu.append(time.perf_counter() - start)

            # CPU diffusion
            start = time.perf_counter()
            field._diffuse_cpu(radius=2)
            diffusion_times_cpu.append(time.perf_counter() - start)

        self.metrics.field_diffusion_time.extend(diffusion_times_gpu)
        logger.info(f"✓ Diffusion complete: GPU {np.mean(diffusion_times_gpu)*1000:.2f} ms, CPU {np.mean(diffusion_times_cpu)*1000:.2f} ms")

        # Test decay
        decay_times = []
        logger.info(f"Testing decay operations ({num_iterations} iterations)...")
        for i in range(num_iterations):
            if i % 10 == 0 and i > 0:
                logger.info(f"  Decay progress: {i}/{num_iterations}...")
            start = time.perf_counter()
            field.decay_all(min_magnitude_threshold=0.01, max_lifetime_seconds=100)
            decay_times.append(time.perf_counter() - start)

        self.metrics.field_decay_time.extend(decay_times)
        logger.info(f"✓ Decay complete: avg {np.mean(decay_times)*1000:.2f} ms")

        # Test aggregation
        aggregation_times = []
        logger.info("Testing pheromone aggregation...")
        for pos, pheromones in list(field.field.items())[:100]:
            if len(pheromones) > 1:
                start = time.perf_counter()
                aggregated = pheromones[0]
                for p in pheromones[1:]:
                    aggregated = aggregated + p
                aggregation_times.append(time.perf_counter() - start)

        self.metrics.pheromone_aggregation_time.extend(aggregation_times)
        logger.info(f"✓ Aggregation complete: {len(aggregation_times)} aggregations tested")

        results = {
            'avg_deposit_time_us': np.mean(deposit_times) * 1e6,
            'avg_diffusion_time_gpu_ms': np.mean(diffusion_times_gpu) * 1000,
            'avg_diffusion_time_cpu_ms': np.mean(diffusion_times_cpu) * 1000,
            'diffusion_speedup_gpu_vs_cpu': np.mean(diffusion_times_cpu) / np.mean(diffusion_times_gpu),
            'avg_decay_time_ms': np.mean(decay_times) * 1000,
            'avg_aggregation_time_us': np.mean(aggregation_times) * 1e6 if aggregation_times else 0
        }

        logger.info(f"✓ Pheromone field profiling complete:")
        logger.info(f"  - GPU speedup: {results['diffusion_speedup_gpu_vs_cpu']:.1f}x")
        logger.info(f"  - Avg decay: {results['avg_decay_time_ms']:.2f} ms")
        logger.info("")

        return results

    def profile_spatial_queries(self, num_queries: int = 1000) -> Dict:
        """Profile environment spatial query performance"""
        logger.info("=" * 80)
        logger.info(f"[3/4] Profiling spatial queries ({num_queries} queries)...")
        logger.info("=" * 80)

        # Simulate resource/hazard lists
        num_resources = 500
        resources = [(np.random.randint(0, 100), np.random.randint(0, 100)) for _ in range(num_resources)]
        logger.info(f"Testing queries on {num_resources} resources...")

        query_times = []
        for i in range(num_queries):
            if i % 200 == 0 and i > 0:
                logger.info(f"  Query progress: {i}/{num_queries}...")
            agent_pos = (np.random.randint(0, 100), np.random.randint(0, 100))
            search_radius = 10

            # Linear search (current implementation)
            start = time.perf_counter()
            nearby_resources = []
            for res_pos in resources:
                distance = np.sqrt((agent_pos[0] - res_pos[0])**2 + (agent_pos[1] - res_pos[1])**2)
                if distance <= search_radius:
                    nearby_resources.append(res_pos)
            query_times.append(time.perf_counter() - start)

        self.metrics.environment_query_time.extend(query_times)

        results = {
            'avg_query_time_us': np.mean(query_times) * 1e6,
            'total_query_time_per_timestep_ms': np.mean(query_times) * 50 * 1000  # 50 agents
        }

        logger.info(f"✓ Spatial query profiling complete:")
        logger.info(f"  - Avg query time: {results['avg_query_time_us']:.2f} μs")
        logger.info(f"  - Total per timestep (50 agents): {results['total_query_time_per_timestep_ms']:.2f} ms")
        logger.info("")

        return results

    def profile_full_simulation(self, max_timesteps: int = 100) -> Dict:
        """Profile a full simulation run"""
        logger.info("=" * 80)
        logger.info(f"[4/4] Profiling full simulation ({max_timesteps} timesteps)...")
        logger.info("=" * 80)

        # Use cProfile for detailed profiling
        profiler = cProfile.Profile()
        profiler.enable()

        # Run experiment
        logger.info("Initializing experiment runner...")
        runner = ExperimentRunner(self.config_path)
        runner.config['environment']['max_timesteps'] = max_timesteps
        logger.info(f"✓ Experiment runner initialized with {len(runner.agents)} agents")

        timestep_times = []
        logger.info(f"Starting simulation loop for {max_timesteps} timesteps...")
        for t in tqdm(range(max_timesteps), desc="Profiling simulation"):
            if t % 10 == 0:
                logger.info(f"  Timestep {t}/{max_timesteps}...")

            start = time.perf_counter()

            # Profile system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            self.metrics.cpu_usage.append(cpu_percent)
            self.metrics.memory_usage.append(memory_percent)

            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                self.metrics.gpu_memory_usage.append(gpu_mem)

            # Run timestep
            runner.run_timestep(t)

            timestep_time = time.perf_counter() - start
            timestep_times.append(timestep_time)

            if t % 10 == 0:
                logger.info(f"    Timestep {t} complete in {timestep_time*1000:.2f} ms")

        profiler.disable()
        logger.info("✓ Simulation complete")

        # Save profiling stats
        logger.info("Saving profiling statistics...")
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')

        stats_file = os.path.join(self.output_dir, 'profile_stats.txt')
        with open(stats_file, 'w') as f:
            stats.stream = f
            stats.print_stats(50)  # Top 50 functions
        logger.info(f"✓ Profile stats saved to {stats_file}")

        # Cleanup
        logger.info("Shutting down Ray...")
        ray.shutdown()
        logger.info("✓ Ray shutdown complete")

        results = {
            'avg_timestep_time_ms': np.mean(timestep_times) * 1000,
            'total_simulation_time_s': sum(timestep_times),
            'avg_cpu_usage_percent': np.mean(self.metrics.cpu_usage),
            'avg_memory_usage_percent': np.mean(self.metrics.memory_usage),
            'avg_gpu_memory_gb': np.mean(self.metrics.gpu_memory_usage) if self.metrics.gpu_memory_usage else 0
        }

        logger.info(f"✓ Full simulation profiling complete:")
        logger.info(f"  - Avg timestep: {results['avg_timestep_time_ms']:.2f} ms")
        logger.info(f"  - Total time: {results['total_simulation_time_s']:.2f} s")
        logger.info(f"  - Avg CPU: {results['avg_cpu_usage_percent']:.1f}%")
        logger.info(f"  - Avg Memory: {results['avg_memory_usage_percent']:.1f}%")
        logger.info("")

        return results

    def analyze_bottlenecks(self) -> Dict:
        """Analyze and identify performance bottlenecks"""
        logger.info("=" * 80)
        logger.info("Analyzing performance bottlenecks...")
        logger.info("=" * 80)

        # Communication bottleneck
        comm_overhead = np.mean(self.metrics.message_serialization_time) + \
                       np.mean(self.metrics.message_deserialization_time)
        comm_overhead_ms = comm_overhead * 1000

        # Field operations bottleneck
        field_overhead = np.mean(self.metrics.field_diffusion_time) + \
                        np.mean(self.metrics.field_decay_time)
        field_overhead_ms = field_overhead * 1000

        # Spatial query bottleneck
        spatial_overhead = np.mean(self.metrics.environment_query_time) if self.metrics.environment_query_time else 0
        spatial_overhead_ms = spatial_overhead * 1000

        total_overhead = comm_overhead_ms + field_overhead_ms + spatial_overhead_ms

        self.bottleneck_analysis = {
            'communication': {
                'time_ms': comm_overhead_ms,
                'percentage': (comm_overhead_ms / total_overhead * 100) if total_overhead > 0 else 0,
                'priority': 'HIGH' if comm_overhead_ms > 50 else 'MEDIUM',
                'cpp_speedup_potential': '4-10x',
                'description': 'Message serialization/deserialization (JSON)'
            },
            'field_operations': {
                'time_ms': field_overhead_ms,
                'percentage': (field_overhead_ms / total_overhead * 100) if total_overhead > 0 else 0,
                'priority': 'HIGH' if field_overhead_ms > 100 else 'MEDIUM',
                'cpp_speedup_potential': '2-5x',
                'description': 'Pheromone diffusion and decay'
            },
            'spatial_queries': {
                'time_ms': spatial_overhead_ms,
                'percentage': (spatial_overhead_ms / total_overhead * 100) if total_overhead > 0 else 0,
                'priority': 'MEDIUM',
                'cpp_speedup_potential': '3-10x',
                'description': 'Environment resource/hazard queries'
            }
        }

        logger.info("Bottleneck analysis results:")
        for category, data in self.bottleneck_analysis.items():
            logger.info(f"  [{category.upper()}]")
            logger.info(f"    - Time: {data['time_ms']:.2f} ms")
            logger.info(f"    - Percentage: {data['percentage']:.1f}%")
            logger.info(f"    - Priority: {data['priority']}")
            logger.info(f"    - C++ speedup potential: {data['cpp_speedup_potential']}")
        logger.info("")

        return self.bottleneck_analysis

    def visualize_performance(self):
        """Generate performance visualization plots"""
        logger.info("=" * 80)
        logger.info("Generating performance visualizations...")
        logger.info("=" * 80)

        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(20, 12))

        # 1. Communication overhead breakdown
        ax1 = plt.subplot(3, 3, 1)
        comm_data = {
            'Serialization': np.mean(self.metrics.message_serialization_time) * 1000,
            'Deserialization': np.mean(self.metrics.message_deserialization_time) * 1000
        }
        ax1.bar(comm_data.keys(), comm_data.values(), color=['#3498db', '#e74c3c'])
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Communication Overhead per Message')
        ax1.grid(True, alpha=0.3)

        # 2. Field operations comparison
        ax2 = plt.subplot(3, 3, 2)
        field_data = {
            'Diffusion\n(GPU)': np.mean(self.metrics.field_diffusion_time) * 1000 if self.metrics.field_diffusion_time else 0,
            'Decay': np.mean(self.metrics.field_decay_time) * 1000 if self.metrics.field_decay_time else 0,
            'Aggregation': np.mean(self.metrics.pheromone_aggregation_time) * 1e6 / 1000 if self.metrics.pheromone_aggregation_time else 0
        }
        ax2.bar(field_data.keys(), field_data.values(), color=['#2ecc71', '#f39c12', '#9b59b6'])
        ax2.set_ylabel('Time (ms)')
        ax2.set_title('Pheromone Field Operations')
        ax2.grid(True, alpha=0.3)

        # 3. Bottleneck pie chart
        ax3 = plt.subplot(3, 3, 3)
        bottleneck_values = [
            self.bottleneck_analysis['communication']['time_ms'],
            self.bottleneck_analysis['field_operations']['time_ms'],
            self.bottleneck_analysis['spatial_queries']['time_ms']
        ]
        labels = ['Communication', 'Field Ops', 'Spatial Queries']
        colors = ['#3498db', '#2ecc71', '#f39c12']
        ax3.pie(bottleneck_values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax3.set_title('Bottleneck Distribution')

        # 4. Message size distribution
        ax4 = plt.subplot(3, 3, 4)
        if self.metrics.message_sizes:
            ax4.hist(self.metrics.message_sizes, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
            ax4.axvline(np.mean(self.metrics.message_sizes), color='red', linestyle='--',
                       label=f'Mean: {np.mean(self.metrics.message_sizes):.0f} bytes')
            ax4.set_xlabel('Message Size (bytes)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Message Size Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. CPU and Memory usage over time
        ax5 = plt.subplot(3, 3, 5)
        if self.metrics.cpu_usage:
            timesteps = range(len(self.metrics.cpu_usage))
            ax5.plot(timesteps, self.metrics.cpu_usage, label='CPU Usage (%)', color='#e74c3c', linewidth=2)
            ax5_twin = ax5.twinx()
            ax5_twin.plot(timesteps, self.metrics.memory_usage, label='Memory Usage (%)',
                         color='#3498db', linewidth=2, linestyle='--')
            ax5.set_xlabel('Timestep')
            ax5.set_ylabel('CPU Usage (%)', color='#e74c3c')
            ax5_twin.set_ylabel('Memory Usage (%)', color='#3498db')
            ax5.set_title('System Resource Usage Over Time')
            ax5.grid(True, alpha=0.3)

        # 6. GPU memory usage
        ax6 = plt.subplot(3, 3, 6)
        if self.metrics.gpu_memory_usage:
            ax6.plot(self.metrics.gpu_memory_usage, color='#2ecc71', linewidth=2)
            ax6.set_xlabel('Timestep')
            ax6.set_ylabel('GPU Memory (GB)')
            ax6.set_title('GPU Memory Usage Over Time')
            ax6.grid(True, alpha=0.3)

        # 7. C++ Optimization Potential
        ax7 = plt.subplot(3, 3, 7)
        optimization_data = {
            'Communication\n(4-10x)': self.bottleneck_analysis['communication']['time_ms'],
            'Field Ops\n(2-5x)': self.bottleneck_analysis['field_operations']['time_ms'],
            'Spatial Queries\n(3-10x)': self.bottleneck_analysis['spatial_queries']['time_ms']
        }
        bars = ax7.bar(optimization_data.keys(), optimization_data.values(),
                      color=['#e74c3c', '#f39c12', '#9b59b6'])
        ax7.set_ylabel('Time (ms)')
        ax7.set_title('C++ Optimization Potential (Speedup Estimates)')
        ax7.grid(True, alpha=0.3)

        # Add annotations
        for i, (bar, (name, value)) in enumerate(zip(bars, optimization_data.items())):
            priority = list(self.bottleneck_analysis.values())[i]['priority']
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{priority}', ha='center', fontweight='bold', fontsize=10)

        # 8. Diffusion time distribution
        ax8 = plt.subplot(3, 3, 8)
        if self.metrics.field_diffusion_time:
            ax8.hist(np.array(self.metrics.field_diffusion_time) * 1000, bins=30,
                    color='#2ecc71', alpha=0.7, edgecolor='black')
            ax8.set_xlabel('Diffusion Time (ms)')
            ax8.set_ylabel('Frequency')
            ax8.set_title('Pheromone Diffusion Time Distribution')
            ax8.grid(True, alpha=0.3)

        # 9. Expected performance improvement with C++
        ax9 = plt.subplot(3, 3, 9)
        categories = ['Communication', 'Field Ops', 'Spatial']
        current_times = [
            self.bottleneck_analysis['communication']['time_ms'],
            self.bottleneck_analysis['field_operations']['time_ms'],
            self.bottleneck_analysis['spatial_queries']['time_ms']
        ]
        # Conservative speedup estimates (lower bound)
        cpp_times = [
            current_times[0] / 4,  # 4x speedup for communication
            current_times[1] / 2,  # 2x speedup for field ops
            current_times[2] / 3   # 3x speedup for spatial
        ]

        x = np.arange(len(categories))
        width = 0.35
        ax9.bar(x - width/2, current_times, width, label='Current (Python)', color='#e74c3c', alpha=0.7)
        ax9.bar(x + width/2, cpp_times, width, label='With C++ (Conservative)', color='#2ecc71', alpha=0.7)
        ax9.set_ylabel('Time (ms)')
        ax9.set_title('Expected Performance with C++ Optimization')
        ax9.set_xticks(x)
        ax9.set_xticklabels(categories)
        ax9.legend()
        ax9.grid(True, alpha=0.3)

        logger.info("Creating final visualization...")
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'performance_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Performance visualization saved to {plot_path}")
        plt.close()
        logger.info("")

    def save_results(self):
        """Save profiling results to files"""
        logger.info("=" * 80)
        logger.info("Saving profiling results...")
        logger.info("=" * 80)

        # Save metrics as JSON
        metrics_dict = {}
        for field_name in self.metrics.__dataclass_fields__:
            values = getattr(self.metrics, field_name)
            if values:
                metrics_dict[field_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'samples': len(values)
                }

        metrics_path = os.path.join(self.output_dir, 'performance_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        logger.info(f"✓ Metrics saved to {metrics_path}")

        # Save bottleneck analysis
        bottleneck_path = os.path.join(self.output_dir, 'bottleneck_analysis.json')
        with open(bottleneck_path, 'w') as f:
            json.dump(self.bottleneck_analysis, f, indent=2)
        logger.info(f"✓ Bottleneck analysis saved to {bottleneck_path}")

        # Save raw metrics for later comparison
        raw_path = os.path.join(self.output_dir, 'raw_metrics.pkl')
        with open(raw_path, 'wb') as f:
            pickle.dump(self.metrics, f)
        logger.info(f"✓ Raw metrics saved to {raw_path}")

        # Generate summary report
        logger.info("Generating summary report...")
        self._generate_summary_report()
        logger.info("")

    def _generate_summary_report(self):
        """Generate a text summary report"""
        report_path = os.path.join(self.output_dir, 'PERFORMANCE_REPORT.md')

        with open(report_path, 'w') as f:
            f.write("# Performance Profiling Report\n")
            f.write("=" * 80 + "\n\n")

            f.write("## Bottleneck Analysis\n\n")
            for name, data in self.bottleneck_analysis.items():
                f.write(f"### {name.replace('_', ' ').title()}\n")
                f.write(f"- **Time**: {data['time_ms']:.2f} ms\n")
                f.write(f"- **Percentage**: {data['percentage']:.1f}%\n")
                f.write(f"- **Priority**: {data['priority']}\n")
                f.write(f"- **C++ Speedup Potential**: {data['cpp_speedup_potential']}\n")
                f.write(f"- **Description**: {data['description']}\n\n")

            f.write("## Recommended C++ Optimization Priorities\n\n")
            sorted_bottlenecks = sorted(
                self.bottleneck_analysis.items(),
                key=lambda x: x[1]['time_ms'],
                reverse=True
            )

            for i, (name, data) in enumerate(sorted_bottlenecks, 1):
                f.write(f"{i}. **{name.replace('_', ' ').title()}** ({data['priority']} priority)\n")
                f.write(f"   - Current overhead: {data['time_ms']:.2f} ms\n")
                f.write(f"   - Expected speedup: {data['cpp_speedup_potential']}\n")
                f.write(f"   - {data['description']}\n\n")

            # Calculate total potential improvement
            current_total = sum(data['time_ms'] for data in self.bottleneck_analysis.values())
            # Conservative estimates: 4x comm, 2x field, 3x spatial
            optimized_total = (
                self.bottleneck_analysis['communication']['time_ms'] / 4 +
                self.bottleneck_analysis['field_operations']['time_ms'] / 2 +
                self.bottleneck_analysis['spatial_queries']['time_ms'] / 3
            )
            overall_speedup = current_total / optimized_total if optimized_total > 0 else 1

            f.write(f"## Overall Performance Improvement Estimate\n\n")
            f.write(f"- **Current total overhead**: {current_total:.2f} ms per timestep\n")
            f.write(f"- **Estimated with C++**: {optimized_total:.2f} ms per timestep\n")
            f.write(f"- **Overall speedup**: {overall_speedup:.1f}x\n")
            f.write(f"- **Time savings per 1000 timesteps**: {(current_total - optimized_total):.2f} seconds\n\n")

        logger.info(f"✓ Summary report saved to {report_path}")

    def run_full_profile(self):
        """Run complete profiling suite"""
        logger.info("\n" + "=" * 80)
        logger.info("DIGITAL PHEROMONE MAS - PERFORMANCE PROFILING")
        logger.info("=" * 80)
        logger.info("Starting comprehensive performance profiling...")
        logger.info("")

        # 1. Profile communication
        comm_results = self.profile_communication_overhead(num_samples=500)

        # 2. Profile pheromone field operations
        field_results = self.profile_pheromone_field_operations(num_iterations=100)

        # 3. Profile spatial queries
        spatial_results = self.profile_spatial_queries(num_queries=2000)

        # 4. Analyze bottlenecks
        self.analyze_bottlenecks()

        # 5. Visualize results
        self.visualize_performance()

        # 6. Save all results
        self.save_results()

        logger.info("=" * 80)
        logger.info("PROFILING COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"Summary report: {os.path.join(self.output_dir, 'PERFORMANCE_REPORT.md')}")
        logger.info(f"Visualization: {os.path.join(self.output_dir, 'performance_analysis.png')}")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Profile Digital Pheromone MAS Performance')
    parser.add_argument('--config', type=str, default='config/experiment_config.yaml',
                       help='Path to experiment configuration')
    parser.add_argument('--output', type=str, default='profiling_results/',
                       help='Output directory for profiling results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick profiling (skip full simulation)')

    args = parser.parse_args()

    logger.info("\n" + "=" * 80)
    logger.info("Performance Profiler initialized")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Mode: {'Quick' if args.quick else 'Full'}")
    logger.info("=" * 80 + "\n")

    profiler = PerformanceProfiler(args.config, args.output)

    if args.quick:
        logger.info("=" * 80)
        logger.info("QUICK PROFILING MODE (without full simulation)")
        logger.info("=" * 80)
        logger.info("")
        profiler.profile_communication_overhead(num_samples=500)
        profiler.profile_pheromone_field_operations(num_iterations=100)
        profiler.profile_spatial_queries(num_queries=2000)
        profiler.analyze_bottlenecks()
        profiler.visualize_performance()
        profiler.save_results()
        logger.info("=" * 80)
        logger.info("QUICK PROFILING COMPLETE!")
        logger.info("=" * 80)
    else:
        profiler.run_full_profile()


if __name__ == '__main__':
    main()
