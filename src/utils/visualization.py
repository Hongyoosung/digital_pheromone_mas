import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import networkx as nx
from matplotlib.animation import FuncAnimation
import torch
from pathlib import Path
import logging


"""
페로몬 필드, 에이전트 네트워크, 성능 지표 등 실험 결과를 시각화하는 유틸리티입니다.
"""


class ExperimentVisualizer:
    """Visualization utilities for experiment results"""
    
    def __init__(self, results_dir='results/figures', style='seaborn-v0_8-darkgrid'):
        """
        시각화 도구 초기화.

        Args:
            results_dir (str): 생성된 그림을 저장할 디렉토리.
            style (str): Matplotlib에 적용할 스타일 시트.
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        logging.info(f"ExperimentVisualizer initialized. Figures will be saved to {self.results_dir}")
        
    def plot_pheromone_field(self, field: np.ndarray, timestep: int, save: bool = True):
        """
        Visualize 4D pheromone field
        
        Args:
            field: 4D pheromone field [4, H, W]
            timestep: Current timestep
            save: Whether to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        dimensions = ['Behavior', 'Emotion', 'Social', 'Context']
        
        # 전체적인 통계 정보 계산
        total_intensity = np.sum(field)
        max_intensity = np.max(field)
        active_cells = np.sum(field > 0.01)  # 활성 셀 개수
        
        for i, (ax, dim) in enumerate(zip(axes.flat, dimensions)):
            im = ax.imshow(field[i], cmap='viridis', aspect='auto', vmin=0, vmax=max_intensity)
            ax.set_title(f'{dim} Dimension (t={timestep})\nMax: {np.max(field[i]):.3f}, Sum: {np.sum(field[i]):.3f}')
            plt.colorbar(im, ax=ax)
            
        # 전체 정보를 상단에 표시
        fig.suptitle(f'Pheromone Field Evolution - Timestep {timestep}\n'
                    f'Total Intensity: {total_intensity:.3f}, Active Cells: {active_cells}, Max: {max_intensity:.3f}', 
                    fontsize=14, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)  # 타이틀 공간 확보
        
        if save:
            # 에포크별 개별 저장을 위한 고유 파일명 생성
            filename = f'pheromone_field_t{timestep:06d}.png'
            save_path = f'{self.results_dir}/{filename}'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            # 최신 파일을 latest로도 저장 (빠른 확인용)
            latest_path = f'{self.results_dir}/latest_pheromone_field.png'
            plt.savefig(latest_path, dpi=150, bbox_inches='tight')
            
        plt.close(fig)
        
    def plot_agent_network(self, positions: np.ndarray, connections: Dict, 
                           pheromone_strengths: Optional[np.ndarray] = None):
        """
        Visualize agent network and connections
        
        Args:
            positions: Agent positions [N, 2]
            connections: Dictionary of agent connections
            pheromone_strengths: Optional pheromone strengths for each agent
        """
        G = nx.Graph()
        
        # Add nodes
        for i in range(len(positions)):
            G.add_node(i, pos=positions[i])
            
        # Add edges
        for agent_id, targets in connections.items():
            for target, weight in targets.items():
                G.add_edge(agent_id, target, weight=weight)
                
        pos = nx.get_node_attributes(G, 'pos')
        
        plt.figure(figsize=(12, 10))
        
        # Node colors based on pheromone strength
        if pheromone_strengths is not None:
            node_colors = pheromone_strengths
        else:
            node_colors = 'lightblue'
            
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                               node_size=100, cmap='viridis')
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        
        plt.title('Agent Communication Network')
        plt.axis('off')
        plt.tight_layout()
        plt.close()
        
    def plot_metrics_evolution(self, metrics_history: Dict):
        """
        Plot evolution of metrics over time
        
        Args:
            metrics_history: Dictionary of metric histories
        """
        num_metrics = len(metrics_history)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics))
        
        if num_metrics == 1:
            axes = [axes]
            
        for ax, (metric_name, values) in zip(axes, metrics_history.items()):
            ax.plot(values, linewidth=2)
            ax.set_title(f'{metric_name} Evolution')
            ax.set_xlabel('Timestep')
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.close()
        
    def create_heatmap(self, data: np.ndarray, title: str, 
                       x_labels: Optional[List] = None, 
                       y_labels: Optional[List] = None):
        """
        Create heatmap visualization
        
        Args:
            data: 2D array for heatmap
            title: Title of the heatmap
            x_labels: Optional x-axis labels
            y_labels: Optional y-axis labels
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, annot=True, fmt='.2f', cmap='coolwarm',
                    xticklabels=x_labels, yticklabels=y_labels)
        plt.title(title)
        plt.tight_layout()
        plt.close()
        
    def plot_convergence_comparison(self, results: Dict[str, List[float]]):
        """
        Compare convergence across different methods
        
        Args:
            results: Dictionary mapping method names to loss histories
        """
        plt.figure(figsize=(12, 6))
        
        for method, losses in results.items():
            plt.plot(losses, label=method, linewidth=2)
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Convergence Comparison')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.close()
        
    def _prepare_plot_data(self, metric_name, values):
        """Helper function to convert metric data into a plottable DataFrame."""
        data_to_plot = []
        for step, value in enumerate(values):
            if isinstance(value, dict):
                # 딕셔너리의 모든 값을 확인
                for sub_metric, sub_value in value.items():
                    # 중첩된 딕셔너리도 처리 (예: mean, std 등의 통계값)
                    if isinstance(sub_value, dict):
                        for stat_name, stat_value in sub_value.items():
                            if isinstance(stat_value, (int, float, np.number)):
                                data_to_plot.append({
                                    'step': step,
                                    'value': stat_value,
                                    'sub_metric': f"{sub_metric}_{stat_name}"
                                })
                    elif isinstance(sub_value, (int, float, np.number)):
                        data_to_plot.append({
                            'step': step,
                            'value': sub_value,
                            'sub_metric': sub_metric
                        })
                    elif isinstance(sub_value, (list, np.ndarray)):
                        # 리스트나 배열인 경우 평균값 사용
                        try:
                            mean_val = np.mean(sub_value)
                            if isinstance(mean_val, (int, float, np.number)):
                                data_to_plot.append({
                                    'step': step,
                                    'value': mean_val,
                                    'sub_metric': f"{sub_metric}_mean"
                                })
                        except (TypeError, ValueError):
                            pass
            elif isinstance(value, (int, float, np.number)):
                data_to_plot.append({
                    'step': step,
                    'value': value,
                })
            elif isinstance(value, (list, np.ndarray)):
                # 리스트나 배열인 경우
                try:
                    mean_val = np.mean(value)
                    if isinstance(mean_val, (int, float, np.number)):
                        data_to_plot.append({
                            'step': step,
                            'value': mean_val,
                        })
                except (TypeError, ValueError):
                    logging.warning(f"Could not compute mean for list/array in metric '{metric_name}' at step {step}.")
            else:
                logging.warning(f"Skipping unsupported data type '{type(value)}' in metric '{metric_name}' at step {step}: {value}")
        
        logging.info(f"Prepared {len(data_to_plot)} data points for metric '{metric_name}'")
        return pd.DataFrame(data_to_plot)

    def create_training_progress_plot(self, metrics_history, current_step, save=False, show=False):
        """
        훈련 과정 중 주요 지표(예: 보상, 손실)의 변화를 시각화합니다.
        이 함수는 숫자 값과 딕셔너리 값이 섞인 이기종 데이터도 처리할 수 있습니다.

        Args:
            metrics_history (dict): 지표 이름과 값 리스트를 담은 딕셔너리.
            current_step (int): 현재 훈련 스텝.
            save (bool): 그림을 파일로 저장할지 여부.
            show (bool): 그림을 화면에 표시할지 여부.
        """
        if not metrics_history or all(not v for v in metrics_history.values()):
            logging.warning("metrics_history is empty or contains no data. Skipping plot generation.")
            return
        
        # 디버깅을 위한 로깅 추가
        logging.info(f"Creating training progress plot for step {current_step}")
        for metric_name, values in metrics_history.items():
            logging.info(f"Metric '{metric_name}': {len(values)} values, sample: {values[:3] if values else 'empty'}")

        plt.style.use(self.style)
        num_metrics = len(metrics_history.keys())
        
        fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 5 * num_metrics), sharex=True)
        if num_metrics == 1:
            axes = [axes]
        fig.suptitle(f'Training Progress at Step {current_step}', fontsize=16)

        for i, (metric, values) in enumerate(metrics_history.items()):
            ax = axes[i]
            
            if not values:
                ax.set_title(f'{metric} (No data)')
                continue

            df = self._prepare_plot_data(metric, values)

            if df.empty:
                ax.set_title(f'{metric} (No plottable data)')
                continue

            # 'sub_metric' 열이 존재하고 고유 값이 2개 이상인 경우에만 hue를 사용
            if 'sub_metric' in df.columns and df['sub_metric'].nunique() > 1:
                sns.lineplot(data=df, x='step', y='value', hue='sub_metric', ax=ax)
                ax.set_title(f'{metric} Details')
                ax.legend(title='Sub-metric')
            else:
                sns.lineplot(data=df, x='step', y='value', ax=ax)
                ax.set_title(metric)

            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        if save:
            save_path = self.results_dir / f'training_progress_step_{current_step}.png'
            plt.savefig(save_path, dpi=300)
            logging.info(f"Saved training progress plot to {save_path}")
        
        if show:
            plt.show()
        
        plt.close(fig)
        
    def create_memory_usage_plot(self, memory_history: List[Dict], timestep: int, save: bool = True):
        """
        메모리 사용량 추이를 시각화
        
        Args:
            memory_history: 메모리 사용량 히스토리
            timestep: 현재 타임스텝
            save: 저장 여부
        """
        if not memory_history:
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Memory Usage Monitoring - Timestep {timestep}', fontsize=14)
        
        # 시스템 메모리 사용률
        system_usage = [entry['system_percent'] for entry in memory_history]
        steps = range(len(system_usage))
        
        ax1.plot(steps, system_usage, linewidth=2, color='red', label='System Memory %')
        ax1.axhline(y=75, color='orange', linestyle='--', alpha=0.7, label='Warning (75%)')
        ax1.axhline(y=85, color='red', linestyle='--', alpha=0.7, label='Critical (85%)')
        ax1.set_ylabel('Memory Usage (%)')
        ax1.set_title('System Memory Usage')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 프로세스 메모리 사용량 (GB)
        process_usage = [entry['process_rss_gb'] for entry in memory_history]
        ax2.plot(steps, process_usage, linewidth=2, color='blue', label='Process Memory (GB)')
        ax2.set_xlabel('Monitoring Steps')
        ax2.set_ylabel('Memory (GB)')
        ax2.set_title('Process Memory Usage')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f'memory_usage_t{timestep:06d}.png'
            save_path = f'{self.results_dir}/{filename}'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        plt.close(fig)
        
    def plot_agent_states(self, agent_states: List[Dict], timestep: int, save: bool = True):
        """
        에이전트들의 상태 변화를 시각화
        
        Args:
            agent_states: 에이전트 상태 정보 리스트
            timestep: 현재 타임스텝
            save: 저장 여부
        """
        if not agent_states:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Agent States Evolution - Timestep {timestep}', fontsize=16)
        
        # Extract data for plotting
        positions = np.array([state['position'] for state in agent_states])
        resources = [state['resources'] for state in agent_states]
        health = [state['health'] for state in agent_states]
        emotions = np.array([state['emotion_state'] for state in agent_states])
        
        # Plot 1: Agent positions
        ax1 = axes[0, 0]
        scatter = ax1.scatter(positions[:, 0], positions[:, 1], 
                            c=resources, s=np.array(health)*2, 
                            cmap='viridis', alpha=0.7)
        ax1.set_title('Agent Positions (Color=Resources, Size=Health)')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        plt.colorbar(scatter, ax=ax1, label='Resources')
        
        # Plot 2: Resources distribution
        ax2 = axes[0, 1]
        ax2.hist(resources, bins=10, alpha=0.7, edgecolor='black')
        ax2.set_title('Resources Distribution')
        ax2.set_xlabel('Resource Level')
        ax2.set_ylabel('Number of Agents')
        ax2.axvline(np.mean(resources), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(resources):.1f}')
        ax2.legend()
        
        # Plot 3: Health distribution
        ax3 = axes[1, 0]
        ax3.hist(health, bins=10, alpha=0.7, edgecolor='black', color='orange')
        ax3.set_title('Health Distribution')
        ax3.set_xlabel('Health Level')
        ax3.set_ylabel('Number of Agents')
        ax3.axvline(np.mean(health), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(health):.1f}')
        ax3.legend()
        
        # Plot 4: Emotional states (radar chart style)
        ax4 = axes[1, 1]
        emotion_labels = ['Joy', 'Fear', 'Anger', 'Sadness', 'Trust']
        mean_emotions = np.mean(emotions, axis=0)
        
        bars = ax4.bar(emotion_labels, mean_emotions, alpha=0.7)
        ax4.set_title('Average Emotional States')
        ax4.set_ylabel('Emotional Intensity')
        ax4.set_ylim(0, 1)
        
        # Color bars based on intensity
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.RdYlBu_r(mean_emotions[i]))
        
        plt.tight_layout()
        
        if save:
            filename = f'agent_states_t{timestep:06d}.png'
            save_path = f'{self.results_dir}/{filename}'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            # Latest states도 저장
            latest_path = f'{self.results_dir}/latest_agent_states.png'
            plt.savefig(latest_path, dpi=150, bbox_inches='tight')
            
        plt.close(fig)
        
    def plot_social_network(self, social_connections: Dict, agent_positions: np.ndarray, 
                           timestep: int, save: bool = True):
        """
        에이전트 간 사회적 연결 네트워크 시각화
        
        Args:
            social_connections: 에이전트별 사회적 연결 정보
            agent_positions: 에이전트 위치 배열
            timestep: 현재 타임스텝
            save: 저장 여부
        """
        plt.figure(figsize=(12, 10))
        
        # Create network graph
        G = nx.Graph()
        num_agents = len(agent_positions)
        
        # Add nodes with positions
        for i in range(num_agents):
            G.add_node(i, pos=agent_positions[i])
            
        # Add edges based on social connections
        for agent_id, connections in social_connections.items():
            for other_id, strength in connections.items():
                if abs(strength) > 0.1:  # Only show significant connections
                    G.add_edge(agent_id, other_id, weight=abs(strength), 
                              color='green' if strength > 0 else 'red')
        
        # Get positions for plotting
        pos = nx.get_node_attributes(G, 'pos')
        pos = {node: pos[node] if node in pos else (0, 0) for node in G.nodes()}
        
        # Draw network
        edges = G.edges()
        colors = [G[u][v]['color'] for u, v in edges]
        weights = [G[u][v]['weight'] * 3 for u, v in edges]
        
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=300, alpha=0.8)
        nx.draw_networkx_labels(G, pos, font_size=8)
        nx.draw_networkx_edges(G, pos, edge_color=colors, width=weights, alpha=0.6)
        
        plt.title(f'Social Network - Timestep {timestep}\n'
                 f'Green=Positive, Red=Negative, Width=Strength')
        plt.axis('off')
        
        if save:
            filename = f'social_network_t{timestep:06d}.png'
            save_path = f'{self.results_dir}/{filename}'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        plt.close()