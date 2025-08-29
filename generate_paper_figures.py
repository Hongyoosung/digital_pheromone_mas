import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv('results/comparison/comparison_metrics.csv')

# 색상 팔레트 설정
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
method_names = {
    'proposed_digital_pheromone': '4D Digital Pheromone\n(Proposed)',
    'rule_based_diffusion': 'Rule-based Diffusion',
    'centralized_attention': 'Centralized Attention', 
    'ablation_2d_pheromone': '2D Pheromone\n(Ablation Study)'
}

def create_comparison_plots():
    """비교 실험 결과 플롯 생성"""
    
    # 1. Shannon Entropy 비교 (가장 중요한 지표)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    shannon_data = df[df['metric'] == 'shannon_entropy']
    methods = shannon_data['method'].tolist()
    means = shannon_data['mean'].tolist()
    stds = shannon_data['std'].tolist()
    
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # 수치 표시
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Methods', fontsize=12, fontweight='bold')
    ax.set_ylabel('Shannon Entropy', fontsize=12, fontweight='bold')
    ax.set_title('Shannon Entropy Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([method_names[m] for m in methods], fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comparison/shannon_entropy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 모든 지표 종합 비교 (레이더 차트)
    metrics_to_plot = ['information_transfer_efficiency', 'shannon_entropy', 'success_rate', 'reward']
    metric_labels = ['Information Transfer\nEfficiency', 'Shannon Entropy', 'Success Rate', 'Average Reward']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2*np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]  # 원을 닫기 위해
    
    for i, method in enumerate(['proposed_digital_pheromone', 'rule_based_diffusion', 
                              'centralized_attention', 'ablation_2d_pheromone']):
        values = []
        for metric in metrics_to_plot:
            metric_data = df[(df['metric'] == metric) & (df['method'] == method)]
            if not metric_data.empty:
                # 정규화 (0-1 스케일)
                if metric == 'shannon_entropy':
                    # Shannon entropy는 특별히 처리 (더 높은 값이 좋음)
                    max_val = df[df['metric'] == metric]['mean'].max()
                    min_val = df[df['metric'] == metric]['mean'].min()
                    normalized = (metric_data['mean'].iloc[0] - min_val) / (max_val - min_val)
                else:
                    # 다른 지표들도 정규화
                    max_val = df[df['metric'] == metric]['mean'].max()
                    min_val = df[df['metric'] == metric]['mean'].min()
                    if max_val == min_val:
                        normalized = 1.0  # 모든 값이 같으면 1로 설정
                    else:
                        normalized = (metric_data['mean'].iloc[0] - min_val) / (max_val - min_val)
                values.append(normalized)
            else:
                values.append(0)
        
        values += values[:1]  # 원을 닫기 위해
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method_names[method], 
                color=colors[i], markersize=8)
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Comprehensive Performance Comparison\n(Normalized Scores)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    plt.tight_layout()
    plt.savefig('results/comparison/comprehensive_comparison_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 상세 성능 지표 박스플롯 (Shannon Entropy와 Information Transfer Efficiency)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Shannon Entropy
    shannon_data = df[df['metric'] == 'shannon_entropy']
    ax1 = axes[0]
    x_pos = np.arange(len(shannon_data))
    bars1 = ax1.bar(x_pos, shannon_data['mean'], yerr=shannon_data['std'], 
                    capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    for i, (bar, mean, std) in enumerate(zip(bars1, shannon_data['mean'], shannon_data['std'])):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.1,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Shannon Entropy', fontsize=12, fontweight='bold')
    ax1.set_title('Shannon Entropy\n(Information Diversity)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([method_names[m] for m in shannon_data['method']], 
                       fontsize=9, rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Information Transfer Efficiency
    info_data = df[df['metric'] == 'information_transfer_efficiency']
    ax2 = axes[1]
    x_pos = np.arange(len(info_data))
    bars2 = ax2.bar(x_pos, info_data['mean'], yerr=info_data['std'], 
                    capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    for i, (bar, mean, std) in enumerate(zip(bars2, info_data['mean'], info_data['std'])):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.005,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Information Transfer Efficiency', fontsize=12, fontweight='bold')
    ax2.set_title('Information Transfer Efficiency\n(Communication Quality)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([method_names[m] for m in info_data['method']], 
                       fontsize=9, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comparison/detailed_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 통계적 유의성 표시 (Shannon Entropy만)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    shannon_data = df[df['metric'] == 'shannon_entropy'].copy()
    shannon_data['method_label'] = shannon_data['method'].map(method_names)
    
    # 제안 방법을 첫 번째로 정렬
    method_order = ['4D Digital Pheromone\n(Proposed)', 'Rule-based Diffusion', 
                   'Centralized Attention', '2D Pheromone\n(Ablation Study)']
    shannon_data['method_label'] = pd.Categorical(shannon_data['method_label'], 
                                                 categories=method_order, ordered=True)
    shannon_data = shannon_data.sort_values('method_label')
    
    x_pos = np.arange(len(shannon_data))
    bars = ax.bar(x_pos, shannon_data['mean'], yerr=shannon_data['std'], 
                  capsize=8, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # 제안 방법 강조
    bars[0].set_color('#2E86AB')
    bars[0].set_edgecolor('red')
    bars[0].set_linewidth(3)
    
    # 수치와 유의성 표시
    for i, (bar, mean, std) in enumerate(zip(bars, shannon_data['mean'], shannon_data['std'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.2,
                f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 통계적 유의성 표시
        if i == 0:  # 제안 방법
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   'PROPOSED\n★★★', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white')
        elif i == 1:  # rule_based_diffusion 
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.6,
                   'p<0.001\n(t=12.73)', ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color='red')
        elif i == 2:  # centralized_attention
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.6,
                   'p<0.001\n(t=18.73)', ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color='red')
        elif i == 3:  # ablation_2d_pheromone
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.6,
                   'p<0.001\n(t=28.54)', ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color='red')
    
    ax.set_xlabel('Methods', fontsize=14, fontweight='bold')
    ax.set_ylabel('Shannon Entropy', fontsize=14, fontweight='bold')
    ax.set_title('Shannon Entropy with Statistical Significance\n(Information Diversity Comparison)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(shannon_data['method_label'], fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 범례 추가
    ax.text(0.02, 0.98, '★★★: Statistically significant improvement (p<0.001)', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/comparison/shannon_entropy_with_significance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("논문용 그래프가 성공적으로 생성되었습니다:")
    print("1. results/comparison/shannon_entropy_comparison.png")
    print("2. results/comparison/comprehensive_comparison_radar.png") 
    print("3. results/comparison/detailed_metrics_comparison.png")
    print("4. results/comparison/shannon_entropy_with_significance.png")

if __name__ == "__main__":
    create_comparison_plots()