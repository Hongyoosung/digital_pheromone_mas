"""
4D 디지털 페로몬 차원별 기여도 분석 (확장된 Ablation Study)

각 차원의 기여도를 체계적으로 분석하여 4D 페로몬의 효과를 정량화합니다.
"""

import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from itertools import combinations
import argparse
from tqdm import tqdm
import ray
from scipy.stats import f_oneway, ttest_ind, chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.experiments.run_experiment import ExperimentRunner
from src.core.pheromone_vector import PheromoneVector
from src.utils.metrics import MetricsTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DimensionAblationStudy:
    """4D 페로몬 차원별 기여도 분석 클래스"""
    
    def __init__(self, config_path: str):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.results_dir = self.config['experiment']['log_dir']
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 실험 구성 정보
        self.dimension_configs = self._prepare_dimension_configurations()
        self.all_results = {}
        self.analysis_results = {}
        
        # Ray 초기화
        if not ray.is_initialized():
            try:
                ray_config = self.config.get('ray', {})
                # 타임아웃 설정 추가
                ray_config.setdefault('_temp_dir', None)  # 기본 임시 디렉토리 사용
                ray.init(**ray_config)
                print(f"Ray 초기화 성공: {ray.cluster_resources()}")
            except Exception as e:
                print(f"Ray 초기화 실패: {e}")
                print("Ray 없이 실행을 시도합니다...")
                # Ray 없이 실행하도록 플래그 설정
                self.use_ray = False
            else:
                self.use_ray = True
        else:
            self.use_ray = True
            
    def _prepare_dimension_configurations(self) -> Dict[str, Dict]:
        """차원 구성 정보 준비"""
        configs = {}
        
        analysis_config = self.config['dimension_analysis']
        
        # 단일 차원
        for config_name, dimensions in analysis_config['single_dimensions'][0].items():
            configs[f"single_{config_name}"] = dimensions
            
        # 2차원 조합
        for config_dict in analysis_config['two_dimensions']:
            for config_name, dimensions in config_dict.items():
                configs[f"two_{config_name}"] = dimensions
                
        # 3차원 조합  
        for config_dict in analysis_config['three_dimensions']:
            for config_name, dimensions in config_dict.items():
                configs[f"three_{config_name}"] = dimensions
                
        # 전체 4차원
        configs["full_4d"] = analysis_config['full_4d']
        
        logger.info(f"준비된 차원 구성 수: {len(configs)}")
        return configs
    
    def run_single_experiment(self, dimension_config: Dict, config_name: str, run_id: int) -> Dict:
        """단일 실험 실행"""
        try:
            # 기본 설정 복사
            experiment_config = self.config.copy()
            
            # 페로몬 차원 설정 업데이트
            experiment_config['pheromone']['dimensions'] = dimension_config
            
            # 실험 이름 업데이트
            experiment_config['experiment']['name'] = f"ablation_{config_name}_run_{run_id}"
            experiment_config['experiment']['log_dir'] = os.path.join(
                self.results_dir, config_name, f"run_{run_id}"
            )
            
            # 시드 설정
            if 'random_seeds' in self.config['execution']:
                seed = self.config['execution']['random_seeds'][run_id % len(self.config['execution']['random_seeds'])]
                np.random.seed(seed)
                torch.manual_seed(seed)
            
            # 임시 설정 파일 생성
            temp_config_path = f"temp_config_{config_name}_{run_id}.yaml"
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(experiment_config, f, allow_unicode=True, indent=2)
            
            try:
                # 실험 실행
                experiment = ExperimentRunner(temp_config_path)
                results = experiment.run_experiment()
            finally:
                # 임시 파일 정리
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)
            
            # 메트릭 추출
            metrics_summary = experiment.metrics_tracker.get_summary()
            
            # 차원별 특성 분석
            dimension_analysis = self._analyze_dimension_usage(
                experiment, dimension_config, metrics_summary
            )
            
            return {
                'config_name': config_name,
                'run_id': run_id,
                'dimension_config': dimension_config,
                'metrics': metrics_summary,
                'dimension_analysis': dimension_analysis,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"실험 실행 오류 ({config_name}, run {run_id}): {e}")
            return {
                'config_name': config_name,
                'run_id': run_id,
                'dimension_config': dimension_config,
                'error': str(e),
                'success': False
            }
    
    def _analyze_dimension_usage(self, experiment: ExperimentRunner, 
                                dimension_config: Dict, metrics: Dict) -> Dict:
        """차원 사용량 분석"""
        analysis = {
            'active_dimensions': [],
            'dimension_weights': {},
            'utilization_rates': {},
            'information_content': {},
        }
        
        # 활성 차원 식별
        for dim_name, dim_size in dimension_config.items():
            if dim_size > 0:
                analysis['active_dimensions'].append(dim_name)
                analysis['dimension_weights'][dim_name] = dim_size
        
        # 페로몬 필드에서 차원별 활용도 계산
        if hasattr(experiment, 'pheromone_field') and experiment.pheromone_field.field:
            total_pheromones = list(experiment.pheromone_field.field.values())
            if total_pheromones:
                # 평균 페로몬 벡터 계산
                avg_vector = sum(total_pheromones, PheromoneVector.zeros(dimension_config))
                avg_vector = avg_vector / len(total_pheromones)
                
                # 차원별 활용률
                dim_start = 0
                for dim_name, dim_size in dimension_config.items():
                    if dim_size > 0:
                        dim_values = avg_vector.to_array()[dim_start:dim_start + dim_size]
                        analysis['utilization_rates'][dim_name] = float(np.mean(np.abs(dim_values)))
                        analysis['information_content'][dim_name] = float(np.std(dim_values))
                        dim_start += dim_size
        
        return analysis
    
    def run_all_experiments(self) -> Dict:
        """모든 차원 조합에 대해 실험 실행"""
        logger.info("차원별 절제 연구 실험 시작")
        
        runs_per_config = self.config['execution']['runs_per_configuration']
        
        for config_name, dimension_config in tqdm(self.dimension_configs.items(), desc="차원 구성"):
            config_results = []
            
            for run_id in tqdm(range(runs_per_config), desc=f"{config_name} 실행", leave=False):
                result = self.run_single_experiment(dimension_config, config_name, run_id)
                config_results.append(result)
                
                if not result['success']:
                    logger.warning(f"실험 실패: {config_name} run {run_id}")
            
            self.all_results[config_name] = config_results
            
            # 중간 결과 저장
            self._save_intermediate_results(config_name, config_results)
        
        logger.info("모든 실험 완료")
        return self.all_results
    
    def _save_intermediate_results(self, config_name: str, results: List[Dict]):
        """중간 결과 저장"""
        results_path = os.path.join(self.results_dir, f"{config_name}_results.yaml")
        
        # 직렬화 가능한 형태로 변환
        serializable_results = []
        for result in results:
            if result['success']:
                serializable_result = {
                    'config_name': result['config_name'],
                    'run_id': result['run_id'],
                    'dimension_config': result['dimension_config'],
                    'dimension_analysis': result['dimension_analysis'],
                    'success': result['success']
                }
                # 메트릭에서 숫자 값만 추출
                if 'metrics' in result:
                    serializable_result['key_metrics'] = self._extract_key_metrics(result['metrics'])
                
                serializable_results.append(serializable_result)
            else:
                serializable_results.append({
                    'config_name': result['config_name'],
                    'run_id': result['run_id'],
                    'error': result['error'],
                    'success': result['success']
                })
        
        with open(results_path, 'w', encoding='utf-8') as f:
            yaml.dump(serializable_results, f, allow_unicode=True, indent=2)
    
    def _extract_key_metrics(self, metrics: Dict) -> Dict:
        """주요 메트릭 추출"""
        key_metrics = {}
        
        # 분석에 중요한 메트릭들
        important_metrics = [
            'information_transfer_efficiency', 'learning_convergence_epochs',
            'communication_overhead', 'network_load', 'shannon_entropy',
            'success_rate', 'average_reward'
        ]
        
        for metric_name in important_metrics:
            if metric_name in metrics:
                metric_data = metrics[metric_name]
                if isinstance(metric_data, dict) and 'mean' in metric_data:
                    key_metrics[metric_name] = {
                        'mean': float(metric_data['mean']),
                        'std': float(metric_data.get('std', 0)),
                        'last': float(metric_data.get('last', metric_data['mean']))
                    }
                elif isinstance(metric_data, (int, float)):
                    key_metrics[metric_name] = float(metric_data)
        
        return key_metrics
    
    def analyze_results(self) -> Dict:
        """결과 분석"""
        logger.info("결과 분석 시작")
        
        # 1. 성능 비교 분석
        performance_analysis = self._analyze_performance_differences()
        
        # 2. 차원별 기여도 분석
        contribution_analysis = self._analyze_dimension_contributions()
        
        # 3. 통계적 유의성 검정
        statistical_analysis = self._perform_statistical_tests()
        
        # 4. 차원 간 시너지 효과 분석
        synergy_analysis = self._analyze_dimension_synergies()
        
        # 5. 정보 이론적 분석
        information_analysis = self._analyze_information_content()
        
        self.analysis_results = {
            'performance_comparison': performance_analysis,
            'dimension_contributions': contribution_analysis,
            'statistical_tests': statistical_analysis,
            'synergy_effects': synergy_analysis,
            'information_analysis': information_analysis,
            'summary': self._generate_analysis_summary()
        }
        
        return self.analysis_results
    
    def _analyze_performance_differences(self) -> Dict:
        """성능 차이 분석"""
        performance_data = {}
        
        # 각 구성에서 성능 메트릭 추출
        for config_name, results in self.all_results.items():
            successful_results = [r for r in results if r['success']]
            if not successful_results:
                continue
                
            metrics_list = []
            for result in successful_results:
                if 'metrics' in result:
                    key_metrics = self._extract_key_metrics(result['metrics'])
                    metrics_list.append(key_metrics)
            
            if metrics_list:
                # 메트릭별 평균과 표준편차 계산
                config_performance = {}
                for metric_name in self.config['analysis_metrics']['performance_indicators']:
                    values = []
                    for metrics in metrics_list:
                        if metric_name in metrics:
                            if isinstance(metrics[metric_name], dict):
                                values.append(metrics[metric_name].get('mean', 0))
                            else:
                                values.append(metrics[metric_name])
                    
                    if values:
                        config_performance[metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'count': len(values)
                        }
                
                performance_data[config_name] = config_performance
        
        return performance_data
    
    def _analyze_dimension_contributions(self) -> Dict:
        """차원별 기여도 분석"""
        contributions = {
            'individual_contributions': {},
            'marginal_contributions': {},
            'interaction_effects': {}
        }
        
        # 개별 차원 기여도 (단일 차원 vs 무차원)
        performance_comparison = self.analysis_results.get('performance_comparison', {})
        single_dim_results = {k: v for k, v in performance_comparison.items() 
                             if k.startswith('single_')}
        
        baseline_performance = 0  # 무차원 기준 (모든 차원이 0인 경우는 실행되지 않으므로 0으로 가정)
        
        for dim_config, performance in single_dim_results.items():
            dim_name = dim_config.replace('single_', '').replace('_only', '')
            if 'shannon_entropy' in performance:
                contribution = performance['shannon_entropy']['mean'] - baseline_performance
                contributions['individual_contributions'][dim_name] = contribution
        
        # 한계 기여도 분석 (n-1차원에서 n차원으로 추가할 때의 개선도)
        full_4d_performance = performance_comparison.get('full_4d', {})
        three_dim_results = {k: v for k, v in performance_comparison.items() 
                           if k.startswith('three_')}
        
        for three_config, three_performance in three_dim_results.items():
            missing_dim = three_config.replace('three_no_', '')
            if 'shannon_entropy' in full_4d_performance and 'shannon_entropy' in three_performance:
                marginal_contribution = (full_4d_performance['shannon_entropy']['mean'] - 
                                       three_performance['shannon_entropy']['mean'])
                contributions['marginal_contributions'][missing_dim] = marginal_contribution
        
        return contributions
    
    def _perform_statistical_tests(self) -> Dict:
        """통계적 유의성 검정"""
        statistical_results = {}
        
        # performance_comparison 변수 정의
        performance_comparison = self.analysis_results.get('performance_comparison', {})
        
        # 차원 수에 따른 성능 차이 (One-way ANOVA)
        dimension_groups = {
            '1D': [k for k in self.all_results.keys() if k.startswith('single_')],
            '2D': [k for k in self.all_results.keys() if k.startswith('two_')],
            '3D': [k for k in self.all_results.keys() if k.startswith('three_')],
            '4D': ['full_4d']
        }
        
        for metric_name in self.config.get('analysis_metrics', {}).get('performance_indicators', ['shannon_entropy']):
            groups_data = []
            group_labels = []
            
            for group_name, config_names in dimension_groups.items():
                group_values = []
                for config_name in config_names:
                    if config_name in performance_comparison:
                        perf_data = performance_comparison[config_name]
                        if metric_name in perf_data:
                            # 모든 실행 결과에서 값 추출
                            for result in self.all_results[config_name]:
                                if result['success'] and 'metrics' in result:
                                    key_metrics = self._extract_key_metrics(result['metrics'])
                                    if metric_name in key_metrics:
                                        if isinstance(key_metrics[metric_name], dict):
                                            group_values.append(key_metrics[metric_name].get('mean', 0))
                                        else:
                                            group_values.append(key_metrics[metric_name])
                
                if group_values:
                    groups_data.append(group_values)
                    group_labels.append(group_name)
            
            # ANOVA 수행
            if len(groups_data) >= 2 and all(len(g) > 1 for g in groups_data):
                f_stat, p_value = f_oneway(*groups_data)
                statistical_results[f'{metric_name}_anova'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < self.config['statistical_analysis']['significance_level'],
                    'groups': group_labels,
                    'group_sizes': [len(g) for g in groups_data]
                }
        
        return statistical_results
    
    def _analyze_dimension_synergies(self) -> Dict:
        """차원 간 시너지 효과 분석"""
        synergies = {}
        
        # performance_comparison 변수 정의
        performance_comparison = self.analysis_results.get('performance_comparison', {})
        
        # 2차원 조합의 성능 vs 개별 차원 성능의 합
        two_dim_configs = [k for k in self.all_results.keys() if k.startswith('two_')]
        
        for config_name in two_dim_configs:
            if config_name in performance_comparison:
                config_performance = performance_comparison[config_name]
                
                # 구성에서 활성 차원 추출
                dimension_config = self.dimension_configs[config_name]
                active_dims = [dim for dim, size in dimension_config.items() if size > 0]
                
                if len(active_dims) == 2:
                    # 개별 차원 성능 가져오기
                    individual_performances = []
                    for dim in active_dims:
                        single_config = f"single_{dim}_only"
                        if single_config in performance_comparison:
                            individual_performances.append(
                                performance_comparison[single_config]
                            )
                    
                    if len(individual_performances) == 2:
                        # Shannon entropy에 대한 시너지 계산
                        if ('shannon_entropy' in config_performance and 
                            all('shannon_entropy' in perf for perf in individual_performances)):
                            
                            combined_performance = config_performance['shannon_entropy']['mean']
                            expected_performance = sum(perf['shannon_entropy']['mean'] 
                                                     for perf in individual_performances)
                            
                            synergy_effect = combined_performance - expected_performance
                            synergies[config_name] = {
                                'dimensions': active_dims,
                                'synergy_effect': synergy_effect,
                                'relative_synergy': synergy_effect / expected_performance if expected_performance != 0 else 0
                            }
        
        return synergies
    
    def _analyze_information_content(self) -> Dict:
        """정보 이론적 분석"""
        information_analysis = {
            'entropy_analysis': {},
            'information_efficiency': {},
            'redundancy_analysis': {}
        }
        
        # 각 차원 조합의 정보 엔트로피 분석
        for config_name, results in self.all_results.items():
            successful_results = [r for r in results if r['success']]
            if not successful_results:
                continue
            
            shannon_entropies = []
            utilization_rates = []
            
            for result in successful_results:
                if 'metrics' in result:
                    key_metrics = self._extract_key_metrics(result['metrics'])
                    if 'shannon_entropy' in key_metrics:
                        if isinstance(key_metrics['shannon_entropy'], dict):
                            shannon_entropies.append(key_metrics['shannon_entropy'].get('mean', 0))
                        else:
                            shannon_entropies.append(key_metrics['shannon_entropy'])
                
                if 'dimension_analysis' in result:
                    dim_analysis = result['dimension_analysis']
                    if 'utilization_rates' in dim_analysis:
                        avg_utilization = np.mean(list(dim_analysis['utilization_rates'].values()))
                        utilization_rates.append(avg_utilization)
            
            if shannon_entropies:
                information_analysis['entropy_analysis'][config_name] = {
                    'mean_entropy': np.mean(shannon_entropies),
                    'entropy_std': np.std(shannon_entropies),
                    'entropy_efficiency': np.mean(shannon_entropies) / len(self.dimension_configs[config_name])  # 차원 수로 정규화
                }
            
            if utilization_rates:
                information_analysis['information_efficiency'][config_name] = {
                    'mean_utilization': np.mean(utilization_rates),
                    'utilization_std': np.std(utilization_rates)
                }
        
        return information_analysis
    
    def _generate_analysis_summary(self) -> Dict:
        """분석 요약 생성"""
        # performance_comparison 변수 정의
        performance_comparison = self.analysis_results.get('performance_comparison', {})
        
        summary = {
            'total_configurations_tested': len(self.dimension_configs),
            'successful_experiments': 0,
            'failed_experiments': 0,
            'best_performing_configuration': None,
            'most_important_dimension': None,
            'key_findings': [],
            'recommendations': []
        }
        
        # 성공/실패 통계
        for results in self.all_results.values():
            for result in results:
                if result['success']:
                    summary['successful_experiments'] += 1
                else:
                    summary['failed_experiments'] += 1
        
        # 최고 성능 구성 찾기
        best_config = None
        best_performance = -float('inf')
        
        for config_name, performance in performance_comparison.items():
            if 'shannon_entropy' in performance:
                entropy = performance['shannon_entropy']['mean']
                if entropy > best_performance:
                    best_performance = entropy
                    best_config = config_name
        
        summary['best_performing_configuration'] = best_config
        
        # 가장 중요한 차원 찾기
        if 'marginal_contributions' in self.analysis_results['dimension_contributions']:
            contributions = self.analysis_results['dimension_contributions']['marginal_contributions']
            if contributions:
                most_important = max(contributions.items(), key=lambda x: x[1])
                summary['most_important_dimension'] = most_important[0]
        
        # 주요 발견사항
        if best_config == 'full_4d':
            summary['key_findings'].append("4D 페로몬이 최고 성능을 보임")
        else:
            summary['key_findings'].append(f"최고 성능: {best_config}")
        
        # 통계적 유의성 확인
        significant_tests = [test for test, result in self.analysis_results['statistical_tests'].items() 
                           if result.get('significant', False)]
        if significant_tests:
            summary['key_findings'].append(f"통계적으로 유의한 차이 발견: {len(significant_tests)}개 테스트")
        
        return summary
    
    def generate_visualizations(self):
        """분석 결과 시각화"""
        logger.info("시각화 생성 시작")
        
        # 1. 성능 비교 히트맵
        self._plot_performance_heatmap()
        
        # 2. 차원별 기여도 바 차트
        self._plot_dimension_contributions()
        
        # 3. 시너지 효과 시각화
        self._plot_synergy_effects()
        
        # 4. 통계적 유의성 플롯
        self._plot_statistical_significance()
        
        # 5. 정보 효율성 분석
        self._plot_information_efficiency()
        
        logger.info("모든 시각화 완료")
    
    def _plot_performance_heatmap(self):
        """성능 비교 히트맵"""
        performance_data = self.analysis_results.get('performance_comparison', {})
        
        # 데이터 준비
        config_names = list(performance_data.keys())
        metrics = self.config['analysis_metrics']['performance_indicators']
        
        heatmap_data = []
        for metric in metrics:
            row = []
            for config in config_names:
                if config in performance_data and metric in performance_data[config]:
                    value = performance_data[config][metric]['mean']
                else:
                    value = 0
                row.append(value)
            heatmap_data.append(row)
        
        # 히트맵 생성
        plt.figure(figsize=(15, 8))
        sns.heatmap(heatmap_data, 
                   xticklabels=config_names, 
                   yticklabels=metrics,
                   annot=True, 
                   cmap='viridis', 
                   fmt='.3f')
        plt.title('차원 구성별 성능 비교')
        plt.xlabel('차원 구성')
        plt.ylabel('성능 지표')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, 'performance_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dimension_contributions(self):
        """차원별 기여도 바 차트"""
        contributions = self.analysis_results['dimension_contributions']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 개별 기여도
        if 'individual_contributions' in contributions:
            dims = list(contributions['individual_contributions'].keys())
            values = list(contributions['individual_contributions'].values())
            
            ax1.bar(dims, values, color='skyblue', alpha=0.7)
            ax1.set_title('개별 차원 기여도')
            ax1.set_xlabel('차원')
            ax1.set_ylabel('기여도 (Shannon Entropy)')
            ax1.tick_params(axis='x', rotation=45)
        
        # 한계 기여도  
        if 'marginal_contributions' in contributions:
            dims = list(contributions['marginal_contributions'].keys())
            values = list(contributions['marginal_contributions'].values())
            
            ax2.bar(dims, values, color='lightcoral', alpha=0.7)
            ax2.set_title('한계 기여도 (3D → 4D)')
            ax2.set_xlabel('추가된 차원')
            ax2.set_ylabel('한계 기여도')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'dimension_contributions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_synergy_effects(self):
        """시너지 효과 시각화"""
        synergies = self.analysis_results['synergy_effects']
        
        if not synergies:
            return
        
        config_names = list(synergies.keys())
        synergy_values = [synergies[config]['synergy_effect'] for config in config_names]
        relative_synergies = [synergies[config]['relative_synergy'] for config in config_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 절대 시너지 효과
        bars1 = ax1.bar(range(len(config_names)), synergy_values, 
                       color=['green' if v > 0 else 'red' for v in synergy_values],
                       alpha=0.7)
        ax1.set_title('차원 간 시너지 효과 (절대값)')
        ax1.set_xlabel('차원 조합')
        ax1.set_ylabel('시너지 효과')
        ax1.set_xticks(range(len(config_names)))
        ax1.set_xticklabels([name.replace('two_', '') for name in config_names], rotation=45)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 상대 시너지 효과
        bars2 = ax2.bar(range(len(config_names)), relative_synergies, 
                       color=['green' if v > 0 else 'red' for v in relative_synergies],
                       alpha=0.7)
        ax2.set_title('차원 간 시너지 효과 (상대값)')
        ax2.set_xlabel('차원 조합')
        ax2.set_ylabel('상대 시너지 효과')
        ax2.set_xticks(range(len(config_names)))
        ax2.set_xticklabels([name.replace('two_', '') for name in config_names], rotation=45)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'synergy_effects.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_significance(self):
        """통계적 유의성 플롯"""
        statistical_results = self.analysis_results['statistical_tests']
        
        # ANOVA 결과만 필터링
        anova_results = {k: v for k, v in statistical_results.items() if k.endswith('_anova')}
        
        if not anova_results:
            return
        
        metrics = [k.replace('_anova', '') for k in anova_results.keys()]
        f_statistics = [anova_results[f'{m}_anova']['f_statistic'] for m in metrics]
        p_values = [anova_results[f'{m}_anova']['p_value'] for m in metrics]
        significance_threshold = self.config['statistical_analysis']['significance_level']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # F-통계량
        ax1.bar(range(len(metrics)), f_statistics, color='lightblue', alpha=0.7)
        ax1.set_title('ANOVA F-통계량')
        ax1.set_xlabel('성능 지표')
        ax1.set_ylabel('F-통계량')
        ax1.set_xticks(range(len(metrics)))
        ax1.set_xticklabels(metrics, rotation=45)
        
        # p-값
        colors = ['green' if p < significance_threshold else 'red' for p in p_values]
        ax2.bar(range(len(metrics)), p_values, color=colors, alpha=0.7)
        ax2.axhline(y=significance_threshold, color='black', linestyle='--', 
                   label=f'α = {significance_threshold}')
        ax2.set_title('ANOVA p-값')
        ax2.set_xlabel('성능 지표')
        ax2.set_ylabel('p-값')
        ax2.set_xticks(range(len(metrics)))
        ax2.set_xticklabels(metrics, rotation=45)
        ax2.legend()
        ax2.set_yscale('log')
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'statistical_significance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_information_efficiency(self):
        """정보 효율성 분석 플롯"""
        info_analysis = self.analysis_results['information_analysis']
        
        entropy_data = info_analysis.get('entropy_analysis', {})
        efficiency_data = info_analysis.get('information_efficiency', {})
        
        if not entropy_data and not efficiency_data:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 엔트로피 효율성
        if entropy_data:
            configs = list(entropy_data.keys())
            efficiencies = [entropy_data[config]['entropy_efficiency'] for config in configs]
            
            axes[0, 0].bar(range(len(configs)), efficiencies, color='purple', alpha=0.7)
            axes[0, 0].set_title('엔트로피 효율성 (엔트로피/차원수)')
            axes[0, 0].set_xlabel('차원 구성')
            axes[0, 0].set_ylabel('효율성')
            axes[0, 0].set_xticks(range(len(configs)))
            axes[0, 0].set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45, ha='right')
        
        # 평균 엔트로피
        if entropy_data:
            entropies = [entropy_data[config]['mean_entropy'] for config in configs]
            
            axes[0, 1].bar(range(len(configs)), entropies, color='orange', alpha=0.7)
            axes[0, 1].set_title('평균 Shannon 엔트로피')
            axes[0, 1].set_xlabel('차원 구성')
            axes[0, 1].set_ylabel('Shannon 엔트로피')
            axes[0, 1].set_xticks(range(len(configs)))
            axes[0, 1].set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45, ha='right')
        
        # 활용률
        if efficiency_data:
            util_configs = list(efficiency_data.keys())
            utilizations = [efficiency_data[config]['mean_utilization'] for config in util_configs]
            
            axes[1, 0].bar(range(len(util_configs)), utilizations, color='green', alpha=0.7)
            axes[1, 0].set_title('평균 차원 활용률')
            axes[1, 0].set_xlabel('차원 구성')
            axes[1, 0].set_ylabel('활용률')
            axes[1, 0].set_xticks(range(len(util_configs)))
            axes[1, 0].set_xticklabels([c.replace('_', '\n') for c in util_configs], rotation=45, ha='right')
        
        # 차원 수 vs 성능 산점도
        if entropy_data:
            dimension_counts = []
            mean_entropies = []
            
            for config in configs:
                # 차원 수 계산
                dim_config = self.dimension_configs[config]
                active_dims = sum(1 for size in dim_config.values() if size > 0)
                dimension_counts.append(active_dims)
                mean_entropies.append(entropy_data[config]['mean_entropy'])
            
            axes[1, 1].scatter(dimension_counts, mean_entropies, alpha=0.7, s=100)
            axes[1, 1].set_title('차원 수 vs 성능')
            axes[1, 1].set_xlabel('활성 차원 수')
            axes[1, 1].set_ylabel('평균 Shannon 엔트로피')
            
            # 추세선 추가
            z = np.polyfit(dimension_counts, mean_entropies, 1)
            p = np.poly1d(z)
            axes[1, 1].plot(sorted(dimension_counts), p(sorted(dimension_counts)), "r--", alpha=0.7)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'information_efficiency.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_final_report(self):
        """최종 보고서 저장"""
        report_path = os.path.join(self.results_dir, 'dimension_ablation_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 4D 디지털 페로몬 차원별 기여도 분석 보고서\n\n")
            
            # 요약 정보
            summary = self.analysis_results['summary']
            f.write("## 🔬 실험 요약\n")
            f.write(f"- 테스트된 차원 구성: {summary['total_configurations_tested']}개\n")
            f.write(f"- 성공한 실험: {summary['successful_experiments']}개\n")
            f.write(f"- 실패한 실험: {summary['failed_experiments']}개\n")
            f.write(f"- 최고 성능 구성: {summary['best_performing_configuration']}\n")
            f.write(f"- 가장 중요한 차원: {summary['most_important_dimension']}\n\n")
            
            # 주요 발견사항
            f.write("## 🎯 주요 발견사항\n")
            for finding in summary['key_findings']:
                f.write(f"- {finding}\n")
            f.write("\n")
            
            # 차원별 기여도
            contributions = self.analysis_results['dimension_contributions']
            f.write("## 📊 차원별 기여도 분석\n\n")
            
            if 'individual_contributions' in contributions:
                f.write("### 개별 차원 기여도\n")
                for dim, contrib in contributions['individual_contributions'].items():
                    f.write(f"- **{dim}**: {contrib:.4f}\n")
                f.write("\n")
            
            if 'marginal_contributions' in contributions:
                f.write("### 한계 기여도 (3D → 4D 추가 시)\n")
                for dim, contrib in contributions['marginal_contributions'].items():
                    f.write(f"- **{dim}**: {contrib:.4f}\n")
                f.write("\n")
            
            # 통계적 유의성
            statistical_tests = self.analysis_results['statistical_tests']
            f.write("## 📈 통계적 유의성 검정\n\n")
            
            significant_count = sum(1 for test in statistical_tests.values() 
                                  if test.get('significant', False))
            f.write(f"통계적으로 유의한 차이: {significant_count}/{len(statistical_tests)}개 테스트\n\n")
            
            for test_name, result in statistical_tests.items():
                if result.get('significant', False):
                    f.write(f"- **{test_name}**: F={result['f_statistic']:.3f}, "
                           f"p={result['p_value']:.4f} ✓\n")
            f.write("\n")
            
            # 시너지 효과
            synergies = self.analysis_results['synergy_effects']
            if synergies:
                f.write("## 🔄 차원 간 시너지 효과\n\n")
                for config, synergy_data in synergies.items():
                    dims = " + ".join(synergy_data['dimensions'])
                    effect = synergy_data['synergy_effect']
                    relative = synergy_data['relative_synergy']
                    
                    f.write(f"- **{dims}**: 시너지 효과 = {effect:.4f} "
                           f"(상대값: {relative:.2%})\n")
                f.write("\n")
            
            # 권장사항
            f.write("## 💡 권장사항\n\n")
            
            if summary['best_performing_configuration'] == 'full_4d':
                f.write("1. **4D 페로몬 사용 권장**: 모든 차원을 사용하는 것이 최적의 성능을 보입니다.\n")
            else:
                f.write(f"1. **최적 구성 적용**: {summary['best_performing_configuration']} 구성이 "
                       "최고 성능을 보입니다.\n")
            
            if summary['most_important_dimension']:
                f.write(f"2. **핵심 차원 집중**: {summary['most_important_dimension']} 차원이 "
                       "가장 큰 기여를 합니다.\n")
            
            f.write("3. **리소스 최적화**: 성능 요구사항과 계산 리소스를 고려하여 "
                   "적절한 차원 조합을 선택하세요.\n")
            
            # 실험 세부사항
            f.write("\n## 📋 실험 세부사항\n\n")
            f.write(f"- 실행 환경: {self.config['experiment']['name']}\n")
            f.write(f"- 반복 실행 횟수: {self.config['execution']['runs_per_configuration']}회\n")
            f.write(f"- 유의수준: {self.config['statistical_analysis']['significance_level']}\n")
            f.write(f"- 분석 일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        logger.info(f"최종 보고서 저장 완료: {report_path}")
    
    def run_complete_analysis(self):
        """전체 분석 파이프라인 실행"""
        logger.info("4D 페로몬 차원별 기여도 분석 시작")
        
        try:
            # 1. 모든 실험 실행
            self.run_all_experiments()
            
            # 2. 결과 분석
            self.analyze_results()
            
            # 3. 시각화 생성
            self.generate_visualizations()
            
            # 4. 최종 보고서 저장
            self.save_final_report()
            
            logger.info("차원별 기여도 분석 완료")
            
            return self.analysis_results
            
        except Exception as e:
            logger.error(f"분석 중 오류 발생: {e}")
            raise
        finally:
            if ray.is_initialized():
                ray.shutdown()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='4D 페로몬 차원별 기여도 분석')
    parser.add_argument('--config', type=str, 
                       default='config/ablation_dimension_config.yaml',
                       help='설정 파일 경로')
    
    args = parser.parse_args()
    
    # 분석 실행
    ablation_study = DimensionAblationStudy(args.config)
    results = ablation_study.run_complete_analysis()
    
    print("\n" + "="*60)
    print("4D 디지털 페로몬 차원별 기여도 분석 완료")
    print("="*60)
    print(f"결과 저장 위치: {ablation_study.results_dir}")
    print(f"최고 성능 구성: {results['summary']['best_performing_configuration']}")
    print(f"가장 중요한 차원: {results['summary']['most_important_dimension']}")
    print("="*60)


if __name__ == "__main__":
    main()