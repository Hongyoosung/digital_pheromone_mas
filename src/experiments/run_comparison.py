import os
import yaml
import torch
import numpy as np
import ray
from typing import Dict, List
import time
import logging
from tqdm import tqdm
import argparse
import pickle
import json
import pandas as pd
from scipy import stats

from src.experiments.run_experiment import ExperimentRunner
from src.utils.metrics import MetricsTracker
from src.utils.visualization import ExperimentVisualizer

logger = logging.getLogger(__name__)

class ComparisonExperimentRunner:
    """연구 계획서 명시 비교 실험 실행기"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = "results/comparison/"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 비교 대상 설정
        self.baseline_methods = self.config['research_design']['baseline_methods']
        self.num_runs = self.config['research_design']['num_runs']
        self.significance_level = self.config['research_design']['statistical_analysis']['significance_level']
        
        # 결과 저장용
        self.comparison_results = {}
        
    def run_baseline_experiment(self, method: str, run_id: int) -> Dict:
        """기준선 실험 실행"""
        logger.info(f"기준선 실험 실행: {method}, Run {run_id}")
        
        # 기준선별 설정 수정
        baseline_config = self.config.copy()
        
        if method == "rule_based_diffusion":
            # 규칙 기반 확산 모델
            baseline_config['pheromone']['decay_rate'] = 0.98  # 단순 감쇠
            baseline_config['attention']['num_heads'] = 1  # 단순 어텐션
            baseline_config['hyperparameters']['communication_period'] = [10]  # 낮은 통신 빈도
            
        elif method == "centralized_attention":
            # 중앙집중 어텐션 네트워크
            baseline_config['attention']['topology_type'] = "centralized"
            baseline_config['hyperparameters']['communication_period'] = [1]  # 높은 통신 빈도
            
        # 기준선 실험 실행
        runner = ExperimentRunner(config_path=None)  # 설정을 직접 전달
        runner.config = baseline_config
        runner.setup_experiment()
        
        results = runner.run_experiment()
        
        # 결과에 메서드 정보 추가
        results['method'] = method
        results['run_id'] = run_id
        
        return results
    
    def run_proposed_method(self, run_id: int) -> Dict:
        """제안 방법 실험 실행"""
        logger.info(f"제안 방법 실험 실행: Run {run_id}")
        
        runner = ExperimentRunner(config_path=None)
        runner.config = self.config
        runner.setup_experiment()
        
        results = runner.run_experiment()
        results['method'] = 'proposed_digital_pheromone'
        results['run_id'] = run_id
        
        return results
    
    def run_comparison_experiment(self):
        """전체 비교 실험 실행"""
        logger.info("연구 계획서 명시 비교 실험 시작")
        
        all_results = []
        
        # 제안 방법 실험 (10회 반복)
        logger.info("제안 방법 실험 실행 중...")
        for run_id in tqdm(range(self.num_runs), desc="제안 방법"):
            try:
                results = self.run_proposed_method(run_id)
                all_results.append(results)
                
                # 중간 결과 저장
                with open(os.path.join(self.results_dir, f"proposed_run_{run_id}.pkl"), 'wb') as f:
                    pickle.dump(results, f)
                    
            except Exception as e:
                logger.error(f"제안 방법 Run {run_id} 실패: {e}")
        
        # 기준선 방법들 실험
        for method in self.baseline_methods:
            logger.info(f"{method} 기준선 실험 실행 중...")
            for run_id in tqdm(range(self.num_runs), desc=method):
                try:
                    results = self.run_baseline_experiment(method, run_id)
                    all_results.append(results)
                    
                    # 중간 결과 저장
                    with open(os.path.join(self.results_dir, f"{method}_run_{run_id}.pkl"), 'wb') as f:
                        pickle.dump(results, f)
                        
                except Exception as e:
                    logger.error(f"{method} Run {run_id} 실패: {e}")
        
        # 결과 분석
        self.analyze_comparison_results(all_results)
        
        # 최종 결과 저장
        with open(os.path.join(self.results_dir, "comparison_results.pkl"), 'wb') as f:
            pickle.dump(all_results, f)
        
        logger.info("비교 실험 완료")
        return all_results
    
    def analyze_comparison_results(self, all_results: List[Dict]):
        """비교 실험 결과 분석"""
        logger.info("비교 실험 결과 분석 중...")
        
        # 결과를 메서드별로 분류
        method_results = {}
        for result in all_results:
            method = result['method']
            if method not in method_results:
                method_results[method] = []
            method_results[method].append(result)
        
        # 주요 지표들 추출
        metrics_to_analyze = [
            'information_transfer_efficiency',
            'learning_convergence_epochs', 
            'communication_overhead',
            'network_load',
            'shannon_entropy',
            'success_rate',
            'reward'
        ]
        
        analysis_results = {}
        
        for metric in metrics_to_analyze:
            metric_data = {}
            
            for method, results in method_results.items():
                values = []
                for result in results:
                    # 메트릭 값 추출
                    if metric in result.get('training_summary', {}).get('research_metrics', {}):
                        value = result['training_summary']['research_metrics'][metric]
                    elif metric in result.get('training_summary', {}).get('performance_analysis', {}):
                        value = result['training_summary']['performance_analysis'][metric].get('final', 0)
                    else:
                        value = 0.0
                    
                    values.append(value)
                
                metric_data[method] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
            
            analysis_results[metric] = metric_data
        
        # 통계적 유의성 검정
        statistical_tests = {}
        
        for metric, metric_data in analysis_results.items():
            if 'proposed_digital_pheromone' in metric_data:
                proposed_values = metric_data['proposed_digital_pheromone']['values']
                
                for method in self.baseline_methods:
                    if method in metric_data:
                        baseline_values = metric_data[method]['values']
                        
                        # t-검정 수행
                        t_stat, p_value = stats.ttest_ind(proposed_values, baseline_values)
                        
                        statistical_tests[f"{metric}_{method}"] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < self.significance_level,
                            'effect_size': (np.mean(proposed_values) - np.mean(baseline_values)) / np.sqrt(
                                (np.var(proposed_values) + np.var(baseline_values)) / 2
                            )
                        }
        
        # 분석 결과 저장
        self.comparison_results = {
            'method_results': method_results,
            'analysis_results': analysis_results,
            'statistical_tests': statistical_tests
        }
        
        # 분석 결과를 CSV로 저장
        self.save_analysis_to_csv(analysis_results, statistical_tests)
        
        # 분석 보고서 생성
        self.generate_comparison_report(analysis_results, statistical_tests)
    
    def save_analysis_to_csv(self, analysis_results: Dict, statistical_tests: Dict):
        """분석 결과를 CSV 파일로 저장"""
        # 메트릭별 결과 테이블
        metric_data = []
        for metric, metric_data_dict in analysis_results.items():
            for method, stats_dict in metric_data_dict.items():
                metric_data.append({
                    'metric': metric,
                    'method': method,
                    'mean': stats_dict['mean'],
                    'std': stats_dict['std'],
                    'min': stats_dict['min'],
                    'max': stats_dict['max']
                })
        
        df_metrics = pd.DataFrame(metric_data)
        df_metrics.to_csv(os.path.join(self.results_dir, "comparison_metrics.csv"), index=False)
        
        # 통계 검정 결과 테이블
        test_data = []
        for test_name, test_result in statistical_tests.items():
            test_data.append({
                'test': test_name,
                't_statistic': test_result['t_statistic'],
                'p_value': test_result['p_value'],
                'significant': test_result['significant'],
                'effect_size': test_result['effect_size']
            })
        
        df_tests = pd.DataFrame(test_data)
        df_tests.to_csv(os.path.join(self.results_dir, "statistical_tests.csv"), index=False)
    
    def generate_comparison_report(self, analysis_results: Dict, statistical_tests: Dict):
        """비교 실험 보고서 생성"""
        report_path = os.path.join(self.results_dir, "comparison_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("4D 디지털 페로몬 MAS 비교 실험 보고서\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("📊 실험 개요\n")
            f.write("-" * 40 + "\n")
            f.write(f"제안 방법: 4D 디지털 페로몬 + 분산 어텐션 네트워크\n")
            f.write(f"기준선 방법: {', '.join(self.baseline_methods)}\n")
            f.write(f"반복 실행 횟수: {self.num_runs}회\n")
            f.write(f"유의수준: α = {self.significance_level}\n\n")
            
            f.write("📈 성능 비교 결과\n")
            f.write("-" * 40 + "\n")
            
            for metric, metric_data in analysis_results.items():
                f.write(f"\n{metric.upper()}:\n")
                for method, stats in metric_data.items():
                    f.write(f"  {method}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                           f"(범위: {stats['min']:.4f} - {stats['max']:.4f})\n")
            
            f.write("\n🔬 통계적 유의성 검정 결과\n")
            f.write("-" * 40 + "\n")
            
            significant_tests = [name for name, result in statistical_tests.items() 
                               if result['significant']]
            
            if significant_tests:
                f.write("통계적으로 유의한 차이를 보인 지표들:\n")
                for test_name in significant_tests:
                    result = statistical_tests[test_name]
                    f.write(f"  - {test_name}: t = {result['t_statistic']:.3f}, "
                           f"p = {result['p_value']:.4f}, "
                           f"효과크기 = {result['effect_size']:.3f}\n")
            else:
                f.write("통계적으로 유의한 차이를 보인 지표가 없습니다.\n")
            
            f.write("\n💡 결론 및 권장사항\n")
            f.write("-" * 40 + "\n")
            
            # 성능 향상도 계산
            improvements = {}
            for metric, metric_data in analysis_results.items():
                if 'proposed_digital_pheromone' in metric_data:
                    proposed_mean = metric_data['proposed_digital_pheromone']['mean']
                    
                    for method in self.baseline_methods:
                        if method in metric_data:
                            baseline_mean = metric_data[method]['mean']
                            improvement = ((proposed_mean - baseline_mean) / baseline_mean) * 100
                            improvements[f"{metric}_{method}"] = improvement
            
            if improvements:
                f.write("제안 방법의 성능 향상도:\n")
                for key, improvement in improvements.items():
                    f.write(f"  - {key}: {improvement:+.2f}%\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("보고서 생성 완료\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"비교 실험 보고서가 저장되었습니다: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Run Digital Pheromone MAS Comparison Experiment")
    parser.add_argument('--config', type=str, default='config/experiment_config.yaml',
                        help='Path to the experiment configuration file.')
    args = parser.parse_args()
    
    # 비교 실험 실행
    comparison_runner = ComparisonExperimentRunner(config_path=args.config)
    results = comparison_runner.run_comparison_experiment()
    
    logger.info("비교 실험 완료!")

if __name__ == "__main__":
    main()
