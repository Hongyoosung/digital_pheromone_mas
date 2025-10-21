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
from src.models.baseline_models import BaselineComparator

logger = logging.getLogger(__name__)

class ComparisonExperimentRunner:
    """연구 계획서 명시 비교 실험 실행기"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 비교군 실험 전용 결과 디렉토리
        self.results_dir = "results/comparison/"
        self.proposed_dir = os.path.join(self.results_dir, "proposed_method")
        self.baseline_dir = os.path.join(self.results_dir, "baseline_methods")
        
        # 각 방법론별 디렉토리 생성
        os.makedirs(self.proposed_dir, exist_ok=True)
        os.makedirs(self.baseline_dir, exist_ok=True)
        
        for method in self.config['research_design']['baseline_methods']:
            method_dir = os.path.join(self.baseline_dir, method)
            os.makedirs(method_dir, exist_ok=True)
        
        logger.info(f"결과 디렉토리 생성 완료: {self.results_dir}")
        logger.info(f"제안 방법 디렉토리: {self.proposed_dir}")
        logger.info(f"기준선 방법 디렉토리: {self.baseline_dir}")
        
        # 비교 대상 설정
        self.baseline_methods = self.config['research_design']['baseline_methods']
        self.num_runs = self.config['research_design']['num_runs']
        self.significance_level = self.config['research_design']['statistical_analysis']['significance_level']
        
        # 결과 저장용
        self.comparison_results = {}
        
        # 비교군 모델 초기화
        self.baseline_comparator = BaselineComparator(self.config)
        
    def run_baseline_experiment(self, method: str, run_id: int) -> Dict:
        """기준선 실험 실행"""
        logger.info(f"기준선 실험 실행: {method}, Run {run_id}")
        
        # 기준선별 설정 수정
        baseline_config = self.config.copy()
        
        if method == "rule_based_diffusion":
            # 규칙 기반 확산 모델 설정
            baseline_config['models']['use_rule_based'] = True
            baseline_config['pheromone']['decay_rate'] = 0.1
            baseline_config['attention']['num_heads'] = 1
            baseline_config['hyperparameters']['communication_period'] = [10]
            
        elif method == "centralized_attention":
            # 중앙집중 어텐션 네트워크 설정
            baseline_config['models']['use_centralized'] = True
            baseline_config['attention']['topology_type'] = "centralized"
            baseline_config['hyperparameters']['communication_period'] = [1]
            
        elif method == "ablation_2d_pheromone":
            # 2D 페로몬 실험 설정 - 사회관계와 환경맥락 제외
            baseline_config['models']['use_2d_pheromone'] = True
            baseline_config['pheromone']['dimensions'] = {
                'behavior': 4,  # 행동 차원
                'emotion': 5,   # 감정 차원
                'social': 0,    # 사회관계 차원 제외 (0으로 설정)
                'context': 0    # 환경맥락 차원 제외 (0으로 설정)
            }
            
        # 임시 설정 파일 생성
        temp_config_path = f"temp_config_{method}_{run_id}.yaml"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(baseline_config, f, allow_unicode=True)
        
        # 기준선 실험 실행
        runner = ExperimentRunner(config_path=temp_config_path)
        
        # 비교군 모델 적용
        if hasattr(runner, 'trainer') and runner.trainer:
            runner.trainer.baseline_comparator = self.baseline_comparator
        
        try:
            results = runner.run_experiment()
            
            # 비교군 실험 결과 추가
            if hasattr(runner, 'attention_router') and runner.attention_router:
                embed_dim = baseline_config.get('embed_dim', 64)
                agent_embeddings = torch.randn(1, 5, embed_dim)  # 설정에 맞는 차원 사용
                pheromone_field = torch.randn(1, 4, 25, 25)  # 빠른 실험 맵 크기에 맞춤
                
                comparison_results = self.baseline_comparator.run_comparison_experiment(
                    agent_embeddings, pheromone_field, timestep=50
                )
                
                comparison_metrics = self.baseline_comparator.get_comparison_metrics(comparison_results)
                results['baseline_comparison'] = {
                    'results': comparison_results,
                    'metrics': comparison_metrics
                }
            
            # 결과에 메서드 정보 추가
            results['method'] = method
            results['run_id'] = run_id
            
            return results
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    
    def run_proposed_method(self, run_id: int) -> Dict:
        """제안 방법 실험 실행"""
        logger.info(f"제안 방법 실험 실행: Run {run_id}")
        
        # 임시 설정 파일 생성
        temp_config_path = f"temp_config_proposed_{run_id}.yaml"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True)
        
        try:
            runner = ExperimentRunner(config_path=temp_config_path)
            results = runner.run_experiment()
            results['method'] = 'proposed_digital_pheromone'
            results['run_id'] = run_id
            return results
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    
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
                
                # 제안 방법 결과를 전용 디렉토리에 저장 (pkl과 json 모두)
                pkl_path = os.path.normpath(os.path.join(self.proposed_dir, f"proposed_run_{run_id}.pkl"))
                with open(pkl_path, 'wb') as f:
                    pickle.dump(results, f)
                
                # JSON 형태로도 저장 (가독성을 위해)
                json_path = os.path.normpath(os.path.join(self.proposed_dir, f"proposed_run_{run_id}.json"))
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
                
                logger.info(f"제안 방법 Run {run_id} 결과 저장 완료: {pkl_path}")
                    
            except Exception as e:
                logger.error(f"제안 방법 Run {run_id} 실패: {e}")
        
        # 기준선 방법들 실험
        for method in self.baseline_methods:
            logger.info(f"{method} 기준선 실험 실행 중...")
            for run_id in tqdm(range(self.num_runs), desc=method):
                try:
                    results = self.run_baseline_experiment(method, run_id)
                    all_results.append(results)
                    
                    # 기준선 방법별 전용 디렉토리에 결과 저장 (pkl과 json 모두)
                    method_dir = os.path.join(self.baseline_dir, method)
                    pkl_path = os.path.normpath(os.path.join(method_dir, f"{method}_run_{run_id}.pkl"))
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(results, f)
                    
                    # JSON 형태로도 저장
                    json_path = os.path.normpath(os.path.join(method_dir, f"{method}_run_{run_id}.json"))
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
                    
                    logger.info(f"기준선 방법 {method} Run {run_id} 결과 저장 완료: {pkl_path}")
                        
                except Exception as e:
                    logger.error(f"{method} Run {run_id} 실패: {e}")
        
        # 결과 분석
        self.analyze_comparison_results(all_results)
        
        # 최종 결과 저장
        final_results_path = os.path.normpath(os.path.join(self.results_dir, "comparison_results.pkl"))
        with open(final_results_path, 'wb') as f:
            pickle.dump(all_results, f)
        
        logger.info(f"최종 비교 결과 저장 완료: {final_results_path}")
        
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
            'ray_communication_overhead',
            'ray_network_load', 
            'ray_bandwidth_utilization',
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
                    # 메트릭 값 추출 - 다양한 경로에서 메트릭 검색
                    value = 0.0
                    
                    # 1. summary.research_metrics에서 검색
                    if 'summary' in result and 'research_metrics' in result['summary']:
                        if metric in result['summary']['research_metrics']:
                            value = result['summary']['research_metrics'][metric]
                    # 2. training_summary 경로에서 검색 (기존)
                    elif metric in result.get('training_summary', {}).get('research_metrics', {}):
                        value = result['training_summary']['research_metrics'][metric]
                    elif metric in result.get('training_summary', {}).get('performance_analysis', {}):
                        perf_metric = result['training_summary']['performance_analysis'][metric]
                        if isinstance(perf_metric, dict):
                            value = perf_metric.get('final', 0)
                        else:
                            value = perf_metric if perf_metric is not None else 0.0
                    # 3. 직접 결과에서 검색
                    elif metric in result:
                        value = result[metric]
                    # 4. metrics 키 아래에서 검색
                    elif 'metrics' in result and isinstance(result['metrics'], list) and result['metrics']:
                        # 마지막 타임스텝의 메트릭 사용
                        last_metrics = result['metrics'][-1] if result['metrics'] else {}
                        if metric in last_metrics:
                            value = last_metrics[metric]
                    
                    # 값이 숫자가 아닌 경우 0.0으로 설정
                    if not isinstance(value, (int, float, np.generic)):
                        value = 0.0

                    values.append(float(value))
                
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
        
        # 분석 결과를 CSV로 저장 (비교군 결과 디렉토리에)
        self.save_analysis_to_csv(analysis_results, statistical_tests)
        
        # 실험 요약 통계 저장
        self.save_experiment_summary(analysis_results, statistical_tests)
        
        # 분석 보고서 생성
        self.generate_comparison_report(analysis_results, statistical_tests)
        
        # JSON 형태로도 결과 저장
        self.save_results_as_json(analysis_results, statistical_tests)
        
        # 비교 실험용 training_summary.txt 생성
        self.save_comparison_training_summary_to_file(all_results, analysis_results, statistical_tests)
        
        logger.info(f"비교 실험 결과가 저장되었습니다: {self.results_dir}")
    
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
            significant_improvements = []
            
            for metric, metric_data in analysis_results.items():
                if 'proposed_digital_pheromone' in metric_data:
                    proposed_mean = metric_data['proposed_digital_pheromone']['mean']
                    
                    for method in self.baseline_methods:
                        if method in metric_data:
                            baseline_mean = metric_data[method]['mean']
                            if baseline_mean != 0:
                                improvement = ((proposed_mean - baseline_mean) / abs(baseline_mean)) * 100
                                improvements[f"{metric}_vs_{method}"] = improvement
                                
                                # 통계적으로 유의한 개선 확인
                                test_key = f"{metric}_{method}"
                                if test_key in statistical_tests and statistical_tests[test_key]['significant']:
                                    significant_improvements.append({
                                        'metric': metric,
                                        'baseline': method,
                                        'improvement': improvement,
                                        'p_value': statistical_tests[test_key]['p_value'],
                                        'effect_size': statistical_tests[test_key]['effect_size']
                                    })
            
            # 전체 성능 향상도 보고
            if improvements:
                f.write("제안 방법의 전체 성능 향상도:\n")
                for key, improvement in improvements.items():
                    metric, baseline = key.replace('_vs_', ' vs ').split(' vs ')
                    significance = ""
                    test_key = f"{metric}_{baseline}"
                    if test_key in statistical_tests and statistical_tests[test_key]['significant']:
                        significance = " (통계적 유의성 확인)"
                    f.write(f"  - {metric} vs {baseline}: {improvement:+.2f}%{significance}\n")
            
            # 주요 성과 요약
            f.write("\n📊 주요 성과 요약:\n")
            if significant_improvements:
                f.write(f"• {len(significant_improvements)}개 지표에서 통계적으로 유의한 성능 향상 확인\n")
                
                # 가장 큰 개선 사항 강조
                best_improvement = max(significant_improvements, key=lambda x: abs(x['improvement']))
                f.write(f"• 최대 개선: {best_improvement['metric']} 지표에서 "
                       f"{best_improvement['improvement']:+.2f}% 향상 "
                       f"(vs {best_improvement['baseline']})\n")
                
                # 효과 크기 분석
                large_effects = [imp for imp in significant_improvements if abs(imp['effect_size']) > 0.8]
                if large_effects:
                    f.write(f"• {len(large_effects)}개 지표에서 큰 효과 크기 (|d| > 0.8) 확인\n")
            else:
                f.write("• 통계적으로 유의한 성능 개선이 확인되지 않음\n")
            
            # 권장사항
            f.write("\n📈 권장사항:\n")
            if len(significant_improvements) >= 3:
                f.write("• 제안된 4D 디지털 페로몬 방법이 우수한 성능을 보이고 있음\n")
                f.write("• 실제 시스템 적용을 위한 추가 검증 권장\n")
            elif len(significant_improvements) > 0:
                f.write("• 일부 지표에서 개선 확인, 추가 최적화 필요\n")
                f.write("• 하이퍼파라미터 튜닝 및 아키텍처 개선 권장\n")
            else:
                f.write("• 현재 구현에서는 기존 방법 대비 명확한 우위 미확인\n")
                f.write("• 모델 아키텍처 재검토 및 훈련 전략 개선 필요\n")
                f.write("• 더 많은 반복 실험으로 데이터 수집 권장\n")
            
            # 실험 메타데이터
            f.write("\n📁 실험 메타데이터:\n")
            total_experiments = self.num_runs * (len(self.baseline_methods) + 1)  # +1 for proposed method
            f.write(f"• 총 실험 횟수: {total_experiments}회\n")
            f.write(f"• 비교 지표 수: {len(analysis_results)}개\n")
            f.write(f"• 통계 검정 수: {len(statistical_tests)}개\n")
            f.write(f"• 보고서 생성 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("보고서 생성 완료\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"비교 실험 보고서가 저장되었습니다: {report_path}")
    
    def save_results_as_json(self, analysis_results: Dict, statistical_tests: Dict):
        """분석 결과를 JSON 파일로 저장"""
        json_results = {
            'experiment_info': {
                'baseline_methods': self.baseline_methods,
                'num_runs': self.num_runs,
                'significance_level': self.significance_level,
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'analysis_results': analysis_results,
            'statistical_tests': statistical_tests,
            'summary': self._generate_summary_statistics(analysis_results, statistical_tests)
        }
        
        # JSON 파일로 저장
        json_path = os.path.join(self.results_dir, "comparison_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"JSON 결과 파일이 저장되었습니다: {json_path}")
    
    def _generate_summary_statistics(self, analysis_results: Dict, statistical_tests: Dict) -> Dict:
        """요약 통계 생성"""
        summary = {
            'total_metrics_analyzed': len(analysis_results),
            'total_statistical_tests': len(statistical_tests),
            'significant_tests': sum(1 for test in statistical_tests.values() if test['significant']),
            'methods_compared': list(set(
                method for metric_data in analysis_results.values() 
                for method in metric_data.keys()
            ))
        }
        
        # 제안 방법의 성능 향상도 계산
        if any('proposed_digital_pheromone' in metric_data for metric_data in analysis_results.values()):
            improvements = {}
            for metric, metric_data in analysis_results.items():
                if 'proposed_digital_pheromone' in metric_data:
                    proposed_mean = metric_data['proposed_digital_pheromone']['mean']
                    
                    for method in self.baseline_methods:
                        if method in metric_data:
                            baseline_mean = metric_data[method]['mean']
                            if baseline_mean != 0:
                                improvement = ((proposed_mean - baseline_mean) / abs(baseline_mean)) * 100
                                improvements[f"{metric}_vs_{method}"] = improvement
            
            summary['performance_improvements'] = improvements
        
        return summary
    
    def save_experiment_summary(self, analysis_results: Dict, statistical_tests: Dict):
        """실험 요약 통계를 간결한 형태로 저장"""
        summary_data = []
        
        for metric, metric_data in analysis_results.items():
            for method, stats in metric_data.items():
                # 통계적 유의성 확인
                is_significant = False
                p_value = None
                effect_size = None
                
                if method != 'proposed_digital_pheromone':
                    test_key = f"{metric}_{method}"
                    if test_key in statistical_tests:
                        is_significant = statistical_tests[test_key]['significant']
                        p_value = statistical_tests[test_key]['p_value']
                        effect_size = statistical_tests[test_key]['effect_size']
                
                summary_data.append({
                    'metric': metric,
                    'method': method,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'is_proposed': method == 'proposed_digital_pheromone',
                    'is_significant': is_significant,
                    'p_value': p_value,
                    'effect_size': effect_size
                })
        
        # DataFrame으로 변환 및 저장
        df_summary = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.results_dir, "experiment_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        
        # 제안 방법 vs 기준선 비교 테이블 생성
        if 'proposed_digital_pheromone' in df_summary['method'].values:
            comparison_data = []
            proposed_data = df_summary[df_summary['method'] == 'proposed_digital_pheromone']
            
            for _, proposed_row in proposed_data.iterrows():
                metric = proposed_row['metric']
                baseline_data = df_summary[
                    (df_summary['metric'] == metric) & 
                    (df_summary['method'] != 'proposed_digital_pheromone')
                ]
                
                for _, baseline_row in baseline_data.iterrows():
                    improvement = 0
                    if baseline_row['mean'] != 0:
                        improvement = ((proposed_row['mean'] - baseline_row['mean']) / abs(baseline_row['mean'])) * 100
                    
                    comparison_data.append({
                        'metric': metric,
                        'baseline_method': baseline_row['method'],
                        'proposed_mean': proposed_row['mean'],
                        'baseline_mean': baseline_row['mean'],
                        'improvement_percent': improvement,
                        'is_significant': baseline_row['is_significant'],
                        'p_value': baseline_row['p_value'],
                        'effect_size': baseline_row['effect_size']
                    })
            
            df_comparison = pd.DataFrame(comparison_data)
            comparison_path = os.path.join(self.results_dir, "method_comparison.csv")
            df_comparison.to_csv(comparison_path, index=False)
            
            logger.info(f"실험 요약 파일이 저장되었습니다: {summary_path}")
            logger.info(f"방법 비교 파일이 저장되었습니다: {comparison_path}")
    
    def save_comparison_training_summary_to_file(self, all_results: List[Dict], analysis_results: Dict, statistical_tests: Dict):
        """베이스라인과 제안된 모델의 비교 결과를 포함한 training_summary.txt 생성"""
        summary_path = os.path.join(self.results_dir, 'training_summary.txt')
        
        # 결과를 메서드별로 분류
        method_results = {}
        for result in all_results:
            method = result['method']
            if method not in method_results:
                method_results[method] = []
            method_results[method].append(result)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("4D 디지털 페로몬 MAS 비교 실험 훈련 요약 보고서\n")
            f.write("=" * 80 + "\n\n")
            
            # 실험 개요
            f.write("🔬 비교 실험 개요\n")
            f.write("-" * 40 + "\n")
            f.write(f"제안 방법: 4D 디지털 페로몬 + 분산 어텐션 네트워크\n")
            f.write(f"기준선 방법: {', '.join(self.baseline_methods)}\n")
            f.write(f"반복 실행 횟수: {self.num_runs}회\n")
            f.write(f"총 실험 수: {len(all_results)}개\n")
            f.write(f"유의수준: α = {self.significance_level}\n\n")
            
            # 제안 방법 요약
            if 'proposed_digital_pheromone' in method_results:
                proposed_results = method_results['proposed_digital_pheromone']
                f.write("🚀 제안 방법 성능 요약\n")
                f.write("-" * 40 + "\n")
                
                # 평균 성능 지표 계산
                avg_metrics = self._calculate_average_metrics(proposed_results)
                f.write(f"정보 전달 효율성: {avg_metrics.get('information_transfer_efficiency', 0):.4f} ± {self._calculate_std_metrics(proposed_results, 'information_transfer_efficiency'):.4f}\n")
                f.write(f"학습 수렴 에포크: {avg_metrics.get('learning_convergence_epochs', 0):.1f} ± {self._calculate_std_metrics(proposed_results, 'learning_convergence_epochs'):.1f}\n")
                f.write(f"통신 오버헤드: {avg_metrics.get('communication_overhead', 0):.4f} ± {self._calculate_std_metrics(proposed_results, 'communication_overhead'):.4f}\n")
                f.write(f"네트워크 부하: {avg_metrics.get('network_load', 0):.4f} ± {self._calculate_std_metrics(proposed_results, 'network_load'):.4f}\n")
                f.write(f"Shannon 엔트로피: {avg_metrics.get('shannon_entropy', 0):.4f} ± {self._calculate_std_metrics(proposed_results, 'shannon_entropy'):.4f}\n")
                f.write(f"성공률: {avg_metrics.get('success_rate', 0):.4f} ± {self._calculate_std_metrics(proposed_results, 'success_rate'):.4f}\n")
                f.write(f"평균 보상: {avg_metrics.get('reward', 0):.4f} ± {self._calculate_std_metrics(proposed_results, 'reward'):.4f}\n\n")
            
            # 베이스라인 방법들 요약
            f.write("🎯 기준선 방법 성능 요약\n")
            f.write("-" * 40 + "\n")
            
            for method in self.baseline_methods:
                if method in method_results:
                    baseline_results = method_results[method]
                    f.write(f"\n[{method.upper()}]\n")
                    
                    avg_metrics = self._calculate_average_metrics(baseline_results)
                    f.write(f"  정보 전달 효율성: {avg_metrics.get('information_transfer_efficiency', 0):.4f} ± {self._calculate_std_metrics(baseline_results, 'information_transfer_efficiency'):.4f}\n")
                    f.write(f"  학습 수렴 에포크: {avg_metrics.get('learning_convergence_epochs', 0):.1f} ± {self._calculate_std_metrics(baseline_results, 'learning_convergence_epochs'):.1f}\n")
                    f.write(f"  통신 오버헤드: {avg_metrics.get('communication_overhead', 0):.4f} ± {self._calculate_std_metrics(baseline_results, 'communication_overhead'):.4f}\n")
                    f.write(f"  네트워크 부하: {avg_metrics.get('network_load', 0):.4f} ± {self._calculate_std_metrics(baseline_results, 'network_load'):.4f}\n")
                    f.write(f"  Shannon 엔트로피: {avg_metrics.get('shannon_entropy', 0):.4f} ± {self._calculate_std_metrics(baseline_results, 'shannon_entropy'):.4f}\n")
                    f.write(f"  성공률: {avg_metrics.get('success_rate', 0):.4f} ± {self._calculate_std_metrics(baseline_results, 'success_rate'):.4f}\n")
                    f.write(f"  평균 보상: {avg_metrics.get('reward', 0):.4f} ± {self._calculate_std_metrics(baseline_results, 'reward'):.4f}\n")
            
            f.write("\n")
            
            # 성능 비교 분석
            f.write("📊 성능 비교 분석\n")
            f.write("-" * 40 + "\n")
            
            if 'proposed_digital_pheromone' in analysis_results:
                for metric, metric_data in analysis_results.items():
                    if 'proposed_digital_pheromone' in metric_data:
                        f.write(f"\n[{metric.upper()}]\n")
                        proposed_mean = metric_data['proposed_digital_pheromone']['mean']
                        f.write(f"  제안 방법: {proposed_mean:.4f}\n")
                        
                        for method in self.baseline_methods:
                            if method in metric_data:
                                baseline_mean = metric_data[method]['mean']
                                improvement = 0
                                if baseline_mean != 0:
                                    improvement = ((proposed_mean - baseline_mean) / abs(baseline_mean)) * 100
                                
                                # 통계적 유의성 확인
                                test_key = f"{metric}_{method}"
                                significance = ""
                                if test_key in statistical_tests and statistical_tests[test_key]['significant']:
                                    p_value = statistical_tests[test_key]['p_value']
                                    significance = f" (p={p_value:.4f}, 통계적 유의성 ✅)"
                                else:
                                    significance = " (통계적 유의성 ❌)"
                                
                                f.write(f"  vs {method}: {baseline_mean:.4f} → {improvement:+.2f}% 차이{significance}\n")
            
            # 통계적 유의성 검정 요약
            f.write("\n🔬 통계적 유의성 검정 요약\n")
            f.write("-" * 40 + "\n")
            
            significant_tests = [name for name, result in statistical_tests.items() if result['significant']]
            total_tests = len(statistical_tests)
            
            f.write(f"전체 통계 검정: {total_tests}개\n")
            f.write(f"통계적 유의한 차이: {len(significant_tests)}개 ({len(significant_tests)/max(total_tests,1)*100:.1f}%)\n\n")
            
            if significant_tests:
                f.write("통계적으로 유의한 개선 지표:\n")
                for test_name in significant_tests[:10]:  # 상위 10개만 표시
                    result = statistical_tests[test_name]
                    metric, baseline = test_name.split('_', 1)
                    f.write(f"  • {metric} vs {baseline}: ")
                    f.write(f"t={result['t_statistic']:.3f}, p={result['p_value']:.4f}, ")
                    f.write(f"효과크기={result['effect_size']:.3f}\n")
            else:
                f.write("통계적으로 유의한 성능 차이가 발견되지 않았습니다.\n")
            
            # 전체 성능 향상도 요약
            f.write("\n📈 전체 성능 향상도 요약\n")
            f.write("-" * 40 + "\n")
            
            significant_improvements = []
            all_improvements = []
            
            for metric, metric_data in analysis_results.items():
                if 'proposed_digital_pheromone' in metric_data:
                    proposed_mean = metric_data['proposed_digital_pheromone']['mean']
                    
                    for method in self.baseline_methods:
                        if method in metric_data:
                            baseline_mean = metric_data[method]['mean']
                            if baseline_mean != 0:
                                improvement = ((proposed_mean - baseline_mean) / abs(baseline_mean)) * 100
                                all_improvements.append(improvement)
                                
                                test_key = f"{metric}_{method}"
                                if test_key in statistical_tests and statistical_tests[test_key]['significant']:
                                    significant_improvements.append({
                                        'metric': metric,
                                        'baseline': method,
                                        'improvement': improvement
                                    })
            
            if all_improvements:
                avg_improvement = np.mean(all_improvements)
                f.write(f"전체 평균 성능 향상: {avg_improvement:+.2f}%\n")
                
                positive_improvements = [imp for imp in all_improvements if imp > 0]
                if positive_improvements:
                    f.write(f"개선된 지표 비율: {len(positive_improvements)}/{len(all_improvements)} ({len(positive_improvements)/len(all_improvements)*100:.1f}%)\n")
                
                if significant_improvements:
                    sig_avg = np.mean([imp['improvement'] for imp in significant_improvements])
                    f.write(f"통계적 유의한 개선 평균: {sig_avg:+.2f}%\n")
            
            # 결론 및 권장사항
            f.write("\n💡 결론 및 권장사항\n")
            f.write("-" * 40 + "\n")
            
            if len(significant_improvements) >= 3:
                f.write("✅ 제안된 4D 디지털 페로몬 방법이 다수 지표에서 통계적으로 유의한 성능 향상을 보입니다.\n")
                f.write("• 연구 가설이 실험적으로 검증되었습니다.\n")
                f.write("• 실제 시스템 적용을 위한 추가 검증을 권장합니다.\n")
            elif len(significant_improvements) > 0:
                f.write("⚠️ 일부 지표에서 통계적 개선이 확인되었으나 전체적인 우위는 제한적입니다.\n")
                f.write("• 추가적인 하이퍼파라미터 최적화가 필요합니다.\n")
                f.write("• 더 많은 반복 실험으로 결과의 신뢰성을 높이는 것을 권장합니다.\n")
            else:
                f.write("❌ 현재 설정에서는 제안 방법의 명확한 우위가 확인되지 않았습니다.\n")
                f.write("• 모델 아키텍처의 재검토가 필요합니다.\n")
                f.write("• 훈련 전략 및 실험 설계의 개선을 권장합니다.\n")
            
            # 실험 메타데이터
            f.write(f"\n📁 실험 메타데이터\n")
            f.write("-" * 40 + "\n")
            f.write(f"실험 완료 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"총 분석된 지표: {len(analysis_results)}개\n")
            f.write(f"결과 저장 위치: {self.results_dir}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("비교 실험 훈련 요약 보고서 생성 완료\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"비교 실험 훈련 요약 보고서가 저장되었습니다: {summary_path}")
    
    def _calculate_average_metrics(self, results: List[Dict]) -> Dict:
        """결과 리스트에서 평균 메트릭 계산"""
        metrics_sum = {}
        metrics_count = {}
        
        for result in results:
            # 다양한 경로에서 메트릭 추출
            metrics_sources = [
                result.get('summary', {}).get('research_metrics', {}),
                result.get('training_summary', {}).get('research_metrics', {}),
                result.get('training_summary', {}).get('performance_analysis', {}),
                result
            ]
            
            for source in metrics_sources:
                if isinstance(source, dict):
                    for key, value in source.items():
                        if isinstance(value, (int, float, np.generic)):
                            if key not in metrics_sum:
                                metrics_sum[key] = 0
                                metrics_count[key] = 0
                            metrics_sum[key] += float(value)
                            metrics_count[key] += 1
        
        return {key: metrics_sum[key] / max(metrics_count[key], 1) for key in metrics_sum}
    
    def _calculate_std_metrics(self, results: List[Dict], metric_name: str) -> float:
        """특정 메트릭의 표준편차 계산"""
        values = []
        
        for result in results:
            value = 0.0
            
            # 다양한 경로에서 메트릭 검색
            if 'summary' in result and 'research_metrics' in result['summary']:
                if metric_name in result['summary']['research_metrics']:
                    value = result['summary']['research_metrics'][metric_name]
            elif metric_name in result.get('training_summary', {}).get('research_metrics', {}):
                value = result['training_summary']['research_metrics'][metric_name]
            elif metric_name in result.get('training_summary', {}).get('performance_analysis', {}):
                perf_metric = result['training_summary']['performance_analysis'][metric_name]
                if isinstance(perf_metric, dict):
                    value = perf_metric.get('final', 0)
                else:
                    value = perf_metric if perf_metric is not None else 0.0
            elif metric_name in result:
                value = result[metric_name]

            if isinstance(value, (int, float, np.generic)):
                values.append(float(value))
        
        return np.std(values) if values else 0.0

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
