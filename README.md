## 주요 기능

### 4D 페로몬 벡터
- **행동**: 행동 확률 분포
- **감정**: 5차원 감정 상태
- **사회적**: 에이전트 간 관계 가중치
- **맥락**: 환경 상태 인코딩

### 시간 확산 모델
- 확산을 위한 학습 가능한 공간 커널
- 적응적 시간 감쇠율
- GPU 가속 확산 프로세스

### 분산 어텐션 라우터
- 페로몬 처리를 위한 멀티헤드 어텐션
- 에이전트 사회적 관계를 위한 그래프 어텐션
- Ray를 사용한 분산 실행

## 설치

1. **저장소 복제:**
```bash
git clone [https://github.com/your-repo/digital-pheromone-mas.git](https://github.com/your-repo/digital-pheromone-mas.git)
cd digital-pheromone-mas
```

2. **실행 Windows 설치 스크립트:**
이 스크립트는 가상 환경을 생성하고 활성화하며 필요한 모든 종속성을 설치합니다.
```batch
run_windows.bat
```

3. **수동 설치:**
```bash
# 가상 환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate

# 종속성 설치
pip install -r requirements.txt

# 편집 가능 모드로 프로젝트 패키지 설치
pip install -e .
```

## 빠른 시작

이 프로젝트에서 실험 수행 방법(C++ 백엔드 가속 사용)

프로젝트 구조에 따라 실험을 실행하는 방법은 다음과 같습니다.

1. 빠른 실험(100개 시간 단계, 5개 에이전트)

빠른 테스트 및 개발을 위해:

# Python 경로 설정
export PYTHONPATH=/home/swim/projects/digital_pheromone_mas:$PYTHONPATH

# C++ 백엔드를 먼저 빌드합니다(아직 빌드되지 않은 경우).
cd cpp_backend && ./build.sh && cd ..

# 빠른 실험 실행
python -m src.experiments.run_experiment --config config/quick_experiment_config.yaml

구성: config/quick_experiment_config.yaml을 사용합니다.
- 맵 크기: 25x25
- 에이전트: 5개
- 시간 단계: 100개
- WandB: 비활성화(더 빠른 실행을 위해)
- 훈련 빈도: 20단계마다

2. 전체 실험(1000개 시간 단계, 에이전트 10개)

종합적인 연구 결과 확인:

# 전체 실험 실행
python -m src.experiments.run_experiment --config config/config.yaml

구성: config/config.yaml 사용:
- 맵 크기: 50x50
- 에이전트: 10개
- 시간 단계: 1000개
- WandB: 활성화됨(실험 추적용)
- 훈련 빈도: 10단계마다

3. 비교 실험(기준선 vs 제안된 방법)

4D 페로몬 방법을 기준선과 비교하려면:

# 비교 실험 실행(각각 10회 실행)
python -m src.experiments.run_comparison --config config/config.yaml

다음이 실행됩니다.
- 제안된 방법: 4D 디지털 페로몬 + 분산 주의
- 기준선:
- 규칙 기반 확산
- 중앙 집중식 주의
- 2D 페로몬 절제

결과 위치: results/comparison/

4. 차원 절제 연구

각 페로몬 차원의 기여도를 분석하려면 다음을 실행합니다.

python -m src.experiments.dimension_ablation_study --config config/config.yaml

C++ 백엔드 가속 활성화

C++ 백엔드는 사용 가능한 경우 자동으로 사용됩니다. 작동 확인 방법:

C++ 백엔드 빌드:

cd /home/swim/projects/digital_pheromone_mas/cpp_backend
./build.sh
cd ..

C++ 백엔드 확인:

# C++ 모듈 사용 가능 여부 확인
python -c "from src.core import cpp_accelerators; print('✓ C++ 백엔드 준비 완료!')"

# 필드 작업 확인
PYTHONPATH=/home/swim/projects/digital_pheromone_mas:$PYTHONPATH python -c "
from src.core.field_operations_wrapper import FieldOperationsWrapper
wrapper = FieldOperationsWrapper()
print(f'✓ 필드 작업 using: {wrapper.backend}')
"

C++ 백엔드 성능 이점:

- 필드 감소: 26배 더 빠름
- 필드 집계: 21배 더 빠름
- 필드 확산: 12배 더 빠름
- 공간 쿼리: 2~30배 더 빠름
- 전체: 시스템 속도 4~6배 향상

주요 파일 및 디렉터리

digital_pheromone_mas/
├── src/experiments/
│ ├── run_experiment.py # 단일 실험 실행기
│ ├── run_comparison.py # 기준선 비교 실행기
│ └── dimension_ablation_study.py # 절제 연구
│
├── config/
│ ├── config.yaml # 전체 실험 구성(1000단계)
│ └── quick_experiment_config.yaml # 빠른 테스트 구성(100단계)
│
├── results/ # 실험 출력
│ ├── training_summary.txt # 요약 보고서
│ ├── comparison/ # 비교 결과
│ └── plots/ # 시각화
│
└── cpp_backend/ # C++ 가속
├── build.sh # 빌드 스크립트
└── src/field_operations.cpp # SIMD 최적화된 연산

실험 출력

실험 실행 후 다음을 확인할 수 있습니다.

1. 훈련 요약: results/training_summary.txt
- 연구 지표(정보 전달 효율, 학습 수렴 등)
- 통신 오버헤드 분석
- 성능 분석
2. 시각화: results/
- 페로몬 필드 히트맵
- 에이전트 상태 플롯
- 학습 곡선
- 통신 분석
3. 원시 데이터: results/*.pkl 및 results/*.json

일반적인 워크플로

개발 및 디버깅:

# 최소한의 리소스로 빠른 테스트
python src/experiments/run_experiment.py \
--config config/quick_experiment_config.yaml

연구 실험:

# C++ 가속을 사용한 전체 실험
python src/experiments/run_experiment.py \
--config config/config.yaml

# 비교 연구 (논문용)
python src/experiments/run_comparison.py \
--config config/config.yaml

성능 벤치마킹:

# C++ 백엔드 구성 요소 벤치마킹
PYTHONPATH=$PWD:$PYTHONPATH python tests/test_field_operations.py
PYTHONPATH=$PWD:$PYTHONPATH python tests/test_spatial_index.py

중요 참고 사항

1. C++ 백엔드 컴파일: 실험을 실행하기 전에 C++ 백엔드를 빌드해야 합니다. 이는
상당한 속도 향상(전체 4~6배)을 제공합니다.
2. 메모리 관리: 구성에는 메모리 관리 설정이 포함되어 있습니다.
- 최대 메모리: 85%
- 경고 임계값: 75%
- 필요 시 자동 정리
3. GPU 가속: GPU를 사용할 수 있는 경우 구성에서 device: "cuda:0"을 설정합니다. CUDA를 사용할 수 없는 경우 프로젝트는
자동으로 CPU로 대체됩니다.
4. WandB 통합: 실험 추적을 위해 설정 파일에서 use_wandb: true를 설정하세요.