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

1. `config/experiment_config.yaml`에서 실험 매개변수를 구성합니다.

2. 단일 실험 실행:
```bash
python -m src.experiments.run_experiment --config config/experiment_config.yaml
```

3. 여러 실험을 순차적으로 실행:
```bash
python -m src.experiments.run_experiment --num_runs 10
```