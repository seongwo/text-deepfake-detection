# 한국어 AI 생성 텍스트 탐지

2025 SW중심대학 디지털 경진대회 AI 부문 — **279팀 중 7위, 후원기업상**.

[팀 저장소](https://github.com/jiwoong218/text-deepfake-detection)를 fork한 저장소입니다. 한국어 텍스트가 사람이 작성한 것인지 AI가 생성한 것인지 구분하는 문제를 다뤘습니다. 클래스 불균형, 긴 문서, 불확실한 라벨에 대응하는 방법을 조합했습니다. 전체 파이프라인과 대회 성과는 팀의 결과입니다.

[English](README.md)

## 방법

| 문제 | 현재 코드에서 확인되는 구성 |
| --- | --- |
| 텍스트 분류 | KoELECTRA |
| 샘플별 난이도 차이 | Hard-first loss weighting |
| 클래스 불균형 | 클래스별 subsampling |
| 긴 입력 | 겹치는 window별 추론과 최대 확률 집계 |
| 여러 모델의 예측 결합 | 예측 확률 평균 |

각 구성의 코드 위치와 실험 기록의 범위는 [구현·재현성 문서](docs/REPRODUCIBILITY.md)에 정리했습니다. 개인별 세부 구현 기여는 아직 문서화하지 않았습니다.

## 코드 구성

- `src/data_preparation.py`: 문단 전처리, 데이터 분할, tokenization
- `src/train.py`: 학습 루프와 클래스별 sampler
- `src/modeling.py`, `src/superloss.py`: KoELECTRA loss wrapper와 loss weighting
- `src/electra.py`, `src/train_9000.py`, `src/train_super_ada.py`: 개별 실험 학습 스크립트
- `src/inference.py`: 기본 추론 및 overlapping-window 추론
- `src/ensemble.py`: 제출 CSV의 확률 평균
- `notebooks/3rd super_Ada.ipynb`: 원래 실험 노트북

대회 데이터, 전처리 데이터, 모델 가중치는 포함하지 않았습니다. `models/` 아래 폴더에는 빈 자리 표시 파일만 있습니다.

## 실행 상태

별도 Python 환경에서 의존성을 설치합니다.

```bash
python -m pip install -r requirements.txt
```

일반 학습 경로에는 수정이 필요한 부분이 있습니다. 전처리는 `save_to_disk`로 저장하지만 학습은 `load_dataset`으로 읽고, validation split 이름도 전처리의 `eval`과 학습의 `test`가 다릅니다. 원래 실험 환경도 버전이 고정되어 있지 않습니다.

추론은 현재 **로컬 checkpoint 디렉터리**를 요구합니다. Hugging Face 모델 ID를 그대로 전달하면 경로 검사에서 거부됩니다. 가변 길이 입력의 batch 구성에도 알려진 문제가 있어, 전체 추론 실행을 검증한 상태는 아닙니다.

기존 예측 CSV는 다음과 같이 결합할 수 있습니다.

```bash
python src/ensemble.py \
  --csvs sub_ada.csv sub_extra.csv sub_42_4000.csv sub_9000.csv \
  --output final_ensemble_submission.csv
```

각 CSV는 행과 순서가 같고 `generated` 확률 열을 가져야 합니다.

## 평가 시 주의할 점

현재 전처리는 문서를 문단으로 나눈 뒤 분할하므로, 같은 문서의 문단이 학습과 validation 양쪽에 들어갈 수 있습니다. 따라서 이 validation을 새로운 문서에 대한 일반화 성능으로 해석하면 안 됩니다.

대회 결과를 재현하려면 원래 데이터 분할, 학습된 모델, 최종 ensemble 구성, 평가 설정이 필요합니다. 공개 코드에는 이 기록이 모두 포함되어 있지 않습니다.
