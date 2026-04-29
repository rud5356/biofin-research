# BIOFIN_TEXT_CLS

국가생물다양성 열린재정 데이터의 workfile 문서(HWP/PDF)를 읽어 예산 분야(16개 카테고리)를 자동 분류하는 BERT 기반 텍스트 분류 파이프라인.

## 개요

workfile.xlsx에는 예산 문서 파일명과 분야명(label)이 함께 있다. 이 파이프라인은 문서 파일에서 텍스트를 추출하고, `klue/bert-base` 모델을 fine-tuning해 분야를 자동으로 예측한다.

**분류 대상 카테고리 (16개)**
공공질서및안전, 과학기술, 교육, 교통및물류, 국방, 국토및지역개발, 농림수산, 문화및관광, 보건, 사회복지, 산업·중소기업및에너지, 예비비, 일반·지방행정, 통신, 통일외교, 환경

## 파이프라인

```
workfile.xlsx
    │
    ▼ 1. build_workfile_dataset.py
workfile_dataset.csv (파일경로 + 라벨 매핑)
    │
    ▼ 2. build_text_dataset.py
workfile_text_dataset.csv (추출된 텍스트 포함)
    │
    ▼ 3. split_workfile_text_dataset.py
data/splits/ (train / val / test)
    │
    ▼ 4. train_kobert_classifier.py
models/kobert_workfile_classifier/ (학습된 모델)
```

## 데이터 현황

- 원본 workfile 행: 9,108개
- 텍스트 추출 성공: 8,294개
- 학습/검증/평가 비율: 80 / 10 / 10
- train: 6,634개 / val: 833개 / test: 826개

## 프로젝트 구조

```
BIOFIN_TEXT_CLS/
├── src/
│   ├── build_workfile_dataset.py      # workfile.xlsx → workfile_dataset.csv
│   ├── build_text_dataset.py          # HWP/PDF 텍스트 추출
│   ├── split_workfile_text_dataset.py # train/val/test 분할
│   ├── train_kobert_classifier.py     # BERT fine-tuning
│   ├── document_text.py               # HWP/PDF 텍스트 추출 유틸
│   ├── config.py                      # 경로 및 하이퍼파라미터 설정
│   └── utils.py                       # 공통 유틸
├── data/                              # gitignore됨 (로컬 전용)
│   ├── workfile_text_dataset.csv
│   └── splits/
│       ├── workfile_train.csv
│       ├── workfile_val.csv
│       └── workfile_test.csv
├── models/                            # gitignore됨 (학습 결과)
├── environment.yml
└── README.md
```

## 실행 방법

### 환경 설정

```bash
conda env create -f environment.yml
conda activate biofin_text_cls
cd src
```

### 1단계: workfile 데이터셋 빌드

workfile.xlsx와 예산 문서 폴더가 필요하다.

```bash
python build_workfile_dataset.py \
  --workfile "C:\Yuna\국가생물다양성_열린재정 데이터_v2\workfile.xlsx" \
  --budget-root "C:\Yuna\국가생물다양성_열린재정 데이터_v2"
```

### 2단계: 텍스트 추출

HWP/PDF 문서에서 텍스트를 추출한다. 문서 수에 따라 수 시간 소요될 수 있다.

```bash
python build_text_dataset.py
```

### 3단계: train/val/test 분할

```bash
python split_workfile_text_dataset.py
```

### 4단계: BERT 학습

CPU에서는 epoch당 30분~1시간 이상 소요된다. GPU 사용 시 자동으로 CUDA를 활용한다.

```bash
python train_kobert_classifier.py \
  --epochs 3 \
  --batch-size 8 \
  --max-length 512
```

**학습 속도 개선 옵션:**

```bash
# 텍스트 앞부분만 사용 (빠른 실험용)
python train_kobert_classifier.py --max-length 128

# 배치 크기 증가 (메모리 허용 시)
python train_kobert_classifier.py --batch-size 16
```

학습 완료 후 `models/kobert_workfile_classifier/` 아래에 저장된다.

| 파일 | 설명 |
|---|---|
| `best_model/` | 검증 macro F1 기준 최고 체크포인트 |
| `training_history.csv` | epoch별 loss/accuracy |
| `test_metrics.json` | 최종 테스트 성능 |
| `test_predictions.csv` | 테스트셋 예측 결과 |
| `label_mapping.json` | 라벨 ID 매핑 |

## 주요 설정 (config.py)

| 상수 | 기본값 | 설명 |
|---|---|---|
| `DEFAULT_KOBERT_MODEL` | `klue/bert-base` | 베이스 모델 |
| `DEFAULT_MAX_LENGTH` | `512` | 최대 토큰 길이 |
| `DEFAULT_BATCH_SIZE` | `8` | 배치 크기 |
| `DEFAULT_NUM_EPOCHS` | `3` | 학습 에폭 수 |
| `DEFAULT_LEARNING_RATE` | `2e-5` | AdamW 학습률 |
| `DEFAULT_TRAIN_RATIO` | `0.8` | 학습 데이터 비율 |
