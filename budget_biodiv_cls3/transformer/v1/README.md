# Transformer v1: BIOFIN 상위 카테고리

예산정보와 사업설명자료를 사용해 상위 카테고리 0~9를 학습·예측합니다.
명령은 `budget_biodiv_cls3`에서 실행합니다. 설치는 [공통 안내](../README.md)를 참고합니다.

## 정답 컬럼 설정

학습 CLI의 기본 `--label_column`은 `BIOFIN 1차 카테고리`입니다.
최근 취합 CSV에는 `1차`라는 이름으로 들어 있으므로 옵션을 명시해야 합니다.
이름 불일치로 발생한 필수 컬럼 오류이며, 삭제한 `BIOFIN분류` 때문이 아닙니다.

- 기본값 지정: `src/train_attention_classifier.py`의 `build_argument_parser()`
- 필수 컬럼 검사 및 정답 읽기: `src/build_dataset.py`의 `load_label_data()`
- 예측 시 평가 정답 지정: `src/predict_attention_classifier.py`의 `--label_column`

필수 컬럼은 `회계연도`, `소관명`, `세부사업명`, 지정한 정답 컬럼입니다.
정답 컬럼 안의 빈칸은 0으로 처리하지만, 컬럼 자체가 없으면 오류입니다.

## 학습 전 점검

```powershell
python transformer/v1/src/train_attention_classifier.py `
  --label_file "document/open/BIOFIN_2023_취합_2026.09.14.csv" `
  --label_column "1차" `
  --doc_dir "document/2023/사업설명자료" `
  --output_dir "transformer/v1/outputs/20260914_check" `
  --dry_run
```

모델을 학습하지 않고 매칭과 레이블 분포를 점검합니다. 문서가 없는 행도
기본적으로 예산정보 입력으로 대체합니다. 문서가 있는 행만 사용하려면
점검과 학습 양쪽에 `--document_only`를 추가합니다.

## 학습

```powershell
python transformer/v1/src/train_attention_classifier.py `
  --label_file "document/open/BIOFIN_2023_취합_2026.09.14.csv" `
  --label_column "1차" `
  --doc_dir "document/2023/사업설명자료" `
  --undersample_majority --class_weight `
  --majority_cap_multiplier 10 --majority_cap_min 1000 `
  --output_dir "transformer/v1/outputs/20260914_model"
```

사업 그룹 기준으로 8:1:1 분할합니다. 위 옵션의 언더샘플링은 학습 세트의
기본 majority label 0에만 적용하며, 상한은
`max(0 이외 클래스 건수의 중앙값 × 10, 1000)`입니다.
검증·테스트 분포는 유지합니다.

주요 결과는 `best_model.pt`, `tokenizer/`, `split_assignments.csv`,
`split_summary.json`, `test_predictions.csv`, `metrics.json`입니다.
`--output_dir`를 생략하면 `transformer/v1/outputs/model_results`에 저장합니다.

## 예측

```powershell
python transformer/v1/src/predict_attention_classifier.py `
  --model_dir "transformer/v1/outputs/20260914_model" `
  --doc_dir "document/2023/사업설명자료" `
  --budget_file "document/open/BIOFIN_2023_취합_2026.09.14.csv" `
  --label_column "1차" `
  --output_dir "transformer/v1/outputs/20260914_predictions" `
  --no-heatmap
```

`classified.csv`는 원본 열에 `예측 BIOFIN 1차 카테고리`를 추가합니다.
예측되지 않은 행의 값은 빈칸으로 남을 수 있습니다. 학습과 달리 예측 입력의
정답 컬럼은 평가용이며 선택적입니다.

기존 `document/2023biofin_label_matched.csv`를 사용한다면 실제 헤더를 확인해
`--label_column "BIOFIN 1차 카테고리"`로 지정합니다.
