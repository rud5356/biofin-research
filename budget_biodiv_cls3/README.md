# budget_biodiv_cls3

기존 예산 CSV의 텍스트와 매칭된 사업설명자료 전체 본문을 함께 사용하는
BIOFIN 1차 카테고리(0~9) Transformer 분류 파이프라인입니다.

## 입력

- 라벨 CSV: `document/2023biofin_label.csv`
- 사업설명자료: `document/2023/사업설명자료/`
- 정답 컬럼: `BIOFIN 1차 카테고리`

예산 메타데이터와 사업설명자료 본문을 결합한 뒤, 긴 문서는 512 token
chunk로 나누고 Attention Pooling으로 문서 단위 분류를 수행합니다.

## 데이터 점검

먼저 라벨 CSV와 수집 문서를 매칭합니다.

```powershell
python match_2023_biofin_documents.py
```

결과는 `document/2023biofin_label_matched.csv`, 미매칭 내역은
`document/2023biofin_label_match_failed.csv`에 저장됩니다.

```powershell
python transformer/src/train_attention_classifier.py `
  --label_file document/2023biofin_label_matched.csv `
  --dry_run --document_only
```

`--document_only`를 빼면 문서가 없거나 파싱에 실패한 행은 예산 CSV
메타데이터만으로 학습 데이터에 포함합니다.

## 모델 학습

기본 분할은 사업 그룹 기준 train/validation/test = 8:1:1입니다.
동일한 소관명·세부사업명 조합은 서로 다른 split에 들어가지 않습니다.

```powershell
python transformer/src/train_attention_classifier.py `
  --label_file document/2023biofin_label_matched.csv `
  --document_only --class_weight
```

주요 결과:

- `outputs/model_results/best_model.pt`
- `outputs/model_results/tokenizer/`
- `outputs/model_results/split_assignments.csv`
- `outputs/model_results/split_summary.json`
- `outputs/model_results/test_predictions.csv`
- `outputs/model_results/metrics.json`

## 새 CSV 분류

문서와 CSV의 key 매칭 컬럼을 준비한 뒤 실행합니다.

```powershell
python transformer/src/predict_attention_classifier.py `
  --model_dir outputs/model_results `
  --doc_dir document/2023/사업설명자료 `
  --budget_file document/2023biofin_label.csv `
  --output_dir outputs/predictions `
  --no-heatmap
```

최종 `outputs/predictions/classified.csv`에는 원본 CSV 컬럼을 그대로
유지하면서 `예측 BIOFIN 1차 카테고리` 컬럼 하나가 추가됩니다.

## Ollama LLM으로 0~9 분류

`llm/classify_biofin_category_with_ollama.py`는 `cls2`의 캐시 기반 LLM
분류 흐름을 참고한 BIOFIN 1차 카테고리(0~9) 분류 스크립트입니다.
`llm/clip_20260724142552447 (1).bmp`의 BIOFIN/GLOBE 분류 기준이
`SYSTEM_PROMPT`에 반영돼 있습니다.

```powershell
# 입력 구조만 확인
python llm/classify_biofin_category_with_ollama.py --dry-run

# 10개 고유 사업만 시험 분류
python llm/classify_biofin_category_with_ollama.py --limit-keys 10

# 전체 분류
python llm/classify_biofin_category_with_ollama.py
```

기본 입력은 `document/2023biofin_label_matched.csv`, 기본 출력은
`outputs/llm/2023biofin_label_matched_llm_classified.csv`입니다.
출력에는 `LLM BIOFIN 1차 카테고리`, `confidence`, `reason`,
`evidence` 컬럼이 추가됩니다. 중단 후 재실행하면
`outputs/llm/category_label_cache.csv`를 재사용합니다.
원본 `BIOFIN 1차 카테고리` 정답이 있으면 `evaluation_metrics.json`,
`confusion_matrix.csv`, `incorrect_predictions.csv`도 자동 생성합니다.
