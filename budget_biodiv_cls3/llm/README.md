# LLM 분류기

Ollama 로컬 LLM으로 예산 사업을 BIOFIN 1차 카테고리 0~9로 분류합니다.
`clip_20260724142552447 (1).bmp`의 BIOFIN/GLOBE 분류표를 해석한 기준이
`classify_biofin_category_with_ollama.py`의 `SYSTEM_PROMPT`에 반영돼
있습니다.

프로젝트 루트에서 실행합니다.

```powershell
python llm/classify_biofin_category_with_ollama.py --dry-run
python llm/classify_biofin_category_with_ollama.py --limit-keys 10
```

데이터는 프로젝트 루트의 `document/`, 결과와 캐시는 `outputs/llm/`을
기본 경로로 사용합니다.

분류가 완료될 때마다 `category_label_cache.csv`를 원자적으로 저장합니다.
중단 후 같은 `--output-dir`, `--cache-csv`, `--model` 설정으로 다시
실행하면 캐시에 있는 완료 사업은 건너뛰고 나머지부터 이어서 처리합니다.

입력에 `BIOFIN 1차 카테고리` 정답 컬럼이 있으면 분류 완료 후 자동으로
정확도를 평가합니다.

- `evaluation_metrics.json`: 정확도, macro precision/recall/F1 및 클래스별 지표
- `confusion_matrix.csv`: 정답 0~9 × 예측 0~9 혼동행렬
- `incorrect_predictions.csv`: 정답과 예측이 다른 사업 목록

정답 컬럼명이 다르면 `--gold-label-col "컬럼명"`으로 지정합니다.
