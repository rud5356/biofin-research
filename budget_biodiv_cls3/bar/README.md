# 귀속 예산(BAR) 계산

분류가 끝난 CSV에 BIOFIN 1차 카테고리별 귀속률을 곱해 귀속 예산 열을 추가합니다. 원본 열과 행은 그대로 둡니다.

## 파일

| 파일 | 역할 |
| --- | --- |
| `bar_rates.json` | 카테고리별 귀속률(%)과 버전. 귀속률이 정해지면 이 파일만 수정 |
| `apply_bar.py` | 계산 스크립트 (Python 3.8+, 표준 라이브러리만 사용) |
| `test_apply_bar.py` | 테스트 |

## 추가되는 열

| 열 | 내용 |
| --- | --- |
| `BAR 귀속률(%)` | 적용한 귀속률 |
| `BAR 귀속 <금액열>` | 금액 × 귀속률 ÷ 100, 반올림(기본 정수). 금액 열마다 하나씩 |
| `BAR 적용상태` | `적용` / `귀속률 미정` / `카테고리 없음` / `알 수 없는 카테고리` |
| `BAR 버전` | `bar_rates.json`의 `version` |

귀속률이 `null`이거나 카테고리가 비어 있으면 금액을 **빈칸**으로 둡니다. 0원과 구분하기 위해서입니다.
`--unit`으로 입력 금액 단위(`원`/`천원`/`백만원`)를 지정하면 항상 원 단위까지 반올림합니다. 지방재정365는 `원`, 열린재정은 `백만원`입니다(예: 1,761백만원 × 50% → `880.5`). 지정하지 않으면 입력 단위의 정수로 반올림합니다.

카테고리 열은 `pipeline_label` → `LLM BIOFIN 1차 카테고리` → `예측 BIOFIN 1차 카테고리` 순으로 자동 탐색합니다.

## 실행

`biofin_menu.py`에서 모델 예측을 실행하면 분류가 끝난 직후 자동으로 이어서 실행됩니다. 단독 실행:

```powershell
python bar/apply_bar.py --input llm/v1/outputs/<폴더>/<이름>_llm_classified.csv
python bar/apply_bar.py --input transformer/v1/outputs/<폴더>/classified.csv --amount-cols 예산액,지출액
python -m unittest discover -s bar -p "test_*.py"
```

결과는 입력 옆에 `<입력이름>_bar.csv`와 카테고리별 합계 `<입력이름>_bar_summary.csv`로 저장되며, 인코딩은 입력과 같습니다. 이미 있으면 `--overwrite`가 필요합니다.
