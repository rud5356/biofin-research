# LLM v1: BIOFIN 상위 카테고리

사업 메타데이터와 사업설명자료를 Ollama에 보내 상위 카테고리 0~9를 분류합니다.
명령은 `budget_biodiv_cls3`에서 실행합니다.
접속정보와 공용 서버 사용 방법은 [LLM 공통 안내](../README.md)에 있습니다.

## 입력과 정답

기본 입력은 `document/2023biofin_label_matched.csv`입니다.
사업 키는 `business_key` 또는 `소관명`, `분야명`, `부문명`, `프로그램명`,
`단위사업명`, `세부사업명`으로 만듭니다.
정답은 추론 입력이 아닌 평가용이며, 기본 컬럼명은 `BIOFIN 1차 카테고리`입니다.
최근 취합 CSV는 `--gold-label-col "1차"`를 지정합니다.
평가 가능한 정답·예측 쌍이 없으면 정확도 평가는 생략됩니다.

## 호출 없는 점검

```powershell
python llm/v1/classify_biofin_category_with_ollama.py `
  --input-file "document/open/BIOFIN_2023_취합_2026.09.14.csv" `
  --gold-label-col "1차" `
  --output-dir "outputs/llm/v1/input_check" `
  --dry-run
```

사업설명자료 본문은 기본 최대 16,000자를 사용합니다.
CSV에 문서 연결 정보가 없다면 문서 매칭 결과를 먼저 준비하거나,
예산정보만 사용할 목적일 때 `--no-document-text`를 명시합니다.

## 공용 서버 소량 테스트

운영자가 사용을 허용한 경우 실행하는 예시입니다. 입력 파일에는 전송 가능한 자료만 넣습니다.

```powershell
python llm/v1/classify_biofin_category_with_ollama.py `
  --input-file "document/open/BIOFIN_2023_취합_2026.09.14.csv" `
  --gold-label-col "1차" `
  --ollama-url "<전달받은 접속 URL>" `
  --model "<서버에서 확인한 모델명>" `
  --workers 1 --limit-keys 1 --retries 0 `
  --output-dir "outputs/llm/v1/remote_test"
```

원격 테스트 전에 [LLM 공통 안내](../README.md)로 접속정보와 사용 가능한 모델명을 확인합니다.
모델의 BIOFIN 분류 정확도는 검증하지 않았습니다.
기본 모델은 여전히 `gemma3:12b`이므로 원격 테스트에서는 모델명을 생략하지 않습니다.
`--limit-keys 1`은 고유 사업 하나를 처리하며 결과 CSV에는 전체 입력 행이 남습니다.
처리되지 않은 행은 기존 캐시가 없다면 예측이 비어 있습니다.

## 결과와 재개

기본 결과 폴더는 `outputs/llm/v1/`이며 파일은 다음과 같습니다.

- `<입력파일명>_llm_classified.csv`: 원본 열과 예측·근거·문서 사용 정보
- `category_label_cache.csv`: 완료 사업 캐시
- `run_summary.json`: 실행 요약
- 평가 가능한 정답이 있으면 `evaluation_metrics.json`, `confusion_matrix.csv`, `incorrect_predictions.csv`

예측 컬럼 기본값은 `LLM BIOFIN 1차 카테고리`이며 `--label-col`로 변경합니다.
`confidence`, `reason`, `evidence`도 출력합니다.
같은 출력 폴더로 재실행하면 유효 캐시를 재사용합니다.
다른 모델을 비교할 때는 출력·캐시 폴더를 분리합니다.
