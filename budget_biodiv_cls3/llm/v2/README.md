# LLM v2: BIOFIN 하위 카테고리

사업정보와 사업설명자료만으로 비해당 `0` 또는 39개 하위 코드 중 하나를 직접 분류합니다.
허용 코드는 스크립트의 `VALID_LABELS`에 정의되어 있으며 0 포함 총 40개입니다.
명령은 `budget_biodiv_cls3`에서 실행합니다.
접속정보와 캐시 관리 방법은 [LLM 공통 안내](../README.md)를 참고합니다.

## 입력과 평가

기본 입력은 `document/2023biofin_label_matched.csv`입니다.
사업 키는 `business_key` 또는 `소관명`, `분야명`, `부문명`, `프로그램명`,
`단위사업명`, `세부사업명`으로 만듭니다.
정답·정답 근거는 프롬프트에 넣지 않고 예측 후 평가에만 사용합니다.

- 기본 정답 컬럼: `하위 카테고리` (`--gold-label-col`)
- 기본 예측 컬럼: `LLM BIOFIN 하위 카테고리` (`--label-col`)
- 정답 형식: 비해당 `0`, 해당 `6.05` 같은 결합 코드
- 기본 결과 폴더: `outputs/llm/v2/`

최근 취합 CSV의 `하위`는 단독 번호이므로 평가용 결합 코드와 다릅니다.
`--gold-label-col "하위"`로만 바꾸어 평가하지 않습니다.
상위와 하위를 결합한 별도 정답 컬럼을 준비해야 합니다.
정답 없이 추론은 가능하지만, 유효한 정답·예측 쌍이 없는 평가는 생략됩니다.
계층 코드 준비는 [Transformer v2 입력 규칙](../../transformer/v2/README.md)을 참고합니다.

## 호출 없는 점검

```powershell
python llm/v2/classify_biofin_subcategory_with_ollama.py `
  --input-file "document/2023biofin_label_matched.csv" `
  --gold-label-col "하위 카테고리" `
  --output-dir "outputs/llm/v2/input_check" `
  --dry-run
```

## 공용 서버 소량 테스트

운영자가 사용을 허용했고 아래 입력자료가 전송 가능한 경우의 예시입니다.
접속정보와 사용 가능한 모델명은 [LLM 공통 안내](../README.md)로 확인합니다.

```powershell
python llm/v2/classify_biofin_subcategory_with_ollama.py `
  --input-file "document/2023biofin_label_matched.csv" `
  --gold-label-col "하위 카테고리" `
  --ollama-url "<전달받은 접속 URL>" `
  --model "<서버에서 확인한 모델명>" `
  --workers 1 --limit-keys 1 --retries 0 `
  --output-dir "outputs/llm/v2/remote_test"
```

해당 모델의 실제 분류 정확도는 검증하지 않았습니다. 상위 숫자만 출력하거나 허용 코드 밖으로
응답하는지 확인합니다. `--limit-keys`는 전체 출력 행 수가 아닌 처리할 고유 사업 수입니다.

## 결과와 재개

`<입력파일명>_llm_classified.csv`, `subcategory_label_cache.csv`, `run_summary.json`을 저장합니다.
평가 가능한 정답이 있으면 `evaluation_metrics.json`, `confusion_matrix.csv`,
`incorrect_predictions.csv`도 생성합니다.

완료 사업은 기본 매 건 캐시에 저장하며 같은 출력 폴더로 재개합니다.
모델이나 프롬프트를 바꾸면 기존 캐시를 공유하지 않도록 출력 폴더를 분리합니다.
`document_status`, `document_path`, `document_chars`로 본문 사용 여부를 확인합니다.
본문은 기본 최대 16,000자이고 `--max-document-chars`로 변경합니다.
