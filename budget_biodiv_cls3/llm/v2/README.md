# LLM v2: BIOFIN 하위 카테고리

사업 메타데이터와 사업설명자료만 사용해 BIOFIN 비해당 `0` 또는 데이터에
존재하는 39개 하위 코드 중 하나를 직접 분류합니다. 정답인 `BIOFIN 1차
카테고리`와 `하위 카테고리`는 프롬프트에 넣지 않으며, 예측 완료 후 평가에만
사용합니다.

- 비해당: `0`
- 해당: 데이터에 존재하며 Transformer v2에서도 사용하는 39개 하위 코드
- 기본 정답 컬럼: `하위 카테고리`
- 기본 예측 컬럼: `LLM BIOFIN 하위 카테고리`
- 기본 결과 위치: `outputs/llm/v2/`

```bash
python llm/v2/classify_biofin_subcategory_with_ollama.py --dry-run
python llm/v2/classify_biofin_subcategory_with_ollama.py --limit-keys 10
```

완료된 사업은 `subcategory_label_cache.csv`에 매 건 저장되므로 같은
명령으로 재실행하면 이어서 처리합니다.

사업설명자료 본문을 최대 16,000자까지 사업 메타데이터와 함께 사용합니다.
`document_status`, `document_path`, `document_chars` 컬럼으로 실제 본문
사용 여부를 확인할 수 있습니다.
