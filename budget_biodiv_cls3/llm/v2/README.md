# LLM v2: BIOFIN 하위 카테고리

BIOFIN/GLOBE 분류표에 제시된 하위 코드 중 하나로 분류합니다.

- 비해당: `0`
- 해당: `1.01`~`9.09` 중 분류표에 정의된 코드
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
