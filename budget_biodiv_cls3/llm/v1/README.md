# LLM v1: BIOFIN 상위 카테고리

BIOFIN 1차 카테고리 0~9 중 하나로 분류합니다.

```bash
python llm/v1/classify_biofin_category_with_ollama.py --limit-keys 10
```

기본 결과 위치는 `outputs/llm/v1/`입니다.
사업설명자료 본문을 최대 16,000자까지 사업 메타데이터와 함께 사용합니다.
