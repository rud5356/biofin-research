# LLM 분류기

- `v1`: BIOFIN 1차 상위 카테고리 0~9 분류
- `v2`: BIOFIN 하위 카테고리 코드 분류

프로젝트 루트에서 실행합니다.

```bash
python llm/v1/classify_biofin_category_with_ollama.py --dry-run
python llm/v2/classify_biofin_subcategory_with_ollama.py --dry-run
```

결과와 캐시는 각각 `outputs/llm/v1/`, `outputs/llm/v2/`에 저장됩니다.

두 버전 모두 CSV의 `사업설명자료_상대경로` 또는
`사업설명자료_파일명`으로 HWP/HWPX/PDF/TXT 본문을 찾아 사업명과 함께
LLM 프롬프트에 넣습니다. 최종 CSV의 `document_status`가 `PARSED`이면
본문을 정상적으로 사용한 것입니다.

## Docker

문서 파싱 라이브러리가 포함된 이미지를 프로젝트 루트에서 한 번 빌드합니다.

```bash
docker build -f llm/Dockerfile -t biofin-llm .
```

v2 실행 예:

```bash
docker run --rm -it \
  --network host \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -w /workspace \
  biofin-llm \
  llm/v2/classify_biofin_subcategory_with_ollama.py \
  --ollama-url http://172.22.0.1:20001 \
  --model gemma3:12b
```

기본적으로 본문은 앞·뒤를 합쳐 최대 16,000자까지 사용하고 Ollama
컨텍스트는 16,384 token으로 요청합니다. 각각 `--max-document-chars`,
`--num-ctx`로 변경할 수 있습니다.
