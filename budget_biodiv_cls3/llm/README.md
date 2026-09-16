# Ollama LLM 분류

- [v1](v1/README.md): BIOFIN 상위 카테고리 0~9
- [v2](v2/README.md): 비해당 0 및 허용된 39개 하위 코드, 총 40개

모든 명령은 `budget_biodiv_cls3` 폴더에서 실행합니다.

## 기본 설정

| 항목 | 기본값 |
| --- | --- |
| 입력 | `document/2023biofin_label_matched.csv` |
| Ollama 주소 | `http://localhost:11434` |
| 모델 | `gemma3:12b` |
| 결과·캐시 | `outputs/llm/v1/` 또는 `outputs/llm/v2/` |
| 동시 작업 | `--workers 1` |
| 문서 최대 길이 | `--max-document-chars 16000` |
| 요청 컨텍스트 | `--num-ctx 16384` |

```powershell
python -m pip install -r llm/requirements.txt
python llm/v1/classify_biofin_category_with_ollama.py --dry-run
python llm/v2/classify_biofin_subcategory_with_ollama.py --dry-run
```

`--dry-run`은 입력과 문서 매칭을 점검하고 Ollama 호출 없이 종료합니다.
출력 폴더는 만들어질 수 있지만 분류 결과와 캐시는 저장하지 않습니다.

## 문서 입력

CSV의 `사업설명자료_상대경로` 또는 `사업설명자료_파일명`으로 HWP/HWPX/PDF/TXT를 찾습니다.
`--doc-dir` 기본값은 `document/2023/사업설명자료`입니다.
최대 길이를 넘는 본문은 앞·뒤를 사용합니다.
`--no-document-text`는 예산 메타데이터만 사용하도록 지정합니다.
출력의 `document_status`, `document_path`, `document_chars`로 실제 사용 여부를 확인합니다.
`document_status=PARSED`는 본문 파싱 성공을 뜻합니다.

## 회사 공용 H200 Ollama 연결

접속 URL·허용 IP 등 실제 접속정보는 문서에 적지 않습니다. 운영자에게 별도로 전달받아
개인 메모나 `.env` 등 커밋되지 않는 곳에 보관합니다. 실제 GPU 배정과 사용 가능 모델도
운영자에게 확인합니다.

모델 목록 조회는 `<접속정보 전달받은 URL>/api/tags`, 관리 화면은 `<...>/manager`,
API 문서는 `<...>/manager/docs`입니다. 접속 허용 외부 IP는 접속하는 PC·서버의 외부 IP
조건이며 API 서버 주소로 넣는 값이 아닙니다.

이 대화에서 수행한 목록 조회에서는 모델 하나가 반환됐습니다. 모델 목록은 변경될 수
있으므로 실행 전 다시 확인합니다. 목록 조회는 인증키 없이 성공했지만 실제 추론과 H200
사용 여부는 검증하지 않았습니다.

PowerShell에서 모델 목록만 조회하려면:

```powershell
$ollamaUrl = "<전달받은 접속 URL>"
(Invoke-RestMethod "$ollamaUrl/api/tags").models | Select-Object name
```

추론에는 `--ollama-url`에 기본 URL만 넣고 `--model`에 설치된 이름을 지정합니다.
코드가 `/api/generate`를 붙입니다. `/manager` 주소를 넣지 않습니다.
클라이언트 옵션 변경은 서버의 모델 설치·삭제·교체가 아닙니다.
실제 추론은 입력을 서버로 보내고 공용 GPU 자원을 사용합니다.
운영자가 허용한 모델·자료·사용량 범위에서 버전별 README의 소량 예시로 시작합니다.
관리 화면의 모델 설치·삭제와 서버 설정 변경은 운영자에게 맡깁니다.

## 모델 변경과 캐시

완료된 사업은 캐시로 재사용합니다. 현재 재사용 여부는 사업 키와 유효한 정답을 기준으로 하며,
모델명이 달라졌다고 기존 캐시가 자동 무효화되지는 않습니다.
모델·프롬프트·입력자료 비교 시 `--output-dir`를 별도로 지정하고,
`--cache-csv`를 직접 지정했다면 그 경로도 분리합니다.

`--limit-keys 1`은 처리할 고유 사업 수 제한으로, CSV 한 행 제한이나 HTTP 요청 1회 보장은 아닙니다.
재시도를 막는 첫 테스트는 `--workers 1 --limit-keys 1 --retries 0`으로 실행합니다.
캐시를 쓰는 같은 실행을 재개할 때는 기존 출력 폴더를 유지합니다.

## Docker

아래는 Linux 셸 예시입니다. 현재 디렉터리는 `budget_biodiv_cls3`입니다.

```bash
docker build -f llm/Dockerfile -t biofin-llm .
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  biofin-llm llm/v1/classify_biofin_category_with_ollama.py --dry-run
```

원격 추론에는 같은 명령에서 `--dry-run`을 제거하고 `--ollama-url`, `--model` 및 입력 옵션을
지정합니다. 이 컨테이너는 클라이언트이므로 원격 GPU 사용에 로컬 `--gpus` 옵션은 필요하지 않습니다.
컨테이너가 실행되는 서버의 외부 IP도 접속 허용 조건을 충족해야 합니다.
