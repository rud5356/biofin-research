# BIOFIN 실행 도우미

Linux 호스트의 Python 3.8 이상과 Docker가 필요합니다. Python 패키지 추가 설치는 없습니다.

`biofin_menu.py`와 `config.json`을 함께 Linux의 `~/biofin/biofin_cls3/`에 복사하고 실행합니다.

```bash
cd ~/biofin/biofin_cls3
python3 biofin_menu.py
```

번호로 학습/예측과 Transformer/Ollama/vLLM API를 고른 뒤 실제 폴더 목록에서 파일과 폴더를 선택합니다. CSV 선택 화면에는 폴더와 CSV 파일만 표시됩니다. 폴더 번호를 입력하면 그 폴더로 이동합니다.

Transformer 학습에서는 원본만 사용할지, 원본+증강 통합 CSV를 사용할지 선택합니다. 통합 CSV는 `build_merged_training_csv.py`로 만든 `row_type`, `training_eligible` 컬럼이 있는 UTF-8 파일입니다. 증강 전체 사용 또는 클래스별 최대 개수를 선택할 수 있습니다. 원본만 선택하면 `--augmentation_per_class 0`을 전달합니다. 4번 데이터 증강 메뉴에서는 원본+증강 합계가 가장 적은 카테고리부터 50개씩 생성해 모든 카테고리가 400개가 되면 종료하며, 실행 전에 서버 저장 경로를 표시합니다.

vLLM은 KT Cloud 등에서 이미 기동한 모델에 API로 분류를 요청합니다. API URL, 모델 ID(`/v1/models`의 `id`), 사업목적 문자 수, 응답 토큰 수, 동시 요청 수, 타임아웃을 입력합니다. GPU 점유·최대 컨텍스트·서버 동시 시퀀스 설정은 변경하지 않습니다. 인증이 필요하면 호스트의 `VLLM_API_KEY` 환경변수가 컨테이너로 전달됩니다. vLLM을 고르면 분류 프롬프트 버전을 묻습니다. V1(기본값)은 `llm/v1/classify_biofin_category_with_vllm.py`, V2(분류지침 개정 2026.10.22 반영)는 `llm/v1/classify_biofin_category_with_vllm_V2.py`를 실행하며, 고른 파일이 프로젝트에 있어야 합니다. V2의 기본 결과 폴더 이름에는 `_v2`가 붙습니다. 버전 목록은 `config.json`의 `VLLM_SCRIPT_VERSIONS`에서 바꿀 수 있습니다.

- `0`: 상위 폴더로 이동
- `s`: 현재 폴더 선택 (문서·모델·결과 폴더)
- `n`: 현재 위치에 새 결과 폴더 지정 (결과 폴더 선택 화면). Enter를 누르면 자동 날짜 이름을 사용합니다.
- `q` 또는 Ctrl+C: 취소

프로젝트 밖으로는 이동할 수 없고 숨김 항목은 표시하지 않습니다. 새 결과 폴더는 선택 시 생성하지 않고 기존 실행 스크립트가 생성합니다. 나머지 옵션은 Enter로 기본값을 사용합니다. 마지막 확인에서 `y`를 입력해야 실행됩니다.

다른 위치에서 실행하거나 명령어만 확인하려면:

```bash
python3 biofin_menu.py --project-dir ~/biofin/biofin_cls3 --dry-run
```

입력 경로는 호스트 프로젝트 안의 경로입니다. Transformer는 호스트와 컨테이너 내부(`/work/biofin_cls3`)가 같은 프로젝트 파일을 가리켜야 합니다. 기존 `biofin` 컨테이너가 실행 중이어야 합니다. LLM은 프로젝트를 `/workspace`에 마운트하고 호스트 UID/GID로 실행합니다. 기존 `.packages` 의존성 및 선택한 Ollama 또는 vLLM 서버/모델이 준비되어 있어야 합니다.

제공된 LLM 학습과 예측은 같은 분류 스크립트를 실행합니다. 별도의 파인튜닝 명령어는 포함하지 않았습니다. 학습은 `--retries 3`, 예측은 원래 명령어처럼 해당 옵션 생략이 기본값입니다.

결과 폴더 생성과 실제 출력 파일 처리는 기존 모델 스크립트가 수행합니다. 호스트에서 컨테이너용 결과 폴더를 미리 생성하지 않습니다.

## 자료 수집 (크롤러)

작업 선택에서 `3. 자료 수집 (크롤러)`를 고르면 수집 대상, 회계연도, 범위, 저장 폴더 이름을 차례로 묻고 크롤러를 실행합니다. Transformer 학습·예측과 같은 `biofin` 컨테이너 안에서 실행하므로 호스트에 Python 패키지를 설치할 필요가 없습니다.

준비:

- 저장소의 `crawlers/` 폴더를 프로젝트 폴더에 복사합니다 (`~/biofin/biofin_cls3/crawlers/`).
- `.env.example`을 `~/biofin/biofin_cls3/.env`로 복사하고 Open API 인증키를 넣습니다. 인증키가 없으면 5건까지만 받을 수 있습니다.
- `biofin` 컨테이너가 실행 중이어야 하고, 호스트 프로젝트와 컨테이너 내부 프로젝트(`/work/biofin_cls3`)가 같은 폴더를 가리켜야 합니다.

상세 PDF나 사업설명자료 수집에는 컨테이너에 Playwright와 Chromium(열린재정은 pandas, openpyxl, tqdm도)이 필요합니다. 실행 직전에 메뉴가 자동으로 확인하고, 없으면 설치할지 물어본 뒤 컨테이너 안에 설치합니다. 컨테이너를 새로 만들면 설치한 내용이 사라지므로 그때는 다시 물어봅니다. 지방재정365 목록과 열린재정 Open API 예산 목록은 추가 설치 없이 동작합니다.

수집 결과는 모두 프로젝트의 `document/` 아래에 저장됩니다.

| 수집 대상 | 저장 위치 |
|---|---|
| 지방재정365 | `document/<폴더 이름>/<폴더 이름>.csv`, `document/<폴더 이름>/사업설명자료/` |
| 열린재정 | `document/<연도>/open_fiscal_<연도>.csv`, `document/<연도>/사업설명자료/` (상위 폴더 이름을 주면 `document/<상위 폴더>/<연도>/`) |
| 열린재정 Open API 예산 목록 | `document/<폴더 이름>/<폴더 이름>.csv` |

지방재정365는 전국 17개 시도 또는 직접 입력한 지역만 수집할 수 있습니다. 폴더 이름의 기본값은 지역이 하나면 `<지역>_<연도>`, 여러 개면 `지방재정_<연도>`입니다. 이미 있는 폴더를 고르면 확인을 거친 뒤 받아 둔 파일은 건너뛰고 이어서 수집합니다.

## 실행 기본값 설정

`biofin_menu.py`와 같은 폴더의 `config.json`을 수정합니다. 실행한 작업 폴더와 관계없이 스크립트 옆의 설정 파일을 읽습니다. 변경 내용은 메뉴를 다시 시작하면 적용됩니다. 메뉴에서 입력한 값은 해당 실행에만 사용되며 파일에 저장되지 않습니다.

- 경로: `PROJECT_DIR`, `DOCUMENT_DIR`, `DOCKER_WORKDIR`, `TRANSFORMER_MODEL_DIR`
- 서버·모델: `VLLM_URL`, `VLLM_MODEL`, `OLLAMA_URL`, `OLLAMA_MODEL`
- 실행 기본값: `MAX_DOCUMENT_CHARS`, `MAX_TOKENS`, `WORKERS`, `NUM_CTX`, `TIMEOUT`
- 프롬프트 버전: `VLLM_SCRIPT_VERSIONS`의 각 항목은 `[메뉴 이름, 스크립트 파일명, 결과 폴더 접미사]`입니다.

`PROJECT_DIR`의 `null`은 현재 작업 폴더를 뜻합니다. JSON에는 주석을 넣을 수 없으며 문자열은 큰따옴표를 사용합니다. API 키 실제 값은 넣지 않고 기존 환경변수나 `.env`로 관리합니다. 설정 파일이 없거나 JSON 형식이 잘못되면 오류를 안내하고 종료합니다.

분류 규칙을 담은 `budget_biodiv_cls3/pipeline/config.json`과는 별개입니다.
