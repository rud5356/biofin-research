# BIOFIN 실행 도우미

Linux 호스트의 Python 3.8 이상과 Docker가 필요합니다. Python 패키지 추가 설치는 없습니다.

`biofin_menu.py`를 Linux의 `~/biofin/biofin_cls3/`에 복사하고 실행합니다.

```bash
cd ~/biofin/biofin_cls3
python3 biofin_menu.py
```

번호로 학습/예측과 Transformer/Ollama/vLLM API를 고른 뒤 실제 폴더 목록에서 파일과 폴더를 선택합니다. CSV 선택 화면에는 폴더와 CSV 파일만 표시됩니다. 폴더 번호를 입력하면 그 폴더로 이동합니다.

Transformer 학습에서는 원본만 사용할지, 원본+증강 통합 CSV를 사용할지 선택합니다. 통합 CSV는 `build_merged_training_csv.py`로 만든 `row_type`, `training_eligible` 컬럼이 있는 UTF-8 파일입니다. 증강 전체 사용 또는 클래스별 최대 개수를 선택할 수 있습니다. 원본만 선택하면 `--augmentation_per_class 0`을 전달합니다. 증강 생성 작업은 실행하지 않습니다.

vLLM은 KT Cloud 등에서 이미 기동한 모델에 API로 분류를 요청합니다. API URL, 모델 ID(`/v1/models`의 `id`), 사업목적 문자 수, 응답 토큰 수, 동시 요청 수, 타임아웃을 입력합니다. GPU 점유·최대 컨텍스트·서버 동시 시퀀스 설정은 변경하지 않습니다. 인증이 필요하면 호스트의 `VLLM_API_KEY` 환경변수가 컨테이너로 전달됩니다. 프로젝트에 `llm/v1/classify_biofin_category_with_vllm.py`가 있어야 합니다.

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
