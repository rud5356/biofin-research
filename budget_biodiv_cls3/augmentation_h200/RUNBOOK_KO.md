> 지정된 BIOFIN CSV의 카테고리 균형 증강은 [BALANCED_RUNBOOK_KO.md](BALANCED_RUNBOOK_KO.md)의 balanced_csv.py를 사용하세요. 아래는 기존 일반 JSONL 방식입니다.

# H200 텍스트 분류 데이터 증강 실행 안내

Python 3.10 이상 Linux 서버용이며 추가 Python 패키지는 필요 없습니다.
원본 형식을 확인하지 못했으므로 학습 전용 JSONL의 `text/label` 형식으로 구현했습니다.
예제 라벨은 동작 테스트용입니다. 실제 연구의 라벨 정의로 간주하지 마세요.

## 저장 및 반복 정책

- 기본 종료는 한국 시간 **2026-10-01 00:00:00**, 즉 9월 30일 끝까지입니다.
- 원본 마지막 행까지 읽으면 처음부터 반복합니다. 요청당 기본 4개의 패러프레이즈를 생성합니다.
- 저장 위치: `/home/work/coreit-workspace/bio-fin/augmented_h200_v1/`.
- `bio-fin` 전체 파일의 논리 크기를 합산합니다. 기본 한도는 십진 1.5TB(1,500,000,000,000 바이트)입니다.
- 다음 쓰기가 한도를 넘으면 생성 shard 파일을 무작위로 골라 최소 500GB를 먼저 삭제합니다. 파일 단위라 기본 shard 크기인 최대 128MB만큼 더 삭제될 수 있습니다. 부족한 공간이 더 크면 추가로 삭제합니다.
- 원본·모델·다른 파일은 삭제하지 않습니다. 삭제 가능한 생성 파일이 부족하면 중단합니다.
- 재시작 시 기존 생성 파일도 계산/삭제 대상입니다. 원본 순회는 처음부터 다시 시작합니다.
- 출력 폴더 잠금으로 중복 실행을 막습니다. API 오류는 지수 지연으로 재시도하며 10회 연속 실패하면 오류 종료합니다.
- 다른 프로세스의 동시 쓰기, 스냅샷, 파일 시스템 할당량은 강제 제한하지 않습니다. 엄격한 물리 한도는 서버 디스크 quota가 필요합니다. 실행 중 다른 프로그램이 출력 폴더를 수정하지 않게 하세요.

## 입력과 품질

한 줄에 JSON 객체 하나:

```json
{"text":"증강할 원문", "label":1, "split":"train"}
```

`split`은 생략 시 train으로 간주하고 다른 값은 거부합니다. 먼저 데이터를 train/validation/test로 나누고 train만 입력하세요.
컬럼명이 다르면 `--text-key 본문 --label-key 분류`를 지정합니다. 출력은 항상 `text/label`입니다.
원본 label, 원문 해시, 행 번호, 모델, 생성 시각, `needs_review: true`를 기록합니다.
라벨은 그대로 복사하지만 생성문의 의미 보존은 모델에 달려 있으므로 학습 전 표본 검수가 필요합니다.
중복 제거는 한 요청과 해당 원문 범위입니다. 전체 기간의 중복 제거나 라벨별 균형 조정은 수행하지 않습니다.

## 1. 서버 준비와 자동 테스트

이 폴더를 **데이터를 저장할 Linux 서버**의 `/home/work/coreit-workspace/augmentation_h200`에 복사합니다.
추론 API를 호출하는 것만으로 원격 디스크에 저장되지 않습니다. 스크립트가 실행되는 서버에 저장됩니다.

```bash
cd /home/work/coreit-workspace/augmentation_h200
python3 --version
python3 -m unittest -v test_augment.py
```

테스트는 임시 폴더와 로컬 가짜 HTTP 서버를 사용하여 두 API 형식, 종료 시각,
용량 정리, 원본 보호, 중복 실행 잠금, 재시작, split 검사를 확인합니다.
로컬 Windows에서 6개 테스트 중 5개 통과, Linux SIGALRM 종료 시각 테스트 1개는 건너뛰었습니다. Python 문법 검사도 통과했습니다. Linux 서버에서 위 명령으로 6개 전체를 확인하세요. 실서버에서도 Ollama exaone3.5:32b 및 vLLM Qwen/Qwen3-8B 각각 예제 1건으로 증강문 2개 생성·저장을 확인했습니다(2026-09-21).

## 2. 작은 용량으로 반복·삭제 테스트

```bash
python3 augment.py --mock --input sample_train.jsonl \
  --output /tmp/biofin-augmentation-smoke \
  --cap-bytes 100000 --delete-bytes 30000 --shard-bytes 10000 \
  --min-free-bytes 0 --max-batches 200 --interval 0
```

`Random cleanup removed ... bytes` 로그와 정상 종료를 확인합니다.
MOCK 결과는 학습용이 아닙니다. 테스트도 기본 종료일 이후에는 동작하지 않으므로 날짜가 지났다면 `--until`에 미래 시각을 지정하세요.

## 3. API 연결과 소량 생성

요청을 보내는 서버의 외부 IP가 허용 IP `27.35.82.130`과 맞아야 합니다.
관리 API 문서는 확인하지 못했으나 공식 표준 경로로 두 서버의 모델 조회와 실제 생성을 확인했습니다.
`/manager`는 추론 주소가 아닙니다. 프록시 prefix가 다르면 `--base-url`을 조정하세요.

```bash
# 인증이 필요한 경우만 실행
read -rsp 'API key: ' AUG_API_KEY; echo
export AUG_API_KEY

# 둘 중 사용할 서비스에서 모델 ID 조회
python3 augment.py --backend vllm --list-models
python3 augment.py --backend ollama --list-models

MODEL='Qwen/Qwen3-8B'
python3 augment.py --backend vllm --model "$MODEL" \
  --input sample_train.jsonl --output /tmp/biofin-api-smoke --max-batches 2
```

Ollama 선택 시 `--backend ollama`와 해당 모델 ID를 사용합니다.
403/접속 실패는 허용 IP·인증을, 404는 API 경로·모델을 확인하세요.
현재 vLLM Qwen/Qwen3-8B의 JSON response_format 동작을 확인했습니다. 서버 모델 변경 후에는 다시 소량 테스트하세요.

## 4. 실전 실행

실제 학습 JSONL을 `/home/work/coreit-workspace/bio-fin/train.jsonl`에 준비합니다.
사용자 systemd가 가능한 Linux 서버 기준입니다. `MODEL`에 조회한 실제 ID를 넣으세요.

```bash
cd /home/work/coreit-workspace/augmentation_h200
MODEL='Qwen/Qwen3-8B'
systemd-run --user --unit=biofin-augment \
  --property=Restart=on-failure --property=RestartSec=60 \
  --property=TimeoutStopSec=150 --property=UMask=0077 \
  --working-directory="$PWD" \
  python3 "$PWD/augment.py" --backend vllm --model "$MODEL" \
  --input /home/work/coreit-workspace/bio-fin/train.jsonl \
  --output /home/work/coreit-workspace/bio-fin \
  --until 2026-10-01T00:00:00+09:00
```

인증이 필요하면 먼저 `systemctl --user import-environment AUG_API_KEY`를 실행하고,
systemd-run에 `--property=PassEnvironment=AUG_API_KEY`를 추가하세요.
로그아웃 후에도 유지하려면 서버 관리 정책에 따라 `loginctl enable-linger "$USER"`가 필요할 수 있습니다.
이 transient 서비스는 서버 재부팅 후에는 실전 실행 명령을 다시 실행해야 합니다.
종료일 도달 시 정상 종료하므로 자동 재시작하지 않습니다.

```bash
journalctl --user -u biofin-augment -f
systemctl --user status biofin-augment
# 수동 중단
systemctl --user stop biofin-augment
```

사용자 systemd가 없다면 tmux 세션 안에서 위 python3 명령을 직접 실행하세요. 이 경우 오류 후 자동 재시작은 없습니다.
1.5TB는 보관 한도이며 목표 생성량이 아닙니다. 실제 속도는 모델과 입력 길이에 따라 다릅니다.
강제 종료나 전원 손실로 JSONL 마지막 줄이 잘릴 수 있으므로 학습 전에 JSON 파싱 검사를 하세요.

표준 API 참고: [Ollama](https://docs.ollama.com/api/chat),
[vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/).

기존 transformer/v1 및 v2 학습 CLI는 JSONL을 직접 읽지 않습니다. 이 코드는 독립 증강 생성기입니다. 기존 학습 파이프라인에서 분할이 끝난 train_records의 text/label을 JSONL로 내보내 입력하고, 생성 결과를 학습에 넣으려면 별도의 로더 연결이 필요합니다.
