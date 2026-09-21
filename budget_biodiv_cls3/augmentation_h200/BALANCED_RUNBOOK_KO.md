# CSV 카테고리 균형 증강

이제 기존 `augment.py` 대신 **`balanced_csv.py`**를 실행합니다. `augment.py`도 공통 모듈이므로 함께 복사하세요.
원본 CSV를 읽기만 하고, 생성 결과는 `bio-fin/balanced_h200_v1/`에 별도 JSONL로 저장합니다.
기존 `augmented_h200_v1/` 결과는 이번 균형 집계와 삭제 대상에 넣지 않습니다. 전체 저장 용량에는 포함합니다.

## 기준과 동작

기본 `--category-level primary`는 **1차 카테고리 0~9** 기준입니다.
`--category-level subcategory`는 1차+하위를 `2.04`처럼 결합합니다. 서로 다른 1차에 속한 하위 번호를 합치지 않습니다.

1. 원본 건수 + 현재 남아 있는 생성 건수가 가장 적은 카테고리를 고릅니다. 동률이면 카테고리 번호순입니다.
2. 그 카테고리의 원본들을 섞어서 순회하며 작은 API 요청 여러 번으로 서로 다른 유효 문장 **50개**를 모읍니다.
3. 완성된 50개를 파일 하나에 원자적으로 저장합니다. JSON 파싱 오류·빈 문장·요청 내/배치 내 동일 문장·원문 그대로인 문장은 제외합니다.
4. 건수를 다시 비교하여 다음 카테고리를 고릅니다. `--balance-batch-size 100`처럼 50의 배수로 늘릴 수 있습니다.

카테고리 50건은 API를 한 번에 50개 요청한다는 뜻이 아닙니다. 기본 요청당 4개이며 총 50개의 검증 통과 결과를 모읍니다.
단어·숫자·의미 보존은 모델 지시로 유도하고 생성 결과를 `needs_review: true`로 표시합니다. 자동 의미 검증까지 제공하지는 않습니다.
원본 사업명·예산정보를 본문으로 사용합니다. **CSV에 연결된 HWP/PDF 본문은 읽지 않습니다.**
카테고리·판단근거·비고는 생성 원문에 포함하지 않습니다. 두 라벨과 원본 business_key, 행 번호, CSV 해시를 결과에 보존합니다.

## 이 CSV의 1차 카테고리 분포

총 3,972건, business_key 중복 0건입니다.

| 카테고리 | 원본 | 초기 최다 3,429건 이상까지 필요한 생성량(50단위) |
|---|---:|---:|
| 1 | 10 | 3,450 |
| 3 | 10 | 3,450 |
| 8 | 17 | 3,450 |
| 4 | 23 | 3,450 |
| 7 | 27 | 3,450 |
| 5 | 69 | 3,400 |
| 6 | 115 | 3,350 |
| 9 | 127 | 3,350 |
| 2 | 145 | 3,300 |
| 0 | 3,429 | 0 |

이 표는 최초 목표까지의 예상량입니다. 50개 저장할 때마다 다시 정렬하므로 한 카테고리의 부족분 전체를 먼저 채우지 않습니다.
기본은 균형을 달성해도 가장 적은 카테고리부터 계속 생성하며 **2026-10-01 00:00:00 +09:00**에 종료합니다.
최초 균형에서 끝내려면 `--stop-when-balanced`를 추가합니다. 각 카테고리가 원본 최다 건수 이상이고 최대·최소 차이가 배치 크기 미만이면 종료합니다.
50개 단위라 정확히 동일한 건수 대신 기본 최대 49건 차이를 허용합니다.

## 서버 준비와 테스트

폴더 전체를 `/home/work/coreit-workspace/augmentation_h200`에 복사합니다. Python 3.10 이상, 추가 패키지 불필요입니다.
원본 CSV는 예를 들어 `/home/work/coreit-workspace/bio-fin/BIOFIN_2023_취합_2026.09.14_document_matched.csv`에 복사해 둡니다.

```bash
cd /home/work/coreit-workspace/augmentation_h200
CSV='/home/work/coreit-workspace/bio-fin/BIOFIN_2023_취합_2026.09.14_document_matched.csv'
python3 -m unittest discover -s . -p 'test_*.py' -v

# 원본 분포와 예상량만 출력: API 호출/저장/삭제 없음
python3 balanced_csv.py --input "$CSV" --category-level primary --plan

# 50개씩 두 배치 가짜 생성: 학습용으로 사용하지 않음
python3 balanced_csv.py --input "$CSV" --mock --max-batches 2 \
  --output /tmp/biofin-balance-mock --interval 0

# 실제 모델로 50개 한 배치 생성
python3 balanced_csv.py --input "$CSV" --backend vllm --model Qwen/Qwen3-8B \
  --max-batches 1 --output /tmp/biofin-balance-api
```

모델이 변경되었으면 `python3 augment.py --backend vllm --list-models`로 조회합니다.
Ollama 사용 시 `--backend ollama --model exaone3.5:32b`로 바꿉니다.

## 실전 실행

기존 증강 서비스가 실행 중이면 먼저 중지하여 하나만 실행합니다.
사용자 systemd 사용 가능 서버 기준이며, 인증·로그아웃 유지·재부팅 관련 사항은 기존 RUNBOOK_KO.md와 같습니다.

```bash
cd /home/work/coreit-workspace/augmentation_h200
CSV='/home/work/coreit-workspace/bio-fin/BIOFIN_2023_취합_2026.09.14_document_matched.csv'
systemd-run --user --unit=biofin-balanced \
  --property=Restart=on-failure --property=RestartSec=60 \
  --property=TimeoutStopSec=150 --property=UMask=0077 \
  --working-directory="$PWD" \
  python3 "$PWD/balanced_csv.py" --input "$CSV" \
  --category-level primary --balance-batch-size 50 \
  --backend vllm --model Qwen/Qwen3-8B \
  --output /home/work/coreit-workspace/bio-fin \
  --until 2026-10-01T00:00:00+09:00

journalctl --user -u biofin-balanced -f
# 중단
systemctl --user stop biofin-balanced
```

## 보관·재시작·원본 보호

- 기존과 동일하게 상위 출력 폴더 전체 1.5TB(십진) 한도이며, 다음 저장 전에 생성 파일 최소 500GB를 무작위 삭제합니다. 완성 배치 파일 단위로 반올림합니다.
- 삭제 후 **남아 있는 파일의 건수**로 다시 계산하므로, 삭제되어 부족해진 카테고리를 보충합니다.
- 저장 중 임시 파일은 `.pending`입니다. 50개가 완성되어 저장될 때 `.jsonl`로 바뀝니다. 종료/실패 시 미완성 배치를 완성 건수로 세지 않습니다.
- 재시작 시 완료 배치는 유지하고 임시 생성 파일만 정리합니다. 데이터 출처 해시·카테고리 기준·mock/real이 다르면 같은 출력 폴더의 재사용을 거부합니다.
- 원본/다른 파일은 삭제하지 않습니다. 생성 파일이 부족해 500GB를 비울 수 없으면 오류로 종료합니다. 원본 SHA-256으로 불변 여부를 확인할 수 있습니다.
- 원본 CSV 해시: `ac6b878b67e053720d5b7186b814e272e6e532d82edc072ac9c7c8d1a67dba42`.
- 파일 개수만큼 작은 배치 메타데이터를 집계하고 동일 프로세스에서는 캐시합니다. 출력 파일은 프로그램 밖에서 수정하지 마세요.
- 전체 실행 기간의 문장 중복 제거는 하지 않습니다. 장기간 반복 생성 결과는 학습 전 중복·품질 검수가 필요합니다.

이 CSV에는 train/valid/test 구분이 없습니다. 출력은 이를 `source_split: unassigned`로 기록합니다.
실제 학습에서는 원본 사업 그룹과 그 증강문을 같은 split에 두고, 검증/시험 원본의 증강문은 train에 넣지 마세요.
원본을 수정하지 않고 별도 train CSV를 만들어 `split=train`을 추가하면 해당 파일의 train 행만 증강합니다.
기존 트랜스포머 학습기의 JSONL 로더 연결은 이번 생성 코드에 포함하지 않았습니다.

## 검증 결과 (2026-09-21)

자동 테스트 12개 중 Windows에서 실행 가능한 11개가 통과했습니다. Linux SIGALRM 시간 제한 테스트 1개는 Linux 서버에서 실행해야 합니다. 작은 용량 테스트에서 실제 무작위 삭제 후 파일의 실제 행 수와 재집계 건수가 일치했고 원본 CSV 바이트 불변을 확인했습니다.

동일 CSV와 실제 vLLM Qwen/Qwen3-8B로 카테고리 1에서 50개 한 배치를 생성·저장했습니다. 저장된 50개는 모두 카테고리 1이고 문자열 중복은 0건입니다. 원본 CSV의 SHA-256이 작업 전후 동일함을 확인했습니다. 이 소량 테스트만 실행했으며 서버 장기 실행은 시작하지 않았습니다.
