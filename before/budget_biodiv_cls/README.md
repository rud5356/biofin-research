# budget_biodiv_cls

정부 예산 사업 문서에서 생물다양성 관련 사업을 분류하는 파이프라인입니다.
Ollama LLM으로 라벨을 생성하고, KLUE/RoBERTa 또는 KoBigBird 모델을 학습합니다.

## 파이프라인 개요

```
[원본 CSV]
사업별결산세출지출현황_파일매칭_최종.csv
    │
    ▼
make_biodiv_labels.py --version v1
    → biodiv_labeled.csv (메타데이터 기반 1차 라벨)
    │
    ▼
build_biodiv_text_dataset.py
    → biodiv_document_text_dataset.csv (HWP/PDF 본문 추출)
    │
    ▼
make_biodiv_labels.py --version v2
    → biodiv_document_text_dataset_labeled_v2.csv (문서 본문 기반 엄격 라벨)
    │
    ├──▶ train_biodiv_cls.py       → model/label_v2/          (RoBERTa-small, 512 토큰)
    ├──▶ train_biodiv_long_cls.py  → model/label_v2_long/     (KoBigBird, 2048 토큰)
    └──▶ PostgreSQL 적재/임베딩     → biodiv_documents / biodiv_document_chunks
```

## 폴더 구조

```
budget_biodiv_cls/
├── src/
│   ├── config.py                    # 경로 및 컬럼 상수
│   ├── build_biodiv_text_dataset.py # HWP/PDF 텍스트 추출
│   ├── make_biodiv_labels.py        # Ollama LLM 라벨링
│   ├── train_biodiv_cls.py          # RoBERTa-small 학습
│   ├── train_biodiv_long_cls.py     # KoBigBird 긴문서 학습
│   ├── load_biodiv_csv_to_postgres.py
│   └── embed_biodiv_chunks_to_postgres.py
├── data/                            # 중간/최종 데이터 (gitignore)
├── model/                           # 학습된 모델 (gitignore)
├── requirements.txt                 # 라벨링·추출용 의존성
├── requirements_train.txt           # 모델 학습용 의존성
├── Dockerfile
└── .dockerignore
```

## 환경 설정

### 로컬 설치

```bash
# 라벨링·문서 추출용
pip install -r requirements.txt

# 모델 학습까지 필요한 경우
pip install -r requirements_train.txt
```

### Docker (GPU 학습 환경)

```bash
# 이미지 빌드 (CUDA 12.8 + cuDNN 9 기반)
docker build -t biodiv-cls .

# 컨테이너 실행 (data, model 폴더를 마운트)
docker run --gpus all -it \
  -v $(pwd)/data:/workspace/code/data \
  -v $(pwd)/model:/workspace/code/model \
  biodiv-cls
```

## 스크립트 사용법

모든 스크립트는 `src/` 폴더에서 실행합니다.

```bash
cd budget_biodiv_cls/src
```

---

### 1. `make_biodiv_labels.py` — LLM 라벨 생성

Ollama에 로컬 실행 중인 LLM으로 생물다양성 관련 여부를 판정합니다 (0: 비관련, 1: 관련, -1: 실패).

**v1** — 메타데이터(소관명, 사업명 등) 기반 1차 라벨링:

```bash
python make_biodiv_labels.py --version v1 --model llama3.2:latest
```

**v2** — 문서 본문(`clean_document_text`) 기반 엄격 라벨링:

```bash
python make_biodiv_labels.py --version v2 --model llama3.2:latest
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--version` | `v1` | `v1`=메타데이터 기반, `v2`=문서 본문 기반 |
| `--model` | `llama3.2:latest` | Ollama 모델명 |
| `--ollama-url` | `http://localhost:11434` | Ollama 서버 주소 |
| `--max-chars` | `3000` | v2 문서 본문 최대 입력 글자 수 |
| `--timeout` | `180` | Ollama 응답 제한 시간(초) |
| `--retries` | `2` | 실패 시 재시도 횟수 |
| `--limit` | `0` (전체) | 처리할 최대 행 수 |
| `--delay` | `0.1` | 요청 간 대기 시간(초) |

중간에 중단해도 이미 처리된 행부터 이어서 재개합니다.

---

### 2. `build_biodiv_text_dataset.py` — 문서 텍스트 추출

HWP·PDF 문서에서 "사업목적" 섹션을 추출합니다. 문서가 없거나 앵커를 찾지 못하면 메타데이터로 대체합니다.

```bash
python build_biodiv_text_dataset.py
```

주요 옵션:

| 옵션 | 설명 |
|------|------|
| `--limit INT` | 처리할 최대 행 수 |
| `--metadata-fallback-on-missing-anchor` | 앵커 미발견 시 메타데이터 사용 |
| `--add-source-prefix` | 입력 앞에 소스 유형 접두사 추가 |

출력 컬럼: `clean_document_text`, `text_source`, `document_status`, `clean_text_char_count`

---

### 3. `train_biodiv_cls.py` — RoBERTa-small 학습

KLUE/RoBERTa-small (512 토큰)로 이진 분류 모델을 학습합니다.

```bash
python train_biodiv_cls.py
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model-name` | `klue/roberta-small` | 허깅페이스 모델 |
| `--output-dir` | `model/label_v2` | 저장 경로 |
| `--label-col` | `label_v2` | 라벨 컬럼명 |
| `--max-len` | `512` | 최대 토큰 길이 |
| `--epochs` | `10` | 에포크 수 |
| `--batch-size` | `16` | 배치 크기 |
| `--balance-mode` | `pos_weight` | `pos_weight` / `undersample` / `none` |
| `--no-cuda` | — | CPU 강제 사용 |

---

### 4. `train_biodiv_long_cls.py` — KoBigBird 긴문서 학습

KoBigBird (최대 2048 토큰)로 긴 문서를 처리하는 분류 모델을 학습합니다. GPU 메모리가 부족할 경우 `--grad-accum-steps`를 늘리거나 `--fp16`을 사용합니다.

```bash
python train_biodiv_long_cls.py
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model-name` | `monologg/kobigbird-bert-base` | 허깅페이스 모델 |
| `--output-dir` | `model/label_v2_long` | 저장 경로 |
| `--max-len` | `2048` | 최대 토큰 길이 |
| `--batch-size` | `1` | 배치 크기 |
| `--grad-accum-steps` | `8` | 그래디언트 누적 스텝 |
| `--fp16` | — | AMP FP16 학습 사용 |
| `--gradient-checkpointing` | — | GPU 메모리 절약 |
| `--attention-type` | `block_sparse` | `block_sparse` / `original_full` |

---

### 5. `load_biodiv_csv_to_postgres.py` — CSV를 PostgreSQL에 적재

학습 입력 CSV(`biodiv_document_text_dataset_labeled_v2.csv`)를 PostgreSQL `biofin` DB의 `biodiv_documents` 테이블에 넣습니다.

```bash
python src/load_biodiv_csv_to_postgres.py --user postgres
```

비밀번호를 터미널에 남기고 싶지 않으면 위 명령처럼 실행하면 프롬프트로 물어봅니다. 연결 문자열을 직접 줄 수도 있습니다.

```bash
python src/load_biodiv_csv_to_postgres.py --database-url postgresql://postgres:비밀번호@localhost:5432/biofin
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--csv` | `data/biodiv_document_text_dataset_labeled_v2.csv` | 적재할 CSV |
| `--db-name` | `biofin` | PostgreSQL DB 이름 |
| `--table` | `biodiv_documents` | 생성/적재할 테이블 이름 |
| `--if-exists` | `fail` | 기존 테이블이 있을 때 `fail` / `append` / `replace` / `truncate` |
| `--chunksize` | `1000` | 한 번에 넣을 행 수 |

처음 실행할 때는 기본값 그대로 사용하면 되고, 같은 테이블을 새로 덮어쓰려면 다음처럼 실행합니다.

```bash
python src/load_biodiv_csv_to_postgres.py --user postgres --if-exists replace
```

---

### 6. `embed_biodiv_chunks_to_postgres.py` — 문서 chunk와 embedding 저장

`biodiv_documents.clean_document_text`를 token 기준으로 쪼개고, transformer embedding을 만들어 `biodiv_document_chunks` 테이블에 저장합니다.

```bash
python src/embed_biodiv_chunks_to_postgres.py --user postgres
```

처음 테스트할 때는 일부 문서만 처리하는 것을 권장합니다.

```bash
python src/embed_biodiv_chunks_to_postgres.py --user postgres --limit 20
```

같은 embedding 모델의 chunk를 다시 만들려면:

```bash
python src/embed_biodiv_chunks_to_postgres.py --user postgres --if-exists replace-model
```

기본 저장 방식은 별도 확장이 필요 없는 PostgreSQL `real[]` 배열입니다. pgvector 확장이 설치되어 있으면 다음처럼 vector 컬럼으로 저장할 수 있습니다.

```bash
python src/embed_biodiv_chunks_to_postgres.py --user postgres --embedding-storage pgvector --recreate-chunks-table
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--embedding-model` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Hugging Face embedding 모델 |
| `--chunks-table` | `biodiv_document_chunks` | chunk/embedding 저장 테이블 |
| `--max-tokens` | `384` | chunk 하나의 최대 토큰 수 |
| `--overlap-tokens` | `64` | 긴 문장 분할 시 겹칠 토큰 수 |
| `--batch-size` | `16` | embedding 배치 크기 |
| `--limit` | `0` | 처리할 문서 수. `0`은 전체 |
| `--device` | `auto` | `auto` / `cpu` / `cuda` |
| `--embedding-storage` | `array` | `array` / `pgvector` |
| `--if-exists` | `skip` | `skip` / `append` / `replace-model` / `replace-all` |

pgAdmin에서 확인:

```sql
SELECT COUNT(*) FROM public.biodiv_document_chunks;

SELECT document_id, chunk_index, embedding_model, embedding_dim, token_count, left(chunk_text, 120)
FROM public.biodiv_document_chunks
ORDER BY document_id, chunk_index
LIMIT 20;
```

---

## 학습 출력물

학습 완료 후 `output-dir`에 다음 파일이 저장됩니다:

```
model/label_v2/
├── model.safetensors        # 학습된 가중치
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── training_metadata.json   # 최적 threshold, F1, AUC 등 기록
```

`training_metadata.json` 예시:

```json
{
  "best_epoch": 3,
  "best_threshold": 0.45,
  "best_f1": 0.82,
  "best_precision": 0.81,
  "best_recall": 0.83,
  "best_auc": 0.89,
  "model_name": "klue/roberta-small",
  "label_col": "label_v2",
  "balance_mode": "pos_weight"
}
```
