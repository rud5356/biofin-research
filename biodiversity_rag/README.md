# Biodiversity RAG

생물다양성 논문 초록을 기반으로 한 RAG(Retrieval-Augmented Generation) 질의응답 시스템입니다.  
`llm_ner_biodiversity`에서 수집한 PubMed 초록을 벡터DB에 인덱싱하고, 자연어 질문에 대해 관련 논문을 검색·인용하며 답변합니다.

## 파이프라인 구조

```
질문 입력
    ↓
[임베딩] sentence-transformers로 질문 벡터화
    ↓
[검색] ChromaDB에서 유사 청크 top-k 반환
    ↓
[생성] ollama llama3.1:8b로 컨텍스트 기반 답변
    ↓
답변 + 참고 문헌 출력
```

## 프로젝트 구조

```
biodiversity_rag/
├── src/
│   ├── index_documents.py  # 1단계: 문서 인덱싱 (최초 1회)
│   ├── rag_pipeline.py     # 2단계: 질의응답 실행
│   ├── chunker.py          # Abstract → 청크 분할
│   ├── embedder.py         # 텍스트 → 임베딩 벡터
│   ├── vector_store.py     # ChromaDB 저장·조회
│   ├── retriever.py        # 질문 검색 로직
│   ├── generator.py        # LLM 답변 생성
│   └── config.py           # 경로·모델·파라미터 설정
├── data/                   # abstracts.csv 위치 (선택)
├── db/                     # ChromaDB 로컬 저장소 (자동 생성)
└── environment.yml
```

## 설치 및 환경 설정

### 1. Conda 환경 생성

```bash
conda env create -f environment.yml
conda activate biodiversity_rag
```

### 2. Ollama 모델 준비

```bash
ollama pull llama3.1:8b
ollama serve
```

### 3. 데이터 준비

`llm_ner_biodiversity`에서 수집한 초록 파일을 그대로 참조합니다.  
`config.py`의 `SOURCE_ABSTRACTS` 경로가 자동으로 아래를 가리킵니다:

```
../llm_ner_biodiversity/data/abstracts.csv
```

경로가 다를 경우 `--source` 옵션으로 직접 지정할 수 있습니다.

## 실행 방법

### 1단계: 문서 인덱싱 (최초 1회)

```bash
cd src

# 기본 실행
python index_documents.py

# 경로 직접 지정
python index_documents.py --source ../../llm_ner_biodiversity/data/abstracts.csv

# DB 초기화 후 재인덱싱
python index_documents.py --reset
```

### 2단계: 질의응답

```bash
# 단일 질문
python rag_pipeline.py --question "What mammal species are found in South Korea?"

# 검색 청크 수 조정
python rag_pipeline.py --question "Where does Lutra lutra live?" --top-k 3

# 대화형 모드 (종료: q)
python rag_pipeline.py
```

**출력 예시**:
```
질문: What mammal species are found in South Korea?

답변:
According to PMID 40584664, the critically endangered mammals in South Korea include
Panthera spp. (tigers and leopards), Vulpes spp. (foxes), Ursus spp. (bears),
and Naemorhedus spp. (gorals).

참고 문서:
  [0.723] PMID 40584664 — Estimating the 15th-Century Potential Habitats of Endangered...
  [0.698] PMID 40584664 — Estimating the 15th-Century Potential Habitats of Endangered...
```

## 주요 옵션

### index_documents.py

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--source` | `../llm_ner_biodiversity/data/abstracts.csv` | 입력 CSV 경로 |
| `--reset` | `False` | DB 초기화 후 재인덱싱 |

### rag_pipeline.py

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--question` | 없음 (대화형 모드) | 단일 질문 문자열 |
| `--top-k` | `5` | 검색할 청크 수 |

## 청킹 전략

| 조건 | 전략 |
|------|------|
| Abstract ≤ 512자 | 전체를 청크 1개로 유지 |
| Abstract > 512자 | 문장 단위로 분리 후 512자 기준으로 합치되, 64자 overlap |

100편 기준 약 398개 청크 생성 (논문당 평균 4개).

## 설정 (`config.py`)

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 임베딩 모델 (로컬, CPU 가능)
CHROMA_COLLECTION = "biodiversity_abstracts"
CHUNK_SIZE = 512        # 최대 청크 크기 (문자 수)
CHUNK_OVERLAP = 64      # 청크 간 겹침 문자 수
TOP_K = 5               # 검색 시 반환할 청크 수
LLM_MODEL = "llama3.1:8b"
```

## 의존성

- Python 3.10+
- `sentence-transformers` — 로컬 임베딩 (all-MiniLM-L6-v2)
- `chromadb` — 로컬 벡터DB
- `ollama` — 로컬 LLM 추론
- `pandas` — 데이터 처리
