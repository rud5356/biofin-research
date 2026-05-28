"""
biodiversity_rag 프로젝트의 경로 및 모델 설정 파일.

RAG(Retrieval-Augmented Generation)란:
  1. 질문과 관련된 문서를 벡터 DB에서 검색(Retrieval)하고
  2. 검색된 내용을 바탕으로 LLM이 답변을 생성(Generation)하는 방식입니다.

이 파일에서 임베딩 모델, 청크 크기, LLM 모델 등을 설정합니다.
"""

from pathlib import Path


# ─── 기본 디렉토리 경로 ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent   # biodiversity_rag/
DATA_DIR = BASE_DIR / "data"                        # 원본 데이터 파일 저장 폴더
DB_DIR = BASE_DIR / "db"                            # ChromaDB 벡터 DB 저장 폴더

# ─── 원본 데이터 경로 (llm_ner_biodiversity 프로젝트가 생성한 파일들) ──────────
# PubMed에서 수집한 생물다양성 논문 초록 CSV
SOURCE_ABSTRACTS = BASE_DIR.parent / "llm_ner_biodiversity" / "data" / "abstracts.csv"
# LLM이 초록에서 추출한 개체명(종명, 지역명 등) 결과 CSV
SOURCE_NER = BASE_DIR.parent / "llm_ner_biodiversity" / "results" / "ner_results_llama3.1-8b_few_shot.csv"

# ─── 임베딩 모델 설정 ─────────────────────────────────────────────────────────
# 텍스트를 벡터(숫자 배열)로 변환하는 모델입니다.
# all-MiniLM-L6-v2: 작고 빠르며 CPU에서도 실행 가능한 문장 임베딩 모델
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ─── ChromaDB 설정 ────────────────────────────────────────────────────────────
# ChromaDB는 벡터(임베딩)를 저장하고 유사도 검색을 제공하는 로컬 벡터 데이터베이스입니다.
# 컬렉션(collection)은 관계형 DB의 테이블처럼 벡터들을 묶는 단위입니다.
CHROMA_COLLECTION = "biodiversity_abstracts"

# ─── 청킹(Chunking) 설정 ─────────────────────────────────────────────────────
# 논문 초록처럼 긴 텍스트를 모델이 처리 가능한 작은 조각(청크)으로 나눕니다.
CHUNK_SIZE = 512        # 청크 하나의 최대 토큰 수 (약 500자)
CHUNK_OVERLAP = 64      # 인접한 청크 사이에 겹치는 문자 수 (문맥 연속성 유지)

# ─── 검색 설정 ────────────────────────────────────────────────────────────────
# 질문이 들어오면 벡터 DB에서 가장 유사한 청크를 몇 개까지 가져올지 설정합니다.
TOP_K = 5

# ─── LLM(대형 언어 모델) 설정 ────────────────────────────────────────────────
# Ollama를 통해 로컬에서 실행하는 LLM 모델 이름
LLM_MODEL = "llama3.1:8b"
# 온도(temperature): 낮을수록 일관되고 보수적인 답변, 높을수록 창의적인 답변
LLM_TEMPERATURE = 0.1
