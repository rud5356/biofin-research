from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"

# 원본 데이터 경로 (llm_ner_biodiversity 결과)
SOURCE_ABSTRACTS = BASE_DIR.parent / "llm_ner_biodiversity" / "data" / "abstracts.csv"
SOURCE_NER = BASE_DIR.parent / "llm_ner_biodiversity" / "results" / "ner_results_llama3.1-8b_few_shot.csv"

# 임베딩 모델 (로컬, CPU 가능)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB 컬렉션명
CHROMA_COLLECTION = "biodiversity_abstracts"

# 청킹 설정
CHUNK_SIZE = 512        # 최대 토큰 수 (근사)
CHUNK_OVERLAP = 64      # 청크 간 겹치는 문자 수

# 검색 설정
TOP_K = 5               # 검색 시 반환할 청크 수

# LLM 설정
LLM_MODEL = "llama3.1:8b"
LLM_TEMPERATURE = 0.1
