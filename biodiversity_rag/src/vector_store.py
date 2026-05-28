"""
ChromaDB 벡터 데이터베이스에 청크를 저장하고 유사도 검색을 수행하는 모듈.

ChromaDB란?
  벡터(숫자 배열)를 저장하고 "이 벡터와 가장 비슷한 벡터는 무엇인가?"를 빠르게
  찾아주는 데이터베이스입니다. 일반 DB의 "SELECT * WHERE name='김철수'"와 달리,
  "의미적으로 비슷한 문장" 을 찾을 수 있습니다.

cosine 유사도:
  두 벡터가 같은 방향을 가리키면 1.0 (완전 일치),
  다른 방향이면 0.0에 가까워집니다.
  ChromaDB는 거리(distance)를 반환하므로 1 - distance = 유사도(score)입니다.
"""

from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
except ModuleNotFoundError as exc:
    # chromadb가 없어도 import는 성공하도록 처리
    chromadb = None
    Settings = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from chunker import Chunk
from config import CHROMA_COLLECTION, DB_DIR


def get_collection(db_dir: Path = DB_DIR) -> "chromadb.Collection":
    """
    ChromaDB 컬렉션을 반환합니다. 없으면 새로 생성합니다.

    PersistentClient: 디스크에 데이터를 영구 저장 (프로그램 종료 후에도 유지)
    anonymized_telemetry=False: 사용 통계를 ChromaDB 서버로 보내지 않음
    hnsw:space=cosine: 벡터 유사도 계산 방식을 코사인 거리로 설정
    """
    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "chromadb가 설치되지 않았습니다. "
            "`conda env create -f biodiversity_rag/environment.yml`로 환경을 만든 뒤 다시 실행하세요."
        ) from _IMPORT_ERROR

    client = chromadb.PersistentClient(
        path=str(db_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},  # 코사인 유사도 기반 인덱스 사용
    )


def add_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """
    청크와 임베딩 벡터를 ChromaDB에 저장합니다.

    이미 저장된 청크(ID 기준)는 건너뜁니다.
    재실행해도 중복 데이터가 쌓이지 않습니다.

    Args:
        chunks: 저장할 Chunk 객체 목록
        embeddings: 각 청크에 대응하는 임베딩 벡터 목록 (순서 일치 필수)
    """
    collection = get_collection()

    # 이미 DB에 있는 ID를 확인하여 중복 저장 방지
    existing_ids = set(collection.get(ids=[c.chunk_id for c in chunks])["ids"])
    new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
    new_embeddings = [embeddings[i] for i, c in enumerate(chunks) if c.chunk_id not in existing_ids]

    if not new_chunks:
        print("  모든 청크가 이미 저장돼 있습니다.")
        return

    collection.add(
        ids=[c.chunk_id for c in new_chunks],           # 고유 ID
        embeddings=new_embeddings,                        # 벡터
        documents=[c.text for c in new_chunks],          # 원문 텍스트
        metadatas=[                                       # 검색 결과에 함께 반환될 메타데이터
            {
                "pmid": c.pmid,
                "title": c.title,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
            }
            for c in new_chunks
        ],
    )
    print(f"  저장 완료: {len(new_chunks)}개 청크")


def query(embedding: list[float], top_k: int) -> list[dict]:
    """
    쿼리 벡터와 가장 유사한 청크를 top_k개 반환합니다.

    Args:
        embedding: 검색 질문의 임베딩 벡터
        top_k: 반환할 최대 결과 수

    Returns:
        유사도 순으로 정렬된 청크 목록.
        각 항목: {"text", "pmid", "title", "chunk_index", "score"}
        score: 0~1 사이 코사인 유사도 (1에 가까울수록 유사)
    """
    collection = get_collection()
    results = collection.query(
        query_embeddings=[embedding],
        # DB에 저장된 청크 수보다 많은 결과를 요청하면 오류 발생하므로 min() 사용
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "pmid": meta["pmid"],
            "title": meta["title"],
            "chunk_index": meta["chunk_index"],
            # cosine distance(0~2)를 similarity(0~1)로 변환: score = 1 - distance
            "score": round(1 - distance, 4),
        })
    return chunks


def count() -> int:
    """ChromaDB에 저장된 총 청크 수를 반환합니다."""
    return get_collection().count()
