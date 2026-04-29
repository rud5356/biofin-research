"""
ChromaDB를 사용해 청크를 저장하고 조회한다.
"""
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
except ModuleNotFoundError as exc:
    chromadb = None
    Settings = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from chunker import Chunk
from config import DB_DIR, CHROMA_COLLECTION


def get_collection(db_dir: Path = DB_DIR) -> "chromadb.Collection":
    """ChromaDB 컬렉션을 반환한다. 없으면 새로 생성한다."""
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
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """청크와 임베딩을 ChromaDB에 저장한다. 이미 존재하는 id는 건너뛴다."""
    collection = get_collection()

    # 중복 제거: 이미 저장된 id 확인
    existing = set(collection.get(ids=[c.chunk_id for c in chunks])["ids"])
    new_chunks = [c for c in chunks if c.chunk_id not in existing]
    new_embeddings = [embeddings[i] for i, c in enumerate(chunks) if c.chunk_id not in existing]

    if not new_chunks:
        print("  모든 청크가 이미 저장돼 있습니다.")
        return

    collection.add(
        ids=[c.chunk_id for c in new_chunks],
        embeddings=new_embeddings,
        documents=[c.text for c in new_chunks],
        metadatas=[
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
    """임베딩과 유사한 청크를 top_k개 반환한다."""
    collection = get_collection()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "pmid": meta["pmid"],
            "title": meta["title"],
            "chunk_index": meta["chunk_index"],
            "score": round(1 - dist, 4),  # cosine distance → similarity
        })
    return chunks


def count() -> int:
    """저장된 청크 수를 반환한다."""
    return get_collection().count()
