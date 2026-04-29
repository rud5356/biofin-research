"""
질문을 임베딩으로 변환하고 유사한 청크를 검색한다.
"""
from embedder import embed_query
from vector_store import query
from config import TOP_K


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """질문과 유사한 청크를 반환한다.

    Returns:
        list of {text, pmid, title, chunk_index, score}
    """
    q_embedding = embed_query(question)
    results = query(q_embedding, top_k=top_k)
    return results


def format_context(chunks: list[dict]) -> str:
    """검색된 청크들을 LLM 프롬프트용 컨텍스트 문자열로 조합한다."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] (PMID: {chunk['pmid']}) {chunk['title']}\n{chunk['text']}"
        )
    return "\n\n".join(parts)
