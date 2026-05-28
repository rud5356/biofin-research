"""
질문을 벡터로 변환하고 ChromaDB에서 유사한 문서 청크를 검색하는 모듈.

RAG 파이프라인에서 "검색(Retrieval)" 단계를 담당합니다:
  질문 텍스트 → 임베딩 벡터 → 벡터 DB 유사도 검색 → 관련 청크 반환
"""

from config import TOP_K
from embedder import embed_query
from vector_store import query


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    질문과 의미적으로 가장 유사한 논문 청크를 검색하여 반환합니다.

    Args:
        question: 사용자의 자연어 질문
        top_k: 반환할 최대 청크 수 (기본값: config.TOP_K)

    Returns:
        유사도 순으로 정렬된 청크 목록.
        각 항목: {"text": ..., "pmid": ..., "title": ..., "chunk_index": ..., "score": ...}
    """
    # 1단계: 질문 텍스트를 벡터로 변환
    query_embedding = embed_query(question)
    # 2단계: 벡터 DB에서 가장 가까운 벡터를 가진 청크들을 검색
    results = query(query_embedding, top_k=top_k)
    return results


def format_context(chunks: list[dict]) -> str:
    """
    검색된 청크들을 LLM 프롬프트에 넣을 수 있는 형식의 문자열로 조합합니다.

    각 청크 앞에 번호와 PMID(논문 고유 ID), 제목을 붙여
    LLM이 출처를 파악하기 쉽게 합니다.

    예시 출력:
        [1] (PMID: 12345678) Biodiversity in Korean wetlands
        This study examines...

        [2] (PMID: 87654321) ...
    """
    formatted_parts = []
    for index, chunk in enumerate(chunks, start=1):
        formatted_parts.append(
            f"[{index}] (PMID: {chunk['pmid']}) {chunk['title']}\n{chunk['text']}"
        )
    # 각 청크 사이에 빈 줄을 넣어 구분
    return "\n\n".join(formatted_parts)
