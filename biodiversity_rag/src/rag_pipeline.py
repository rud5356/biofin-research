"""
RAG(Retrieval-Augmented Generation) 질의응답 파이프라인.

RAG 동작 방식 (3단계):
  1. 검색(Retrieval): 질문을 벡터로 변환 → ChromaDB에서 유사한 논문 청크 검색
  2. 조합(Augmentation): 검색된 청크들을 LLM 프롬프트에 컨텍스트로 삽입
  3. 생성(Generation): LLM이 컨텍스트를 바탕으로 답변 생성

단일 질문 모드와 대화형(interactive) 모드 모두 지원합니다.

사용법:
    python rag_pipeline.py
    python rag_pipeline.py --question "What species are found in South Korea?"
    python rag_pipeline.py --question "..." --top-k 3
"""

import argparse

from config import TOP_K


def ask(question: str, top_k: int = TOP_K) -> dict:
    """
    질문에 대한 RAG 답변을 생성합니다.

    Args:
        question: 사용자의 자연어 질문
        top_k: 검색할 논문 청크 수

    Returns:
        {"answer": 답변 텍스트, "sources": 참고 논문 목록}
        관련 문서가 없으면 answer에 안내 메시지 반환
    """
    from generator import generate
    from retriever import format_context, retrieve

    # 1단계: 질문과 유사한 논문 청크 검색
    retrieved_chunks = retrieve(question, top_k=top_k)

    if not retrieved_chunks:
        return {"answer": "관련 문서를 찾지 못했습니다.", "sources": []}

    # 2단계: 검색된 청크들을 LLM 프롬프트용 컨텍스트 문자열로 조합
    context = format_context(retrieved_chunks)

    # 3단계: LLM이 컨텍스트를 참고하여 답변 생성
    answer = generate(question, context)

    # 출처 정보 (PMID, 제목, 유사도 점수)
    sources = [
        {"pmid": chunk["pmid"], "title": chunk["title"], "score": chunk["score"]}
        for chunk in retrieved_chunks
    ]
    return {"answer": answer, "sources": sources}


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description="생물다양성 논문 기반 RAG 질의응답 시스템")
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="질문 (생략 시 대화형 모드로 실행)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"검색할 청크 수 (기본값: {TOP_K})",
    )
    return parser.parse_args()


def main() -> None:
    """RAG 파이프라인 실행: 단일 질문 또는 대화형 모드."""
    args = parse_args()

    try:
        from vector_store import count
    except ModuleNotFoundError as exc:
        print(f"오류: 필요한 패키지가 없습니다: {exc.name}")
        print("`conda env create -f biodiversity_rag/environment.yml`로 환경을 만든 뒤 다시 실행하세요.")
        return

    total_chunks = count()
    print(f"벡터DB 청크 수: {total_chunks}개")
    if total_chunks == 0:
        print("오류: 인덱싱이 필요합니다. index_documents.py를 먼저 실행하세요.")
        return

    def _print_result(result: dict) -> None:
        """결과를 터미널에 출력하는 내부 헬퍼 함수."""
        print(f"\n답변:\n{result['answer']}")
        print("\n참고 문서:")
        for source in result["sources"]:
            print(f"  [{source['score']:.3f}] PMID {source['pmid']} — {source['title'][:60]}")

    if args.question:
        # 단일 질문 모드: 질문 하나만 처리하고 종료
        try:
            result = ask(args.question, top_k=args.top_k)
        except ModuleNotFoundError as exc:
            print(f"오류: 필요한 패키지가 없습니다: {exc.name}")
            print("`conda env create -f biodiversity_rag/environment.yml`로 환경을 만든 뒤 다시 실행하세요.")
            return
        print(f"\n질문: {args.question}")
        _print_result(result)
    else:
        # 대화형 모드: 사용자가 'q'를 입력할 때까지 반복
        print("생물다양성 RAG 질의응답 시스템 (종료: q)\n")
        while True:
            question = input("질문: ").strip()
            if question.lower() in {"q", "quit", "exit"}:
                break
            if not question:
                continue

            try:
                result = ask(question, top_k=args.top_k)
            except ModuleNotFoundError as exc:
                print(f"오류: 필요한 패키지가 없습니다: {exc.name}")
                print("`conda env create -f biodiversity_rag/environment.yml`로 환경을 만든 뒤 다시 실행하세요.")
                return
            _print_result(result)
            print()


if __name__ == "__main__":
    main()
