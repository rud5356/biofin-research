"""
RAG 파이프라인 — 질문 입력 → 검색 → 생성 → 답변 출력

실행:
    python rag_pipeline.py
    python rag_pipeline.py --question "What species are found in South Korea?"
    python rag_pipeline.py --question "..." --top-k 3
"""
import argparse

from config import TOP_K


def ask(question: str, top_k: int = TOP_K) -> dict:
    """질문에 대한 RAG 답변을 반환한다."""
    from retriever import retrieve, format_context
    from generator import generate

    # 1. 검색
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return {"answer": "관련 문서를 찾지 못했습니다.", "sources": []}

    # 2. 컨텍스트 조합
    context = format_context(chunks)

    # 3. 생성
    answer = generate(question, context)

    sources = [
        {"pmid": c["pmid"], "title": c["title"], "score": c["score"]}
        for c in chunks
    ]
    return {"answer": answer, "sources": sources}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Biodiversity RAG 질의응답")
    parser.add_argument("--question", type=str, default=None, help="질문 (생략 시 대화형 모드)")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="검색할 청크 수")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from vector_store import count
    except ModuleNotFoundError as exc:
        print(f"ERROR: 필요한 패키지가 없습니다: {exc.name}")
        print("`conda env create -f biodiversity_rag/environment.yml`로 환경을 만든 뒤 다시 실행하세요.")
        return

    total = count()
    print(f"벡터DB 청크 수: {total}개")
    if total == 0:
        print("ERROR: 인덱싱이 필요합니다. index_documents.py를 먼저 실행하세요.")
        return

    if args.question:
        # 단일 질문 모드
        try:
            result = ask(args.question, top_k=args.top_k)
        except ModuleNotFoundError as exc:
            print(f"ERROR: 필요한 패키지가 없습니다: {exc.name}")
            print("`conda env create -f biodiversity_rag/environment.yml`로 환경을 만든 뒤 다시 실행하세요.")
            return
        print(f"\n질문: {args.question}")
        print(f"\n답변:\n{result['answer']}")
        print("\n참고 문서:")
        for s in result["sources"]:
            print(f"  [{s['score']:.3f}] PMID {s['pmid']} — {s['title'][:60]}")
    else:
        # 대화형 모드
        print("Biodiversity RAG (종료: q)\n")
        while True:
            question = input("질문: ").strip()
            if question.lower() in {"q", "quit", "exit"}:
                break
            if not question:
                continue

            try:
                result = ask(question, top_k=args.top_k)
            except ModuleNotFoundError as exc:
                print(f"ERROR: 필요한 패키지가 없습니다: {exc.name}")
                print("`conda env create -f biodiversity_rag/environment.yml`로 환경을 만든 뒤 다시 실행하세요.")
                return
            print(f"\n답변:\n{result['answer']}")
            print("\n참고 문서:")
            for s in result["sources"]:
                print(f"  [{s['score']:.3f}] PMID {s['pmid']} — {s['title'][:60]}")
            print()


if __name__ == "__main__":
    main()
