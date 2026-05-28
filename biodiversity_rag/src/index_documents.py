"""
논문 초록 CSV를 읽어 청킹·임베딩 후 ChromaDB에 저장하는 인덱싱 스크립트.

RAG 시스템을 처음 구축하거나 데이터가 추가될 때 실행합니다.
한 번 인덱싱하면 rag_pipeline.py에서 검색에 활용할 수 있습니다.

실행 순서:
  1. fetch_abstracts.py → abstracts.csv 수집
  2. index_documents.py → ChromaDB에 인덱싱  ← 현재 파일
  3. rag_pipeline.py    → 질의응답

사용법:
    python index_documents.py
    python index_documents.py --source ../data/abstracts.csv
    python index_documents.py --reset  # 기존 DB 삭제 후 처음부터 인덱싱
"""

import argparse
import shutil
from pathlib import Path

from config import DATA_DIR, DB_DIR, SOURCE_ABSTRACTS


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="abstracts.csv를 청킹·임베딩하여 ChromaDB에 인덱싱합니다."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_ABSTRACTS,
        help="abstracts CSV 파일 경로",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 ChromaDB를 삭제하고 처음부터 다시 인덱싱합니다.",
    )
    return parser.parse_args()


def main() -> None:
    """인덱싱 파이프라인 실행: 로드 → 청킹 → 임베딩 → 저장."""
    args = parse_args()

    try:
        import pandas as pd
        from chunker import chunk_dataframe
    except ModuleNotFoundError as exc:
        print(f"오류: 필요한 패키지가 없습니다: {exc.name}")
        print("`conda env create -f biodiversity_rag/environment.yml`로 환경을 만든 뒤 다시 실행하세요.")
        return

    # --reset 옵션: 기존 벡터 DB를 완전히 삭제하고 새로 시작
    if args.reset and DB_DIR.exists():
        shutil.rmtree(DB_DIR)
        print("기존 ChromaDB 삭제 완료")

    # abstracts.csv 파일을 찾습니다 (지정 경로 → data/ 폴더 순서로 확인)
    source_path = args.source
    if not source_path.exists():
        local_path = DATA_DIR / "abstracts.csv"
        if local_path.exists():
            source_path = local_path
        else:
            print(f"오류: 파일을 찾을 수 없습니다: {source_path}")
            print("llm_ner_biodiversity/data/abstracts.csv를 data/ 폴더에 복사하거나 --source 옵션을 사용하세요.")
            return

    # 1단계: 데이터 로드
    print(f"데이터 로드: {source_path}")
    df = pd.read_csv(source_path, encoding="utf-8-sig")
    print(f"  논문 수: {len(df)}편")
    if df.empty:
        print("오류: 입력 CSV가 비어 있습니다.")
        return

    # 2단계: 청킹 (긴 초록을 문장 단위 청크로 분할)
    print("청킹 중...")
    chunks = chunk_dataframe(df)
    if not chunks:
        print("오류: 인덱싱 가능한 초록이 없습니다. 빈 값 또는 결측치를 확인하세요.")
        return

    avg_chunks_per_paper = len(chunks) / len(df)
    print(f"  청크 수: {len(chunks)}개 (논문당 평균 {avg_chunks_per_paper:.1f}개)")

    try:
        from embedder import embed_texts
        from vector_store import add_chunks, count
    except ModuleNotFoundError as exc:
        print(f"오류: 필요한 패키지가 없습니다: {exc.name}")
        print("`conda env create -f biodiversity_rag/environment.yml`로 환경을 만든 뒤 다시 실행하세요.")
        return

    # 3단계: 임베딩 (각 청크를 벡터로 변환)
    print("임베딩 중...")
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(chunk_texts)

    # 4단계: ChromaDB에 저장
    print("ChromaDB에 저장 중...")
    add_chunks(chunks, embeddings)

    print(f"\n인덱싱 완료. 총 저장된 청크: {count()}개")


if __name__ == "__main__":
    main()
