"""
문서 인덱싱 — abstracts.csv를 읽어 청킹·임베딩 후 ChromaDB에 저장한다.
최초 1회 또는 데이터 업데이트 시 실행한다.

실행:
    python index_documents.py
    python index_documents.py --source ../data/abstracts.csv
"""
import argparse
import shutil
from pathlib import Path

from config import SOURCE_ABSTRACTS, DATA_DIR, DB_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="abstracts.csv를 ChromaDB에 인덱싱한다.")
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_ABSTRACTS,
        help="abstracts CSV 파일 경로",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 DB를 삭제하고 처음부터 다시 인덱싱",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import pandas as pd
        from chunker import chunk_dataframe
    except ModuleNotFoundError as exc:
        print(f"ERROR: 필요한 패키지가 없습니다: {exc.name}")
        print("`conda env create -f biodiversity_rag/environment.yml`로 환경을 만든 뒤 다시 실행하세요.")
        return

    # DB 초기화
    if args.reset and DB_DIR.exists():
        shutil.rmtree(DB_DIR)
        print("기존 DB 삭제 완료")

    # 데이터 로드
    source = args.source
    if not source.exists():
        # data/ 폴더에서도 확인
        local = DATA_DIR / "abstracts.csv"
        if local.exists():
            source = local
        else:
            print(f"ERROR: 파일을 찾을 수 없습니다: {source}")
            print("llm_ner_biodiversity/data/abstracts.csv를 data/ 폴더에 복사하거나 --source 옵션을 사용하세요.")
            return

    print(f"데이터 로드: {source}")
    df = pd.read_csv(source, encoding="utf-8-sig")
    print(f"  논문 수: {len(df)}편")
    if df.empty:
        print("ERROR: 입력 CSV가 비어 있습니다. 인덱싱할 문서가 없습니다.")
        return

    # 청킹
    print("청킹 중...")
    chunks = chunk_dataframe(df)
    if not chunks:
        print("ERROR: 인덱싱 가능한 abstract가 없습니다. 빈 값 또는 결측치를 확인하세요.")
        return

    avg_chunks = len(chunks) / len(df)
    print(f"  청크 수: {len(chunks)}개 (논문당 평균 {avg_chunks:.1f}개)")

    try:
        from embedder import embed_texts
        from vector_store import add_chunks, count
    except ModuleNotFoundError as exc:
        print(f"ERROR: 필요한 패키지가 없습니다: {exc.name}")
        print("`conda env create -f biodiversity_rag/environment.yml`로 환경을 만든 뒤 다시 실행하세요.")
        return

    # 임베딩
    print("임베딩 중...")
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    # 저장
    print("ChromaDB에 저장 중...")
    add_chunks(chunks, embeddings)

    print(f"\n인덱싱 완료. 총 저장된 청크: {count()}개")


if __name__ == "__main__":
    main()
