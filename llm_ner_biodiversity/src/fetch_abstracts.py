"""
PubMed에서 생물다양성 관련 논문 초록을 수집하여 CSV로 저장하는 스크립트.

PubMed는 미국 국립의학도서관(NLM)이 운영하는 의생명과학 논문 데이터베이스입니다.
이 스크립트는 키워드 검색으로 논문을 찾고, 생물다양성과 무관한 논문을 필터링합니다.

사용법:
    python fetch_abstracts.py
    python fetch_abstracts.py --keyword "endangered species Korea" --limit 200
    python fetch_abstracts.py --no-filter-biodiversity  # 필터링 없이 전체 수집
"""

import argparse
from pathlib import Path

import pandas as pd

from config import DATA_DIR, DEFAULT_KEYWORD, DEFAULT_LIMIT
from ner_pipeline import ensure_directory
from pubmed_client import fetch_abstracts


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="PubMed에서 생물다양성 논문 초록을 수집하여 CSV로 저장합니다."
    )
    parser.add_argument(
        "--keyword",
        default=DEFAULT_KEYWORD,
        help=f"PubMed 검색 키워드 (기본값: '{DEFAULT_KEYWORD}')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"수집할 최대 논문 수 (기본값: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="abstracts.csv를 저장할 폴더 (기본값: data/)",
    )
    parser.add_argument(
        "--filter-biodiversity",
        action="store_true",
        default=True,
        help=(
            "의학·임상 등 생물다양성과 무관한 논문을 제거하고 "
            "종, 생태, 야생동물 관련 논문만 저장합니다 (기본값: True)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """PubMed 초록 수집 실행."""
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)

    print(f"PubMed 초록 수집 중 (키워드='{args.keyword}', 최대 {args.limit}건)...")
    abstracts = fetch_abstracts(
        keyword=args.keyword,
        limit=args.limit,
        filter_biodiversity=args.filter_biodiversity,
    )
    print(f"수집 완료: {len(abstracts)}건")

    # ID, 제목, 초록 열만 저장 (NER 파이프라인에서 이 형식을 기대합니다)
    output_path = output_dir / "abstracts.csv"
    pd.DataFrame(abstracts)[["id", "title", "abstract"]].to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )
    print(f"저장 완료: {output_path}")


if __name__ == "__main__":
    main()
