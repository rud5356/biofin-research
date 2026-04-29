import argparse
import pandas as pd
from pathlib import Path
from config import DEFAULT_KEYWORD, DEFAULT_LIMIT, DATA_DIR
from pubmed_client import fetch_abstracts
from ner_pipeline import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch abstracts from PubMed and save to CSV.")
    parser.add_argument("--keyword", default=DEFAULT_KEYWORD, help="PubMed literature search keyword.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of records to fetch.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory to save abstracts.csv.",
    )
    parser.add_argument(
        "--filter-biodiversity",
        action="store_true",
        default=True,
        help=(
            "Filter out non-biodiversity papers (biomedical, clinical, etc.) "
            "and save only papers relevant to species, ecology, or wildlife."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)

    print(f"Fetching abstracts from PubMed (keyword='{args.keyword}', limit={args.limit})...")
    abstracts = fetch_abstracts(
        keyword=args.keyword,
        limit=args.limit,
        filter_biodiversity=args.filter_biodiversity,
    )
    print(f"Fetched: {len(abstracts)} abstracts")

    abstracts_path = output_dir / "abstracts.csv"
    pd.DataFrame(abstracts)[["id", "title", "abstract"]].to_csv(
        abstracts_path, index=False, encoding="utf-8-sig"
    )
    print(f"Saved: {abstracts_path}")


if __name__ == "__main__":
    main()
