from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import WORKFILE_DATASET_PATH, WORKFILE_TEXT_DATASET_PATH
from document_text import count_words, extract_document_text
from utils import console_safe
from workfile_text_processing import build_model_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract document text from HWP/PDF files and build a text dataset."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=WORKFILE_DATASET_PATH,
        help="Input dataset CSV with file paths.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=WORKFILE_TEXT_DATASET_PATH,
        help="Output CSV with extracted text.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N rows. 0 means all rows.",
    )
    return parser.parse_args()


def build_text_rows(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = len(dataframe)

    for index, raw_row in enumerate(dataframe.to_dict(orient="records"), start=1):
        file_path = Path(str(raw_row["file_path"]))
        if index == 1 or index == total or index % 50 == 0:
            print(f"[{index}/{total}] Extracting text: {console_safe(file_path.name)}")

        text = ""
        extract_method = ""
        extract_status = "ok"
        extract_error = ""

        try:
            if not file_path.exists():
                extract_status = "missing_file"
            else:
                text, extract_method = extract_document_text(file_path)
                if not text:
                    extract_status = "empty_text"
        except Exception as exc:  # pragma: no cover - file level error path
            extract_status = "extract_error"
            extract_error = str(exc)

        row = dict(raw_row)
        model_text, model_text_method = build_model_text(text, row)
        row.update(
            {
                "extract_status": extract_status,
                "extract_method": extract_method,
                "extract_error": extract_error,
                "text_char_count": len(text),
                "text_word_count": count_words(text),
                "text": text,
                "model_text_method": model_text_method,
                "model_text_char_count": len(model_text),
                "model_text_word_count": count_words(model_text),
                "model_text": model_text,
            }
        )
        rows.append(row)

    return rows


def run(args: argparse.Namespace) -> int:
    input_csv = args.input_csv.resolve()
    output_csv = args.output_csv.resolve()

    dataframe = pd.read_csv(input_csv, encoding="utf-8-sig")
    if args.limit and args.limit > 0:
        dataframe = dataframe.head(args.limit)

    rows = build_text_rows(dataframe)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved text dataset: {output_csv}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
