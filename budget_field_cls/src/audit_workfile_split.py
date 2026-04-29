from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import (
    DEFAULT_CLASSIFICATION_TEXT_COLUMN,
    DEFAULT_GROUP_COLUMNS,
    WORKFILE_TEST_SPLIT_PATH,
    WORKFILE_TRAIN_SPLIT_PATH,
    WORKFILE_VAL_SPLIT_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit leakage signals across BIOFIN train/val/test splits."
    )
    parser.add_argument("--train-csv", type=Path, default=WORKFILE_TRAIN_SPLIT_PATH, help="Training split CSV.")
    parser.add_argument("--val-csv", type=Path, default=WORKFILE_VAL_SPLIT_PATH, help="Validation split CSV.")
    parser.add_argument("--test-csv", type=Path, default=WORKFILE_TEST_SPLIT_PATH, help="Test split CSV.")
    parser.add_argument(
        "--text-column",
        default=DEFAULT_CLASSIFICATION_TEXT_COLUMN,
        help="Text column to inspect for residual label mentions.",
    )
    parser.add_argument(
        "--group-columns",
        nargs="+",
        default=list(DEFAULT_GROUP_COLUMNS),
        help="Columns that should not overlap across splits.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Label column name.",
    )
    return parser.parse_args()


def normalize_value(raw_value: object) -> str:
    value = str(raw_value or "").strip()
    if value.lower() in {"", "nan", "none"}:
        return ""
    return value


def normalize_for_match(raw_value: object) -> str:
    value = normalize_value(raw_value)
    return "".join(character for character in value if character.isalnum())


def build_group_key(row: pd.Series, group_columns: list[str]) -> str:
    parts = []
    for column in group_columns:
        value = normalize_value(row.get(column, ""))
        parts.append(value if value else "__MISSING__")
    return "||".join(parts)


def count_group_overlap(left: pd.DataFrame, right: pd.DataFrame, group_columns: list[str]) -> int:
    if not group_columns:
        return 0
    left_keys = {build_group_key(row, group_columns) for _, row in left.iterrows()}
    right_keys = {build_group_key(row, group_columns) for _, row in right.iterrows()}
    return len(left_keys.intersection(right_keys))


def count_label_mentions(frame: pd.DataFrame, text_column: str, label_column: str) -> int:
    count = 0
    for row in frame.to_dict(orient="records"):
        text = normalize_for_match(row.get(text_column, ""))
        label = normalize_for_match(row.get(label_column, ""))
        if text and label and label in text:
            count += 1
    return count


def run(args: argparse.Namespace) -> int:
    train_df = pd.read_csv(args.train_csv.resolve(), encoding="utf-8-sig")
    val_df = pd.read_csv(args.val_csv.resolve(), encoding="utf-8-sig")
    test_df = pd.read_csv(args.test_csv.resolve(), encoding="utf-8-sig")

    print(f"train_rows={len(train_df)}")
    print(f"val_rows={len(val_df)}")
    print(f"test_rows={len(test_df)}")
    print(f"group_columns={args.group_columns}")
    print(f"train_val_group_overlap={count_group_overlap(train_df, val_df, args.group_columns)}")
    print(f"train_test_group_overlap={count_group_overlap(train_df, test_df, args.group_columns)}")
    print(f"val_test_group_overlap={count_group_overlap(val_df, test_df, args.group_columns)}")
    print(f"train_label_mentions={count_label_mentions(train_df, args.text_column, args.label_column)}")
    print(f"val_label_mentions={count_label_mentions(val_df, args.text_column, args.label_column)}")
    print(f"test_label_mentions={count_label_mentions(test_df, args.text_column, args.label_column)}")
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
