from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import (
    DEFAULT_CLASSIFICATION_TEXT_COLUMN,
    DEFAULT_GROUP_COLUMNS,
    DEFAULT_MIN_LABEL_COUNT,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    WORKFILE_DROPPED_LABELS_PATH,
    WORKFILE_SPLIT_SUMMARY_PATH,
    WORKFILE_TEST_SPLIT_PATH,
    WORKFILE_TEXT_DATASET_PATH,
    WORKFILE_TRAIN_SPLIT_PATH,
    WORKFILE_VAL_SPLIT_PATH,
)
from utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-aware train/val/test splits from the BIOFIN workfile text dataset."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=WORKFILE_TEXT_DATASET_PATH,
        help="Input CSV with extracted document text.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=WORKFILE_TRAIN_SPLIT_PATH,
        help="Output CSV for the training split.",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=WORKFILE_VAL_SPLIT_PATH,
        help="Output CSV for the validation split.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=WORKFILE_TEST_SPLIT_PATH,
        help="Output CSV for the test split.",
    )
    parser.add_argument(
        "--dropped-output",
        type=Path,
        default=WORKFILE_DROPPED_LABELS_PATH,
        help="Output CSV for labels that were dropped before splitting.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=WORKFILE_SPLIT_SUMMARY_PATH,
        help="Output JSON summary for split statistics.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Training split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--min-label-count",
        type=int,
        default=DEFAULT_MIN_LABEL_COUNT,
        help="Minimum row count required for a label to be kept.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Random seed for deterministic shuffling.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column containing the label name.",
    )
    parser.add_argument(
        "--label-id-column",
        default="label_id",
        help="Column containing the numeric label id.",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_CLASSIFICATION_TEXT_COLUMN,
        help="Column containing the model-ready document text.",
    )
    parser.add_argument(
        "--group-columns",
        nargs="+",
        default=list(DEFAULT_GROUP_COLUMNS),
        help="Columns that define a leakage-safe group key. Rows in the same group stay in one split.",
    )
    parser.add_argument(
        "--disable-group-split",
        action="store_true",
        help="Ignore group columns and split rows independently.",
    )
    parser.add_argument(
        "--allow-non-ok",
        action="store_true",
        help="Keep rows even if extract_status is not 'ok'.",
    )
    return parser.parse_args()


def normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")

    total = sum(ratios)
    if total <= 0:
        raise ValueError("At least one split ratio must be greater than zero.")

    return tuple(ratio / total for ratio in ratios)


def allocate_split_counts(count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    positive_indexes = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if not positive_indexes:
        raise ValueError("At least one split ratio must be greater than zero.")
    if count < len(positive_indexes):
        raise ValueError("Count is smaller than the number of required non-empty splits.")

    desired = [count * ratio for ratio in ratios]
    counts = [int(value) for value in desired]
    remaining_units = count - sum(counts)
    fractional_indexes = sorted(
        range(3),
        key=lambda index: (desired[index] - counts[index], ratios[index]),
        reverse=True,
    )
    for index in fractional_indexes:
        if remaining_units <= 0:
            break
        if ratios[index] <= 0:
            continue
        counts[index] += 1
        remaining_units -= 1

    deficit = 0
    for index in positive_indexes:
        if counts[index] == 0:
            counts[index] = 1
            deficit += 1

    if deficit > 0:
        removable_indexes = sorted(
            range(3),
            key=lambda index: (counts[index] - desired[index], counts[index]),
            reverse=True,
        )
        for index in removable_indexes:
            while deficit > 0 and counts[index] > 1:
                counts[index] -= 1
                deficit -= 1
            if deficit <= 0:
                break

    if deficit > 0 or sum(counts) != count:
        raise ValueError("Failed to allocate split counts.")

    return counts[0], counts[1], counts[2]


def filter_rows(dataframe: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    required_columns = {args.label_column, args.text_column}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    filtered = dataframe.copy()
    filtered[args.text_column] = filtered[args.text_column].fillna("").astype(str)
    filtered[args.label_column] = filtered[args.label_column].fillna("").astype(str)
    filtered = filtered[filtered[args.text_column].str.strip() != ""]
    filtered = filtered[filtered[args.label_column].str.strip() != ""]

    if not args.allow_non_ok and "extract_status" in filtered.columns:
        filtered = filtered[filtered["extract_status"] == "ok"]

    return filtered.reset_index(drop=True)


def resolve_group_columns(dataframe: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    if args.disable_group_split:
        return []

    group_columns = [column for column in args.group_columns if column]
    if not group_columns:
        return []

    missing_columns = [column for column in group_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing group columns: {missing_columns}")

    return group_columns


def normalize_group_value(raw_value: object) -> str:
    value = str(raw_value or "").strip()
    if value.lower() in {"", "nan", "none"}:
        return ""
    return value


def build_group_key(row: pd.Series, group_columns: list[str]) -> str:
    if not group_columns:
        return str(row.name)
    values = []
    for column in group_columns:
        raw_value = normalize_group_value(row.get(column, ""))
        values.append(raw_value if raw_value else "__MISSING__")
    return "||".join(values)


def split_rows_without_grouping(
    dataframe: pd.DataFrame,
    label_column: str,
    label_id_column: str,
    ratios: tuple[float, float, float],
    min_required: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    dropped_rows: list[dict[str, object]] = []

    grouped = dataframe.groupby(label_column, sort=True)
    for label, group in grouped:
        sample_count = len(group)
        label_id = group.iloc[0][label_id_column] if label_id_column in group.columns else ""

        if sample_count < min_required:
            dropped_rows.append(
                {
                    "label": label,
                    "label_id": label_id,
                    "sample_count": sample_count,
                    "group_count": sample_count,
                    "reason": f"sample_count_lt_{min_required}",
                }
            )
            continue

        shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        train_count, val_count, test_count = allocate_split_counts(sample_count, ratios)

        train_parts.append(shuffled.iloc[:train_count].copy())
        val_parts.append(shuffled.iloc[train_count : train_count + val_count].copy())
        test_parts.append(shuffled.iloc[train_count + val_count : train_count + val_count + test_count].copy())

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else dataframe.head(0).copy()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else dataframe.head(0).copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else dataframe.head(0).copy()
    dropped_df = pd.DataFrame(dropped_rows)

    return train_df, val_df, test_df, dropped_df


def split_rows_by_group(
    dataframe: pd.DataFrame,
    label_column: str,
    label_id_column: str,
    ratios: tuple[float, float, float],
    min_required: int,
    seed: int,
    group_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = dataframe.copy()
    working["split_group_key"] = working.apply(lambda row: build_group_key(row, group_columns), axis=1)

    positive_split_count = sum(1 for ratio in ratios if ratio > 0)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    dropped_rows: list[dict[str, object]] = []

    grouped = working.groupby(label_column, sort=True)
    for label, group in grouped:
        sample_count = len(group)
        label_id = group.iloc[0][label_id_column] if label_id_column in group.columns else ""
        group_table = (
            group.groupby("split_group_key", sort=False)
            .size()
            .reset_index(name="row_count")
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
        unique_group_count = len(group_table)

        if sample_count < min_required:
            dropped_rows.append(
                {
                    "label": label,
                    "label_id": label_id,
                    "sample_count": sample_count,
                    "group_count": unique_group_count,
                    "reason": f"sample_count_lt_{min_required}",
                }
            )
            continue

        if unique_group_count < positive_split_count:
            dropped_rows.append(
                {
                    "label": label,
                    "label_id": label_id,
                    "sample_count": sample_count,
                    "group_count": unique_group_count,
                    "reason": f"group_count_lt_{positive_split_count}",
                }
            )
            continue

        train_group_count, val_group_count, test_group_count = allocate_split_counts(unique_group_count, ratios)

        train_group_keys = set(group_table.iloc[:train_group_count]["split_group_key"].tolist())
        val_group_keys = set(
            group_table.iloc[train_group_count : train_group_count + val_group_count]["split_group_key"].tolist()
        )
        test_group_keys = set(
            group_table.iloc[
                train_group_count + val_group_count : train_group_count + val_group_count + test_group_count
            ]["split_group_key"].tolist()
        )

        train_parts.append(group[group["split_group_key"].isin(train_group_keys)].copy())
        val_parts.append(group[group["split_group_key"].isin(val_group_keys)].copy())
        test_parts.append(group[group["split_group_key"].isin(test_group_keys)].copy())

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else working.head(0).copy()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else working.head(0).copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else working.head(0).copy()
    dropped_df = pd.DataFrame(dropped_rows)

    return train_df, val_df, test_df, dropped_df


def split_dataframe(
    dataframe: pd.DataFrame,
    args: argparse.Namespace,
    group_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratios = normalize_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    positive_split_count = sum(1 for ratio in ratios if ratio > 0)
    min_required = max(args.min_label_count, positive_split_count)

    if group_columns:
        train_df, val_df, test_df, dropped_df = split_rows_by_group(
            dataframe,
            label_column=args.label_column,
            label_id_column=args.label_id_column,
            ratios=ratios,
            min_required=min_required,
            seed=args.seed,
            group_columns=group_columns,
        )
    else:
        train_df, val_df, test_df, dropped_df = split_rows_without_grouping(
            dataframe,
            label_column=args.label_column,
            label_id_column=args.label_id_column,
            ratios=ratios,
            min_required=min_required,
            seed=args.seed,
        )

    train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    return train_df, val_df, test_df, dropped_df


def build_summary(
    source_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dropped_df: pd.DataFrame,
    args: argparse.Namespace,
    group_columns: list[str],
) -> dict[str, object]:
    def label_counts(frame: pd.DataFrame) -> dict[str, int]:
        if frame.empty:
            return {}
        counts = frame.groupby(args.label_column).size().sort_index()
        return {str(label): int(count) for label, count in counts.items()}

    def unique_group_count(frame: pd.DataFrame) -> int:
        if frame.empty or "split_group_key" not in frame.columns:
            return 0
        return int(frame["split_group_key"].nunique())

    return {
        "input_csv": str(args.input_csv.resolve()),
        "text_column": args.text_column,
        "group_columns": group_columns,
        "row_count_after_filtering": int(len(source_df)),
        "train_row_count": int(len(train_df)),
        "val_row_count": int(len(val_df)),
        "test_row_count": int(len(test_df)),
        "train_group_count": unique_group_count(train_df),
        "val_group_count": unique_group_count(val_df),
        "test_group_count": unique_group_count(test_df),
        "dropped_label_count": int(len(dropped_df)),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "min_label_count": args.min_label_count,
        "seed": args.seed,
        "train_label_counts": label_counts(train_df),
        "val_label_counts": label_counts(val_df),
        "test_label_counts": label_counts(test_df),
    }


def run(args: argparse.Namespace) -> int:
    input_csv = args.input_csv.resolve()
    train_output = args.train_output.resolve()
    val_output = args.val_output.resolve()
    test_output = args.test_output.resolve()
    dropped_output = args.dropped_output.resolve()
    summary_output = args.summary_output.resolve()

    dataframe = pd.read_csv(input_csv, encoding="utf-8-sig")
    filtered = filter_rows(dataframe, args)
    group_columns = resolve_group_columns(filtered, args)
    train_df, val_df, test_df, dropped_df = split_dataframe(filtered, args, group_columns)
    summary = build_summary(filtered, train_df, val_df, test_df, dropped_df, args, group_columns)

    for path in (train_output, val_output, test_output, dropped_output, summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_output, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_output, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_output, index=False, encoding="utf-8-sig")
    dropped_df.to_csv(dropped_output, index=False, encoding="utf-8-sig")
    save_json(summary_output, summary)

    print(f"Saved train split: {train_output}")
    print(f"Saved val split: {val_output}")
    print(f"Saved test split: {test_output}")
    print(f"Saved dropped labels: {dropped_output}")
    print(f"Saved split summary: {summary_output}")
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
