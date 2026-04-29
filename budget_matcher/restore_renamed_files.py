from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


CSV_ENCODINGS = ("utf-8-sig", "cp949", "utf-8")


@dataclass(frozen=True)
class RenameLogRow:
    old_path: Path
    old_filename: str
    related_source_nos: tuple[str, ...]
    related_row_nos: tuple[str, ...]
    matched_row_count: int
    status: str
    new_path: Path
    new_filename: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore filenames previously changed by match_budget_files.py."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Yuna\budget_matcher\output"),
        help="Directory that contains auto_rename_log.csv and matched result CSVs.",
    )
    parser.add_argument(
        "--mode",
        choices=("normalized-only", "all"),
        default="normalized-only",
        help="Restore only files tied to normalized matches, or restore all renamed files.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually rename files. Without this flag the script only reports what would change.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Failed to read CSV {path}") from last_error


def split_multi_value(raw: str) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split("|") if part.strip())


def load_rename_log(path: Path) -> list[RenameLogRow]:
    rows = read_csv_rows(path)
    parsed: list[RenameLogRow] = []
    for row in rows:
        matched_row_count_raw = str(row.get("matched_row_count", "")).strip() or "0"
        parsed.append(
            RenameLogRow(
                old_path=Path(str(row.get("old_path", "")).strip()),
                old_filename=str(row.get("old_filename", "")).strip(),
                related_source_nos=split_multi_value(str(row.get("related_source_nos", "")).strip()),
                related_row_nos=split_multi_value(str(row.get("related_row_nos", "")).strip()),
                matched_row_count=int(matched_row_count_raw),
                status=str(row.get("status", "")).strip(),
                new_path=Path(str(row.get("new_path", "")).strip()),
                new_filename=str(row.get("new_filename", "")).strip(),
                note=str(row.get("note", "")).strip(),
            )
        )
    return parsed


def load_normalized_source_numbers(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"matched_normalized.csv not found: {path}")
    rows = read_csv_rows(path)
    numbers: set[str] = set()
    for row in rows:
        source_no = str(row.get("source_no", "") or row.get("No.", "")).strip()
        if source_no:
            numbers.add(source_no)
    return numbers


def should_restore(log_row: RenameLogRow, mode: str, normalized_source_nos: set[str]) -> bool:
    if log_row.status not in {"renamed", "already_prefixed"}:
        return False
    if mode == "all":
        return True
    return any(source_no in normalized_source_nos for source_no in log_row.related_source_nos)


def restore(log_rows: list[RenameLogRow], mode: str, execute: bool, output_dir: Path) -> dict[str, object]:
    normalized_source_nos = (
        load_normalized_source_numbers(output_dir / "matched_normalized.csv")
        if mode == "normalized-only"
        else set()
    )

    planned_rows: list[dict[str, object]] = []
    status_counter: Counter[str] = Counter()

    for log_row in log_rows:
        if not should_restore(log_row, mode, normalized_source_nos):
            continue

        current_path = log_row.new_path
        target_path = log_row.old_path
        row: dict[str, object] = {
            "mode": mode,
            "old_path": str(log_row.old_path),
            "new_path": str(log_row.new_path),
            "related_source_nos": " | ".join(log_row.related_source_nos),
            "related_row_nos": " | ".join(log_row.related_row_nos),
            "original_status": log_row.status,
            "restore_status": "",
            "note": "",
        }

        if not current_path.exists():
            row["restore_status"] = "missing_current_file"
            row["note"] = "current_prefixed_file_not_found"
            planned_rows.append(row)
            status_counter[row["restore_status"]] += 1
            continue

        if target_path.exists():
            row["restore_status"] = "target_already_exists"
            row["note"] = "original_filename_already_exists"
            planned_rows.append(row)
            status_counter[row["restore_status"]] += 1
            continue

        if execute:
            current_path.rename(target_path)
            row["restore_status"] = "restored"
        else:
            row["restore_status"] = "planned"
        planned_rows.append(row)
        status_counter[row["restore_status"]] += 1

    restore_log_path = output_dir / (
        "restore_normalized_rename_log.csv"
        if mode == "normalized-only"
        else "restore_all_rename_log.csv"
    )
    write_csv(restore_log_path, planned_rows)

    summary = {
        "mode": mode,
        "execute": execute,
        "rename_log_path": str(output_dir / "auto_rename_log.csv"),
        "restore_log_path": str(restore_log_path),
        "selected_entry_count": len(planned_rows),
        "planned_count": status_counter.get("planned", 0),
        "restored_count": status_counter.get("restored", 0),
        "missing_current_file_count": status_counter.get("missing_current_file", 0),
        "target_already_exists_count": status_counter.get("target_already_exists", 0),
    }
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    try:
        output_dir = args.output_dir.resolve()
        log_rows = load_rename_log(output_dir / "auto_rename_log.csv")
        summary = restore(
            log_rows=log_rows,
            mode=args.mode,
            execute=args.execute,
            output_dir=output_dir,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
