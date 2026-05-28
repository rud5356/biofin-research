"""
match_budget_files.py 가 자동으로 변경한 파일명을 원래대로 되돌리는 스크립트.

match_budget_files.py 는 파일을 매칭할 때 이름을 정규화(rename)합니다.
이 스크립트는 그 이름 변경 기록(auto_rename_log.csv)을 읽어서 되돌립니다.

두 가지 모드:
  normalized-only : matched_normalized.csv 에 포함된 source_no 에 해당하는 파일만 복원
  all             : 이름이 변경된 모든 파일 복원

실행 예:
    python restore_renamed_files.py --execute                     # normalized-only 모드로 실제 복원
    python restore_renamed_files.py --mode all --execute          # 전체 복원
    python restore_renamed_files.py                               # dry-run (실제 변경 없이 계획만 출력)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


# CSV 읽기 시 시도할 인코딩 순서 (BOM 포함 UTF-8 → 한글 Windows → 일반 UTF-8)
CSV_ENCODINGS = ("utf-8-sig", "cp949", "utf-8")


@dataclass(frozen=True)
class RenameLogRow:
    """
    auto_rename_log.csv 한 행을 표현하는 불변(frozen) 데이터 클래스.

    frozen=True: 생성 후 필드를 변경할 수 없음 → 안전한 값 객체로 사용
    """
    old_path: Path             # 원래 파일 경로 (복원할 목적지)
    old_filename: str          # 원래 파일명
    related_source_nos: tuple[str, ...]  # 연관된 사업 번호(들)
    related_row_nos: tuple[str, ...]     # 연관된 CSV 행 번호(들)
    matched_row_count: int     # 매칭된 CSV 행 수
    status: str                # 이름 변경 상태 ('renamed' | 'already_prefixed')
    new_path: Path             # 변경된 파일 경로 (현재 존재하는 파일)
    new_filename: str          # 변경된 파일명
    note: str                  # 비고


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="match_budget_files.py 가 변경한 파일명을 원래대로 복원합니다."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Yuna\budget_matcher\output"),
        help="auto_rename_log.csv 와 matched 결과 CSV 가 저장된 폴더",
    )
    parser.add_argument(
        "--mode",
        choices=("normalized-only", "all"),
        default="normalized-only",
        help="normalized 매칭된 파일만 복원(기본값) vs 전체 복원",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="실제로 파일명을 변경합니다. 없으면 dry-run (계획만 출력).",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """CSV 파일을 여러 인코딩으로 시도하여 읽습니다."""
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"CSV 읽기 실패: {path}") from last_error


def split_multi_value(raw: str) -> tuple[str, ...]:
    """
    '|' 구분자로 연결된 여러 값을 튜플로 분리합니다.

    예) "123 | 456" → ("123", "456")
    """
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split("|") if part.strip())


def load_rename_log(path: Path) -> list[RenameLogRow]:
    """auto_rename_log.csv 를 읽어 RenameLogRow 목록으로 반환합니다."""
    rows = read_csv_rows(path)
    parsed: list[RenameLogRow] = []
    for row in rows:
        # matched_row_count 가 비어있으면 0으로 처리
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
    """
    matched_normalized.csv 에서 source_no 집합을 읽습니다.

    normalized-only 모드에서 어떤 번호에 해당하는 파일만 복원할지 결정하는 데 사용.
    """
    if not path.exists():
        raise FileNotFoundError(f"matched_normalized.csv 를 찾을 수 없습니다: {path}")
    rows = read_csv_rows(path)
    numbers: set[str] = set()
    for row in rows:
        # source_no 또는 No. 컬럼에서 번호 추출
        source_no = str(row.get("source_no", "") or row.get("No.", "")).strip()
        if source_no:
            numbers.add(source_no)
    return numbers


def should_restore(log_row: RenameLogRow, mode: str, normalized_source_nos: set[str]) -> bool:
    """
    이 파일을 복원해야 하는지 여부를 반환합니다.

    복원 대상 조건:
    - status 가 'renamed' 또는 'already_prefixed' 인 경우
    - mode='all' 이면 무조건 복원
    - mode='normalized-only' 이면 source_no 가 normalized 매칭 목록에 있어야 함
    """
    if log_row.status not in {"renamed", "already_prefixed"}:
        return False
    if mode == "all":
        return True
    return any(source_no in normalized_source_nos for source_no in log_row.related_source_nos)


def restore(log_rows: list[RenameLogRow], mode: str, execute: bool, output_dir: Path) -> dict[str, object]:
    """
    로그를 기반으로 파일명을 원래대로 복원합니다.

    execute=True 이면 실제 파일 rename 수행,
    execute=False 이면 dry-run (계획만 기록, 실제 변경 없음).
    """
    # normalized-only 모드일 때만 source_no 집합 로드
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

        # 현재 파일(변경된 이름) → 목적지(원래 이름)
        current_path = log_row.new_path
        target_path  = log_row.old_path

        row: dict[str, object] = {
            "mode": mode,
            "old_path": str(log_row.old_path),
            "new_path": str(log_row.new_path),
            "related_source_nos": " | ".join(log_row.related_source_nos),
            "related_row_nos":    " | ".join(log_row.related_row_nos),
            "original_status":    log_row.status,
            "restore_status":     "",
            "note":               "",
        }

        # 현재 파일이 존재하지 않으면 복원 불가
        if not current_path.exists():
            row["restore_status"] = "missing_current_file"
            row["note"] = "current_prefixed_file_not_found"
            planned_rows.append(row)
            status_counter[row["restore_status"]] += 1
            continue

        # 목적지(원래 이름)가 이미 존재하면 충돌
        if target_path.exists():
            row["restore_status"] = "target_already_exists"
            row["note"] = "original_filename_already_exists"
            planned_rows.append(row)
            status_counter[row["restore_status"]] += 1
            continue

        if execute:
            # 실제 rename 수행 (현재 이름 → 원래 이름)
            current_path.rename(target_path)
            row["restore_status"] = "restored"
        else:
            # dry-run: 파일 변경 없이 계획만 기록
            row["restore_status"] = "planned"

        planned_rows.append(row)
        status_counter[row["restore_status"]] += 1

    # 복원 로그 파일명은 모드에 따라 결정
    restore_log_path = output_dir / (
        "restore_normalized_rename_log.csv"
        if mode == "normalized-only"
        else "restore_all_rename_log.csv"
    )
    write_csv(restore_log_path, planned_rows)

    return {
        "mode": mode,
        "execute": execute,
        "rename_log_path": str(output_dir / "auto_rename_log.csv"),
        "restore_log_path": str(restore_log_path),
        "selected_entry_count": len(planned_rows),
        "planned_count":         status_counter.get("planned", 0),
        "restored_count":        status_counter.get("restored", 0),
        "missing_current_file_count":  status_counter.get("missing_current_file", 0),
        "target_already_exists_count": status_counter.get("target_already_exists", 0),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """딕셔너리 목록을 CSV 파일로 저장합니다."""
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    # 모든 행의 키를 순서 유지하며 합산 (dict.fromkeys 로 중복 제거)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """파일명 복원 스크립트 진입점."""
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
        # 결과를 JSON 형식으로 출력
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
