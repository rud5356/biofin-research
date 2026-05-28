"""
workfile.xlsx 에서 파일 분류 데이터셋을 구축하는 스크립트.

동작 순서:
  1. workfile.xlsx (match_workfile 시트) 읽기
  2. budget_root 폴더 재귀 탐색 — HWP/PDF 파일 인덱스 구성
  3. matched_filename → 실제 파일 경로 해결 (1:1 매칭 또는 분야 폴더 기준)
  4. 데이터셋 CSV + 라벨 CSV + 요약 JSON + 미해결 CSV 저장

budget_field_cls/src/build_workfile_dataset.py 와 유사하지만
  - XLSX 전용 (CSV/TSV 미지원)
  - 출력 경로가 workfile.xlsx 위치 기준으로 자동 결정됨

실행 예:
    python build_workfile_dataset.py
    python build_workfile_dataset.py --budget-root /path/to/v2
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


# ─── 기본 경로 설정 ───────────────────────────────────────────────────────────
DEFAULT_BUDGET_ROOT      = Path(r"C:\Yuna\국가생물다양성_열린재정 데이터_v2")
DEFAULT_WORKFILE         = DEFAULT_BUDGET_ROOT / "workfile.xlsx"
DEFAULT_SHEET_NAME       = "match_workfile"
DEFAULT_EXTENSIONS       = (".hwp", ".pdf")
# 포함할 자동 매칭 상태
DEFAULT_INCLUDE_STATUSES = ("exact", "manual", "normalized")
# 유효하지 않은 파일명 (이 값이면 해결 불가로 처리)
PLACEHOLDER_FILENAMES    = {"x", "X", "-", "없음", "미입력", "na", "n/a"}


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="workfile.xlsx 에서 파일 분류 데이터셋을 구축합니다."
    )
    parser.add_argument("--workfile",    type=Path, default=DEFAULT_WORKFILE,    help="workfile.xlsx 경로")
    parser.add_argument("--sheet-name",             default=DEFAULT_SHEET_NAME,  help="Excel 시트 이름")
    parser.add_argument("--budget-root", type=Path, default=DEFAULT_BUDGET_ROOT, help="분야 폴더가 있는 루트 폴더")
    # 출력 경로 미지정 시 workfile.xlsx 위치에 저장
    parser.add_argument("--output-csv",     type=Path, default=None, help="데이터셋 CSV 경로 (기본: workfile.xlsx 폴더)")
    parser.add_argument("--labels-csv",     type=Path, default=None, help="라벨 CSV 경로")
    parser.add_argument("--summary-json",   type=Path, default=None, help="요약 JSON 경로")
    parser.add_argument("--unresolved-csv", type=Path, default=None, help="미해결 행 CSV 경로")
    parser.add_argument("--extensions",     nargs="+", default=list(DEFAULT_EXTENSIONS), help="스캔할 파일 확장자")
    parser.add_argument("--include-statuses", nargs="+", default=list(DEFAULT_INCLUDE_STATUSES), help="포함할 auto_match_status 값")
    parser.add_argument(
        "--keep-duplicate-paths",
        action="store_true",
        help="같은 파일 경로 중복 행을 유지합니다 (기본: 첫 번째만 유지)",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    """
    문자열을 정규화하여 폴더명 비교에 사용합니다.

    NFKC 정규화 → 소문자 → 영숫자·한글 이외 제거
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return re.sub(r"[^0-9a-z가-힣]+", "", text)


def read_workfile(workfile_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    workfile.xlsx 를 읽어 DataFrame 으로 반환합니다.

    필수 컬럼: 분야명, 세부사업명, auto_match_status, matched_filename
    """
    if not workfile_path.exists():
        raise FileNotFoundError(f"workfile 을 찾을 수 없습니다: {workfile_path}")

    dataframe        = pd.read_excel(workfile_path, sheet_name=sheet_name)
    required_columns = {"분야명", "세부사업명", "auto_match_status", "matched_filename"}
    missing_columns  = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"workfile 에 필수 컬럼이 없습니다: {', '.join(sorted(missing_columns))}")
    return dataframe


def build_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """
    출력 경로를 결정합니다.

    미지정 옵션은 workfile.xlsx 파일이 있는 폴더에 기본 이름으로 저장됩니다.
    """
    base_dir    = args.workfile.resolve().parent
    output_csv  = args.output_csv.resolve()   if args.output_csv   else base_dir / "workfile_dataset.csv"
    labels_csv  = args.labels_csv.resolve()   if args.labels_csv   else base_dir / "workfile_labels.csv"
    summary_json = (
        args.summary_json.resolve() if args.summary_json
        else base_dir / "workfile_dataset_summary.json"
    )
    unresolved_csv = (
        args.unresolved_csv.resolve() if args.unresolved_csv
        else base_dir / "workfile_dataset_unresolved.csv"
    )
    return output_csv, labels_csv, summary_json, unresolved_csv


def scan_files(
    budget_root: Path,
    extensions:  Iterable[str],
) -> dict[str, list[Path]]:
    """
    budget_root 를 재귀 탐색하여 파일명 → 경로 목록 인덱스를 반환합니다.

    동일한 파일명이 여러 분야 폴더에 있을 수 있으므로 목록으로 관리합니다.
    """
    normalized_extensions = {
        ext.lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
        for ext in extensions
    }
    files_by_name: dict[str, list[Path]] = defaultdict(list)

    if not budget_root.exists():
        raise FileNotFoundError(f"루트 폴더를 찾을 수 없습니다: {budget_root}")

    for path in budget_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in normalized_extensions:
            continue
        files_by_name[path.name].append(path.resolve())

    return files_by_name


def resolve_file_path(
    matched_filename:  str,
    budget_field_name: str,
    files_by_name:     dict[str, list[Path]],
) -> tuple[Path | None, str]:
    """
    matched_filename 으로 실제 파일 경로를 해결합니다.

    해결 순서:
    1. 파일이 1개뿐 → unique_filename
    2. 파일이 여러 개 → 분야명으로 상위 폴더 필터
       - 1개 일치 → filename_and_field_folder
       - 2개 이상 → multiple_candidates_same_field (실패)
    3. 분야 폴더 매칭 불가 → multiple_candidates_across_folders (실패)
    """
    candidates = sorted(files_by_name.get(matched_filename, []))
    if not candidates:
        return None, "filename_not_found"
    if len(candidates) == 1:
        return candidates[0], "unique_filename"

    field_key      = normalize_text(budget_field_name)
    folder_matched = [
        path for path in candidates if normalize_text(path.parent.name) == field_key
    ]
    if len(folder_matched) == 1:
        return folder_matched[0], "filename_and_field_folder"
    if len(folder_matched) > 1:
        return None, "multiple_candidates_same_field"

    return None, "multiple_candidates_across_folders"


def safe_relative_path(path: Path, root: Path) -> str:
    """path 가 root 내에 있으면 상대 경로, 없으면 절대 경로를 반환합니다."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_dataset_rows(
    dataframe:        pd.DataFrame,
    budget_root:      Path,
    files_by_name:    dict[str, list[Path]],
    include_statuses: set[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """작업표 각 행의 파일 경로를 해결하고 데이터셋/미해결 행을 반환합니다."""
    dataset_rows:    list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []

    for _, raw_row in dataframe.iterrows():
        matched_filename  = str(raw_row.get("matched_filename", "") or "").strip()
        auto_match_status = str(raw_row.get("auto_match_status", "") or "").strip().lower()

        if not matched_filename:
            continue
        if auto_match_status and auto_match_status not in include_statuses:
            continue

        budget_field_name = str(raw_row.get("분야명", "") or "").strip()

        if matched_filename in PLACEHOLDER_FILENAMES:
            unresolved_rows.append({
                "row_no":            raw_row.get("row_no", ""),
                "source_no":         raw_row.get("source_no", ""),
                "분야명":             budget_field_name,
                "세부사업명":         raw_row.get("세부사업명", ""),
                "auto_match_status": auto_match_status,
                "matched_filename":  matched_filename,
                "resolution_status": "placeholder_filename",
            })
            continue

        resolved_path, resolution_status = resolve_file_path(
            matched_filename=matched_filename,
            budget_field_name=budget_field_name,
            files_by_name=files_by_name,
        )

        if resolved_path is None:
            unresolved_rows.append({
                "row_no":            raw_row.get("row_no", ""),
                "source_no":         raw_row.get("source_no", ""),
                "분야명":             budget_field_name,
                "세부사업명":         raw_row.get("세부사업명", ""),
                "auto_match_status": auto_match_status,
                "matched_filename":  matched_filename,
                "resolution_status": resolution_status,
            })
            continue

        folder_label = resolved_path.parent.name  # 파일이 있는 폴더 이름 = 분야 라벨
        dataset_rows.append({
            "row_no":                     raw_row.get("row_no", ""),
            "source_no":                  raw_row.get("source_no", ""),
            "fiscal_year":                raw_row.get("회계연도", ""),
            "ministry_name":              raw_row.get("소관명", ""),
            "account_name":               raw_row.get("회계코드명", ""),
            "detail_account_name":        raw_row.get("계정명", ""),
            "budget_field_name":          budget_field_name,
            "sector_name":                raw_row.get("부문명", ""),
            "program_name":               raw_row.get("프로그램명", ""),
            "unit_project_name":          raw_row.get("단위사업명", ""),
            "detail_project_name":        raw_row.get("세부사업명", ""),
            "auto_match_status":          auto_match_status,
            "matched_filename":           matched_filename,
            "resolution_status":          resolution_status,
            "file_path":                  str(resolved_path),
            "relative_path":              safe_relative_path(resolved_path, budget_root),
            "file_ext":                   resolved_path.suffix.lower(),
            "is_hwp":                     resolved_path.suffix.lower() == ".hwp",
            "is_pdf":                     resolved_path.suffix.lower() == ".pdf",
            "label":                      folder_label,
            "label_matches_budget_field": normalize_text(folder_label) == normalize_text(budget_field_name),
        })

    return dataset_rows, unresolved_rows


def assign_label_ids(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """라벨 문자열에 정수 ID를 부여하고 라벨 매핑 목록을 반환합니다."""
    label_counts = Counter(str(row["label"]) for row in rows)
    labels       = sorted(label_counts)
    label_to_id  = {label: index for index, label in enumerate(labels)}

    for row in rows:
        row["label_id"] = label_to_id[str(row["label"])]

    return [
        {
            "label_id":     label_to_id[label],
            "label":        label,
            "sample_count": label_counts[label],
        }
        for label in labels
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """딕셔너리 목록을 CSV 파일로 저장합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(path, index=False, encoding="utf-8-sig")


def build_summary(
    workfile_path:      Path,
    budget_root:        Path,
    dataset_rows:       list[dict[str, object]],
    unresolved_rows:    list[dict[str, object]],
    original_dataframe: pd.DataFrame,
    include_statuses:   set[str],
) -> dict[str, object]:
    """처리 결과 통계를 딕셔너리로 반환합니다."""
    status_counts           = Counter(str(row.get("auto_match_status", ""))  for row in dataset_rows)
    extension_counts        = Counter(str(row.get("file_ext", ""))           for row in dataset_rows)
    label_counts            = Counter(str(row.get("label", ""))              for row in dataset_rows)
    unresolved_status_counts = Counter(str(row.get("resolution_status", "")) for row in unresolved_rows)
    duplicate_path_count    = len(dataset_rows) - len({str(row.get("file_path", "")) for row in dataset_rows})

    return {
        "workfile_path":                str(workfile_path),
        "budget_root":                  str(budget_root),
        "original_row_count":           int(len(original_dataframe)),
        "dataset_row_count":            len(dataset_rows),
        "unresolved_row_count":         len(unresolved_rows),
        "label_count":                  len(label_counts),
        "duplicate_file_path_row_count": duplicate_path_count,
        "included_statuses":            sorted(include_statuses),
        "status_counts":                dict(sorted(status_counts.items())),
        "extension_counts":             dict(sorted(extension_counts.items())),
        "label_counts":                 dict(sorted(label_counts.items())),
        "unresolved_status_counts":     dict(sorted(unresolved_status_counts.items())),
    }


def drop_duplicate_paths(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """같은 file_path 의 중복 행을 제거합니다 (처음 등장한 행만 유지)."""
    deduplicated: list[dict[str, object]] = []
    seen_paths:   set[str] = set()
    for row in rows:
        file_path = str(row.get("file_path", ""))
        if file_path in seen_paths:
            continue
        seen_paths.add(file_path)
        deduplicated.append(row)
    return deduplicated


def run(args: argparse.Namespace) -> int:
    """데이터셋 구축 파이프라인을 실행합니다."""
    workfile_path                                = args.workfile.resolve()
    budget_root                                  = args.budget_root.resolve()
    output_csv, labels_csv, summary_json, unresolved_csv = build_output_paths(args)

    dataframe        = read_workfile(workfile_path, args.sheet_name)
    files_by_name    = scan_files(budget_root, args.extensions)
    include_statuses = {status.strip().lower() for status in args.include_statuses if status.strip()}

    dataset_rows, unresolved_rows = build_dataset_rows(
        dataframe=dataframe,
        budget_root=budget_root,
        files_by_name=files_by_name,
        include_statuses=include_statuses,
    )

    if not args.keep_duplicate_paths:
        dataset_rows = drop_duplicate_paths(dataset_rows)

    label_rows = assign_label_ids(dataset_rows)
    summary    = build_summary(
        workfile_path=workfile_path,
        budget_root=budget_root,
        dataset_rows=dataset_rows,
        unresolved_rows=unresolved_rows,
        original_dataframe=dataframe,
        include_statuses=include_statuses,
    )

    write_csv(output_csv,     dataset_rows)
    write_csv(labels_csv,     label_rows)
    write_csv(unresolved_csv, unresolved_rows)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"데이터셋 저장: {output_csv}")
    print(f"라벨 저장: {labels_csv}")
    print(f"미해결 행 저장: {unresolved_csv}")
    print(f"요약 저장: {summary_json}")
    return 0


def main() -> int:
    """데이터셋 구축 스크립트 진입점."""
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
