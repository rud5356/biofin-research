"""2023 BIOFIN 라벨 CSV를 열린재정 사업설명자료와 매칭한다.

라벨 CSV에는 business_key가 없으므로 먼저 열린재정 목록 CSV와 사업 계층
정보를 매칭하고, 목록의 business_key와 파일 경로를 라벨 행에 붙인다.
모호한 후보는 임의 선택하지 않고 상태와 후보 수를 기록한다.
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DOCUMENT_DIR = PROJECT_DIR / "document"
DEFAULT_LABEL_CSV = DEFAULT_DOCUMENT_DIR / "2023biofin_label.csv"
DEFAULT_FISCAL_CSV = DEFAULT_DOCUMENT_DIR / "2023" / "open_fiscal_2023.csv"
DEFAULT_DOC_DIR = DEFAULT_DOCUMENT_DIR / "2023" / "사업설명자료"
DEFAULT_OUTPUT_CSV = DEFAULT_DOCUMENT_DIR / "2023biofin_label_matched.csv"
DEFAULT_FAILURE_CSV = DEFAULT_DOCUMENT_DIR / "2023biofin_label_match_failed.csv"

LOGGER = logging.getLogger("biofin_document_matcher")

HIERARCHY_COLUMNS = (
    "회계연도",
    "소관명",
    "회계명",
    "프로그램명",
    "단위사업명",
    "세부사업명",
)
OUTPUT_COLUMNS = (
    "business_key",
    "사업설명자료_파일명",
    "사업설명자료_상대경로",
    "사업설명자료_절대경로",
    "문서매칭상태",
    "문서매칭방식",
    "문서매칭후보수",
)


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeError(f"CSV 인코딩 판별 실패: {path}") from last_error


def normalize(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def normalize_year(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", str(value))
    return match.group(1) if match else ""


def make_key(row: pd.Series, columns: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        normalize_year(row.get(column)) if column == "회계연도" else normalize(row.get(column))
        for column in columns
    )


def parse_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    cleaned = re.sub(r"[^0-9.+-]", "", str(value))
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def disambiguate_by_amount(
    label_row: pd.Series,
    fiscal: pd.DataFrame,
    candidates: list[int],
) -> list[int]:
    """라벨의 천원 금액과 열린재정의 백만원 금액이 유일하게 같은 후보를 고른다."""

    label_amount = parse_number(label_row.get("국회확정금액(천원)"))
    if label_amount is None:
        return candidates
    matched = []
    for index in candidates:
        fiscal_amount = parse_number(fiscal.at[index, "예산액"])
        if fiscal_amount is not None and abs(label_amount - fiscal_amount * 1000) < 0.5:
            matched.append(index)
    return matched if len(matched) == 1 else candidates


def build_indexes(
    fiscal: pd.DataFrame,
) -> list[tuple[str, tuple[str, ...], dict[tuple[str, ...], list[int]]]]:
    strategies = [
        ("FULL_HIERARCHY", HIERARCHY_COLUMNS),
        (
            "WITHOUT_UNIT",
            ("회계연도", "소관명", "회계명", "프로그램명", "세부사업명"),
        ),
        (
            "MINISTRY_ACCOUNT_ACTIVITY",
            ("회계연도", "소관명", "회계명", "세부사업명"),
        ),
        ("MINISTRY_ACTIVITY", ("회계연도", "소관명", "세부사업명")),
    ]
    result = []
    for name, columns in strategies:
        index: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for row_index, row in fiscal.iterrows():
            key = make_key(row, columns)
            if all(key):
                index[key].append(int(row_index))
        result.append((name, columns, index))
    return result


def business_key_suffix(value: object) -> str:
    match = re.search(r"-(\d{2})$", str(value or "").strip())
    return match.group(1) if match else ""


def infer_account_suffixes(
    labels: pd.DataFrame,
    fiscal: pd.DataFrame,
    full_index: dict[tuple[str, ...], list[int]],
) -> dict[str, str]:
    """유일한 금액 매칭 사례에서 계정명별 business_key 상세계정 suffix를 학습한다."""

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for _, row in labels.iterrows():
        account = normalize(row.get("계정명"))
        if not account:
            continue
        candidates = full_index.get(make_key(row, HIERARCHY_COLUMNS), [])
        selected = disambiguate_by_amount(row, fiscal, candidates)
        if len(selected) != 1:
            continue
        suffix = business_key_suffix(fiscal.at[selected[0], "business_key"])
        if suffix:
            counts[account][suffix] += 1
    return {
        account: suffix_counts.most_common(1)[0][0]
        for account, suffix_counts in counts.items()
        if suffix_counts
    }


def resolve_document_path(
    fiscal_row: pd.Series,
    year_dir: Path,
    doc_dir: Path,
    documents_by_name: dict[str, Path],
) -> Path | None:
    relative_path = str(fiscal_row.get("사업설명자료_상대경로", "") or "").strip()
    filename = str(fiscal_row.get("사업설명자료_파일명", "") or "").strip()
    candidates = []
    if relative_path:
        candidates.append(year_dir.joinpath(*PurePosixPath(relative_path).parts))
    if filename:
        candidates.append(doc_dir / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if filename:
        return documents_by_name.get(Path(filename).name.casefold())
    return None


def match_rows(
    labels: pd.DataFrame,
    fiscal: pd.DataFrame,
    year_dir: Path,
    doc_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexes = build_indexes(fiscal)
    account_suffixes = infer_account_suffixes(labels, fiscal, indexes[0][2])
    documents_by_name = {
        path.name.casefold(): path.resolve()
        for path in doc_dir.rglob("*")
        if path.is_file()
    }

    output = labels.copy()
    for column in OUTPUT_COLUMNS:
        output[column] = ""

    failure_rows: list[dict[str, object]] = []
    for label_index, label_row in labels.iterrows():
        matched_indices: list[int] = []
        match_strategy = ""
        row_number = parse_number(label_row.get("No."))
        if row_number is not None and row_number.is_integer():
            fiscal_index = int(row_number) - 1
            if (
                fiscal_index in fiscal.index
                and make_key(label_row, HIERARCHY_COLUMNS)
                == make_key(fiscal.loc[fiscal_index], HIERARCHY_COLUMNS)
            ):
                matched_indices = [fiscal_index]
                match_strategy = "NO_AND_FULL_HIERARCHY"

        if not matched_indices:
            for strategy, columns, index in indexes:
                candidates = index.get(make_key(label_row, columns), [])
                if len(candidates) == 1:
                    matched_indices = candidates
                    match_strategy = strategy
                    break
                if len(candidates) > 1:
                    amount_matched = disambiguate_by_amount(label_row, fiscal, candidates)
                    account_suffix = account_suffixes.get(
                        normalize(label_row.get("계정명")), ""
                    )
                    suffix_matched = [
                        candidate
                        for candidate in amount_matched
                        if account_suffix
                        and business_key_suffix(fiscal.at[candidate, "business_key"])
                        == account_suffix
                    ]
                    if len(suffix_matched) == 1:
                        matched_indices = suffix_matched
                        match_strategy = f"{strategy}_AMOUNT_ACCOUNT"
                    else:
                        matched_indices = amount_matched
                        match_strategy = (
                            f"{strategy}_AMOUNT"
                            if len(amount_matched) == 1
                            else strategy
                        )
                    # 더 느슨한 전략은 후보를 줄이지 못하므로 여기서 모호 처리한다.
                    break

        if not matched_indices:
            status = "NO_FISCAL_MATCH"
            output.at[label_index, "문서매칭상태"] = status
            output.at[label_index, "문서매칭후보수"] = 0
        elif len(matched_indices) > 1:
            status = "AMBIGUOUS_FISCAL_MATCH"
            output.at[label_index, "문서매칭상태"] = status
            output.at[label_index, "문서매칭방식"] = match_strategy
            output.at[label_index, "문서매칭후보수"] = len(matched_indices)
        else:
            fiscal_row = fiscal.loc[matched_indices[0]]
            business_key = str(fiscal_row.get("business_key", "") or "").strip()
            filename = str(fiscal_row.get("사업설명자료_파일명", "") or "").strip()
            relative_path = str(
                fiscal_row.get("사업설명자료_상대경로", "") or ""
            ).strip()
            document_path = resolve_document_path(
                fiscal_row, year_dir, doc_dir, documents_by_name
            )
            download_status = str(fiscal_row.get("다운로드상태", "") or "").strip()

            output.at[label_index, "business_key"] = business_key
            output.at[label_index, "사업설명자료_파일명"] = filename
            output.at[label_index, "사업설명자료_상대경로"] = relative_path
            output.at[label_index, "문서매칭방식"] = match_strategy
            output.at[label_index, "문서매칭후보수"] = 1
            if document_path is not None:
                status = "MATCHED"
                output.at[label_index, "사업설명자료_절대경로"] = str(document_path)
            elif not filename or download_status == "no_document":
                status = "NO_DOCUMENT"
            else:
                status = "DOCUMENT_NOT_FOUND"
            output.at[label_index, "문서매칭상태"] = status

        if output.at[label_index, "문서매칭상태"] != "MATCHED":
            failure_rows.append(
                {
                    "원본행번호": int(label_index) + 2,
                    "No.": label_row.get("No.", ""),
                    "회계연도": label_row.get("회계연도", ""),
                    "소관명": label_row.get("소관명", ""),
                    "회계명": label_row.get("회계명", ""),
                    "프로그램명": label_row.get("프로그램명", ""),
                    "단위사업명": label_row.get("단위사업명", ""),
                    "세부사업명": label_row.get("세부사업명", ""),
                    "문서매칭상태": output.at[label_index, "문서매칭상태"],
                    "문서매칭방식": output.at[label_index, "문서매칭방식"],
                    "문서매칭후보수": output.at[label_index, "문서매칭후보수"],
                }
            )
    return output, pd.DataFrame(failure_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2023 BIOFIN 라벨 CSV와 열린재정 사업설명자료 매칭"
    )
    parser.add_argument("--label-csv", type=Path, default=DEFAULT_LABEL_CSV)
    parser.add_argument("--fiscal-csv", type=Path, default=DEFAULT_FISCAL_CSV)
    parser.add_argument("--doc-dir", type=Path, default=DEFAULT_DOC_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--failure-csv", type=Path, default=DEFAULT_FAILURE_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for path in (args.label_csv, args.fiscal_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.doc_dir.is_dir():
        raise NotADirectoryError(args.doc_dir)

    labels = read_csv_flexible(args.label_csv)
    fiscal = read_csv_flexible(args.fiscal_csv)
    required_label = set(HIERARCHY_COLUMNS)
    required_fiscal = required_label | {
        "business_key",
        "사업설명자료_파일명",
        "사업설명자료_상대경로",
    }
    if missing := sorted(required_label - set(labels.columns)):
        raise ValueError(f"라벨 CSV 필수 컬럼 누락: {', '.join(missing)}")
    if missing := sorted(required_fiscal - set(fiscal.columns)):
        raise ValueError(f"열린재정 CSV 필수 컬럼 누락: {', '.join(missing)}")

    matched, failed = match_rows(
        labels,
        fiscal,
        args.fiscal_csv.parent,
        args.doc_dir,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.failure_csv.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    failed.to_csv(args.failure_csv, index=False, encoding="utf-8-sig")

    counts = matched["문서매칭상태"].value_counts(dropna=False).to_dict()
    document_count = sum(1 for path in args.doc_dir.rglob("*") if path.is_file())
    LOGGER.info(
        "라벨 행=%d, 열린재정 행=%d, 문서 파일=%d",
        len(labels),
        len(fiscal),
        document_count,
    )
    LOGGER.info("매칭 결과=%s", counts)
    LOGGER.info("결과 CSV=%s", args.output_csv.resolve())
    LOGGER.info("미매칭 CSV=%s", args.failure_csv.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
