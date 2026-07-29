"""온전한 2023 category CSV와 부분 BIOFIN 취합본을 합쳐 전체 라벨을 복원한다."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FULL_SOURCE = (
    PROJECT_DIR.parent
    / "budget_biodiv_cls2"
    / "outputs"
    / "사업설명자료"
    / "세부사업 예산편성현황(총액)_2023_category.csv"
)
DEFAULT_DETAIL_SOURCE = PROJECT_DIR / "document" / "BIOFIN_전체취합.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "document" / "2023biofin_label.csv"

BASE_COLUMNS = [
    "No.",
    "회계연도",
    "소관명",
    "회계명",
    "계정명",
    "분야명",
    "부문명",
    "프로그램명",
    "단위사업명",
    "세부사업명",
    "경비구분",
    "지출구분",
    "정부안금액(천원)",
    "국회확정금액(천원)",
]
DETAIL_COLUMNS = ["하위 카테고리", "가중치(BAR,%)", "분류 근거"]


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeError(f"CSV 인코딩 판별 실패: {path}") from last_error


def normalize_no(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "No." not in frame.columns:
        raise ValueError(f"{source_name}에 No. 컬럼이 없습니다")
    result = frame.copy()
    number = pd.to_numeric(result["No."], errors="raise")
    if (number % 1 != 0).any():
        raise ValueError(f"{source_name}의 No.에 정수가 아닌 값이 있습니다")
    result["No."] = number.astype(int)
    if result["No."].duplicated().any():
        duplicates = result.loc[result["No."].duplicated(False), "No."].tolist()[:20]
        raise ValueError(f"{source_name}의 No.가 중복됩니다: {duplicates}")
    return result


def rebuild(full_source: Path, detail_source: Path) -> pd.DataFrame:
    full = normalize_no(read_csv_flexible(full_source), "전체 category CSV")
    detail = normalize_no(read_csv_flexible(detail_source), "BIOFIN 상세 취합 CSV")
    required_full = set(BASE_COLUMNS) | {"biofin_category"}
    if missing := sorted(required_full - set(full.columns)):
        raise ValueError(f"전체 category CSV 컬럼 누락: {', '.join(missing)}")
    if missing := sorted(set(DETAIL_COLUMNS) - set(detail.columns)):
        raise ValueError(f"BIOFIN 상세 취합 CSV 컬럼 누락: {', '.join(missing)}")
    if len(full) != 9074:
        raise ValueError(f"전체 category CSV 행 수가 9074가 아닙니다: {len(full)}")

    category = pd.to_numeric(full["biofin_category"], errors="raise")
    if not ((category % 1 == 0) & category.between(0, 9)).all():
        raise ValueError("biofin_category에 0~9 이외의 값이 있습니다")
    full["BIOFIN 1차 카테고리"] = category.astype(int)
    full["BIOFIN분류"] = (full["BIOFIN 1차 카테고리"] != 0).astype(int)

    detail_subset = detail[["No.", *DETAIL_COLUMNS]].copy()
    result = full.merge(
        detail_subset,
        on="No.",
        how="left",
        validate="one_to_one",
    )
    result = result[
        [
            *BASE_COLUMNS,
            "BIOFIN분류",
            "BIOFIN 1차 카테고리",
            *DETAIL_COLUMNS,
        ]
    ]
    if len(result) != len(full) or result["No."].nunique() != 9074:
        raise RuntimeError("병합 후 행 수 또는 No. 고유성이 손상됐습니다")
    return result


def atomic_write_with_backup(frame: pd.DataFrame, output: Path) -> Path | None:
    output.parent.mkdir(parents=True, exist_ok=True)
    backup = output.with_name(f"{output.stem}.before_rebuild{output.suffix}")
    if output.exists() and not backup.exists():
        shutil.copy2(output, backup)

    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    verification = pd.read_csv(temporary, encoding="utf-8-sig", low_memory=False)
    if len(verification) != 9074:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"임시 출력 검증 실패: {len(verification)}행")
    if verification["No."].nunique() != 9074:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("임시 출력 No. 고유성 검증 실패")
    temporary.replace(output)
    return backup if backup.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2023 BIOFIN 전체 9074행 라벨 복원")
    parser.add_argument("--full-source", type=Path, default=DEFAULT_FULL_SOURCE)
    parser.add_argument("--detail-source", type=Path, default=DEFAULT_DETAIL_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = rebuild(args.full_source, args.detail_source)
    backup = atomic_write_with_backup(result, args.output)
    print(f"복원 완료: {args.output.resolve()}")
    print(f"행 수: {len(result)}, No. 고유값: {result['No.'].nunique()}")
    print(f"UTF-8 BOM: yes")
    if backup:
        print(f"기존 파일 백업: {backup.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
