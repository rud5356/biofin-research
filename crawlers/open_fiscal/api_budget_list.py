#!/usr/bin/env python3
"""열린재정 Open API로 회계연도별 세부사업 예산 목록을 CSV로 저장한다.

기존 `crawl_business_docs.py`를 대체하지 않는 별도 도구다. Open API는
사업설명자료 파일과 사업 코드를 제공하지 않으므로 문서 다운로드와
`business_key` 생성은 기존 크롤러가 계속 맡는다. 이 파일은 예산액의 공식
출처를 확보하거나 기존 목록의 누락을 점검할 때 쓴다.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

try:
    from ..openapi import get_api_key, iter_rows
    from .config import SAVE_DIR
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from openapi import get_api_key, iter_rows  # type: ignore[no-redef]
    from config import SAVE_DIR  # type: ignore[no-redef]


LOG = logging.getLogger("open_fiscal_api")

# 세부사업 예산편성현황(총액). 인증키는 저장소 루트의 `.env`에서 읽는다.
API_URL = "https://openapi.openfiscaldata.go.kr/ExpenditureBudgetInit5"
API_KEY_ENV = "OPENFISCAL_API_KEY"

# CSV 컬럼명 → Open API 출력 항목 ID
COLUMNS = {
    "회계연도": "FSCL_YY",
    "소관명": "OFFC_NM",
    "회계명": "FSCL_NM",
    "계정명": "ACCT_NM",
    "분야명": "FLD_NM",
    "부문명": "SECT_NM",
    "프로그램명": "PGM_NM",
    "단위사업명": "ACTV_NM",
    "세부사업명": "SACTV_NM",
    "경비구분": "BZ_CLS_NM",
    "지출구분": "FIN_DE_EP_NM",
    "정부안금액": "Y_YY_MEDI_KCUR_AMT",
    "국회확정금액": "Y_YY_DFN_MEDI_KCUR_AMT",
}


def default_output(year: int) -> Path:
    return SAVE_DIR / str(year) / f"open_fiscal_{year}_api.csv"


def fetch_budget_rows(year: int, api_key: str, limit: int = 0) -> tuple[int, list[dict[str, str]]]:
    """(전체 건수, CSV 행 목록)을 돌려준다."""
    rows: list[dict[str, str]] = []
    total = 0
    for total, raw in iter_rows(API_URL, {"FSCL_YY": str(year)}, api_key, limit):
        rows.append(
            {
                column: "" if raw.get(item) is None else str(raw[item])
                for column, item in COLUMNS.items()
            }
        )
    return total, rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="열린재정 Open API로 세부사업 예산편성현황(총액)을 CSV로 저장"
    )
    parser.add_argument("--year", type=int, required=True, help="회계연도(예: 2024)")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="최대 사업 수(기본 5, 전체는 0). 인증키가 없으면 5까지만 가능",
    )
    parser.add_argument("--output", type=Path, help="저장할 CSV 경로")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.limit < 0:
        parser.error("--limit은 0 이상이어야 합니다")
    try:
        total, rows = fetch_budget_rows(args.year, get_api_key(API_KEY_ENV), args.limit)
    except RuntimeError as exc:
        parser.error(str(exc))
        return 2
    output = (args.output or default_output(args.year)).resolve()
    write_csv(output, rows)
    LOG.info("%d년 조회 결과 %s건 중 %s건을 저장했습니다: %s", args.year, f"{total:,}", f"{len(rows):,}", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
