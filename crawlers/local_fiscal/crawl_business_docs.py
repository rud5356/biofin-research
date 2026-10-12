#!/usr/bin/env python3
"""지방재정365 Open API로 세부사업 목록을 수집하고 상세 화면을 PDF로 저장한다."""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

try:
    from ..openapi import get_api_key, iter_rows
    from .config import (
        API_KEY_ENV,
        API_URL,
        BASE_DIR,
        DEFAULT_DATE,
        DEFAULT_LIMIT,
        DETAIL_SETTLE_MS,
        DETAIL_URL,
        MANIFEST_PATH,
        PAGE_TIMEOUT_MS,
        SAVE_DIR,
        TARGET_REGIONS,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from openapi import get_api_key, iter_rows  # type: ignore[no-redef]
    from config import (  # type: ignore[no-redef]
        API_KEY_ENV,
        API_URL,
        BASE_DIR,
        DEFAULT_DATE,
        DEFAULT_LIMIT,
        DETAIL_SETTLE_MS,
        DETAIL_URL,
        MANIFEST_PATH,
        PAGE_TIMEOUT_MS,
        SAVE_DIR,
        TARGET_REGIONS,
    )

try:
    from playwright.async_api import (
        Browser,
        Page,
        async_playwright,
    )
except ImportError:
    Browser = Any  # type: ignore[assignment,misc]
    Page = Any  # type: ignore[assignment,misc]
    async_playwright = None


LOG = logging.getLogger("local_fiscal")
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
WHITESPACE = re.compile(r"\s+")

# CSV 컬럼명 → 내부 항목명. 예측·LLM 분류 코드가 읽는 한글 컬럼명에 맞춘다.
MANIFEST_COLUMNS = {
    "회계연도": "fiscal_year",
    "소관명": "local_government",
    "회계명": "account",
    "분야명": "field",
    "부문명": "sector",
    "세부사업명": "business_name",
    "예산액": "budget_total",
    "business_key": "key",
    "사업설명자료_파일명": "pdf_file",
    "다운로드상태": "status",
    "오류내용": "error",
    "시도코드": "region_code",
    "시도명": "region",
    "지방자치단체코드": "local_code",
    "기준일자": "reference_date",
    "국비": "budget_national",
    "시도비": "budget_province",
    "시군구비": "budget_local",
    "기타재원": "budget_other",
    "지출액": "expenditure",
    "잔액": "balance",
    "세부사업코드": "dbiz_code",
    "상세페이지_URL": "detail_url",
    "갱신일시": "updated_at",
}


@dataclass(frozen=True)
class RegionSpec:
    name: str
    code: str


@dataclass(frozen=True)
class CrawlConfig:
    reference_date: str
    regions: tuple[RegionSpec, ...]
    limit: int
    output_dir: Path
    manifest_path: Path
    headed: bool
    browser_channel: str | None
    timeout_ms: int
    min_delay: float
    pdf_scale: float
    overwrite: bool
    retry_failed: bool
    list_only: bool
    api_key: str = ""

    @property
    def compact_date(self) -> str:
        return self.reference_date.replace("-", "")

    @property
    def fiscal_year(self) -> str:
        return self.reference_date[:4]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "지방재정365 Open API로 지정 지역의 세부사업별 세출현황을 수집하고 "
            "상세 화면을 PDF로 저장"
        )
    )
    parser.add_argument("--date", default=DEFAULT_DATE, help="기준일(YYYY-MM-DD)")
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=list(TARGET_REGIONS),
        default=list(TARGET_REGIONS),
        help="수집할 지역(기본: 전국 17개 시도)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="지역별 최대 사업 수(기본 5, 전체는 0). 인증키가 없으면 5까지만 가능",
    )
    parser.add_argument("--output-dir", type=Path, default=SAVE_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--list-only", action="store_true", help="PDF 없이 목록만 수집")
    parser.add_argument("--overwrite", action="store_true", help="기존 PDF도 다시 저장")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="기존 manifest의 failed 항목만 다시 처리",
    )
    parser.add_argument("--headed", action="store_true", help="브라우저 창 표시")
    parser.add_argument(
        "--browser-channel",
        default="chrome",
        help="Playwright 브라우저 채널(기본 chrome, 내장 Chromium은 빈 문자열)",
    )
    parser.add_argument("--timeout", type=int, default=PAGE_TIMEOUT_MS)
    parser.add_argument("--min-delay", type=float, default=0.8)
    parser.add_argument(
        "--pdf-scale",
        type=float,
        default=0.70,
        help="PDF 출력 배율(기본 0.70 = 70%%)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def validate_date(value: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("날짜는 YYYY-MM-DD 형식이어야 합니다") from exc
    return parsed.strftime("%Y-%m-%d")


def parse_config(args: argparse.Namespace) -> CrawlConfig:
    reference_date = validate_date(args.date)
    if args.limit < 0:
        raise ValueError("--limit은 0 이상이어야 합니다")
    if args.min_delay < 0:
        raise ValueError("--min-delay는 0 이상이어야 합니다")
    if not 0.1 <= args.pdf_scale <= 2.0:
        raise ValueError("--pdf-scale은 0.1 이상 2.0 이하여야 합니다")
    return CrawlConfig(
        reference_date=reference_date,
        regions=tuple(RegionSpec(name, TARGET_REGIONS[name]) for name in args.regions),
        limit=args.limit,
        output_dir=args.output_dir.resolve(),
        manifest_path=args.manifest.resolve(),
        headed=args.headed,
        browser_channel=args.browser_channel or None,
        timeout_ms=args.timeout,
        min_delay=args.min_delay,
        pdf_scale=args.pdf_scale,
        overwrite=args.overwrite,
        retry_failed=args.retry_failed,
        list_only=args.list_only,
        api_key=get_api_key(API_KEY_ENV),
    )


def detail_url(dbiz_code: str, local_code: str, fiscal_year: str, date: str) -> str:
    query = urlencode(
        {
            "dbizCd": dbiz_code,
            "lafCd": local_code,
            "fyr": fiscal_year,
            "inqYmd": date.replace("-", ""),
        }
    )
    return f"{DETAIL_URL}?{query}"


def safe_filename(value: str, max_length: int = 110) -> str:
    cleaned = INVALID_FILENAME_CHARS.sub("_", value)
    cleaned = WHITESPACE.sub(" ", cleaned).strip(" ._")
    return (cleaned or "이름없음")[:max_length].rstrip(" .")


def pdf_filename(row: dict[str, str]) -> str:
    parts = [
        row["fiscal_year"],
        row["region"],
        row["local_government"],
        row["account"],
        safe_filename(row["business_name"]),
        row["local_code"],
        row["dbiz_code"],
    ]
    return safe_filename("_".join(parts), max_length=190) + ".pdf"


def _money(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _balance(budget: Any, expenditure: Any) -> str:
    # Open API는 집행잔액을 주지 않으므로 예산현액에서 지출액을 뺀다.
    try:
        return str(int(budget) - int(expenditure))
    except (TypeError, ValueError):
        return ""


def normalize_business(
    raw: dict[str, Any],
    cfg: CrawlConfig,
    region: RegionSpec,
) -> dict[str, str]:
    dbiz_code = str(raw.get("dbiz_cd") or "")
    local_code = str(raw.get("laf_cd") or "")
    fiscal_year = str(raw.get("fyr") or cfg.fiscal_year)
    business_name = str(raw.get("dbiz_nm") or "").strip()
    region_name = str(raw.get("wa_laf_hg_nm") or region.name).strip()
    local_unit = str(raw.get("laf_hg_nm") or "").strip()
    local_government = (
        local_unit if local_unit.startswith(region_name) else f"{region_name}{local_unit}"
    )
    key = "|".join([dbiz_code, local_code, fiscal_year, cfg.compact_date])
    row = {
        "key": key,
        "region_code": region.code,
        "region": region_name,
        "local_government": local_government,
        "local_code": local_code,
        "fiscal_year": fiscal_year,
        "reference_date": cfg.reference_date,
        "account": str(raw.get("acnt_dv_nm") or ""),
        "business_name": business_name,
        "field": str(raw.get("fld_nm") or ""),
        "sector": str(raw.get("part_nm") or ""),
        "budget_total": _money(raw.get("bdg_cash_amt")),
        "budget_national": _money(raw.get("bdg_ntep")),
        "budget_province": _money(raw.get("capep")),
        "budget_local": _money(raw.get("sggep")),
        "budget_other": _money(raw.get("etc_amt")),
        "expenditure": _money(raw.get("ep_amt")),
        "balance": _balance(raw.get("bdg_cash_amt"), raw.get("ep_amt")),
        "dbiz_code": dbiz_code,
        "detail_url": detail_url(dbiz_code, local_code, fiscal_year, cfg.reference_date),
        "pdf_file": "",
        "status": "pending",
        "error": "",
        "updated_at": "",
    }
    row["pdf_file"] = pdf_filename(row)
    return row


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    manifest: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            # 영문 컬럼으로 저장된 예전 매니페스트도 이어받을 수 있게 둘 다 읽는다.
            row = {name: raw.get(column) or raw.get(name) or "" for column, name in MANIFEST_COLUMNS.items()}
            if row["key"]:
                manifest[row["key"]] = row
    return manifest


def write_manifest(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_COLUMNS)
        for row in rows:
            writer.writerow([row.get(name, "") for name in MANIFEST_COLUMNS.values()])
    temp_path.replace(path)


def merge_manifest(
    collected: list[dict[str, str]],
    existing: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for row in collected:
        previous = existing.get(row["key"])
        if previous:
            for field in ("pdf_file", "status", "error", "updated_at"):
                row[field] = previous.get(field, row[field])
        merged.append(row)
    return merged


def collect_region_list(cfg: CrawlConfig, region: RegionSpec) -> list[dict[str, str]]:
    params = {
        "fyr": cfg.fiscal_year,
        "exe_ymd": cfg.compact_date,
        "wa_laf_cd": region.code,
    }
    collected: list[dict[str, str]] = []
    total = 0
    for total, raw in iter_rows(API_URL, params, cfg.api_key, cfg.limit):
        collected.append(normalize_business(raw, cfg, region))
        if len(collected) % 10_000 == 0:
            LOG.info("%s 목록 누적 %s건", region.name, f"{len(collected):,}")
    LOG.info(
        "%s 조회 결과 %s건 중 %s건을 수집했습니다.",
        region.name,
        f"{total:,}",
        f"{len(collected):,}",
    )
    return collected


def collect_business_list(cfg: CrawlConfig) -> list[dict[str, str]]:
    collected: list[dict[str, str]] = []
    for region in cfg.regions:
        collected.extend(collect_region_list(cfg, region))
    return collected


def should_process(row: dict[str, str], cfg: CrawlConfig) -> bool:
    pdf_path = cfg.output_dir / row["pdf_file"]
    if cfg.retry_failed:
        return row["status"] == "failed"
    if cfg.overwrite:
        return True
    if row["status"] == "success" and pdf_path.exists() and pdf_path.stat().st_size > 0:
        return False
    return True


async def render_detail_pdf(page: Page, row: dict[str, str], cfg: CrawlConfig) -> Path:
    await page.goto(row["detail_url"], wait_until="networkidle", timeout=cfg.timeout_ms)
    await page.get_by_text("사업개요", exact=True).first.wait_for(
        state="visible", timeout=cfg.timeout_ms
    )
    await page.wait_for_timeout(DETAIL_SETTLE_MS)
    body_text = await page.locator("body").inner_text()
    # 지방재정 목록명과 상세 화면의 현행 사업명이 다른 사례가 있으므로
    # 사업명 완전 일치 대신 상세 화면의 필수 섹션과 회계연도를 검증한다.
    if "사업개요" not in body_text or row["fiscal_year"] not in body_text:
        raise RuntimeError("상세 페이지의 사업개요 또는 회계연도를 확인하지 못했습니다")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = cfg.output_dir / row["pdf_file"]
    await page.emulate_media(media="screen")
    await page.pdf(
        path=str(pdf_path),
        format="A4",
        print_background=True,
        landscape=True,
        scale=cfg.pdf_scale,
        margin={"top": "10mm", "right": "8mm", "bottom": "10mm", "left": "8mm"},
    )
    if not pdf_path.exists() or pdf_path.stat().st_size < 1_000:
        raise RuntimeError("생성된 PDF가 없거나 비정상적으로 작습니다")
    return pdf_path


async def launch_browser(playwright: Any, cfg: CrawlConfig) -> Browser:
    options: dict[str, Any] = {"headless": not cfg.headed}
    if cfg.browser_channel:
        options["channel"] = cfg.browser_channel
    try:
        return await playwright.chromium.launch(**options)
    except Exception:
        if not cfg.browser_channel:
            raise
        LOG.warning(
            "브라우저 채널 %s 실행 실패. Playwright Chromium으로 재시도합니다.",
            cfg.browser_channel,
        )
        return await playwright.chromium.launch(headless=not cfg.headed)


async def run(cfg: CrawlConfig) -> int:
    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_manifest(cfg.manifest_path)

    collected = collect_business_list(cfg)
    rows = merge_manifest(collected, existing)
    write_manifest(cfg.manifest_path, rows)
    LOG.info("목록을 저장했습니다: %s", cfg.manifest_path)

    if cfg.list_only:
        return 0

    if async_playwright is None:
        raise RuntimeError(
            "PDF 저장에는 Playwright가 필요합니다. `pip install playwright`와 "
            "`playwright install chromium`을 실행하거나 --list-only로 실행하세요."
        )

    async with async_playwright() as playwright:
        browser = await launch_browser(playwright, cfg)
        try:
            context = await browser.new_context(viewport={"width": 1440, "height": 1000})
            detail_page = await context.new_page()
            for index, row in enumerate(rows, start=1):
                if not should_process(row, cfg):
                    LOG.info("[%d/%d] 건너뜀: %s", index, len(rows), row["business_name"])
                    continue
                try:
                    pdf_path = await render_detail_pdf(detail_page, row, cfg)
                    row["status"] = "success"
                    row["error"] = ""
                    LOG.info("[%d/%d] 저장: %s", index, len(rows), pdf_path.name)
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = f"{type(exc).__name__}: {exc}"[:500]
                    LOG.error(
                        "[%d/%d] 실패: %s - %s",
                        index,
                        len(rows),
                        row["business_name"],
                        exc,
                    )
                row["updated_at"] = datetime.now().isoformat(timespec="seconds")
                write_manifest(cfg.manifest_path, rows)
                if cfg.min_delay:
                    await asyncio.sleep(cfg.min_delay)

            failures = sum(row["status"] == "failed" for row in rows)
            successes = sum(row["status"] == "success" for row in rows)
            LOG.info("완료: 성공 %d건, 실패 %d건", successes, failures)
            return 1 if failures else 0
        finally:
            await browser.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        cfg = parse_config(args)
        return asyncio.run(run(cfg))
    except (ValueError, argparse.ArgumentTypeError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
