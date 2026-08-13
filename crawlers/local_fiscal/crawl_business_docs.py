#!/usr/bin/env python3
"""지방재정365의 세부사업 목록을 수집하고 상세 화면을 PDF로 저장한다."""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

try:
    from .config import (
        BASE_DIR,
        DEFAULT_DATE,
        DEFAULT_LIMIT,
        DETAIL_SETTLE_MS,
        DETAIL_URL,
        LIST_URL,
        MANIFEST_PATH,
        PAGE_TIMEOUT_MS,
        ROWS_PER_PAGE,
        SAVE_DIR,
        TARGET_REGIONS,
    )
except ImportError:
    from config import (  # type: ignore[no-redef]
        BASE_DIR,
        DEFAULT_DATE,
        DEFAULT_LIMIT,
        DETAIL_SETTLE_MS,
        DETAIL_URL,
        LIST_URL,
        MANIFEST_PATH,
        PAGE_TIMEOUT_MS,
        ROWS_PER_PAGE,
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

MANIFEST_COLUMNS = [
    "key",
    "region_code",
    "region",
    "local_government",
    "local_code",
    "fiscal_year",
    "reference_date",
    "account",
    "business_name",
    "field",
    "sector",
    "budget_total",
    "budget_national",
    "budget_province",
    "budget_local",
    "budget_other",
    "expenditure",
    "balance",
    "dbiz_code",
    "detail_url",
    "pdf_file",
    "status",
    "error",
    "updated_at",
]


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

    @property
    def compact_date(self) -> str:
        return self.reference_date.replace("-", "")

    @property
    def fiscal_year(self) -> str:
        return self.reference_date[:4]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "지방재정365 지정 지역의 세부사업별 세출현황을 수집하고 "
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
        help="처리할 최대 사업 수(기본 5, 전체는 0)",
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


def normalize_business(
    raw: dict[str, Any],
    cfg: CrawlConfig,
    region: RegionSpec,
) -> dict[str, str]:
    dbiz_code = str(raw.get("code") or "")
    local_code = str(raw.get("lafCd") or "")
    fiscal_year = str(raw.get("rsltYr") or cfg.fiscal_year)
    business_name = str(raw.get("codeNm4") or "").strip()
    region_name = str(raw.get("codeNm") or region.name).strip()
    local_unit = str(raw.get("codeNm2") or "").strip()
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
        "account": str(raw.get("codeNm3") or ""),
        "business_name": business_name,
        "field": str(raw.get("codeNm5") or ""),
        "sector": str(raw.get("codeNm6") or ""),
        "budget_total": _money(raw.get("amt1")),
        "budget_national": _money(raw.get("amt2")),
        "budget_province": _money(raw.get("amt3")),
        "budget_local": _money(raw.get("amt4")),
        "budget_other": _money(raw.get("amt5")),
        "expenditure": _money(raw.get("amt6")),
        "balance": _money(raw.get("amt7")),
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
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["key"]: row
            for row in csv.DictReader(handle)
            if row.get("key")
        }


def write_manifest(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
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


async def configure_search(page: Page, cfg: CrawlConfig, region: RegionSpec) -> None:
    await page.goto(LIST_URL, wait_until="networkidle", timeout=cfg.timeout_ms)

    # 날짜 변경 이벤트가 자치단체 목록을 비동기로 다시 만들기 때문에 화면에서
    # 순차 클릭하면 선택값이 초기화될 수 있다. 검색 폼이 서버로 전송하는 값을
    # 한 번에 설정한 뒤 기존 검색 함수를 호출한다.
    async with page.expect_response(
        lambda response: "retvLstTxrvSituSgg.do" in response.url,
        timeout=cfg.timeout_ms,
    ):
        await page.evaluate(
            """(values) => {
                const setValue = (selector, value) => {
                    const element = document.querySelector(selector);
                    if (element) element.value = value;
                };
                const regionSelect = document.querySelector('#inqCap');
                if (regionSelect && ![...regionSelect.options].some(
                    option => option.value === values.regionCode
                )) {
                    regionSelect.add(new Option(values.regionName, values.regionCode));
                }
                setValue('#subCode', values.regionCode);
                setValue('#inqCap', values.regionCode);
                setValue('#inqSgg', '');
                setValue('#inqDtaCd', '');
                setValue('#inqDtaCd2', '');
                setValue('#inqYmd', values.referenceDate);
                setValue('#inqBgngYmd', values.compactDate);
                setValue('#inqYr', values.fiscalYear);
                window.lf.search('1');
            }""",
            {
                "regionCode": region.code,
                "regionName": region.name,
                "referenceDate": cfg.reference_date,
                "compactDate": cfg.compact_date,
                "fiscalYear": cfg.fiscal_year,
            },
        )
    await page.wait_for_timeout(300)
    await page.wait_for_function(
        "() => Array.isArray(window.list)", timeout=cfg.timeout_ms
    )


async def current_list(page: Page) -> list[dict[str, Any]]:
    return await page.evaluate(
        """() => window.list.map(row => ({
            code: row.code,
            lafCd: row.lafCd,
            codeNm: row.codeNm,
            codeNm2: row.codeNm2,
            codeNm3: row.codeNm3,
            codeNm4: row.codeNm4,
            codeNm5: row.codeNm5,
            codeNm6: row.codeNm6,
            rsltYr: row.rsltYr,
            amt1: row.amt1,
            amt2: row.amt2,
            amt3: row.amt3,
            amt4: row.amt4,
            amt5: row.amt5,
            amt6: row.amt6,
            amt7: row.amt7
        }))"""
    )


async def collect_region_list(
    page: Page,
    cfg: CrawlConfig,
    region: RegionSpec,
) -> list[dict[str, str]]:
    await configure_search(page, cfg, region)
    total_text = await page.locator(".stats em").inner_text()
    total = int(total_text.replace(",", "").strip())
    wanted = min(total, cfg.limit) if cfg.limit else total
    total_pages = math.ceil(wanted / ROWS_PER_PAGE)
    LOG.info(
        "%s 조회 결과 %s건 중 %s건을 수집합니다.",
        region.name,
        f"{total:,}",
        f"{wanted:,}",
    )

    collected: list[dict[str, str]] = []
    for page_number in range(1, total_pages + 1):
        if page_number > 1:
            async with page.expect_response(
                lambda response: "retvLstTxrvSituSgg.do" in response.url,
                timeout=cfg.timeout_ms,
            ):
                await page.evaluate("(number) => window.lf.search(String(number))", page_number)
            await page.wait_for_timeout(300)
            await page.wait_for_function(
                "() => Array.isArray(window.list)", timeout=cfg.timeout_ms
            )

        raw_rows = await current_list(page)
        for raw in raw_rows:
            row = normalize_business(raw, cfg, region)
            collected.append(row)
            if len(collected) >= wanted:
                break
        LOG.info(
            "%s 목록 %d/%d페이지: 누적 %d건",
            region.name,
            page_number,
            total_pages,
            len(collected),
        )
        if len(collected) >= wanted:
            break

    return collected


async def collect_business_list(page: Page, cfg: CrawlConfig) -> list[dict[str, str]]:
    collected: list[dict[str, str]] = []
    for region in cfg.regions:
        collected.extend(await collect_region_list(page, cfg, region))
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
    if async_playwright is None:
        raise RuntimeError(
            "Playwright가 필요합니다. `pip install playwright`와 "
            "`playwright install chromium`을 실행하세요."
        )

    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_manifest(cfg.manifest_path)

    async with async_playwright() as playwright:
        browser = await launch_browser(playwright, cfg)
        try:
            context = await browser.new_context(viewport={"width": 1440, "height": 1000})
            list_page = await context.new_page()
            collected = await collect_business_list(list_page, cfg)
            rows = merge_manifest(collected, existing)
            write_manifest(cfg.manifest_path, rows)
            LOG.info("목록을 저장했습니다: %s", cfg.manifest_path)

            if cfg.list_only:
                return 0

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
