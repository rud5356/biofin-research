#!/usr/bin/env python3
"""열린재정에서 생물다양성 관련 세부사업의 사업설명자료를 내려받는다.

입력 파일에서 label(또는 현재 프로젝트에서 쓰는 biodiv_label)이 1인 행을
정규화한 뒤, 회계연도별 열린재정 목록과 안전하게 매칭한다. 사이트 DOM이
바뀔 가능성을 고려해 DOM 행과 XHR JSON 응답을 함께 탐색하고, 구조를 찾지
못한 경우 debug_html 폴더에 당시 HTML을 남긴다.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import logging
import random
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from email.message import Message
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import unquote, urlencode, urljoin, urlparse

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # 설치 전에도 --help와 오류 로그는 동작하게 한다.
    tqdm = None

try:
    from playwright.async_api import (
        Download,
        Error as PlaywrightError,
        Locator,
        Page,
        TimeoutError as PlaywrightTimeoutError,
        async_playwright,
    )
except ImportError:  # main에서 친절한 설치 안내를 출력하기 위한 지연 실패 처리다.
    Download = Any  # type: ignore[assignment,misc]
    Locator = Any  # type: ignore[assignment,misc]
    Page = Any  # type: ignore[assignment,misc]
    PlaywrightError = Exception  # type: ignore[assignment,misc]
    PlaywrightTimeoutError = TimeoutError  # type: ignore[assignment,misc]
    async_playwright = None


# 기본 경로와 사이트 주소는 CLI에서 덮어쓸 수 있게 상수로 분리했다.
BASE_DIR = Path(r"C:\repos\biofin-research\budget_biodiv_cls2\outputs")
SAVE_DIR = BASE_DIR / "사업설명자료"
SITE_URL = "https://www.openfiscaldata.go.kr/op/ko/bs/UOPKOBSA02"
ALLOWED_EXTENSIONS = {".hwp", ".pdf", ".zip", ".xlsx"}

SUCCESS_COLUMNS = [
    "year",
    "source_file",
    "ministry",
    "program_name",
    "activity_name",
    "downloaded_file",
    "source_url",
    "crawled_at",
]
FAILED_COLUMNS = [
    "year",
    "source_file",
    "ministry",
    "activity_name",
    "reason",
    "searched_url",
    "crawled_at",
]
TARGET_COLUMNS = [
    "year",
    "source_file",
    "source_row",
    "label_column",
    "ministry",
    "account_name",
    "field_name",
    "sector_name",
    "program_name",
    "unit_name",
    "activity_name",
]

# 실제 데이터의 한글명과 열린재정 영문 필드명을 모두 허용한다.
COLUMN_CANDIDATES = {
    "year": ["회계연도", "회계년도", "연도", "acntYr", "accountYear", "year"],
    "ministry": ["소관명", "부처명", "중앙관서명", "소관기관명", "offcNm", "ministry"],
    "account_name": ["회계명", "회계", "acntNm", "accountName"],
    "field_name": ["분야명", "분야", "fldNm", "fieldName"],
    "sector_name": ["부문명", "부문", "sectNm", "sectorName"],
    "program_name": ["프로그램명", "프로그램", "pgmNm", "programName"],
    "unit_name": ["단위사업명", "단위사업", "unitBizNm", "unitName"],
    "activity_name": ["세부사업명", "세부사업", "actvNm", "사업명", "activityName"],
}

# 표, jqGrid, ARIA grid, ag-grid 등 흔한 렌더러를 한 번에 탐색한다.
ROW_SELECTOR = ", ".join(
    [
        "table tbody tr",
        ".ui-jqgrid-btable tbody tr",
        "[role='rowgroup'] [role='row']",
        ".ag-center-cols-container .ag-row",
        ".grid-body .row",
        ".tbl_data tbody tr",
        ".board-list tbody tr",
    ]
)

ATTACHMENT_SELECTOR = ", ".join(
    [
        "a[download]",
        "a[href$='.hwp' i]",
        "a[href*='.hwp?' i]",
        "a[href$='.pdf' i]",
        "a[href*='.pdf?' i]",
        "a[href$='.zip' i]",
        "a[href*='.zip?' i]",
        "a[href$='.xlsx' i]",
        "a[href*='.xlsx?' i]",
        "a:has-text('HWP')",
        "a:has-text('PDF')",
        "a:has-text('첨부파일')",
        "a:has-text('다운로드')",
        "button:has-text('다운로드')",
        "[role='button']:has-text('다운로드')",
    ]
)


@dataclass
class RuntimeConfig:
    """함수 시그니처를 단순하게 유지하면서 CLI 옵션을 공유한다."""

    timeout_ms: int = 30_000
    dry_run: bool = False
    min_delay: float = 1.0
    max_delay: float = 2.0


RUNTIME = RuntimeConfig()
BUILD_FAILURES: list[dict[str, Any]] = []
LOGGER = logging.getLogger("business-doc-crawler")


# ---------------------------------------------------------------------------
# 입력 파일 탐색·정규화
# ---------------------------------------------------------------------------
def find_input_files(base_dir: Path) -> list[Path]:
    """BASE_DIR 바로 아래의 대상 CSV/XLSX만 안정적인 순서로 반환한다."""

    if not base_dir.exists() or not base_dir.is_dir():
        return []
    return sorted(
        (
            path
            for path in base_dir.iterdir()
            if path.is_file()
            and not path.name.startswith("~$")
            and "세부사업 예산편성현황" in path.name
            and path.suffix.lower() in {".csv", ".xlsx"}
        ),
        key=lambda path: path.name,
    )


def read_budget_file(path: Path) -> pd.DataFrame:
    """공공데이터에서 자주 쓰는 UTF-8/CP949 CSV와 XLSX를 모두 읽는다."""

    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")

    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError("CSV 인코딩 판별 실패: " + " | ".join(errors))


def normalize_text(text: str) -> str:
    """NFKC 후 공백·괄호·특수문자를 없애 표시 차이를 흡수한다."""

    if text is None or pd.isna(text):
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    # \w는 언더스코어도 포함하므로, 숫자·영문·한글만 명시적으로 남긴다.
    return re.sub(r"[^0-9a-z가-힣]", "", normalized)


def detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """완전일치→포함관계→유사도 순으로 열 이름을 자동 탐색한다."""

    columns = [str(column).strip() for column in df.columns]
    normalized_columns = {column: normalize_text(column) for column in columns}
    normalized_candidates = [normalize_text(candidate) for candidate in candidates]

    for candidate in normalized_candidates:
        for column, normalized_column in normalized_columns.items():
            if candidate and normalized_column == candidate:
                return column

    scored: list[tuple[float, str]] = []
    for column, normalized_column in normalized_columns.items():
        for candidate in normalized_candidates:
            if not candidate or not normalized_column:
                continue
            if candidate in normalized_column or normalized_column in candidate:
                score = 0.92 - abs(len(candidate) - len(normalized_column)) / 100
            else:
                score = difflib.SequenceMatcher(None, candidate, normalized_column).ratio()
            scored.append((score, column))

    if not scored:
        return None
    best_score, best_column = max(scored, key=lambda item: item[0])
    return best_column if best_score >= 0.70 else None


def _parse_year(value: Any) -> int | None:
    """숫자/문자 혼합 값에서 현실적인 4자리 회계연도만 꺼낸다."""

    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        year = int(value)
        return year if 1900 <= year <= 2100 else None
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", str(value))
    return int(match.group(1)) if match else None


def extract_year(df: pd.DataFrame, file_path: Path) -> int | None:
    """요구된 우선순위대로 연도 열을 먼저 보고, 그다음 파일명을 본다."""

    year_column = detect_column(df, COLUMN_CANDIDATES["year"])
    if year_column:
        years = [_parse_year(value) for value in df[year_column].dropna().tolist()]
        years = [year for year in years if year is not None]
        if years:
            return int(pd.Series(years).mode().iloc[0])

    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", file_path.stem)
    return int(match.group(1)) if match else None


def _clean_value(value: Any) -> str:
    """NaN을 빈 문자열로 바꾸고 화면 검색에 불필요한 양끝 공백을 없앤다."""

    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _failed_row(
    *,
    year: Any = "",
    source_file: str = "",
    ministry: str = "",
    activity_name: str = "",
    reason: str,
    searched_url: str = "",
) -> dict[str, Any]:
    return {
        "year": year,
        "source_file": source_file,
        "ministry": ministry,
        "activity_name": activity_name,
        "reason": reason,
        "searched_url": searched_url,
        "crawled_at": _now(),
    }


def build_targets(base_dir: Path) -> pd.DataFrame:
    """label=1 행을 표준 열 구조로 합치고, 잘못된 파일/행은 실패 목록에 남긴다."""

    BUILD_FAILURES.clear()
    input_files = find_input_files(base_dir)
    if not input_files:
        BUILD_FAILURES.append(
            _failed_row(reason=f"입력 파일 없음: {base_dir}", searched_url=SITE_URL)
        )
        return pd.DataFrame(columns=TARGET_COLUMNS)

    target_rows: list[dict[str, Any]] = []
    for file_path in input_files:
        try:
            df = read_budget_file(file_path)
        except Exception as exc:  # 한 파일 실패가 다른 연도 처리를 막지 않게 한다.
            BUILD_FAILURES.append(
                _failed_row(
                    source_file=file_path.name,
                    reason=f"입력 파일 읽기 실패: {type(exc).__name__}: {exc}",
                )
            )
            continue

        # 프로젝트 실데이터의 biodiv_label도 label의 명시적 별칭으로 지원한다.
        label_column = detect_column(
            df, ["label", "biodiv_label", "biodiversity_label", "분류라벨", "라벨"]
        )
        if not label_column:
            BUILD_FAILURES.append(
                _failed_row(
                    year=extract_year(df, file_path) or "",
                    source_file=file_path.name,
                    reason="label 컬럼 없음",
                )
            )
            continue

        numeric_labels = pd.to_numeric(df[label_column], errors="coerce")
        string_labels = df[label_column].astype(str).str.strip()
        positive_df = df[(numeric_labels == 1) | (string_labels == "1")].copy()
        if positive_df.empty:
            LOGGER.info("label=1 행 없음: %s", file_path.name)
            continue

        detected = {
            name: detect_column(df, candidates)
            for name, candidates in COLUMN_CANDIDATES.items()
        }
        activity_column = detected["activity_name"]
        if not activity_column:
            BUILD_FAILURES.append(
                _failed_row(
                    year=extract_year(df, file_path) or "",
                    source_file=file_path.name,
                    reason="세부사업명 컬럼 탐색 실패",
                )
            )
            continue

        fallback_year = extract_year(df, file_path)
        year_column = detected["year"]
        for index, row in positive_df.iterrows():
            row_year = _parse_year(row.get(year_column)) if year_column else None
            year = row_year or fallback_year
            activity_name = _clean_value(row.get(activity_column))
            ministry_column = detected["ministry"]
            ministry = _clean_value(row.get(ministry_column)) if ministry_column else ""

            if not year:
                BUILD_FAILURES.append(
                    _failed_row(
                        source_file=file_path.name,
                        ministry=ministry,
                        activity_name=activity_name,
                        reason=f"연도 추출 실패(원본 행 {index + 2})",
                    )
                )
                continue
            if not activity_name:
                BUILD_FAILURES.append(
                    _failed_row(
                        year=year,
                        source_file=file_path.name,
                        ministry=ministry,
                        reason=f"세부사업명 값 없음(원본 행 {index + 2})",
                    )
                )
                continue

            canonical = {
                "year": int(year),
                "source_file": file_path.name,
                "source_row": int(index) + 2,
                "label_column": label_column,
                "ministry": ministry,
                "activity_name": activity_name,
            }
            for name in (
                "account_name",
                "field_name",
                "sector_name",
                "program_name",
                "unit_name",
            ):
                column = detected[name]
                canonical[name] = _clean_value(row.get(column)) if column else ""
            target_rows.append(canonical)

    return pd.DataFrame(target_rows, columns=TARGET_COLUMNS)


# ---------------------------------------------------------------------------
# 열린재정 목록 탐색과 후보 매칭
# ---------------------------------------------------------------------------
def _year_url(year: int, page_index: int = 1) -> str:
    params = {
        "pageIndex": page_index,
        "pageSize": 5000,
        "totalCnt": 0,
        "acntYr": year,
        "chkoffcNm": "Y",
        "chkacntNm": "Y",
        "chkfldNm": "Y",
        "chksectNm": "Y",
        "chkpgmNm": "Y",
        "chkactvNm": "Y",
    }
    return f"{SITE_URL}?{urlencode(params)}"


def _dict_value(record: dict[str, Any], aliases: Iterable[str]) -> str:
    """JSON 필드도 컬럼명과 같은 정규화 규칙으로 찾는다."""

    normalized = {normalize_text(key): value for key, value in record.items()}
    for alias in aliases:
        value = normalized.get(normalize_text(alias))
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            cleaned = _clean_value(value)
            if cleaned:
                return cleaned
    return ""


def _iter_dicts(payload: Any) -> Iterator[dict[str, Any]]:
    """응답 포맷이 data/list/rows 중 무엇이든 내부 레코드를 순회한다."""

    stack = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            yield value
            stack.extend(child for child in value.values() if isinstance(child, (dict, list)))
        elif isinstance(value, list):
            stack.extend(value)


def _candidate_from_record(
    record: dict[str, Any], response_url: str, expected_year: int
) -> dict[str, Any] | None:
    activity = _dict_value(record, COLUMN_CANDIDATES["activity_name"])
    if not activity:
        return None
    record_year = _parse_year(_dict_value(record, COLUMN_CANDIDATES["year"]))
    if record_year and record_year != expected_year:
        return None

    href = _dict_value(
        record,
        ["detailUrl", "detail_url", "fileUrl", "downloadUrl", "href", "url", "link"],
    )
    if href and not href.lower().startswith(("http://", "https://", "/")):
        href = ""

    id_parts = []
    for key, value in record.items():
        normalized_key = normalize_text(key)
        if (
            (normalized_key.endswith("id") or normalized_key.endswith("seq") or normalized_key.endswith("cd"))
            and isinstance(value, (str, int, float))
            and _clean_value(value)
        ):
            id_parts.append(f"{key}={value}")

    return {
        "activity": activity,
        "ministry": _dict_value(record, COLUMN_CANDIDATES["ministry"]),
        "account": _dict_value(record, COLUMN_CANDIDATES["account_name"]),
        "field": _dict_value(record, COLUMN_CANDIDATES["field_name"]),
        "sector": _dict_value(record, COLUMN_CANDIDATES["sector_name"]),
        "program": _dict_value(record, COLUMN_CANDIDATES["program_name"]),
        "unit": _dict_value(record, COLUMN_CANDIDATES["unit_name"]),
        "href": urljoin(SITE_URL, href) if href else "",
        "source_url": response_url,
        "record_id": "|".join(sorted(id_parts)),
        "raw_text": json.dumps(record, ensure_ascii=False, default=str)[:4000],
        "origin": "network",
        "row_index": None,
    }


async def _consume_json_response(
    response: Any,
    expected_year: int,
    sink: list[dict[str, Any]],
) -> None:
    """가상 스크롤 표에도 대응하려고 XHR/Fetch JSON의 사업 레코드를 수집한다."""

    try:
        content_type = response.headers.get("content-type", "").lower()
        if "json" not in content_type:
            return
        payload = await response.json()
        for record in _iter_dicts(payload):
            candidate = _candidate_from_record(record, response.url, expected_year)
            if candidate:
                sink.append(candidate)
    except Exception:
        # 일부 응답은 압축/스트리밍되어 body를 다시 읽지 못할 수 있으므로 무시한다.
        return


async def _extract_dom_candidates(page: Page) -> list[dict[str, Any]]:
    """표 헤더를 읽어 열 위치를 추론하고 각 결과 행의 링크까지 보존한다."""

    locator = page.locator(ROW_SELECTOR)
    try:
        rows = await locator.evaluate_all(
            """
            (elements) => elements.map((row, rowIndex) => {
                const clean = (value) => (value || '').replace(/\\s+/g, ' ').trim();
                const cells = Array.from(row.querySelectorAll(':scope > td, :scope > [role="gridcell"]'));
                const cellTexts = cells.map(cell => clean(cell.innerText || cell.textContent));
                const table = row.closest('table');
                let headers = [];
                if (table) {
                    const headerRows = Array.from(table.querySelectorAll('thead tr'));
                    const headerRow = headerRows.length ? headerRows[headerRows.length - 1] : null;
                    if (headerRow) {
                        headers = Array.from(headerRow.querySelectorAll('th, [role="columnheader"]'))
                            .map(cell => clean(cell.innerText || cell.textContent));
                    }
                }
                const headerIndex = (words) => headers.findIndex(
                    header => words.some(word => header.replace(/\\s+/g, '').includes(word))
                );
                const valueFor = (words) => {
                    const index = headerIndex(words);
                    return index >= 0 && index < cellTexts.length ? cellTexts[index] : '';
                };
                const links = Array.from(row.querySelectorAll('a, button, [role="button"]')).map(el => ({
                    text: clean(el.innerText || el.textContent || el.getAttribute('title')),
                    href: el.tagName === 'A' ? (el.href || el.getAttribute('href') || '') : '',
                    onclick: el.getAttribute('onclick') || ''
                }));
                let activity = valueFor(['세부사업', '사업명', 'actv']);
                let activityLink = null;
                if (activity) {
                    activityLink = links.find(link => link.text && activity.includes(link.text));
                }
                if (!activityLink) {
                    activityLink = links
                        .filter(link => link.text && !/다운로드|엑셀|excel|첨부파일/i.test(link.text))
                        .sort((a, b) => b.text.length - a.text.length)[0] || null;
                }
                if (!activity && activityLink) activity = activityLink.text;
                return {
                    row_index: rowIndex,
                    raw_text: clean(row.innerText || row.textContent),
                    activity,
                    ministry: valueFor(['소관', '부처', '관서', 'offc']),
                    account: valueFor(['회계', 'acnt']),
                    field: valueFor(['분야', 'fld']),
                    sector: valueFor(['부문', 'sect']),
                    program: valueFor(['프로그램', 'pgm']),
                    unit: valueFor(['단위사업']),
                    href: activityLink ? activityLink.href : '',
                    onclick: activityLink ? activityLink.onclick : '',
                    links
                };
            })
            """
        )
    except PlaywrightError:
        return []

    candidates: list[dict[str, Any]] = []
    for row in rows:
        raw_text = _clean_value(row.get("raw_text"))
        if not raw_text or re.search(r"총\s*\d+\s*건|조회된.*없", raw_text):
            continue
        if not row.get("activity") and len(raw_text) < 2:
            continue
        row.update(
            {
                "source_url": page.url,
                "record_id": "",
                "origin": "dom",
            }
        )
        candidates.append(row)
    return candidates


def _business_key(candidate: dict[str, Any]) -> tuple[str, ...]:
    return (
        normalize_text(candidate.get("ministry", "")),
        normalize_text(candidate.get("activity", "")),
        normalize_text(candidate.get("account", "")),
        normalize_text(candidate.get("program", "")),
        normalize_text(candidate.get("unit", "")),
    )


def _deduplicate_candidates(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """동일 JSON 객체의 중복 캡처만 제거하고, 서로 다른 사업 후보는 유지한다."""

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in candidates:
        if candidate.get("origin") == "dom":
            unique_marker = "|".join(
                [
                    _clean_value(candidate.get("origin")),
                    _clean_value(candidate.get("listing_url")),
                    _clean_value(candidate.get("row_index")),
                ]
            )
        else:
            unique_marker = (
                normalize_text(candidate.get("record_id", ""))
                or normalize_text(candidate.get("href", ""))
                or normalize_text(candidate.get("raw_text", ""))
            )
        key = _business_key(candidate) + (unique_marker,)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _merge_dom_hints(
    network_candidates: list[dict[str, Any]], dom_candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """전체 JSON 목록을 쓰되, 상세 클릭에 필요한 DOM href/행 번호를 보강한다."""

    if not network_candidates:
        return _deduplicate_candidates(dom_candidates)
    for candidate in network_candidates:
        matches = [dom for dom in dom_candidates if _business_key(dom)[:2] == _business_key(candidate)[:2]]
        if len(matches) == 1:
            candidate["href"] = candidate.get("href") or matches[0].get("href", "")
            candidate["row_index"] = matches[0].get("row_index")
    return _deduplicate_candidates(network_candidates)


def _activity_match(candidate: dict[str, Any], target: dict[str, Any]) -> int:
    """2=완전일치, 1=정규화 포함관계, 0=불일치로 점수화한다."""

    target_name = normalize_text(target.get("activity_name", ""))
    candidate_name = normalize_text(candidate.get("activity", ""))
    if candidate_name and candidate_name == target_name:
        return 2
    if candidate_name and target_name and (
        target_name in candidate_name or candidate_name in target_name
    ):
        return 1
    if not candidate_name and target_name in normalize_text(candidate.get("raw_text", "")):
        return 1
    return 0


def _ministry_matches(candidate: dict[str, Any], target: dict[str, Any]) -> bool:
    target_ministry = normalize_text(target.get("ministry", ""))
    if not target_ministry:
        return True
    candidate_ministry = normalize_text(candidate.get("ministry", ""))
    if candidate_ministry:
        return (
            target_ministry == candidate_ministry
            or target_ministry in candidate_ministry
            or candidate_ministry in target_ministry
        )
    return target_ministry in normalize_text(candidate.get("raw_text", ""))


def _select_candidates(
    candidates: list[dict[str, Any]], target: dict[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    """완전일치와 부처 일치를 우선하며 모호하면 후보 전체를 반환한다."""

    scored = [(candidate, _activity_match(candidate, target)) for candidate in candidates]
    best_activity_score = max((score for _, score in scored), default=0)
    if best_activity_score == 0:
        return [], "검색 결과 없음"
    activity_candidates = [
        candidate for candidate, score in scored if score == best_activity_score
    ]

    if target.get("ministry"):
        candidates_with_ministry = [
            candidate
            for candidate in activity_candidates
            if candidate.get("ministry")
            or normalize_text(target["ministry"])
            in normalize_text(candidate.get("raw_text", ""))
        ]
        ministry_candidates = [
            candidate for candidate in activity_candidates if _ministry_matches(candidate, target)
        ]
        if ministry_candidates:
            activity_candidates = ministry_candidates
        elif candidates_with_ministry:
            return [], "사업명은 일치하지만 부처명이 일치하지 않음"

    match_type = "완전일치" if best_activity_score == 2 else "정규화 포함일치"
    return activity_candidates, match_type


def _candidate_summary(candidate: dict[str, Any], index: int, total: int) -> str:
    details = {
        "candidate": f"{index}/{total}",
        "ministry": candidate.get("ministry", ""),
        "activity": candidate.get("activity", ""),
        "account": candidate.get("account", ""),
        "program": candidate.get("program", ""),
        "unit": candidate.get("unit", ""),
        "href": candidate.get("href", ""),
        "record_id": candidate.get("record_id", ""),
    }
    return "다중 후보 - 다운로드 보류: " + json.dumps(details, ensure_ascii=False)


async def _save_debug_html(page: Page, save_dir: Path, label: str) -> Path | None:
    """셀렉터 실패 시 재현 가능한 HTML을 남기되 파일명은 안전하게 제한한다."""

    try:
        debug_dir = save_dir / "debug_html"
        debug_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = _safe_component(label, max_length=80)
        path = debug_dir / f"{timestamp}_{name}.html"
        parts = [f"<!-- URL: {page.url} -->\n", await page.content()]
        for frame_index, frame in enumerate(page.frames[1:], start=1):
            try:
                parts.append(
                    f"\n<!-- FRAME {frame_index}: {frame.url} -->\n{await frame.content()}"
                )
            except PlaywrightError:
                pass
        path.write_text("\n".join(parts), encoding="utf-8")
        return path
    except Exception as exc:
        LOGGER.warning("debug HTML 저장 실패: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 상세 진입과 첨부파일 다운로드
# ---------------------------------------------------------------------------
def _safe_component(value: Any, max_length: int = 80) -> str:
    """Windows 금지문자·제어문자·예약 끝문자를 제거한다."""

    text = unicodedata.normalize("NFKC", _clean_value(value))
    text = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    return (text or "미상")[:max_length].rstrip(" .")


def _extension_from_content_type(content_type: str) -> str:
    lowered = content_type.lower().split(";", 1)[0].strip()
    return {
        "application/pdf": ".pdf",
        "application/zip": ".zip",
        "application/x-zip-compressed": ".zip",
        "application/haansofthwp": ".hwp",
        "application/x-hwp": ".hwp",
        "application/vnd.hancom.hwp": ".hwp",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    }.get(lowered, "")


def _filename_from_disposition(content_disposition: str) -> str:
    """filename과 RFC 5987 filename*을 모두 해석한다."""

    if not content_disposition:
        return ""
    message = Message()
    message["content-disposition"] = content_disposition
    filename = message.get_filename()
    if filename:
        if isinstance(filename, tuple):
            return unquote(filename[2])
        return str(filename)
    match = re.search(r"filename\*\s*=\s*[^']*''([^;]+)", content_disposition, re.I)
    return unquote(match.group(1)) if match else ""


def _filename_from_url(url: str) -> str:
    return unquote(Path(urlparse(url).path).name)


def _destination_path(save_dir: Path, target: dict[str, Any], original_name: str) -> Path:
    original = _safe_component(Path(original_name).name, max_length=110)
    extension = Path(original_name).suffix.lower()
    if extension and not original.lower().endswith(extension):
        original += extension
    filename = "_".join(
        [
            _safe_component(target.get("year"), 4),
            _safe_component(target.get("ministry") or "부처미상", 45),
            _safe_component(target.get("activity_name"), 75),
            original,
        ]
    )
    # Windows MAX_PATH 여유를 위해 최종 파일명도 제한한다.
    if len(filename) > 230:
        suffix = Path(filename).suffix
        filename = filename[: 230 - len(suffix)].rstrip(" ._") + suffix
    return save_dir / filename


async def _save_playwright_download(
    download: Download, target: dict[str, Any], save_dir: Path
) -> Path:
    original_name = download.suggested_filename or "사업설명자료"
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"지원하지 않는 다운로드 확장자: {extension or '없음'}")
    destination = _destination_path(save_dir, target, original_name)
    if destination.exists():
        await download.cancel()
        return destination
    await download.save_as(str(destination))
    return destination


async def _download_url(
    page: Page,
    url: str,
    target: dict[str, Any],
    save_dir: Path,
    name_hint: str = "",
) -> tuple[Path | None, str]:
    """동일 브라우저 세션의 쿠키를 쓰는 Playwright request로 정적 링크를 받는다."""

    try:
        response = await page.request.get(url, timeout=RUNTIME.timeout_ms)
        if not response.ok:
            return None, f"HTTP {response.status} {response.status_text}"
        headers = response.headers
        original_name = (
            _filename_from_disposition(headers.get("content-disposition", ""))
            or _filename_from_url(url)
            or name_hint
            or "사업설명자료"
        )
        extension = Path(original_name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            detected = _extension_from_content_type(headers.get("content-type", ""))
            if detected:
                original_name = f"{Path(original_name).stem or '사업설명자료'}{detected}"
                extension = detected
        if extension not in ALLOWED_EXTENSIONS:
            return None, f"지원 파일 형식 아님({headers.get('content-type', '')})"

        destination = _destination_path(save_dir, target, original_name)
        if destination.exists():
            return destination, "이미 존재하여 건너뜀"
        destination.write_bytes(await response.body())
        return destination, "다운로드 완료"
    except Exception as exc:
        return None, f"URL 다운로드 실패: {type(exc).__name__}: {exc}"


async def _find_clickable_in_row(row: Locator, activity_name: str) -> Locator | None:
    links = row.locator("a, button, [role='button']")
    count = await links.count()
    normalized_activity = normalize_text(activity_name)
    fallback: Locator | None = None
    for index in range(count):
        locator = links.nth(index)
        try:
            if not await locator.is_visible():
                continue
            text = _clean_value(await locator.inner_text())
            if not text:
                text = _clean_value(await locator.get_attribute("title"))
            normalized_text = normalize_text(text)
            if normalized_text == normalized_activity:
                return locator
            if normalized_text and (
                normalized_text in normalized_activity or normalized_activity in normalized_text
            ):
                fallback = fallback or locator
        except PlaywrightError:
            continue
    return fallback


async def _filter_listing_by_activity(page: Page, activity_name: str) -> bool:
    """가상 표에서 행이 보이지 않을 때 세부사업 검색 입력을 보조 경로로 사용한다."""

    input_selectors = [
        "input[name='actvNm']",
        "#actvNm",
        "input[name*='actv' i]",
        "input[title*='세부사업']",
        "input[placeholder*='세부사업']",
    ]
    for selector in input_selectors:
        locator = page.locator(selector)
        for index in range(await locator.count()):
            candidate = locator.nth(index)
            try:
                if not await candidate.is_visible():
                    continue
                await candidate.fill(activity_name)
                await candidate.press("Enter")
                await page.wait_for_timeout(1500)
                return True
            except PlaywrightError:
                continue
    return False


async def _open_candidate(
    page: Page, target: dict[str, Any], candidate: dict[str, Any]
) -> tuple[Page, list[Path]]:
    """직접 href를 우선하고, 없으면 새로 읽은 결과 행을 클릭한다."""

    href = _clean_value(candidate.get("href"))
    parsed_href = urlparse(href)
    # href="#"인 JavaScript 링크는 현재 목록을 상세 화면으로 오인하지 않게 클릭 경로로 보낸다.
    href_is_placeholder = (
        not href
        or href.lower().startswith("javascript:")
        or href in {"#", "/#"}
        or (
            parsed_href.path == urlparse(target["_search_url"]).path
            and not parsed_href.query
            and bool(parsed_href.fragment)
        )
    )
    if not href_is_placeholder:
        extension = Path(urlparse(href).path).suffix.lower()
        if extension in ALLOWED_EXTENSIONS:
            path, reason = await _download_url(page, href, target, Path(target["_save_dir"]))
            if path:
                return page, [path]
            raise RuntimeError(reason)
        await page.goto(href, wait_until="domcontentloaded", timeout=RUNTIME.timeout_ms)
        try:
            await page.wait_for_load_state("networkidle", timeout=min(RUNTIME.timeout_ms, 10_000))
        except PlaywrightTimeoutError:
            pass
        return page, []

    search_url = target["_search_url"]
    await page.goto(search_url, wait_until="domcontentloaded", timeout=RUNTIME.timeout_ms)
    try:
        await page.wait_for_selector(ROW_SELECTOR, timeout=RUNTIME.timeout_ms)
    except PlaywrightTimeoutError:
        pass

    dom_candidates = await _extract_dom_candidates(page)
    matching, _ = _select_candidates(dom_candidates, target)
    if len(matching) != 1:
        await _filter_listing_by_activity(page, target["activity_name"])
        dom_candidates = await _extract_dom_candidates(page)
        matching, _ = _select_candidates(dom_candidates, target)
    if len(matching) != 1 or matching[0].get("row_index") is None:
        raise RuntimeError("매칭된 결과 행을 현재 DOM에서 다시 찾지 못함")

    row = page.locator(ROW_SELECTOR).nth(int(matching[0]["row_index"]))
    clickable = await _find_clickable_in_row(row, target["activity_name"])
    if clickable is None:
        raise RuntimeError("결과 행에서 상세 링크를 찾지 못함")

    before_pages = list(page.context.pages)
    direct_downloads: list[Path] = []
    try:
        async with page.expect_download(timeout=3000) as download_info:
            await clickable.click(timeout=RUNTIME.timeout_ms)
        direct_downloads.append(
            await _save_playwright_download(
                await download_info.value, target, Path(target["_save_dir"])
            )
        )
        return page, direct_downloads
    except PlaywrightTimeoutError:
        # 상세 화면 이동은 다운로드 이벤트가 발생하지 않는 것이 정상이다.
        pass

    await page.wait_for_timeout(1000)
    new_pages = [candidate_page for candidate_page in page.context.pages if candidate_page not in before_pages]
    detail_page = new_pages[-1] if new_pages else page
    try:
        await detail_page.wait_for_load_state("domcontentloaded", timeout=RUNTIME.timeout_ms)
    except PlaywrightTimeoutError:
        pass
    return detail_page, direct_downloads


async def _attachment_locators(page: Page) -> list[tuple[Any, Locator, str, str]]:
    """메인 문서와 iframe 안의 파일 링크/버튼을 중복 없이 찾는다."""

    results: list[tuple[Any, Locator, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for frame in page.frames:
        locators = frame.locator(ATTACHMENT_SELECTOR)
        try:
            count = min(await locators.count(), 100)
        except PlaywrightError:
            continue
        for index in range(count):
            locator = locators.nth(index)
            try:
                if not await locator.is_visible():
                    continue
                href = _clean_value(await locator.get_attribute("href"))
                if href:
                    href = urljoin(frame.url or page.url, href)
                text = _clean_value(await locator.inner_text()) or _clean_value(
                    await locator.get_attribute("title")
                )
                onclick = _clean_value(await locator.get_attribute("onclick"))
                key = (href, text, onclick)
                if key in seen:
                    continue
                seen.add(key)
                results.append((frame, locator, href, text))
            except PlaywrightError:
                continue
    return results


async def download_business_doc(page: Page, target: dict, save_dir: Path) -> dict:
    """선택한 사업 상세로 들어가 허용 형식의 첨부파일을 모두 저장한다."""

    target = dict(target)
    target["_save_dir"] = str(save_dir)
    candidate = target["_candidate"]
    detail_page: Page = page
    opened_new_page = False
    downloaded: list[Path] = []
    errors: list[str] = []

    try:
        original_pages = set(page.context.pages)
        detail_page, direct_downloads = await _open_candidate(page, target, candidate)
        opened_new_page = detail_page is not page and detail_page not in original_pages
        downloaded.extend(direct_downloads)

        if not direct_downloads:
            attachments = await _attachment_locators(detail_page)
            if not attachments:
                debug_path = await _save_debug_html(
                    detail_page,
                    save_dir,
                    f"{target['year']}_{target['activity_name']}_download_button_missing",
                )
                return {
                    "ok": False,
                    "reason": "다운로드 버튼 없음"
                    + (f" (debug: {debug_path.name})" if debug_path else ""),
                    "files": [],
                    "source_url": detail_page.url,
                }

            for _, locator, href, text in attachments:
                if href and not href.lower().startswith("javascript:"):
                    path, reason = await _download_url(
                        detail_page, href, target, save_dir, name_hint=text
                    )
                    if path:
                        if path not in downloaded:
                            downloaded.append(path)
                    else:
                        errors.append(f"{text or href}: {reason}")
                    continue

                try:
                    async with detail_page.expect_download(
                        timeout=RUNTIME.timeout_ms
                    ) as download_info:
                        await locator.click(timeout=RUNTIME.timeout_ms)
                    path = await _save_playwright_download(
                        await download_info.value, target, save_dir
                    )
                    if path not in downloaded:
                        downloaded.append(path)
                except Exception as exc:
                    errors.append(f"{text or '동적 버튼'}: {type(exc).__name__}: {exc}")

        if downloaded:
            return {
                "ok": True,
                "reason": "; ".join(errors),
                "files": downloaded,
                "source_url": detail_page.url,
            }
        return {
            "ok": False,
            "reason": "다운로드 실패: " + ("; ".join(errors) or "원인 불명"),
            "files": [],
            "source_url": detail_page.url,
        }
    except PlaywrightTimeoutError as exc:
        await _save_debug_html(
            detail_page, save_dir, f"{target['year']}_{target['activity_name']}_timeout"
        )
        return {
            "ok": False,
            "reason": f"사이트 응답 timeout({RUNTIME.timeout_ms}ms): {exc}",
            "files": [],
            "source_url": detail_page.url,
        }
    except Exception as exc:
        await _save_debug_html(
            detail_page, save_dir, f"{target['year']}_{target['activity_name']}_download_error"
        )
        return {
            "ok": False,
            "reason": f"다운로드 실패: {type(exc).__name__}: {exc}",
            "files": [],
            "source_url": detail_page.url,
        }
    finally:
        if opened_new_page and not detail_page.is_closed():
            await detail_page.close()


async def crawl_one_year(
    page: Page, year: int, targets: pd.DataFrame, save_dir: Path
) -> tuple[list[dict], list[dict]]:
    """연도 목록을 한 번 읽고, 고유 사업별로 매칭·다운로드를 계속 수행한다."""

    success_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    search_url = _year_url(year)
    network_candidates: list[dict[str, Any]] = []
    response_tasks: set[asyncio.Task[Any]] = set()

    def on_response(response: Any) -> None:
        if response.request.resource_type not in {"xhr", "fetch"}:
            return
        task = asyncio.create_task(
            _consume_json_response(response, year, network_candidates)
        )
        response_tasks.add(task)
        task.add_done_callback(response_tasks.discard)

    page.on("response", on_response)
    all_dom_candidates: list[dict[str, Any]] = []
    try:
        # 한 연도의 원자료가 5,000건을 넘을 수 있으므로 총 건수에 따라 다음 페이지도 읽는다.
        page_index = 1
        total_pages: int | None = None
        while page_index <= (total_pages or 20):
            current_url = _year_url(year, page_index)
            network_start = len(network_candidates)
            await page.goto(
                current_url, wait_until="domcontentloaded", timeout=RUNTIME.timeout_ms
            )
            try:
                await page.wait_for_load_state(
                    "networkidle", timeout=min(RUNTIME.timeout_ms, 15_000)
                )
            except PlaywrightTimeoutError:
                pass
            try:
                await page.wait_for_selector(ROW_SELECTOR, timeout=RUNTIME.timeout_ms)
            except PlaywrightTimeoutError:
                pass
            await page.wait_for_timeout(1000)
            if response_tasks:
                await asyncio.gather(*list(response_tasks), return_exceptions=True)

            current_dom = await _extract_dom_candidates(page)
            for candidate in current_dom:
                candidate["listing_url"] = current_url
            all_dom_candidates.extend(current_dom)
            for candidate in network_candidates[network_start:]:
                candidate["listing_url"] = current_url

            try:
                body_text = _clean_value(await page.locator("body").inner_text())
            except PlaywrightError:
                body_text = ""
            total_match = re.search(r"총\s*([\d,]+)\s*건", body_text)
            if total_match:
                total_count = int(total_match.group(1).replace(",", ""))
                total_pages = max(1, (total_count + 4999) // 5000)

            current_network = _deduplicate_candidates(
                network_candidates[network_start:]
            )
            page_record_count = len(current_network) or len(current_dom)
            if total_pages is not None and page_index >= total_pages:
                break
            if total_pages is None and page_record_count < 5000:
                break
            page_index += 1
    except PlaywrightTimeoutError as exc:
        debug_path = await _save_debug_html(page, save_dir, f"{year}_year_page_timeout")
        reason = f"연도 페이지 timeout({RUNTIME.timeout_ms}ms): {exc}"
        if debug_path:
            reason += f" (debug: {debug_path.name})"
        for target in targets.to_dict("records"):
            failed_rows.append(
                _failed_row(
                    year=year,
                    source_file=target["source_file"],
                    ministry=target["ministry"],
                    activity_name=target["activity_name"],
                    reason=reason,
                    searched_url=search_url,
                )
            )
        return success_rows, failed_rows
    except Exception as exc:
        reason = f"연도 페이지 접근 실패: {type(exc).__name__}: {exc}"
        for target in targets.to_dict("records"):
            failed_rows.append(
                _failed_row(
                    year=year,
                    source_file=target["source_file"],
                    ministry=target["ministry"],
                    activity_name=target["activity_name"],
                    reason=reason,
                    searched_url=search_url,
                )
            )
        return success_rows, failed_rows
    finally:
        page.remove_listener("response", on_response)

    candidates = _merge_dom_hints(network_candidates, all_dom_candidates)
    if not candidates:
        body_text = _clean_value(await page.locator("body").inner_text())
        reason = "검색 결과 없음" if re.search(r"총\s*0\s*건", body_text) else "검색 결과 목록 셀렉터 탐색 실패"
        debug_path = await _save_debug_html(page, save_dir, f"{year}_result_structure_missing")
        if debug_path:
            reason += f" (debug: {debug_path.name})"
        for target in targets.to_dict("records"):
            failed_rows.append(
                _failed_row(
                    year=year,
                    source_file=target["source_file"],
                    ministry=target["ministry"],
                    activity_name=target["activity_name"],
                    reason=reason,
                    searched_url=search_url,
                )
            )
        return success_rows, failed_rows

    # 동일 핵심 키는 한 번만 내려받고 전체 원본 행은 crawl_targets.csv에 보존한다.
    work = targets.copy()
    work["_crawl_key"] = work.apply(
        lambda row: "|".join(
            [
                str(row["year"]),
                normalize_text(row["ministry"]),
                normalize_text(row["activity_name"]),
            ]
        ),
        axis=1,
    )
    unique_targets = work.drop_duplicates("_crawl_key").drop(columns="_crawl_key")
    records = unique_targets.to_dict("records")
    iterator = tqdm(records, desc=f"{year}년", unit="사업") if tqdm else records

    for target in iterator:
        try:
            matched, match_type = _select_candidates(candidates, target)
            if not matched:
                failed_rows.append(
                    _failed_row(
                        year=year,
                        source_file=target["source_file"],
                        ministry=target["ministry"],
                        activity_name=target["activity_name"],
                        reason=match_type,
                        searched_url=search_url,
                    )
                )
                continue

            if len(matched) > 1:
                # 요구사항대로 후보를 하나 임의 선택하지 않고, 후보별 행을 모두 기록한다.
                for index, candidate in enumerate(matched, start=1):
                    failed_rows.append(
                        _failed_row(
                            year=year,
                            source_file=target["source_file"],
                            ministry=target["ministry"],
                            activity_name=target["activity_name"],
                            reason=_candidate_summary(candidate, index, len(matched)),
                            searched_url=search_url,
                        )
                    )
                continue

            candidate = matched[0]
            if RUNTIME.dry_run:
                success_rows.append(
                    {
                        "year": year,
                        "source_file": target["source_file"],
                        "ministry": target["ministry"],
                        "program_name": target["program_name"],
                        "activity_name": target["activity_name"],
                        "downloaded_file": f"[DRY-RUN] {match_type}: {candidate.get('activity', '')}",
                        "source_url": candidate.get("href") or search_url,
                        "crawled_at": _now(),
                    }
                )
                continue

            download_target = dict(target)
            download_target["_candidate"] = candidate
            download_target["_search_url"] = candidate.get("listing_url") or search_url
            result = await download_business_doc(page, download_target, save_dir)
            if result["ok"]:
                for downloaded_file in result["files"]:
                    success_rows.append(
                        {
                            "year": year,
                            "source_file": target["source_file"],
                            "ministry": target["ministry"],
                            "program_name": target["program_name"],
                            "activity_name": target["activity_name"],
                            "downloaded_file": str(Path(downloaded_file).name),
                            "source_url": result["source_url"],
                            "crawled_at": _now(),
                        }
                    )
                if result.get("reason"):
                    failed_rows.append(
                        _failed_row(
                            year=year,
                            source_file=target["source_file"],
                            ministry=target["ministry"],
                            activity_name=target["activity_name"],
                            reason="일부 첨부파일 다운로드 실패: " + result["reason"],
                            searched_url=search_url,
                        )
                    )
            else:
                failed_rows.append(
                    _failed_row(
                        year=year,
                        source_file=target["source_file"],
                        ministry=target["ministry"],
                        activity_name=target["activity_name"],
                        reason=result["reason"],
                        searched_url=search_url,
                    )
                )
        except Exception as exc:
            failed_rows.append(
                _failed_row(
                    year=year,
                    source_file=target["source_file"],
                    ministry=target["ministry"],
                    activity_name=target["activity_name"],
                    reason=f"사업 처리 예외: {type(exc).__name__}: {exc}",
                    searched_url=search_url,
                )
            )
        finally:
            # 성공/실패와 무관하게 요청 간 간격을 두어 사이트 부하를 낮춘다.
            await asyncio.sleep(random.uniform(RUNTIME.min_delay, RUNTIME.max_delay))

    return success_rows, failed_rows


# ---------------------------------------------------------------------------
# 로그 저장과 CLI 진입점
# ---------------------------------------------------------------------------
def save_logs(
    success_rows: list[dict],
    failed_rows: list[dict],
    targets: pd.DataFrame,
    save_dir: Path,
) -> None:
    """Excel에서도 한글이 바로 열리도록 UTF-8 BOM CSV로 매 실행 결과를 저장한다."""

    save_dir.mkdir(parents=True, exist_ok=True)
    success_df = pd.DataFrame(success_rows).reindex(columns=SUCCESS_COLUMNS)
    failed_df = pd.DataFrame(failed_rows).reindex(columns=FAILED_COLUMNS)
    target_df = targets.reindex(columns=TARGET_COLUMNS).copy()
    success_df.to_csv(save_dir / "crawl_success.csv", index=False, encoding="utf-8-sig")
    failed_df.to_csv(save_dir / "crawl_failed.csv", index=False, encoding="utf-8-sig")
    target_df.to_csv(save_dir / "crawl_targets.csv", index=False, encoding="utf-8-sig")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="label=1 세부사업의 열린재정 사업설명자료 다운로드"
    )
    parser.add_argument("--year", type=int, help="특정 회계연도만 실행(예: 2026)")
    parser.add_argument("--headed", action="store_true", help="Chromium 창을 표시")
    parser.add_argument("--dry-run", action="store_true", help="매칭만 확인하고 다운로드하지 않음")
    parser.add_argument("--timeout", type=int, default=30_000, help="timeout(ms), 기본 30000")
    parser.add_argument("--min-delay", type=float, default=1.0, help="사업별 최소 대기(초)")
    parser.add_argument("--max-delay", type=float, default=2.0, help="사업별 최대 대기(초)")
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR, help="입력 파일 폴더")
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR, help="다운로드/로그 폴더")
    return parser.parse_args()


async def main() -> int:
    """대상을 먼저 확정·로그화한 뒤 브라우저를 열어 모든 연도를 순차 처리한다."""

    global RUNTIME
    args = _parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout은 1 이상이어야 합니다.")
    if args.min_delay < 0 or args.max_delay < args.min_delay:
        raise SystemExit("대기시간은 0 이상이고 max-delay >= min-delay 여야 합니다.")
    RUNTIME = RuntimeConfig(
        timeout_ms=args.timeout,
        dry_run=args.dry_run,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    targets = build_targets(args.base_dir)
    if args.year is not None and not targets.empty:
        targets = targets[targets["year"] == args.year].copy()

    success_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = list(BUILD_FAILURES)
    if targets.empty:
        reason = (
            f"선택 연도({args.year})의 label=1 대상 없음"
            if args.year is not None and find_input_files(args.base_dir)
            else "수집 가능한 label=1 대상 없음"
        )
        if not failed_rows:
            failed_rows.append(_failed_row(year=args.year or "", reason=reason))
        save_logs(success_rows, failed_rows, targets, args.save_dir)
        LOGGER.error(reason)
        return 1

    # tqdm은 화면 표시용일 뿐이므로 없어도 계속하고, Playwright만 필수로 둔다.
    if tqdm is None:
        LOGGER.warning("tqdm 미설치: 진행률 표시 없이 계속합니다.")
    if async_playwright is None:
        reason = "필수 패키지 미설치: playwright"
        failed_rows.append(_failed_row(reason=reason))
        save_logs(success_rows, failed_rows, targets, args.save_dir)
        LOGGER.error("%s (pip install pandas openpyxl playwright tqdm)", reason)
        return 2

    LOGGER.info(
        "대상 %d행, 고유 핵심키 %d개, 연도 %s",
        len(targets),
        targets.apply(
            lambda row: (row["year"], normalize_text(row["ministry"]), normalize_text(row["activity_name"])),
            axis=1,
        ).nunique(),
        sorted(targets["year"].unique().tolist()),
    )

    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=not args.headed)
            context = await browser.new_context(
                accept_downloads=True,
                locale="ko-KR",
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
                ),
            )
            page = await context.new_page()
            page.set_default_timeout(args.timeout)
            page.set_default_navigation_timeout(args.timeout)

            for year, year_targets in targets.groupby("year", sort=True):
                try:
                    year_success, year_failed = await crawl_one_year(
                        page, int(year), year_targets.copy(), args.save_dir
                    )
                    success_rows.extend(year_success)
                    failed_rows.extend(year_failed)
                except Exception as exc:
                    # 연도 단위의 예상 밖 실패도 다음 연도로 계속 진행한다.
                    for target in year_targets.to_dict("records"):
                        failed_rows.append(
                            _failed_row(
                                year=year,
                                source_file=target["source_file"],
                                ministry=target["ministry"],
                                activity_name=target["activity_name"],
                                reason=f"연도 처리 예외: {type(exc).__name__}: {exc}",
                                searched_url=_year_url(int(year)),
                            )
                        )
                finally:
                    # 긴 작업 도중 종료되어도 완료된 연도 로그는 보존한다.
                    save_logs(success_rows, failed_rows, targets, args.save_dir)

            await context.close()
            await browser.close()
    except Exception as exc:
        failed_rows.append(
            _failed_row(reason=f"브라우저 시작/실행 실패: {type(exc).__name__}: {exc}")
        )
        save_logs(success_rows, failed_rows, targets, args.save_dir)
        LOGGER.exception("브라우저 실행 실패")
        return 2

    save_logs(success_rows, failed_rows, targets, args.save_dir)
    LOGGER.info(
        "완료: 성공 로그 %d건, 실패 로그 %d건, 저장 경로 %s",
        len(success_rows),
        len(failed_rows),
        args.save_dir,
    )
    return 0 if not failed_rows else 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    raise SystemExit(asyncio.run(main()))
