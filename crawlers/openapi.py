"""열린재정·지방재정365 Open API 공통 호출 모듈.

두 사이트의 Open API는 요청 인자(Key, Type, pIndex, pSize)와 응답 구조가
같으므로 인증키 읽기와 페이지 단위 호출을 한 곳에서 처리한다.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


LOG = logging.getLogger("openapi")

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"

# 한 번에 요청할 수 있는 최대 건수(초과 시 ERROR-336).
MAX_PAGE_SIZE = 1000
# 인증키 없이 호출하면 견본 키가 적용되어 1페이지 5건만 받을 수 있다.
SAMPLE_PAGE_SIZE = 5

NO_DATA_CODE = "INFO-200"
OK_CODE = "INFO-000"


class OpenApiError(RuntimeError):
    """Open API가 오류 코드를 돌려주었거나 응답을 해석하지 못했을 때."""


def load_env(path: Path = ENV_PATH) -> None:
    """`.env`의 `이름=값`을 환경 변수로 읽는다. 이미 설정된 값은 유지한다."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(name.strip(), value)


def get_api_key(env_name: str) -> str:
    """환경 변수 또는 `.env`에서 인증키를 읽는다. 없으면 빈 문자열."""
    load_env()
    return os.environ.get(env_name, "").strip()


def _result(payload: Any) -> dict[str, Any] | None:
    # 자료가 없거나 오류일 때는 서비스명 없이 RESULT만 온다.
    result = payload.get("RESULT") if isinstance(payload, dict) else None
    if isinstance(result, list):
        result = result[0] if result else None
    return result if isinstance(result, dict) else None


def parse_response(text: str, service: str) -> tuple[int, list[dict[str, Any]]]:
    """응답 본문에서 (전체 건수, 행 목록)을 꺼낸다."""
    try:
        payload = json.loads(text)
        # 열린재정은 JSON을 문자열로 한 번 더 감싸서 돌려준다.
        if isinstance(payload, str):
            payload = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise OpenApiError(f"응답을 JSON으로 해석하지 못했습니다: {text[:200]}") from exc

    result = _result(payload)
    if result is not None:
        if result.get("CODE") == NO_DATA_CODE:
            return 0, []
        raise OpenApiError(f"{result.get('CODE')}: {result.get('MESSAGE')}")

    try:
        head, body = payload[service][0]["head"], payload[service][1]["row"]
        total = int(head[0]["list_total_count"])
        code = head[1]["RESULT"]["CODE"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise OpenApiError(f"예상하지 못한 응답 구조입니다: {text[:200]}") from exc
    if code != OK_CODE:
        raise OpenApiError(f"{code}: {head[1]['RESULT'].get('MESSAGE')}")
    return total, list(body)


def fetch_page(
    url: str,
    params: dict[str, str],
    api_key: str = "",
    page: int = 1,
    page_size: int = MAX_PAGE_SIZE,
    timeout: float = 60.0,
    retries: int = 3,
) -> tuple[int, list[dict[str, Any]]]:
    """한 페이지를 호출해 (전체 건수, 행 목록)을 돌려준다."""
    query = {"Type": "json", "pIndex": str(page), "pSize": str(page_size), **params}
    if api_key:
        query["Key"] = api_key
    request = Request(f"{url}?{urlencode(query)}", headers={"User-Agent": "biofin-crawler"})
    service = url.rstrip("/").rsplit("/", 1)[-1]

    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                text = response.read().decode("utf-8")
            return parse_response(text, service)
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == retries:
                raise OpenApiError(f"{service} 호출 실패: {exc}") from exc
            LOG.warning("%s 호출 실패(%d/%d): %s", service, attempt, retries, exc)
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def iter_rows(
    url: str,
    params: dict[str, str],
    api_key: str = "",
    limit: int = 0,
    delay: float = 0.2,
    timeout: float = 60.0,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """조건에 맞는 행을 (전체 건수, 행)으로 차례로 돌려준다. `limit=0`은 전체."""
    if not api_key and not 0 < limit <= SAMPLE_PAGE_SIZE:
        raise OpenApiError(
            f"인증키 없이는 {SAMPLE_PAGE_SIZE}건까지만 받을 수 있습니다. "
            f"{ENV_PATH}에 인증키를 넣어 주세요."
        )
    page_size = MAX_PAGE_SIZE if api_key else SAMPLE_PAGE_SIZE
    if limit:
        page_size = min(page_size, limit)

    yielded = 0
    page = 1
    while True:
        total, rows = fetch_page(url, params, api_key, page, page_size, timeout)
        for row in rows:
            yield total, row
            yielded += 1
            if limit and yielded >= limit:
                return
        if not rows or yielded >= total:
            return
        page += 1
        if delay:
            time.sleep(delay)
