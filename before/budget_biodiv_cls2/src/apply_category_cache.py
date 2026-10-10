"""category_cache를 연도별 category CSV에 확정 반영하는 유지보수 도구.

``biodiv_label=0``은 BIOFIN category 0으로, ``biodiv_label=1``은
``classify_biofin_category.py``와 동일한 hash key로 cache를 조회한다.
반영이 끝나면 중간 라벨링 컬럼 4개를 제거한다.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from classify_biofin_category import (  # noqa: E402
    build_key,
    hash_key,
    load_cache,
    read_csv,
    write_csv,
)


REMOVE_COLUMNS = {"biodiv_label", "confidence", "reason", "evidence"}
CATEGORY_PATTERN = re.compile(r'"category"\s*:\s*(\d+)')


def cache_category(cached: dict, cache_hash: str) -> int:
    """cache category가 빈 과거 행은 raw_response JSON에서 0을 복원한다."""
    raw_value = str(cached.get("category", "")).strip()
    if raw_value:
        category = int(raw_value)
    else:
        match = CATEGORY_PATTERN.search(str(cached.get("raw_response", "")))
        if not match:
            raise ValueError(f"cache category를 복원할 수 없습니다: hash={cache_hash}")
        category = int(match.group(1))
    if not 0 <= category <= 9:
        raise ValueError(f"cache category가 0~9 범위를 벗어납니다: {category}")
    return category


def apply_to_file(path: Path, cache: dict[str, dict]) -> Counter[int]:
    headers, rows = read_csv(path)
    missing_columns = REMOVE_COLUMNS - set(headers)
    if missing_columns:
        raise ValueError(f"{path.name}: 삭제 대상 컬럼 없음: {sorted(missing_columns)}")
    if "biofin_category" not in headers:
        headers.append("biofin_category")

    counts: Counter[int] = Counter()
    for row_number, row in enumerate(rows, start=2):
        biodiv_label = str(row.get("biodiv_label", "")).strip()
        if biodiv_label == "0":
            category = 0
        elif biodiv_label == "1":
            cache_hash = hash_key(build_key(row))
            cached = cache.get(cache_hash)
            if cached is None:
                raise ValueError(
                    f"{path.name}:{row_number}: category cache 누락(hash={cache_hash})"
                )
            category = cache_category(cached, cache_hash)
        else:
            raise ValueError(
                f"{path.name}:{row_number}: 잘못된 biodiv_label={biodiv_label!r}"
            )
        row["biofin_category"] = str(category)
        counts[category] += 1

    output_headers = [header for header in headers if header not in REMOVE_COLUMNS]
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    write_csv(temporary_path, output_headers, rows)

    # 기록 수와 헤더를 다시 읽은 뒤에만 원본을 교체한다.
    check_headers, check_rows = read_csv(temporary_path)
    if check_headers != output_headers or len(check_rows) != len(rows):
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"임시 CSV 검증 실패: {path.name}")
    temporary_path.replace(path)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_DIR / "outputs" / "사업설명자료"
    )
    parser.add_argument("--cache", type=Path, default=None)
    args = parser.parse_args()

    cache_path = args.cache or args.data_dir / "category_cache.csv"
    cache = load_cache(cache_path)
    if not cache:
        raise FileNotFoundError(f"category cache가 없거나 비어 있습니다: {cache_path}")
    files = sorted(args.data_dir.glob("세부사업 예산편성현황(총액)_*_category.csv"))
    if not files:
        raise FileNotFoundError(f"수정할 category CSV가 없습니다: {args.data_dir}")

    grand_total: Counter[int] = Counter()
    for path in files:
        counts = apply_to_file(path, cache)
        grand_total.update(counts)
        print(f"UPDATED {path.name} rows={sum(counts.values())} categories={dict(sorted(counts.items()))}")
    print(f"TOTAL rows={sum(grand_total.values())} categories={dict(sorted(grand_total.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
