"""
label_cache.csv를 읽어 5년치 CSV 전체 행에 레이블을 적용하는 스크립트.

label_biodiv_with_ollama.py가 만든 label_cache.csv를 이용해
사업별결산세출지출현황_20XX.csv 각 행에 biodiv_label 컬럼을 추가하고
outputs/{원본파일명}_labeled.csv 로 저장합니다.

사용법:
    python apply_cache_labels.py
    python apply_cache_labels.py --label-col my_label
    python apply_cache_labels.py --input-dir ../other_dir --out-dir ./out
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path
from typing import Any

# ─── 기본 경로 설정 ───────────────────────────────────────────────────────────
SCRIPT_DIR       = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR
DEFAULT_OUT_DIR   = SCRIPT_DIR / "outputs"
DEFAULT_CACHE     = SCRIPT_DIR / "outputs" / "label_cache.csv"
DEFAULT_INPUT_GLOB = "사업별결산세출지출현황_*.csv"
DEFAULT_LABEL_COL  = "biodiv_label"

# label_biodiv_with_ollama.py 와 동일한 KEY_COLUMNS
KEY_COLUMNS = (
    "소관명",
    "분야명",
    "부문명",
    "프로그램명",
    "단위사업명",
    "세부사업명",
)


# ─── 헬퍼 함수 (label_biodiv_with_ollama.py 와 동일 로직) ─────────────────────

def clean_surrogates(value: Any) -> str:
    """UTF-16 서로게이트 쌍을 올바른 유니코드 문자로 변환합니다."""
    text    = str(value or "")
    cleaned: list[str] = []
    index = 0
    while index < len(text):
        code = ord(text[index])
        if 0xD800 <= code <= 0xDBFF:
            if index + 1 < len(text):
                low = ord(text[index + 1])
                if 0xDC00 <= low <= 0xDFFF:
                    cleaned.append(chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)))
                    index += 2
                    continue
            index += 1
            continue
        cleaned.append(text[index])
        index += 1
    return "".join(cleaned)


def clean_cell(value: Any) -> str:
    """값을 문자열로 변환하고 연속 공백을 단일 공백으로 정리합니다."""
    return re.sub(r"\s+", " ", clean_surrogates(value).strip())


def build_key(row: dict[str, str]) -> str:
    """KEY_COLUMNS 값을 U+241F(␟) 구분자로 연결해 고유 사업 키를 만듭니다."""
    return "␟".join(clean_cell(row.get(col, "")) for col in KEY_COLUMNS)


def hash_key(key: str) -> str:
    """키 문자열을 SHA256 해시값(24자)으로 변환합니다."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


# ─── CSV 읽기/쓰기 ────────────────────────────────────────────────────────────

def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """UTF-8 BOM → CP949 → UTF-8 순으로 시도해 CSV를 읽습니다."""
    for encoding in ("utf-8-sig", "cp949", "utf-8"):
        try:
            with path.open(encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                rows = [dict(row) for row in reader]
            return list(reader.fieldnames or []), rows
        except Exception:
            continue
    raise RuntimeError(f"CSV 읽기 실패: {path}")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """행 목록을 UTF-8 BOM CSV로 저장합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {k: clean_surrogates(v) for k, v in row.items()}
            for row in rows
        )


# ─── 캐시 로드 ───────────────────────────────────────────────────────────────

def load_cache(cache_path: Path) -> dict[str, str]:
    """
    label_cache.csv를 읽어 {key_hash: label} 딕셔너리를 반환합니다.

    label이 0 또는 1인 행만 포함합니다. (-1, 빈값 제외)
    """
    if not cache_path.exists():
        print(f"[오류] 캐시 파일을 찾을 수 없습니다: {cache_path}")
        sys.exit(1)

    cache: dict[str, str] = {}
    with cache_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            label = str(row.get("label", "")).strip()
            if label in {"0", "1"}:
                cache[row["key_hash"]] = label

    return cache


# ─── 메인 처리 ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="캐시 기반 전체 행 레이블 적용")
    p.add_argument("--input-dir",  type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    p.add_argument("--out-dir",    type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--cache",      type=Path, default=DEFAULT_CACHE)
    p.add_argument("--label-col",  default=DEFAULT_LABEL_COL,
                   help="추가할 레이블 컬럼명")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1) 캐시 로드
    cache = load_cache(args.cache)
    print(f"캐시 로드 완료: {len(cache):,}개 (label=0/1만 포함)")

    # 2) 입력 CSV 목록 수집 (outputs 폴더 내 파일 및 _labeled.csv 제외)
    csv_paths = sorted(
        p for p in args.input_dir.glob(args.input_glob)
        if p.is_file()
        and args.out_dir not in p.parents
        and not p.name.endswith("_labeled.csv")
    )
    if not csv_paths:
        print(f"[오류] 입력 CSV를 찾지 못했습니다: {args.input_dir / args.input_glob}")
        sys.exit(1)

    print(f"입력 파일: {len(csv_paths)}개\n")

    total_rows   = 0
    total_labeled = 0
    total_missing = 0

    # 3) 파일별로 레이블 적용 후 저장
    for csv_path in csv_paths:
        headers, rows = read_csv(csv_path)

        # 레이블 컬럼이 이미 있으면 제거 후 맨 끝에 추가
        out_headers = [h for h in headers if h != args.label_col] + [args.label_col]

        labeled   = 0
        missing   = 0
        rows_out: list[dict] = []

        for row in rows:
            key_hash = hash_key(build_key(row))
            label    = cache.get(key_hash, "")   # 캐시에 없으면 빈 문자열
            if label:
                labeled += 1
            else:
                missing += 1
            row_out = dict(row)
            row_out[args.label_col] = label
            rows_out.append(row_out)

        out_path = args.out_dir / f"{csv_path.stem}_labeled.csv"
        write_csv(out_path, out_headers, rows_out)

        print(f"  {csv_path.name}")
        print(f"    전체 {len(rows):,}행 → 레이블 {labeled:,}개 / 미매칭 {missing:,}개")
        print(f"    저장: {out_path}")

        total_rows    += len(rows)
        total_labeled += labeled
        total_missing += missing

    # 4) 최종 요약
    print(f"\n{'─'*55}")
    print(f"전체 {total_rows:,}행  레이블 적용: {total_labeled:,}개 ({total_labeled/total_rows*100:.1f}%)")
    if total_missing:
        print(f"미매칭(캐시 없음 또는 -1): {total_missing:,}개 → 빈 값으로 저장")


if __name__ == "__main__":
    main()
