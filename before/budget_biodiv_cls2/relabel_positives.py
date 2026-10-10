"""
label_cache_6.csv에서 label=1인 항목을 gemma3:12b(V5 프롬프트)로 재분류해
label_cache_7.csv를 생성하고, 세부사업 예산편성현황(총액)_*_labeled.csv를 재생성합니다.

동작 방식:
    - label=0 행: cache_6에서 그대로 cache_7으로 복사 (재호출 없음)
    - label=1 행: gemma3:12b로 재분류 → 결과로 교체
    - 재실행 시 cache_7에 model=gemma3:12b로 이미 처리된 항목은 건너뜀 (중단 재개 가능)

사용 예:
    python relabel_positives.py
    python relabel_positives.py --workers 2 --timeout 180
    python relabel_positives.py --src-cache outputs/label_cache_6.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from label_biodiv_with_ollama import (
    SYSTEM_PROMPT,
    PROMPT_TEMPLATE,
    KEY_COLUMNS,
    ENCODINGS,
    call_ollama,
    parse_jsonish_response,
    build_prompt_values,
    build_key,
    hash_key,
    clean_cell,
)

# ─── 기본값 ────────────────────────────────────────────────────────────────
DEFAULT_SRC_CACHE  = Path(__file__).parent / "outputs" / "label_cache_6.csv"
DEFAULT_DST_CACHE  = Path(__file__).parent / "outputs" / "label_cache_7.csv"
DEFAULT_INPUT_GLOB = "세부사업 예산편성현황(총액)_*.csv"
DEFAULT_INPUT_DIR  = Path(__file__).parent
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs"
DEFAULT_LABEL_COL  = "biodiv_label"
EXTRA_COLS         = ("confidence", "reason", "evidence")
DEFAULT_MODEL      = "gemma3:12b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_WORKERS    = 1
DEFAULT_TIMEOUT    = 120
DEFAULT_RETRIES    = 2
DEFAULT_RETRY_DELAY = 3
SAVE_INTERVAL      = 100  # N건 처리마다 중간 저장


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="label=1 항목을 gemma3:12b로 재분류")
    p.add_argument("--src-cache",   type=Path, default=DEFAULT_SRC_CACHE)
    p.add_argument("--dst-cache",   type=Path, default=DEFAULT_DST_CACHE)
    p.add_argument("--input-dir",   type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--input-glob",  default=DEFAULT_INPUT_GLOB)
    p.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--label-col",   default=DEFAULT_LABEL_COL)
    p.add_argument("--model",       default=DEFAULT_MODEL)
    p.add_argument("--cache-csv",   type=Path, dest="dst_cache",
                   help="--dst-cache 의 별칭 (label_biodiv_with_ollama.py 호환)")
    p.add_argument("--ollama-url",  default=DEFAULT_OLLAMA_URL)
    p.add_argument("--workers",     type=int, default=DEFAULT_WORKERS)
    p.add_argument("--timeout",     type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--retries",     type=int, default=DEFAULT_RETRIES)
    p.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY)
    return p.parse_args()


def read_csv_file(path: Path) -> list[dict[str, str]]:
    for enc in ENCODINGS:
        try:
            with open(path, encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    raise RuntimeError(f"파일 읽기 실패: {path}")


def write_csv_file(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def parse_input_text(input_text: str) -> dict[str, str]:
    """'key: value | key: value' 형식을 dict로 변환합니다."""
    row: dict[str, str] = {}
    for part in input_text.split(" | "):
        if ": " in part:
            k, v = part.split(": ", 1)
            row[k.strip()] = v.strip()
    return row


def rerun_one(cache_row: dict[str, str], args: argparse.Namespace) -> dict[str, str]:
    """cache_row를 gemma3:12b로 재분류하고 캐시 행 형식으로 반환합니다."""
    row_dict = parse_input_text(cache_row.get("input_text", ""))
    prompt_values = build_prompt_values(row_dict)
    prompt = PROMPT_TEMPLATE.format(**prompt_values)

    last_exc: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            raw = call_ollama(
                prompt=prompt,
                model=args.model,
                ollama_url=args.ollama_url,
                timeout=args.timeout,
                use_json_format=True,
            )
            result = parse_jsonish_response(raw)
            if result["label"] in {0, 1}:
                return {
                    "key_hash":     cache_row["key_hash"],
                    "label":        result["label"],
                    "confidence":   result["confidence"],
                    "reason":       result["reason"],
                    "evidence":     result["evidence"],
                    "model":        args.model,
                    "input_text":   cache_row["input_text"],
                    "raw_response": raw,
                    "updated_at":   datetime.now().isoformat(timespec="seconds"),
                }
            last_exc = RuntimeError("유효하지 않은 label 응답")
        except Exception as exc:
            last_exc = exc
        if attempt < args.retries:
            time.sleep(args.retry_delay)

    # 모든 재시도 실패 → 보수적으로 label=0 처리
    print(f"  WARN: 타임아웃 → label=0 처리 ({cache_row['key_hash'][:12]}…): {last_exc}")
    return {
        "key_hash":     cache_row["key_hash"],
        "label":        0,
        "confidence":   0.0,
        "reason":       "타임아웃으로 판단 불가 — 보수적 0 처리",
        "evidence":     "",
        "model":        f"{args.model}_timeout",
        "input_text":   cache_row["input_text"],
        "raw_response": "",
        "updated_at":   datetime.now().isoformat(timespec="seconds"),
    }


def apply_cache_to_csv(
    input_csv: Path,
    cache: dict[str, dict],
    output_dir: Path,
    label_col: str,
) -> tuple[int, int]:
    """원본 CSV에 cache를 적용해 _labeled.csv를 재생성합니다. (적용 건수, 미적용 건수) 반환."""
    rows = read_csv_file(input_csv)
    if not rows:
        return 0, 0

    base_cols = list(rows[0].keys())
    out_cols = [c for c in base_cols if c not in {label_col} | set(EXTRA_COLS)]
    out_cols += [label_col] + list(EXTRA_COLS)

    out_rows = []
    hit = miss = 0
    for row in rows:
        key = hash_key(build_key(row))
        cached = cache.get(key)
        out_row = {c: row.get(c, "") for c in base_cols}
        if cached:
            out_row[label_col]    = cached["label"]
            out_row["confidence"] = cached.get("confidence", "")
            out_row["reason"]     = cached.get("reason", "")
            out_row["evidence"]   = cached.get("evidence", "")
            hit += 1
        else:
            out_row[label_col]    = ""
            out_row["confidence"] = ""
            out_row["reason"]     = ""
            out_row["evidence"]   = ""
            miss += 1
        out_rows.append(out_row)

    out_path = output_dir / f"{input_csv.stem}_labeled.csv"
    write_csv_file(out_path, out_rows, out_cols)
    return hit, miss


def main() -> None:
    args = parse_args()

    # ── 1. cache_6 로드 ────────────────────────────────────────────────────
    if not args.src_cache.exists():
        print(f"ERROR: {args.src_cache} 없음"); sys.exit(1)

    cache6_rows = read_csv_file(args.src_cache)
    cache6_fields = list(cache6_rows[0].keys()) if cache6_rows else []

    negatives = [r for r in cache6_rows if str(r.get("label", "")).strip() != "1"]
    positives = [r for r in cache6_rows if str(r.get("label", "")).strip() == "1"]
    print(f"cache_6: 전체 {len(cache6_rows)}건 — label=0: {len(negatives)}건, label=1: {len(positives)}건")

    # ── 2. cache_7 기존 항목 로드 (중단 재개용) ────────────────────────────
    done: dict[str, dict] = {}
    if args.dst_cache.exists():
        for r in read_csv_file(args.dst_cache):
            done[r["key_hash"]] = r
        print(f"cache_7 기존 항목: {len(done)}건 (재개)")

    # ── 3. 재분류 대상 추출 (gemma3:12b 성공 완료된 것만 건너뜀, timeout은 재처리) ─
    def is_done(key_hash: str) -> bool:
        r = done.get(key_hash, {})
        m = str(r.get("model", ""))
        conf = r.get("confidence", "")
        try:
            conf_val = float(conf)
        except (ValueError, TypeError):
            conf_val = 0.0
        # 모델명 일치 + confidence > 0 인 경우만 완료로 간주 (timeout은 confidence=0)
        return m == args.model and conf_val > 0.0

    todo = [r for r in positives if not is_done(r["key_hash"])]
    print(f"재분류 대상: {len(todo)}건 (이미 완료: {len(positives) - len(todo)}건)\n")

    # ── 4. 병렬 재분류 ────────────────────────────────────────────────────
    processed = 0
    failed = 0

    def save_progress() -> None:
        merged = (
            list(done.values())
            + negatives
        )
        # done에 없는 negatives만 추가 (중복 방지)
        done_keys = set(done.keys())
        extra_neg = [r for r in negatives if r["key_hash"] not in done_keys]
        all_rows = list(done.values()) + extra_neg
        write_csv_file(args.dst_cache, all_rows, cache6_fields)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(rerun_one, row, args): row for row in todo}
        for future in as_completed(futures):
            result = future.result()
            done[result["key_hash"]] = result
            processed += 1
            if str(result.get("model", "")).endswith("_failed"):
                failed += 1
            label = result.get("label", "?")
            key_short = result["key_hash"][:12]
            row_dict = parse_input_text(result.get("input_text", ""))
            name = row_dict.get("세부사업명", "?")[:35]
            print(f"[{processed:5d}/{len(todo)}] label={label}  {name}")

            if processed % SAVE_INTERVAL == 0:
                save_progress()
                print(f"  → 중간 저장 완료 ({processed}건)")

    # ── 5. 최종 cache_7 저장 ─────────────────────────────────────────────
    done_keys = set(done.keys())
    extra_neg = [r for r in negatives if r["key_hash"] not in done_keys]
    all_rows  = list(done.values()) + extra_neg
    write_csv_file(args.dst_cache, all_rows, cache6_fields)

    new_ones  = sum(1 for r in done.values() if str(r.get("label")) == "1" and r["key_hash"] in {p["key_hash"] for p in positives})
    new_zeros = sum(1 for r in done.values() if str(r.get("label")) == "0" and r["key_hash"] in {p["key_hash"] for p in positives})

    print(f"\n{'='*60}")
    print(f"cache_7 저장: {args.dst_cache}")
    print(f"  재분류 결과 — 여전히 1: {new_ones}건 / 0으로 변경: {new_zeros}건 / 실패: {failed}건")
    print(f"  최종 cache_7 전체: {len(all_rows)}건")

    # ── 6. 세부사업 예산편성현황 labeled.csv 재생성 ───────────────────────
    cache7_dict = {r["key_hash"]: r for r in all_rows}
    input_files = sorted(args.input_dir.glob(args.input_glob))
    if not input_files:
        print(f"\nWARN: 입력 CSV 없음 ({args.input_glob}) — labeled 재생성 건너뜀")
        return

    print(f"\n{'='*60}")
    print(f"labeled.csv 재생성: {len(input_files)}개 파일")
    total_hit = total_miss = 0
    for csv_path in input_files:
        hit, miss = apply_cache_to_csv(csv_path, cache7_dict, args.output_dir, args.label_col)
        total_hit  += hit
        total_miss += miss
        rate = hit / (hit + miss) * 100 if (hit + miss) else 0
        print(f"  {csv_path.name}: {hit}건 적용 / {miss}건 미적용 ({rate:.1f}%)")

    print(f"\n완료 — 전체 {total_hit + total_miss}건 중 {total_hit}건 캐시 적용")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
