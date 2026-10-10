"""
두 모델(llama3.1:8b vs gemma3:12b)의 라벨링 결과를 side-by-side로 비교하는 테스트 스크립트.

label_cache_6.csv에서 label=1인 행을 N개 샘플링해 동일 조건으로 두 모델을 실행하고
outputs/model_comparison_YYYYMMDD_HHMMSS.csv에 저장합니다.

사용 예:
    python test_model_comparison.py
    python test_model_comparison.py --n 10 --seed 99
    python test_model_comparison.py --cache outputs/label_cache_6.csv --n 20
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

# 메인 스크립트에서 공통 로직 임포트
sys.path.insert(0, str(Path(__file__).parent))
from label_biodiv_with_ollama import (
    SYSTEM_PROMPT,
    PROMPT_TEMPLATE,
    call_ollama,
    parse_jsonish_response,
    build_prompt_values,
    ENCODINGS,
)

# ─── 기본값 ────────────────────────────────────────────────────────────────
DEFAULT_CACHE    = Path(__file__).parent / "outputs" / "label_cache_6.csv"
DEFAULT_OUTPUT   = Path(__file__).parent / "outputs"
MODELS           = ["llama3.1:8b", "gemma3:12b"]
DEFAULT_N        = 20
DEFAULT_SEED     = 42
DEFAULT_OLLAMA   = "http://localhost:11434"
DEFAULT_TIMEOUT  = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="두 모델 라벨링 결과 비교 테스트")
    parser.add_argument("--cache",   type=Path, default=DEFAULT_CACHE,  help="샘플링할 캐시 CSV 경로")
    parser.add_argument("--n",       type=int,  default=DEFAULT_N,      help="샘플링 건수 (기본 20)")
    parser.add_argument("--seed",    type=int,  default=DEFAULT_SEED,   help="랜덤 시드 (기본 42)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT, help="결과 저장 폴더")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA)
    parser.add_argument("--timeout",    type=int, default=DEFAULT_TIMEOUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    for enc in ENCODINGS:
        try:
            with open(path, encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    raise RuntimeError(f"파일을 읽을 수 없습니다: {path}")


def parse_input_text(input_text: str) -> dict[str, str]:
    """'key: value | key: value' 형식의 input_text를 dict로 변환합니다."""
    row: dict[str, str] = {}
    for part in input_text.split(" | "):
        if ": " in part:
            key, value = part.split(": ", 1)
            row[key.strip()] = value.strip()
    return row


def run_model(prompt: str, model: str, ollama_url: str, timeout: int) -> dict:
    """한 모델에 대해 프롬프트를 실행하고 파싱 결과를 반환합니다."""
    try:
        raw = call_ollama(
            prompt=prompt,
            model=model,
            ollama_url=ollama_url,
            timeout=timeout,
            use_json_format=True,
        )
        result = parse_jsonish_response(raw)
        result["raw_response"] = raw
        result["error"] = ""
    except Exception as exc:
        result = {
            "label": -1,
            "confidence": 0.0,
            "reason": "",
            "evidence": "",
            "biofin_category": "",
            "raw_response": "",
            "error": str(exc),
        }
    return result


def main() -> None:
    args = parse_args()

    # ── 1. 캐시 로드 및 샘플링 ──────────────────────────────────────────────
    if not args.cache.exists():
        print(f"ERROR: 캐시 파일 없음: {args.cache}")
        sys.exit(1)

    all_rows = read_csv(args.cache)
    label1_rows = [r for r in all_rows if str(r.get("label", "")).strip() == "1"]

    if len(label1_rows) < args.n:
        print(f"WARN: label=1인 행이 {len(label1_rows)}개뿐 (요청 {args.n}개). 전체 사용.")
        args.n = len(label1_rows)

    import random
    random.seed(args.seed)
    samples = random.sample(label1_rows, args.n)

    print(f"샘플링 완료: label=1 전체 {len(label1_rows)}건 중 {args.n}건 선택 (seed={args.seed})")

    # ── 2. 두 모델 실행 ─────────────────────────────────────────────────────
    results: list[dict] = []

    for idx, cache_row in enumerate(samples, 1):
        row_dict = parse_input_text(cache_row.get("input_text", ""))
        prompt_values = build_prompt_values(row_dict)
        prompt = PROMPT_TEMPLATE.format(**prompt_values)

        print(f"[{idx:2d}/{args.n}] {row_dict.get('세부사업명', '?')[:40]}")

        row_result: dict = {
            "idx":          idx,
            "input_text":   cache_row.get("input_text", ""),
            "세부사업명":    row_dict.get("세부사업명", ""),
            "단위사업명":    row_dict.get("단위사업명", ""),
            "프로그램명":    row_dict.get("프로그램명", ""),
            "소관명":        row_dict.get("소관명", ""),
        }

        for model in MODELS:
            short = model.replace(":", "_").replace(".", "_")
            print(f"       → {model} 실행 중...", end=" ", flush=True)
            res = run_model(prompt, model, args.ollama_url, args.timeout)
            print(f"label={res['label']}  confidence={res['confidence']:.2f}")

            row_result[f"{short}_label"]      = res["label"]
            row_result[f"{short}_confidence"] = res["confidence"]
            row_result[f"{short}_reason"]     = res["reason"]
            row_result[f"{short}_evidence"]   = res["evidence"]
            row_result[f"{short}_error"]      = res["error"]

        # 두 모델 label 일치 여부
        labels = [row_result.get(f"{m.replace(':', '_').replace('.', '_')}_label") for m in MODELS]
        row_result["label_match"] = "Y" if len(set(labels)) == 1 else "N"

        results.append(row_result)

    # ── 3. 결과 저장 ────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"model_comparison_{timestamp}.csv"

    if not results:
        print("저장할 결과 없음.")
        return

    fieldnames = list(results[0].keys())
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ── 4. 요약 출력 ────────────────────────────────────────────────────────
    match_count = sum(1 for r in results if r["label_match"] == "Y")
    print()
    print("=" * 60)
    print(f"결과 저장: {out_path}")
    print(f"총 {len(results)}건 비교")
    print(f"  두 모델 일치: {match_count}건 ({match_count/len(results)*100:.1f}%)")
    print(f"  불일치:       {len(results)-match_count}건")

    for model in MODELS:
        short = model.replace(":", "_").replace(".", "_")
        ones  = sum(1 for r in results if str(r.get(f"{short}_label")) == "1")
        zeros = sum(1 for r in results if str(r.get(f"{short}_label")) == "0")
        errs  = sum(1 for r in results if r.get(f"{short}_error"))
        print(f"\n  [{model}]")
        print(f"    1(관련):   {ones}건")
        print(f"    0(비관련): {zeros}건")
        if errs:
            print(f"    오류:      {errs}건")
    print("=" * 60)


if __name__ == "__main__":
    main()
