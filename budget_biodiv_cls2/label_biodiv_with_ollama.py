from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import URLError


DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_INPUT_GLOB = "사업별결산세출지출현황_*.csv"
DEFAULT_LABEL_COLUMN = "biodiv_label"

ENCODINGS = ("utf-8-sig", "cp949", "utf-8")

KEY_COLUMNS = (
    "소관명",
    "분야명",
    "부문명",
    "프로그램명",
    "단위사업명",
    "세부사업명",
)

OUTPUT_COLUMNS = (
    "label",
    "confidence",
    "reason",
    "evidence",
    "raw_response",
)


PROMPT_TEMPLATE = """\
너는 대한민국 재정사업을 생물다양성 관련 여부로 엄격하게 분류하는 검수자다.

다음 예산 사업 정보가 생물다양성(biodiversity) 보전과 직접 관련이 있으면 1, 아니면 0으로 분류하라.

판단 기준:
- 1: 생물다양성, 생태계 보전, 자연환경 보전, 야생생물 보호, 멸종위기종, 서식지 보호/복원, 보호지역, 습지/갯벌/해양생태/산림생태 조사·관리·복원과 직접 관련.
- 0: 단순 행정지원, 인건비, 기본경비, 운영지원, 위원회 운영, 시설 유지관리, 여유자금운용, 예치, 전출, 보상, 일반 연구기관 지원.
- 0: 환경/산림/수산/농업 분야라도 사업명만으로 생물다양성 또는 자연생태 보전의 직접 근거가 약하면 0.
- 0: 기후변화, 탄소중립, 에너지, 폐기물, 대기/수질오염, 재난안전, 산불/산사태 대응은 생태계 보전·복원 목적이 직접 드러나지 않으면 0.
- 판단은 반드시 제공된 사업 정보에 근거해야 한다. 추측하지 말라.

예산 사업 정보:
회계연도: {회계연도}
소관명: {소관명}
회계코드명: {회계코드명}
계정명: {계정명}
분야명: {분야명}
부문명: {부문명}
프로그램명: {프로그램명}
단위사업명: {단위사업명}
세부사업명: {세부사업명}

JSON만 출력하라. 설명 문장을 JSON 밖에 쓰지 말라.
형식:
{{
  "label": 0 또는 1,
  "confidence": 0.0부터 1.0 사이 숫자,
  "reason": "짧은 한국어 판단 사유",
  "evidence": "근거가 된 사업명/분야명 일부"
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSV 예산 사업 행을 Ollama LLM으로 생물다양성 관련 여부 라벨링합니다."
    )
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--label-col", default=DEFAULT_LABEL_COLUMN)
    parser.add_argument("--cache-csv", type=Path, default=None)
    parser.add_argument("--audit-csv", type=Path, default=None)
    parser.add_argument("--review-csv", type=Path, default=None)
    parser.add_argument("--review-threshold", type=float, default=0.7)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--limit-keys", type=int, default=0, help="테스트용: 앞 N개 고유 사업 조합만 라벨링")
    parser.add_argument("--dry-run", action="store_true", help="Ollama 호출 없이 입력 구조와 중복키만 확인")
    parser.add_argument("--overwrite", action="store_true", help="기존 캐시 라벨도 다시 생성")
    parser.add_argument("--no-json-format", action="store_true", help="Ollama format=json 옵션을 끕니다")
    return parser.parse_args()


def read_csv_file(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                reader = csv.DictReader(file)
                if not reader.fieldnames:
                    raise ValueError("CSV header not found")
                rows = [dict(row) for row in reader]
            return list(reader.fieldnames), rows, encoding
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"CSV 읽기 실패: {path}") from last_error


def write_csv_file(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean_cell(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def build_prompt_values(row: dict[str, str]) -> dict[str, str]:
    columns = (
        "회계연도",
        "소관명",
        "회계코드명",
        "계정명",
        "분야명",
        "부문명",
        "프로그램명",
        "단위사업명",
        "세부사업명",
    )
    return {column: clean_cell(row.get(column, "")) for column in columns}


def build_input_text(row: dict[str, str]) -> str:
    values = build_prompt_values(row)
    return " | ".join(f"{key}: {value}" for key, value in values.items() if value)


def build_key(row: dict[str, str]) -> str:
    values = [clean_cell(row.get(column, "")) for column in KEY_COLUMNS]
    return "\u241f".join(values)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def build_prompt(row: dict[str, str]) -> str:
    values = build_prompt_values(row)
    return PROMPT_TEMPLATE.format(**values)


def parse_jsonish_response(text: str) -> dict[str, Any]:
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            label_match = re.search(r"\b([01])\b", raw)
            if not label_match:
                return {
                    "label": -1,
                    "confidence": 0.0,
                    "reason": "응답 파싱 실패",
                    "evidence": "",
                    "raw_response": text,
                }
            return {
                "label": int(label_match.group(1)),
                "confidence": 0.5,
                "reason": "JSON이 아닌 응답에서 숫자만 추출",
                "evidence": "",
                "raw_response": text,
            }
        data = json.loads(match.group(0))

    label = data.get("label", -1)
    try:
        label = int(label)
    except (TypeError, ValueError):
        label = -1
    if label not in {0, 1}:
        label = -1

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "label": label,
        "confidence": confidence,
        "reason": clean_cell(data.get("reason", ""))[:240],
        "evidence": clean_cell(data.get("evidence", ""))[:240],
        "raw_response": text,
    }


def call_ollama(
    prompt: str,
    model: str,
    ollama_url: str,
    timeout: int,
    use_json_format: bool,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.1,
            "num_ctx": 4096,
        },
    }
    if use_json_format:
        payload["format"] = "json"

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        f"{ollama_url.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body.get("response", ""))


def classify_with_retries(
    row: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = build_prompt(row)
    last_error: Exception | None = None

    for attempt in range(args.retries + 1):
        try:
            raw_response = call_ollama(
                prompt=prompt,
                model=args.model,
                ollama_url=args.ollama_url,
                timeout=args.timeout,
                use_json_format=not args.no_json_format,
            )
            result = parse_jsonish_response(raw_response)
            if result["label"] in {0, 1}:
                return result
            last_error = RuntimeError("invalid label response")
        except (URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc

        if attempt < args.retries:
            time.sleep(args.retry_delay)

    return {
        "label": -1,
        "confidence": 0.0,
        "reason": f"Ollama 호출 실패: {last_error}",
        "evidence": "",
        "raw_response": "",
    }


def default_output_paths(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_csv is None:
        args.cache_csv = args.output_dir / "label_cache.csv"
    if args.audit_csv is None:
        args.audit_csv = args.output_dir / "label_audit.csv"
    if args.review_csv is None:
        args.review_csv = args.output_dir / "review_needed.csv"


def load_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return {row["key_hash"]: dict(row) for row in reader if row.get("key_hash")}


def save_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    fieldnames = [
        "key_hash",
        "label",
        "confidence",
        "reason",
        "evidence",
        "model",
        "input_text",
        "raw_response",
        "updated_at",
    ]
    rows = sorted(cache.values(), key=lambda row: row.get("key_hash", ""))
    write_csv_file(path, fieldnames, rows)


def collect_inputs(args: argparse.Namespace) -> tuple[dict[Path, dict[str, Any]], dict[str, dict[str, Any]]]:
    csv_paths = sorted(
        path
        for path in args.input_dir.glob(args.input_glob)
        if path.is_file() and args.output_dir not in path.parents and not path.name.endswith("_labeled.csv")
    )
    if not csv_paths:
        raise FileNotFoundError(f"입력 CSV를 찾지 못했습니다: {args.input_dir / args.input_glob}")

    files: dict[Path, dict[str, Any]] = {}
    key_map: dict[str, dict[str, Any]] = {}

    for path in csv_paths:
        headers, rows, encoding = read_csv_file(path)
        missing = [column for column in KEY_COLUMNS if column not in headers]
        if missing:
            raise ValueError(f"{path.name}에 필수 컬럼이 없습니다: {', '.join(missing)}")

        files[path] = {
            "headers": headers,
            "rows": rows,
            "encoding": encoding,
        }

        for row in rows:
            key = build_key(row)
            key_hash = hash_key(key)
            if key_hash not in key_map:
                key_map[key_hash] = {
                    "key_hash": key_hash,
                    "key": key,
                    "row": row,
                    "input_text": build_input_text(row),
                    "count": 0,
                }
            key_map[key_hash]["count"] += 1

    return files, key_map


def print_input_summary(files: dict[Path, dict[str, Any]], key_map: dict[str, dict[str, Any]]) -> None:
    total_rows = sum(len(item["rows"]) for item in files.values())
    print("입력 CSV")
    for path, item in files.items():
        print(f"  - {path.name}: {len(item['rows']):,}행, encoding={item['encoding']}")
    print(f"전체 행 수: {total_rows:,}")
    print(f"고유 사업 조합: {len(key_map):,}")

    reuse_counts = Counter(int(item["count"]) for item in key_map.values())
    reused_keys = sum(1 for item in key_map.values() if int(item["count"]) > 1)
    reused_rows = sum(int(item["count"]) for item in key_map.values() if int(item["count"]) > 1)
    print(f"2회 이상 재사용되는 조합: {reused_keys:,}개 / {reused_rows:,}행")
    print(f"재사용 분포 상위: {reuse_counts.most_common(5)}")


def build_cache_record(
    key_hash: str,
    key_item: dict[str, Any],
    result: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    return {
        "key_hash": key_hash,
        "label": str(result["label"]),
        "confidence": f"{float(result.get('confidence', 0.0)):.3f}",
        "reason": result.get("reason", ""),
        "evidence": result.get("evidence", ""),
        "model": model,
        "input_text": key_item["input_text"],
        "raw_response": result.get("raw_response", ""),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def label_unique_keys(
    key_map: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    keys = list(key_map.items())
    if args.limit_keys > 0:
        keys = keys[: args.limit_keys]

    pending = [
        (key_hash, item)
        for key_hash, item in keys
        if args.overwrite or key_hash not in cache or str(cache[key_hash].get("label", "")) not in {"0", "1"}
    ]
    print(f"라벨링 대상 고유 조합: {len(pending):,}개")
    if args.limit_keys > 0:
        print(f"주의: --limit-keys {args.limit_keys} 적용 중")

    for index, (key_hash, item) in enumerate(pending, start=1):
        result = classify_with_retries(item["row"], args)
        cache[key_hash] = build_cache_record(key_hash, item, result, args.model)

        label = cache[key_hash]["label"]
        confidence = cache[key_hash]["confidence"]
        print(f"[{index:,}/{len(pending):,}] {label} conf={confidence} {item['input_text'][:100]}")

        if args.save_every > 0 and index % args.save_every == 0:
            save_cache(args.cache_csv, cache)
        if args.delay > 0:
            time.sleep(args.delay)

    save_cache(args.cache_csv, cache)
    return cache


def output_headers(headers: list[str], label_col: str) -> list[str]:
    return [column for column in headers if column != label_col] + [label_col]


def write_labeled_outputs(
    files: dict[Path, dict[str, Any]],
    key_map: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    missing_labels = 0
    for path, item in files.items():
        headers = output_headers(item["headers"], args.label_col)
        rows_out: list[dict[str, Any]] = []
        for row in item["rows"]:
            row_out = dict(row)
            key_hash = hash_key(build_key(row))
            cached = cache.get(key_hash)
            label = str(cached.get("label", "")) if cached else ""
            if label not in {"0", "1"}:
                missing_labels += 1
                label = ""
            row_out[args.label_col] = label
            rows_out.append(row_out)

        output_path = args.output_dir / f"{path.stem}_labeled.csv"
        write_csv_file(output_path, headers, rows_out)
        print(f"저장: {output_path}")

    if missing_labels:
        print(f"WARN: 라벨이 비어 있는 행 {missing_labels:,}개가 있습니다.")


def write_audit_files(
    key_map: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    audit_rows = []
    review_rows = []
    for key_hash, item in sorted(key_map.items(), key=lambda kv: kv[1]["input_text"]):
        cached = cache.get(key_hash, {})
        row = {
            "key_hash": key_hash,
            "row_count": item["count"],
            "label": cached.get("label", ""),
            "confidence": cached.get("confidence", ""),
            "reason": cached.get("reason", ""),
            "evidence": cached.get("evidence", ""),
            "input_text": item["input_text"],
            "raw_response": cached.get("raw_response", ""),
        }
        audit_rows.append(row)

        label = str(row["label"])
        try:
            confidence = float(row["confidence"])
        except (TypeError, ValueError):
            confidence = 0.0
        if label not in {"0", "1"} or confidence < args.review_threshold:
            review_rows.append(row)

    fieldnames = [
        "key_hash",
        "row_count",
        "label",
        "confidence",
        "reason",
        "evidence",
        "input_text",
        "raw_response",
    ]
    write_csv_file(args.audit_csv, fieldnames, audit_rows)
    write_csv_file(args.review_csv, fieldnames, review_rows)
    print(f"검수 파일: {args.audit_csv}")
    print(f"확인 필요: {args.review_csv} ({len(review_rows):,}건)")


def write_summary(
    files: dict[Path, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    counts = Counter(str(row.get("label", "")) for row in cache.values())
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "input_files": {path.name: len(item["rows"]) for path, item in files.items()},
        "cache_rows": len(cache),
        "label_counts_in_cache": dict(sorted(counts.items())),
        "output_dir": str(args.output_dir),
        "label_column": args.label_col,
    }
    path = args.output_dir / "run_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"요약: {path}")


def main() -> int:
    args = parse_args()
    default_output_paths(args)

    files, key_map = collect_inputs(args)
    print_input_summary(files, key_map)

    if args.dry_run:
        print("\n--dry-run: Ollama 호출과 파일 저장 없이 종료합니다.")
        return 0

    cache = load_cache(args.cache_csv)
    print(f"기존 캐시: {len(cache):,}개 ({args.cache_csv})")

    cache = label_unique_keys(key_map, cache, args)
    write_labeled_outputs(files, key_map, cache, args)
    write_audit_files(key_map, cache, args)
    write_summary(files, cache, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
