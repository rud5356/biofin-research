from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

from config import (
    BIODIV_LABELED_CSV,
    BIODIV_TEXT_DATASET_CSV,
    BIODIV_TEXT_LABELED_V2_CSV,
    DOCUMENT_TEXT_COLUMN,
    LABEL_COLUMN,
    METADATA_COLUMNS,
    SOURCE_MATCHED_CSV,
)


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:latest"
DEFAULT_TIMEOUT = 180
DEFAULT_V2_MAX_CHARS = 3000
DEFAULT_OUTPUT_COLUMN = LABEL_COLUMN

INPUT_CSV = SOURCE_MATCHED_CSV
OUTPUT_CSV = BIODIV_LABELED_CSV

TEXT_COLUMNS = list(METADATA_COLUMNS[1:])
V2_LABEL_COLUMN = "label_v2"

PROMPT_TEMPLATE = """\
다음은 대한민국 정부 예산 사업 정보입니다.
이 사업이 생물다양성(biodiversity) 보전, 생태계, 자연환경, 야생생물과 관련된 사업인지 판단하세요.

관련 있으면 1, 관련 없으면 0만 출력하세요. 숫자 하나만 출력하세요.

사업 정보:
{text}

답변:"""

PROMPT_TEMPLATE_V2 = """\
다음은 대한민국 정부 예산 사업 문서에서 추출한 본문입니다.
이 사업이 생물다양성(biodiversity) 보전, 생태계 보전, 자연환경 보전, 야생생물 보호,
서식지 보호·복원, 생태계 조사·관리와 직접 관련된 사업인지 판단하세요.

판단 기준:
- 문서 본문 안에 생물다양성/생태계/자연환경/야생생물/서식지/보호지역/생태복원 등 직접 근거가 있으면 1입니다.
- 단순 행정지원, 인건비, 기본경비, 운영지원, 위원회 운영, 연구기관 일반지원, 시설 유지관리만 있으면 0입니다.
- 환경 분야 사업이라도 생물다양성 또는 자연생태 보전과 직접 관련된 근거가 없으면 0입니다.
- 관련 근거 문장을 찾을 수 없으면 0입니다.

관련 있으면 1, 관련 없으면 0만 출력하세요. 숫자 하나만 출력하세요.

문서 본문:
{text}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ollama LLM으로 생물다양성 이진 라벨을 생성합니다."
    )
    parser.add_argument(
        "--version",
        choices=["v1", "v2"],
        default="v1",
        help="v1=메타데이터 기반 기존 라벨링, v2=clean_document_text 기반 엄격 라벨링",
    )
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--text-col", default=DOCUMENT_TEXT_COLUMN)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_V2_MAX_CHARS, help="v2 문서 본문 최대 입력 글자 수")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Ollama 응답 제한 시간(초)")
    parser.add_argument("--retries", type=int, default=2, help="Ollama 호출 실패 시 재시도 횟수")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="재시도 전 대기 시간(초)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--limit", type=int, default=0, help="처리할 최대 행 수 (0=전체)")
    parser.add_argument("--delay", type=float, default=0.1, help="요청 간 대기 시간(초)")
    return parser.parse_args()


def clean_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def build_text(row: object) -> str:
    parts = []
    for col in TEXT_COLUMNS:
        val = clean_cell(row.get(col, ""))
        if val:
            parts.append(f"{col}: {val}")
    return "\n".join(parts)


def call_ollama(text: str, model: str, ollama_url: str, timeout: int) -> int:
    prompt = PROMPT_TEMPLATE.format(text=text)
    response = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 4},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    raw = response.json().get("response", "").strip()
    match = re.search(r"[01]", raw)
    return int(match.group()) if match else -1


def call_ollama_v2(text: str, model: str, ollama_url: str, timeout: int) -> int:
    prompt = PROMPT_TEMPLATE_V2.format(text=text)
    response = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 4},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    raw = response.json().get("response", "").strip()
    match = re.search(r"[01]", raw)
    return int(match.group()) if match else -1


def call_with_retries(callable_fn, args: tuple, retries: int, retry_delay: float) -> int:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return callable_fn(*args)
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            print(f"\nWARN: 호출 실패, 재시도 {attempt + 1}/{retries} → {exc}")
            time.sleep(retry_delay)
    raise last_exc if last_exc else RuntimeError("Ollama 호출 실패")


def read_csv_records(path: Path, encoding: str = "utf-8-sig", delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding=encoding, newline="") as file:
        return list(csv.DictReader(file, delimiter=delimiter))


def write_csv_records(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_io_paths(args: argparse.Namespace) -> None:
    if args.input_csv is None:
        args.input_csv = INPUT_CSV if args.version == "v1" else BIODIV_TEXT_DATASET_CSV
    if args.output_csv is None:
        args.output_csv = OUTPUT_CSV if args.version == "v1" else BIODIV_TEXT_LABELED_V2_CSV


def run(args: argparse.Namespace) -> int:
    import pandas as pd

    df = pd.read_csv(args.input_csv, encoding="cp949", sep="\t")
    print(f"원본 파일: {len(df)}행, {df.shape[1]}개 컬럼")

    if args.limit > 0:
        df = df.head(args.limit)
        print(f"--limit {args.limit} 적용")

    start_index = 0
    if args.output_csv.exists():
        done = pd.read_csv(args.output_csv, encoding="utf-8-sig")
        start_index = len(done)
        print(f"이미 처리된 행 {start_index}개 → 이어서 시작")
        results = done.to_dict(orient="records")
    else:
        results = []

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    remaining = df.iloc[start_index:].reset_index(drop=True)

    for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="라벨 생성"):
        text = build_text(row)
        try:
            label = call_with_retries(
                call_ollama,
                (text, args.model, args.ollama_url, args.timeout),
                args.retries,
                args.retry_delay,
            )
        except Exception as exc:
            print(f"\nWARN: 호출 실패 → -1 ({exc})")
            label = -1

        record = row.to_dict()
        record[DEFAULT_OUTPUT_COLUMN] = label
        results.append(record)

        if len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(args.output_csv, index=False, encoding="utf-8-sig")

        time.sleep(args.delay)

    pd.DataFrame(results).to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    total = len(results)
    print(f"\n완료: {total}행")
    print(f"  생물다양성 관련(1): {sum(1 for r in results if r.get(DEFAULT_OUTPUT_COLUMN) == 1)}행")
    print(f"  관련 없음      (0): {sum(1 for r in results if r.get(DEFAULT_OUTPUT_COLUMN) == 0)}행")
    print(f"  실패          (-1): {sum(1 for r in results if r.get(DEFAULT_OUTPUT_COLUMN) == -1)}행")
    print(f"저장: {args.output_csv}")
    return 0


def run_v2(args: argparse.Namespace) -> int:
    rows = read_csv_records(args.input_csv)
    print(f"v2 입력 파일: {len(rows)}행")

    if args.limit > 0:
        rows = rows[: args.limit]
        print(f"--limit {args.limit} 적용")

    start_index = 0
    if args.output_csv.exists():
        results = read_csv_records(args.output_csv)
        start_index = len(results)
        print(f"이미 처리된 행 {start_index}개 → 이어서 시작")
    else:
        results = []

    remaining = rows[start_index:]
    for row in tqdm(remaining, total=len(remaining), desc="v2 라벨 생성"):
        text = clean_cell(row.get(args.text_col, ""))
        if not text:
            text = build_text(row)
        if args.max_chars > 0:
            text = text[: args.max_chars]

        try:
            label = call_with_retries(
                call_ollama_v2,
                (text, args.model, args.ollama_url, args.timeout),
                args.retries,
                args.retry_delay,
            )
        except Exception as exc:
            print(f"\nWARN: v2 호출 실패 → -1 ({exc})")
            label = -1

        record = dict(row)
        record[V2_LABEL_COLUMN] = label
        results.append(record)

        if len(results) % 50 == 0:
            write_csv_records(args.output_csv, results)

        time.sleep(args.delay)

    write_csv_records(args.output_csv, results)

    total = len(results)
    print(f"\nv2 완료: {total}행")
    print(f"  생물다양성 관련(1): {sum(1 for r in results if str(r.get(V2_LABEL_COLUMN)) == '1')}행")
    print(f"  관련 없음      (0): {sum(1 for r in results if str(r.get(V2_LABEL_COLUMN)) == '0')}행")
    print(f"  실패          (-1): {sum(1 for r in results if str(r.get(V2_LABEL_COLUMN)) == '-1')}행")
    print(f"저장: {args.output_csv}")
    return 0


def main() -> int:
    args = parse_args()
    resolve_io_paths(args)
    try:
        if args.version == "v2":
            return run_v2(args)
        return run(args)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
