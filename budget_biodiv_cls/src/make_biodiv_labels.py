from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from config import BIODIV_LABELED_CSV, LABEL_COLUMN, METADATA_COLUMNS, SOURCE_MATCHED_CSV


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:latest"
DEFAULT_OUTPUT_COLUMN = LABEL_COLUMN

INPUT_CSV = SOURCE_MATCHED_CSV
OUTPUT_CSV = BIODIV_LABELED_CSV

TEXT_COLUMNS = list(METADATA_COLUMNS[1:])

PROMPT_TEMPLATE = """\
다음은 대한민국 정부 예산 사업 정보입니다.
이 사업이 생물다양성(biodiversity) 보전, 생태계, 자연환경, 야생생물과 관련된 사업인지 판단하세요.

관련 있으면 1, 관련 없으면 0만 출력하세요. 숫자 하나만 출력하세요.

사업 정보:
{text}

답변:"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ollama LLM으로 생물다양성 이진 라벨을 생성합니다."
    )
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--limit", type=int, default=0, help="처리할 최대 행 수 (0=전체)")
    parser.add_argument("--delay", type=float, default=0.1, help="요청 간 대기 시간(초)")
    return parser.parse_args()


def build_text(row: pd.Series) -> str:
    parts = []
    for col in TEXT_COLUMNS:
        val = str(row.get(col, "") or "").strip()
        if val:
            parts.append(f"{col}: {val}")
    return "\n".join(parts)


def call_ollama(text: str, model: str, ollama_url: str) -> int:
    prompt = PROMPT_TEMPLATE.format(text=text)
    response = requests.post(
        f"{ollama_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    raw = response.json().get("response", "").strip()
    match = re.search(r"[01]", raw)
    return int(match.group()) if match else -1


def run(args: argparse.Namespace) -> int:
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
            label = call_ollama(text, args.model, args.ollama_url)
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


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
