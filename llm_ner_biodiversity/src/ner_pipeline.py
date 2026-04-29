import json
import re
import time
from pathlib import Path
from typing import Any

import ollama
import pandas as pd

from prompting import build_prompt


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_output(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {"entities": [], "parse_error": True, "raw": raw[:200]}


def run_ner(text: str, model: str = "llama3.1:8b", mode: str = "few_shot") -> dict[str, Any]:
    prompt = build_prompt(text, mode=mode)

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        raw_output = response["message"]["content"]
        return parse_output(raw_output)
    except Exception as exc:
        return {"entities": [], "error": str(exc)}


def run_batch(
    abstracts: list[dict[str, Any]],
    output_dir: Path,
    model: str = "llama3.1:8b",
    mode: str = "few_shot",
    sample_size: int = 30,
    checkpoint_every: int = 10,
) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    sample = abstracts[:sample_size]
    ensure_directory(output_dir)

    for index, item in enumerate(sample, start=1):
        print(f"[{index}/{len(sample)}] Processing abstract...")

        start_time = time.time()
        ner_result = run_ner(item["abstract"], model=model, mode=mode)
        elapsed = time.time() - start_time

        results.append(
            {
                "id": item["id"],
                "title": item["title"],
                "abstract": item["abstract"],
                "entities": json.dumps(ner_result.get("entities", []), ensure_ascii=False),
                "parse_error": ner_result.get("parse_error", False),
                "error": ner_result.get("error", ""),
                "elapsed_sec": round(elapsed, 2),
            }
        )

        if checkpoint_every > 0 and index % checkpoint_every == 0:
            checkpoint_path = output_dir / f"ner_results_checkpoint_{index}.csv"
            pd.DataFrame(results).to_csv(checkpoint_path, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(results)
    safe_model = model.replace(":", "-")  # Windows 파일명에 콜론 사용 불가
    result_path = output_dir / f"ner_results_{safe_model}_{mode}.csv"
    df.to_csv(result_path, index=False, encoding="utf-8-sig")
    return df
