"""CSV runner; no model execution until adapters are connected."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from adapters import llm_predictor, transformer_predictor
from core import Config, OUTPUT_COLUMNS, run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--encoding", default="utf-8-sig")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new output path")
    config = Config(**json.loads(args.config.read_text(encoding="utf-8-sig")))
    with args.input.open(encoding=args.encoding, newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or len(set(fields)) != len(fields):
            parser.error("Input must have unique CSV column headers")
        if set(fields) & set(OUTPUT_COLUMNS):
            parser.error("Input contains reserved pipeline output columns")
        missing = {rule["column"] for rule in config.zero_rules} - set(fields)
        if missing:
            parser.error(f"Missing rule columns: {sorted(missing)}")
        rows = list(reader)
        if any(None in row or None in row.values() for row in rows):
            parser.error("CSV row lengths do not match the header")
    result = run_pipeline(rows, config, transformer_predictor, llm_predictor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(result)
    counts = Counter(row["pipeline_source"] or "pending" for row in result)
    print(json.dumps({"total": len(result), **counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
