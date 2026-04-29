import argparse
import pandas as pd
from pathlib import Path
from config import (
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_MODE,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_SIZE,
    PROMPTS_DIR,
    DATA_DIR,
    RESULTS_DIR
)
from ner_pipeline import ensure_directory, run_batch
from prompting import save_prompt_template
from result_analysis import summarize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-based NER on saved abstracts.")
    parser.add_argument(
        "--abstracts-file",
        type=Path,
        default=DATA_DIR / "abstracts.csv",
        help="Path to abstracts CSV (output of fetch_abstracts.py).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of abstracts to process.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument(
        "--mode",
        choices=["zero_shot", "few_shot"],
        default=DEFAULT_MODE,
        help="Prompting mode.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="Write checkpoint CSV every N processed abstracts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for NER result CSV outputs.",
    )
    parser.add_argument(
        "--save-prompt",
        action="store_true",
        help="Save the active prompt template to the prompts directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)

    if args.save_prompt:
        prompt_path = save_prompt_template(PROMPTS_DIR, args.mode)
        print(f"Saved prompt template to: {prompt_path}")

    if not args.abstracts_file.exists():
        print(f"ERROR: abstracts file not found: {args.abstracts_file}")
        print("Run fetch_abstracts.py first to collect and save abstracts.")
        return

    abstracts = pd.read_csv(args.abstracts_file, encoding="utf-8-sig").to_dict(orient="records")
    print(f"Loaded {len(abstracts)} abstracts from {args.abstracts_file}")

    df = run_batch(
        abstracts=abstracts,
        output_dir=output_dir,
        model=args.model,
        mode=args.mode,
        sample_size=args.sample_size,
        checkpoint_every=args.checkpoint_every,
    )
    summarize_results(df)


if __name__ == "__main__":
    main()
