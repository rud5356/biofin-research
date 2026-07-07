"""V1 학습 파이프라인 실행 후 Attention 히트맵을 자동 생성하는 V2 CLI."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import train_attention_classifier as v1
from generate_attention_heatmap import generate_attention_heatmap


LOGGER = logging.getLogger("budget_document_classifier")


def build_v2_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--no_heatmap",
        action="store_true",
        help="학습 완료 후 Attention 히트맵 생성을 생략",
    )
    parser.add_argument(
        "--heatmap_output",
        default="attention_heatmap.html",
        help="output_dir 기준 히트맵 HTML 파일명 또는 절대 경로",
    )
    parser.add_argument(
        "--heatmap_max_documents",
        type=int,
        default=50,
        help="히트맵에 표시할 최대 문서 수",
    )
    parser.add_argument(
        "--heatmap_min_chunks",
        type=int,
        default=2,
        help="히트맵에 포함할 문서의 최소 조각 수",
    )
    parser.add_argument(
        "--heatmap_errors_only",
        action="store_true",
        help="기준 라벨과 예측 라벨이 다른 문서만 표시",
    )
    return parser


def _print_help() -> None:
    print("V2: V1 문서 분류 학습 완료 후 Attention 히트맵 HTML 자동 생성\n")
    print(v1.build_argument_parser().format_help())
    print("V2 히트맵 옵션:")
    print(build_v2_parser().format_help())


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in raw_argv or "--help" in raw_argv:
        _print_help()
        return 0

    heatmap_args, train_argv = build_v2_parser().parse_known_args(raw_argv)
    train_args = v1.build_argument_parser().parse_args(train_argv)
    status = v1.main(train_argv)
    if status != 0 or train_args.dry_run or heatmap_args.no_heatmap:
        return status

    output_dir = Path(train_args.output_dir)
    input_csv = output_dir / "attention_outputs.csv"
    heatmap_path = Path(heatmap_args.heatmap_output)
    if not heatmap_path.is_absolute():
        heatmap_path = output_dir / heatmap_path

    try:
        summary = generate_attention_heatmap(
            input_csv,
            heatmap_path,
            max_documents=heatmap_args.heatmap_max_documents,
            min_chunks=heatmap_args.heatmap_min_chunks,
            errors_only=heatmap_args.heatmap_errors_only,
        )
    except Exception:
        LOGGER.exception("학습은 완료됐지만 Attention 히트맵 생성에 실패했습니다.")
        return 1

    LOGGER.info("Attention 히트맵 저장: %s", heatmap_path.resolve())
    LOGGER.info("Attention 히트맵 요약: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
