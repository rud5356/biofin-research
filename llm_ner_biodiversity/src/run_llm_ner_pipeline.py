"""
LLM 기반 NER(개체명 인식) 파이프라인 실행 스크립트.

전체 파이프라인:
  1. fetch_abstracts.py  → PubMed에서 논문 초록 수집 (abstracts.csv)
  2. run_llm_ner_pipeline.py → LLM으로 개체명 추출 (ner_results_*.csv)  ← 현재 파일
  3. visualize_locations.py → 지역명 지오코딩 및 지도 시각화

사용법:
    python run_llm_ner_pipeline.py
    python run_llm_ner_pipeline.py --sample-size 50 --mode few_shot
    python run_llm_ner_pipeline.py --save-prompt  # 사용 중인 프롬프트를 파일로 저장
"""

import argparse
from pathlib import Path

import pandas as pd

from config import (
    DATA_DIR,
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_MODE,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_SIZE,
    PROMPTS_DIR,
    RESULTS_DIR,
)
from ner_pipeline import ensure_directory, run_batch
from prompting import save_prompt_template
from result_analysis import summarize_results


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="저장된 논문 초록에서 LLM으로 개체명(종명, 지역명 등)을 추출합니다."
    )
    parser.add_argument(
        "--abstracts-file",
        type=Path,
        default=DATA_DIR / "abstracts.csv",
        help="초록 CSV 파일 경로 (fetch_abstracts.py 출력)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"처리할 초록 수 (기본값: {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"사용할 Ollama 모델 이름 (기본값: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--mode",
        choices=["zero_shot", "few_shot"],
        default=DEFAULT_MODE,
        help=f"프롬프팅 방식: few_shot(예시 포함) / zero_shot(예시 없음) (기본값: {DEFAULT_MODE})",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help=f"N건 처리마다 중간 결과를 저장 (기본값: {DEFAULT_CHECKPOINT_EVERY})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="NER 결과 CSV를 저장할 폴더 (기본값: results/)",
    )
    parser.add_argument(
        "--save-prompt",
        action="store_true",
        help="현재 사용 중인 프롬프트 템플릿을 prompts/ 폴더에 저장합니다.",
    )
    return parser.parse_args()


def main() -> None:
    """NER 파이프라인 실행."""
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)

    # 프롬프트 템플릿 저장 (디버깅·검토용)
    if args.save_prompt:
        prompt_path = save_prompt_template(PROMPTS_DIR, args.mode)
        print(f"프롬프트 템플릿 저장 완료: {prompt_path}")

    # 초록 파일 존재 확인
    if not args.abstracts_file.exists():
        print(f"오류: 초록 파일을 찾을 수 없습니다: {args.abstracts_file}")
        print("먼저 fetch_abstracts.py를 실행하여 초록을 수집하세요.")
        return

    # 초록 CSV를 딕셔너리 목록으로 로드
    abstracts = pd.read_csv(args.abstracts_file, encoding="utf-8-sig").to_dict(orient="records")
    print(f"{len(abstracts)}건의 초록을 로드했습니다: {args.abstracts_file}")

    # NER 배치 실행
    # 처리 중 오류가 나도 체크포인트 덕분에 진행상황이 유지됩니다.
    result_df = run_batch(
        abstracts=abstracts,
        output_dir=output_dir,
        model=args.model,
        mode=args.mode,
        sample_size=args.sample_size,
        checkpoint_every=args.checkpoint_every,
    )

    # 결과 통계 요약 출력
    summarize_results(result_df)


if __name__ == "__main__":
    main()
