"""
예산 문서(HWP/PDF)에서 텍스트를 추출하여 학습용 텍스트 데이터셋을 만드는 스크립트.

입력: workfile_dataset.csv (파일 경로가 포함된 메타데이터)
출력: workfile_text_dataset.csv (추출된 텍스트 + 모델 입력용 텍스트 포함)

사용법:
    python build_text_dataset.py
    python build_text_dataset.py --limit 100  # 처음 100개만 처리 (테스트용)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import WORKFILE_DATASET_PATH, WORKFILE_TEXT_DATASET_PATH
from document_text import count_words, extract_document_text
from utils import console_safe
from workfile_text_processing import build_model_text


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="HWP/PDF 문서에서 텍스트를 추출하여 텍스트 데이터셋을 만듭니다."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=WORKFILE_DATASET_PATH,
        help="파일 경로가 포함된 입력 CSV (기본값: workfile_dataset.csv)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=WORKFILE_TEXT_DATASET_PATH,
        help="텍스트가 추가된 출력 CSV (기본값: workfile_text_dataset.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="처리할 최대 행 수. 0이면 전체 처리 (기본값: 0)",
    )
    return parser.parse_args()


def build_text_rows(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """
    데이터프레임의 각 행에서 HWP/PDF 파일 텍스트를 추출하고
    모델 입력용 텍스트까지 포함한 결과 딕셔너리 목록을 반환합니다.

    추출 결과로 각 행에 추가되는 열:
    - extract_status: 추출 결과 (ok / missing_file / empty_text / extract_error)
    - extract_method: 추출 방식 (hwp_bodytext / hwp_preview / pdf_pypdf 등)
    - extract_error: 오류 발생 시 메시지
    - text: 추출된 원시 텍스트
    - text_char_count / text_word_count: 텍스트 길이 통계
    - model_text: 모델에 입력할 가공된 텍스트
    - model_text_method / model_text_char_count / model_text_word_count: 모델 텍스트 통계
    """
    result_rows: list[dict[str, object]] = []
    total_count = len(dataframe)

    for index, raw_row in enumerate(dataframe.to_dict(orient="records"), start=1):
        file_path = Path(str(raw_row["file_path"]))

        # 진행 상황을 첫 행, 마지막 행, 50행마다 출력
        if index == 1 or index == total_count or index % 50 == 0:
            print(f"[{index}/{total_count}] 텍스트 추출 중: {console_safe(file_path.name)}")

        # 추출 결과 초기화
        extracted_text = ""
        extract_method = ""
        extract_status = "ok"
        extract_error = ""

        try:
            if not file_path.exists():
                # 파일 자체가 없는 경우
                extract_status = "missing_file"
            else:
                extracted_text, extract_method = extract_document_text(file_path)
                if not extracted_text:
                    # 파일은 있지만 텍스트가 비어있는 경우 (스캔본 PDF 등)
                    extract_status = "empty_text"
        except Exception as exc:
            # 파싱 중 예외 발생 (손상된 파일 등)
            extract_status = "extract_error"
            extract_error = str(exc)

        # 원본 행 정보에 추출 결과 열을 추가
        row = dict(raw_row)
        model_text, model_text_method = build_model_text(extracted_text, row)
        row.update(
            {
                "extract_status": extract_status,
                "extract_method": extract_method,
                "extract_error": extract_error,
                "text_char_count": len(extracted_text),
                "text_word_count": count_words(extracted_text),
                "text": extracted_text,
                "model_text_method": model_text_method,
                "model_text_char_count": len(model_text),
                "model_text_word_count": count_words(model_text),
                "model_text": model_text,
            }
        )
        result_rows.append(row)

    return result_rows


def run(args: argparse.Namespace) -> int:
    """메인 실행 로직: 데이터 로드 → 텍스트 추출 → CSV 저장."""
    input_csv = args.input_csv.resolve()
    output_csv = args.output_csv.resolve()

    dataframe = pd.read_csv(input_csv, encoding="utf-8-sig")

    # --limit 옵션이 지정된 경우 앞에서 N행만 사용 (빠른 테스트용)
    if args.limit and args.limit > 0:
        dataframe = dataframe.head(args.limit)

    result_rows = build_text_rows(dataframe)

    # 출력 폴더가 없으면 자동 생성
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {output_csv}")
    return 0


def main() -> int:
    """CLI 진입점."""
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"오류: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
