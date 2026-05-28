"""
train/val/test 분할 데이터셋의 데이터 누출(leakage) 위험을 감사하는 스크립트.

데이터 누출이란?
  학습(train) 데이터와 검증(val)/테스트(test) 데이터가 겹치는 경우를 말합니다.
  같은 부처·사업이 train과 test 모두에 있으면, 모델이 해당 데이터를 '기억'해서
  실제 성능보다 높게 평가될 수 있습니다.

이 스크립트가 확인하는 것:
1. 그룹(부처명 + 세부사업명) 겹침: train-val, train-test, val-test 간 중복 그룹 수
2. 텍스트 내 라벨 언급: 모델 입력 텍스트에 정답 라벨명이 노출된 행 수

사용법:
    python audit_workfile_split.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import (
    DEFAULT_CLASSIFICATION_TEXT_COLUMN,
    DEFAULT_GROUP_COLUMNS,
    WORKFILE_TEST_SPLIT_PATH,
    WORKFILE_TRAIN_SPLIT_PATH,
    WORKFILE_VAL_SPLIT_PATH,
)


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="train/val/test 분할 데이터셋의 데이터 누출 위험을 검사합니다."
    )
    parser.add_argument("--train-csv", type=Path, default=WORKFILE_TRAIN_SPLIT_PATH)
    parser.add_argument("--val-csv", type=Path, default=WORKFILE_VAL_SPLIT_PATH)
    parser.add_argument("--test-csv", type=Path, default=WORKFILE_TEST_SPLIT_PATH)
    parser.add_argument(
        "--text-column",
        default=DEFAULT_CLASSIFICATION_TEXT_COLUMN,
        help="라벨 언급 여부를 확인할 텍스트 열 이름",
    )
    parser.add_argument(
        "--group-columns",
        nargs="+",
        default=list(DEFAULT_GROUP_COLUMNS),
        help="분할 간 겹침을 확인할 그룹 기준 열 이름들",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="라벨(정답) 열 이름",
    )
    return parser.parse_args()


def normalize_value(raw_value: object) -> str:
    """값을 문자열로 변환하고, NaN·None은 빈 문자열로 처리합니다."""
    value = str(raw_value or "").strip()
    if value.lower() in {"", "nan", "none"}:
        return ""
    return value


def normalize_for_match(raw_value: object) -> str:
    """
    값에서 영문·숫자·한글만 남깁니다.

    '환경부'와 '환 경 부'처럼 공백이나 특수문자가 다를 때도
    동일하게 비교하기 위해 알파벳·숫자·한글만 남깁니다.
    """
    value = normalize_value(raw_value)
    return "".join(character for character in value if character.isalnum())


def build_group_key(row: pd.Series, group_columns: list[str]) -> str:
    """
    행에서 그룹 키를 만듭니다.

    예: 부처명="환경부", 세부사업명="생태복원" → "환경부||생태복원"
    값이 없으면 "__MISSING__"을 사용합니다.
    """
    parts = []
    for column in group_columns:
        value = normalize_value(row.get(column, ""))
        parts.append(value if value else "__MISSING__")
    return "||".join(parts)


def count_group_overlap(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    group_columns: list[str],
) -> int:
    """
    두 데이터프레임 사이에 겹치는 그룹 키의 수를 반환합니다.

    집합(set) 교집합으로 계산하므로 행 수가 아닌 고유 그룹 수를 셉니다.
    """
    if not group_columns:
        return 0
    left_keys = {build_group_key(row, group_columns) for _, row in left_df.iterrows()}
    right_keys = {build_group_key(row, group_columns) for _, row in right_df.iterrows()}
    return len(left_keys.intersection(right_keys))


def count_label_mentions(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
) -> int:
    """
    텍스트 열에 라벨명이 직접 언급된 행 수를 셉니다.

    예: 라벨이 "환경"인데 텍스트에 "환경부" 또는 "환경 관련" 같은
    단어가 포함된 경우를 감지합니다.
    마스킹이 제대로 됐다면 이 수가 0에 가까워야 합니다.
    """
    count = 0
    for row in df.to_dict(orient="records"):
        # 특수문자를 제거해 비교 (공백, 기호 차이를 무시)
        text = normalize_for_match(row.get(text_column, ""))
        label = normalize_for_match(row.get(label_column, ""))
        if text and label and label in text:
            count += 1
    return count


def run(args: argparse.Namespace) -> int:
    """감사 실행 로직."""
    train_df = pd.read_csv(args.train_csv.resolve(), encoding="utf-8-sig")
    val_df = pd.read_csv(args.val_csv.resolve(), encoding="utf-8-sig")
    test_df = pd.read_csv(args.test_csv.resolve(), encoding="utf-8-sig")

    print(f"train 행 수:                    {len(train_df)}")
    print(f"val 행 수:                      {len(val_df)}")
    print(f"test 행 수:                     {len(test_df)}")
    print(f"그룹 기준 열:                   {args.group_columns}")
    print(f"train-val 그룹 겹침:            {count_group_overlap(train_df, val_df, args.group_columns)}")
    print(f"train-test 그룹 겹침:           {count_group_overlap(train_df, test_df, args.group_columns)}")
    print(f"val-test 그룹 겹침:             {count_group_overlap(val_df, test_df, args.group_columns)}")
    print(f"train 텍스트 내 라벨 언급 수:   {count_label_mentions(train_df, args.text_column, args.label_column)}")
    print(f"val 텍스트 내 라벨 언급 수:     {count_label_mentions(val_df, args.text_column, args.label_column)}")
    print(f"test 텍스트 내 라벨 언급 수:    {count_label_mentions(test_df, args.text_column, args.label_column)}")
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
