"""
예산 분야 분류 데이터셋을 학습/검증/테스트 셋으로 분리하는 스크립트.

단순히 행을 무작위로 나누면 같은 문서에서 추출된 텍스트가
학습과 검증 셋에 동시에 들어갈 수 있어 성능이 부풀려집니다.
이를 '데이터 누수(data leakage)'라고 합니다.

이 스크립트는 그룹 컬럼(--group-columns)을 기반으로
같은 그룹의 행이 반드시 같은 셋에만 들어가도록 분리합니다.

예시:
    python src/split_workfile_text_dataset.py
    python src/split_workfile_text_dataset.py --disable-group-split  # 그룹 없이 분리
    python src/split_workfile_text_dataset.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import (
    DEFAULT_CLASSIFICATION_TEXT_COLUMN,
    DEFAULT_GROUP_COLUMNS,
    DEFAULT_MIN_LABEL_COUNT,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    WORKFILE_DROPPED_LABELS_PATH,
    WORKFILE_SPLIT_SUMMARY_PATH,
    WORKFILE_TEST_SPLIT_PATH,
    WORKFILE_TEXT_DATASET_PATH,
    WORKFILE_TRAIN_SPLIT_PATH,
    WORKFILE_VAL_SPLIT_PATH,
)
from utils import save_json


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="BIOFIN 워크파일 텍스트 데이터셋을 데이터 누수 없이 학습/검증/테스트로 분리합니다."
    )
    parser.add_argument("--input-csv",      type=Path, default=WORKFILE_TEXT_DATASET_PATH,
                        help="문서 텍스트가 추출된 입력 CSV")
    parser.add_argument("--train-output",   type=Path, default=WORKFILE_TRAIN_SPLIT_PATH,
                        help="학습 셋 저장 경로")
    parser.add_argument("--val-output",     type=Path, default=WORKFILE_VAL_SPLIT_PATH,
                        help="검증 셋 저장 경로")
    parser.add_argument("--test-output",    type=Path, default=WORKFILE_TEST_SPLIT_PATH,
                        help="테스트 셋 저장 경로")
    parser.add_argument("--dropped-output", type=Path, default=WORKFILE_DROPPED_LABELS_PATH,
                        help="샘플 수 부족으로 제외된 라벨 저장 경로")
    parser.add_argument("--summary-output", type=Path, default=WORKFILE_SPLIT_SUMMARY_PATH,
                        help="분리 통계 요약 JSON 저장 경로")
    parser.add_argument("--train-ratio",    type=float, default=DEFAULT_TRAIN_RATIO,
                        help="학습 셋 비율")
    parser.add_argument("--val-ratio",      type=float, default=DEFAULT_VAL_RATIO,
                        help="검증 셋 비율")
    parser.add_argument("--test-ratio",     type=float, default=DEFAULT_TEST_RATIO,
                        help="테스트 셋 비율")
    parser.add_argument("--min-label-count", type=int, default=DEFAULT_MIN_LABEL_COUNT,
                        help="라벨을 유지하기 위한 최소 샘플 수")
    parser.add_argument("--seed",           type=int,  default=DEFAULT_SPLIT_SEED,
                        help="재현성을 위한 난수 시드")
    parser.add_argument("--label-column",   default="label",
                        help="라벨 이름이 담긴 컬럼")
    parser.add_argument("--label-id-column", default="label_id",
                        help="라벨 숫자 ID가 담긴 컬럼")
    parser.add_argument("--text-column",    default=DEFAULT_CLASSIFICATION_TEXT_COLUMN,
                        help="모델 입력 텍스트 컬럼")
    parser.add_argument(
        "--group-columns", nargs="+", default=list(DEFAULT_GROUP_COLUMNS),
        help="데이터 누수 방지를 위한 그룹 키 컬럼들. 같은 그룹은 같은 셋에 배치됩니다.",
    )
    parser.add_argument("--disable-group-split", action="store_true",
                        help="그룹 컬럼을 무시하고 행 단위로 분리합니다.")
    parser.add_argument("--allow-non-ok",  action="store_true",
                        help="extract_status가 'ok'가 아닌 행도 포함합니다.")
    return parser.parse_args()


def normalize_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[float, float, float]:
    """
    비율 3개를 합이 1.0이 되도록 정규화합니다.

    예: (0.7, 0.1, 0.2) → (0.7, 0.1, 0.2)  # 이미 합이 1.0
        (7, 1, 2) → (0.7, 0.1, 0.2)         # 정규화
    """
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("분리 비율은 0 이상이어야 합니다.")

    total = sum(ratios)
    if total <= 0:
        raise ValueError("분리 비율의 합은 0보다 커야 합니다.")

    return tuple(ratio / total for ratio in ratios)


def allocate_split_counts(
    count: int,
    ratios: tuple[float, float, float],
) -> tuple[int, int, int]:
    """
    전체 샘플 수를 3개 셋에 비율대로 정수 배분합니다.

    단순히 반올림하면 합이 count와 달라질 수 있습니다.
    이를 해결하기 위해 소수점 이하가 큰 셋부터 +1씩 나눠줍니다 (최대 잔차법).

    비율이 0인 셋은 0개를 받으며, 비율이 양수인 셋은 최소 1개를 보장합니다.
    count가 양수 비율의 셋 수보다 작으면 ValueError를 발생시킵니다.
    """
    positive_indexes = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if not positive_indexes:
        raise ValueError("양수 비율이 하나 이상 있어야 합니다.")
    if count < len(positive_indexes):
        raise ValueError("샘플 수가 비어있지 않은 셋의 수보다 작습니다.")

    # 각 셋의 소수점 비율 계산
    desired = [count * ratio for ratio in ratios]
    counts  = [int(value) for value in desired]
    remaining_units = count - sum(counts)

    # 소수점이 큰 순서대로 나머지 1개씩 배분
    fractional_indexes = sorted(
        range(3),
        key=lambda index: (desired[index] - counts[index], ratios[index]),
        reverse=True,
    )
    for index in fractional_indexes:
        if remaining_units <= 0:
            break
        if ratios[index] <= 0:
            continue
        counts[index] += 1
        remaining_units -= 1

    # 양수 비율이지만 0개가 된 셋에 최소 1개 보장
    deficit = 0
    for index in positive_indexes:
        if counts[index] == 0:
            counts[index] = 1
            deficit += 1

    # 초과분을 가장 여유 있는 셋에서 차감
    if deficit > 0:
        removable_indexes = sorted(
            range(3),
            key=lambda index: (counts[index] - desired[index], counts[index]),
            reverse=True,
        )
        for index in removable_indexes:
            while deficit > 0 and counts[index] > 1:
                counts[index] -= 1
                deficit -= 1
            if deficit <= 0:
                break

    if deficit > 0 or sum(counts) != count:
        raise ValueError("샘플 수 배분에 실패했습니다.")

    return counts[0], counts[1], counts[2]


def filter_rows(dataframe: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    학습에 적합한 행만 남기고 필터링합니다.

    - 텍스트/라벨 컬럼이 존재하는지 확인
    - 텍스트나 라벨이 비어있는 행 제거
    - extract_status가 'ok'가 아닌 행 제거 (--allow-non-ok로 스킵 가능)
    """
    required_columns   = {args.label_column, args.text_column}
    missing_columns    = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing_columns)}")

    filtered = dataframe.copy()
    filtered[args.text_column]  = filtered[args.text_column].fillna("").astype(str)
    filtered[args.label_column] = filtered[args.label_column].fillna("").astype(str)
    filtered = filtered[filtered[args.text_column].str.strip()  != ""]
    filtered = filtered[filtered[args.label_column].str.strip() != ""]

    # extract_status 컬럼이 있으면 'ok'인 행만 유지 (--allow-non-ok로 해제 가능)
    if not args.allow_non_ok and "extract_status" in filtered.columns:
        filtered = filtered[filtered["extract_status"] == "ok"]

    return filtered.reset_index(drop=True)


def resolve_group_columns(
    dataframe: pd.DataFrame,
    args: argparse.Namespace,
) -> list[str]:
    """
    실제로 사용할 그룹 컬럼 목록을 반환합니다.

    --disable-group-split이면 빈 목록 반환.
    지정된 컬럼이 데이터프레임에 없으면 ValueError.
    """
    if args.disable_group_split:
        return []

    group_columns   = [column for column in args.group_columns if column]
    if not group_columns:
        return []

    missing_columns = [column for column in group_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"그룹 컬럼이 없습니다: {missing_columns}")

    return group_columns


def normalize_group_value(raw_value: object) -> str:
    """그룹 값의 공백을 제거하고 nan/none 값은 빈 문자열로 변환합니다."""
    value = str(raw_value or "").strip()
    if value.lower() in {"", "nan", "none"}:
        return ""
    return value


def build_group_key(row: pd.Series, group_columns: list[str]) -> str:
    """
    행의 그룹 컬럼 값들을 '||'로 연결해 고유 그룹 키를 만듭니다.

    group_columns가 없으면 행 인덱스를 키로 사용합니다 (개별 분리).
    빈 값은 '__MISSING__'으로 대체해 None/빈 문자열을 동일하게 처리합니다.

    예: ["사업명A", "2023"] → "사업명A||2023"
    """
    if not group_columns:
        return str(row.name)
    values = []
    for column in group_columns:
        raw_value = normalize_group_value(row.get(column, ""))
        values.append(raw_value if raw_value else "__MISSING__")
    return "||".join(values)


def split_rows_without_grouping(
    dataframe: pd.DataFrame,
    label_column: str,
    label_id_column: str,
    ratios: tuple[float, float, float],
    min_required: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    그룹 없이 라벨별로 행을 균등하게 분리합니다.

    각 라벨의 행을 섞은 후 train/val/test 비율대로 나눕니다.
    샘플 수가 min_required보다 적은 라벨은 제외(dropped)됩니다.
    """
    train_parts: list[pd.DataFrame]     = []
    val_parts:   list[pd.DataFrame]     = []
    test_parts:  list[pd.DataFrame]     = []
    dropped_rows: list[dict[str, object]] = []

    grouped = dataframe.groupby(label_column, sort=True)
    for label, group in grouped:
        sample_count = len(group)
        label_id = group.iloc[0][label_id_column] if label_id_column in group.columns else ""

        if sample_count < min_required:
            # 샘플 수가 부족한 라벨은 dropped 목록에 기록하고 건너뜀
            dropped_rows.append({
                "label": label, "label_id": label_id,
                "sample_count": sample_count, "group_count": sample_count,
                "reason": f"sample_count_lt_{min_required}",
            })
            continue

        # 라벨 내 행을 무작위로 섞기
        shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        train_count, val_count, test_count = allocate_split_counts(sample_count, ratios)

        train_parts.append(shuffled.iloc[:train_count].copy())
        val_parts.append(shuffled.iloc[train_count : train_count + val_count].copy())
        test_parts.append(shuffled.iloc[train_count + val_count :].copy())

    train_df   = pd.concat(train_parts,  ignore_index=True) if train_parts  else dataframe.head(0).copy()
    val_df     = pd.concat(val_parts,    ignore_index=True) if val_parts    else dataframe.head(0).copy()
    test_df    = pd.concat(test_parts,   ignore_index=True) if test_parts   else dataframe.head(0).copy()
    dropped_df = pd.DataFrame(dropped_rows)

    return train_df, val_df, test_df, dropped_df


def split_rows_by_group(
    dataframe: pd.DataFrame,
    label_column: str,
    label_id_column: str,
    ratios: tuple[float, float, float],
    min_required: int,
    seed: int,
    group_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    그룹 단위로 데이터 누수 없이 분리합니다.

    '그룹'이란 같은 문서에서 나온 여러 행의 묶음입니다.
    (예: 같은 사업번호의 여러 페이지 텍스트 청크)

    그룹을 먼저 셋에 배정한 뒤 해당 그룹의 모든 행을 함께 배치하므로
    같은 문서의 행이 학습/검증에 동시에 들어가는 것을 막습니다.

    제외 조건:
    - 샘플 수 < min_required
    - 고유 그룹 수 < positive_split_count (각 셋에 최소 1그룹 필요)
    """
    working = dataframe.copy()
    # 각 행에 그룹 키 컬럼 추가
    working["split_group_key"] = working.apply(
        lambda row: build_group_key(row, group_columns), axis=1
    )

    positive_split_count = sum(1 for ratio in ratios if ratio > 0)
    train_parts:  list[pd.DataFrame]      = []
    val_parts:    list[pd.DataFrame]      = []
    test_parts:   list[pd.DataFrame]      = []
    dropped_rows: list[dict[str, object]] = []

    grouped = working.groupby(label_column, sort=True)
    for label, group in grouped:
        sample_count = len(group)
        label_id = group.iloc[0][label_id_column] if label_id_column in group.columns else ""

        # 그룹별 행 수를 계산한 테이블 (그룹 키 → 행 수)
        group_table = (
            group.groupby("split_group_key", sort=False)
            .size()
            .reset_index(name="row_count")
            .sample(frac=1.0, random_state=seed)  # 그룹 순서도 무작위로 섞기
            .reset_index(drop=True)
        )
        unique_group_count = len(group_table)

        if sample_count < min_required:
            dropped_rows.append({
                "label": label, "label_id": label_id,
                "sample_count": sample_count, "group_count": unique_group_count,
                "reason": f"sample_count_lt_{min_required}",
            })
            continue

        # 각 셋에 최소 1그룹이 필요하므로 그룹 수가 셋 수보다 적으면 제외
        if unique_group_count < positive_split_count:
            dropped_rows.append({
                "label": label, "label_id": label_id,
                "sample_count": sample_count, "group_count": unique_group_count,
                "reason": f"group_count_lt_{positive_split_count}",
            })
            continue

        # 그룹을 셋 비율로 배분
        train_group_count, val_group_count, test_group_count = allocate_split_counts(
            unique_group_count, ratios
        )

        # 각 셋에 배정된 그룹 키 집합
        train_group_keys = set(group_table.iloc[:train_group_count]["split_group_key"].tolist())
        val_group_keys   = set(
            group_table.iloc[train_group_count : train_group_count + val_group_count]["split_group_key"].tolist()
        )
        test_group_keys  = set(
            group_table.iloc[
                train_group_count + val_group_count :
                train_group_count + val_group_count + test_group_count
            ]["split_group_key"].tolist()
        )

        # 해당 그룹 키를 가진 행들을 각 셋에 추가
        train_parts.append(group[group["split_group_key"].isin(train_group_keys)].copy())
        val_parts.append(  group[group["split_group_key"].isin(val_group_keys)].copy())
        test_parts.append( group[group["split_group_key"].isin(test_group_keys)].copy())

    train_df   = pd.concat(train_parts,  ignore_index=True) if train_parts  else working.head(0).copy()
    val_df     = pd.concat(val_parts,    ignore_index=True) if val_parts    else working.head(0).copy()
    test_df    = pd.concat(test_parts,   ignore_index=True) if test_parts   else working.head(0).copy()
    dropped_df = pd.DataFrame(dropped_rows)

    return train_df, val_df, test_df, dropped_df


def split_dataframe(
    dataframe: pd.DataFrame,
    args: argparse.Namespace,
    group_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    그룹 컬럼 유무에 따라 적절한 분리 함수를 호출합니다.

    분리 후 각 셋을 다시 한번 무작위로 섞어서 라벨 순서로 인한 편향을 제거합니다.
    """
    ratios = normalize_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    # 각 셋에 최소 1개가 필요하므로 min_required는 양수 셋 수 이상이어야 함
    positive_split_count = sum(1 for ratio in ratios if ratio > 0)
    min_required = max(args.min_label_count, positive_split_count)

    if group_columns:
        train_df, val_df, test_df, dropped_df = split_rows_by_group(
            dataframe, label_column=args.label_column, label_id_column=args.label_id_column,
            ratios=ratios, min_required=min_required, seed=args.seed, group_columns=group_columns,
        )
    else:
        train_df, val_df, test_df, dropped_df = split_rows_without_grouping(
            dataframe, label_column=args.label_column, label_id_column=args.label_id_column,
            ratios=ratios, min_required=min_required, seed=args.seed,
        )

    # 각 셋 내부 순서를 섞어 학습 시 라벨 순서 편향 방지
    train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val_df   = val_df.sample(  frac=1.0, random_state=args.seed).reset_index(drop=True)
    test_df  = test_df.sample( frac=1.0, random_state=args.seed).reset_index(drop=True)
    return train_df, val_df, test_df, dropped_df


def build_summary(
    source_df: pd.DataFrame,
    train_df:  pd.DataFrame,
    val_df:    pd.DataFrame,
    test_df:   pd.DataFrame,
    dropped_df: pd.DataFrame,
    args: argparse.Namespace,
    group_columns: list[str],
) -> dict[str, object]:
    """분리 결과 통계 요약 딕셔너리를 만듭니다."""

    def label_counts(frame: pd.DataFrame) -> dict[str, int]:
        """셋 내 라벨별 행 수를 집계합니다."""
        if frame.empty:
            return {}
        counts = frame.groupby(args.label_column).size().sort_index()
        return {str(label): int(count) for label, count in counts.items()}

    def unique_group_count(frame: pd.DataFrame) -> int:
        """셋 내 고유 그룹 수를 반환합니다."""
        if frame.empty or "split_group_key" not in frame.columns:
            return 0
        return int(frame["split_group_key"].nunique())

    return {
        "input_csv":                 str(args.input_csv.resolve()),
        "text_column":               args.text_column,
        "group_columns":             group_columns,
        "row_count_after_filtering": int(len(source_df)),
        "train_row_count":           int(len(train_df)),
        "val_row_count":             int(len(val_df)),
        "test_row_count":            int(len(test_df)),
        "train_group_count":         unique_group_count(train_df),
        "val_group_count":           unique_group_count(val_df),
        "test_group_count":          unique_group_count(test_df),
        "dropped_label_count":       int(len(dropped_df)),
        "train_ratio":               args.train_ratio,
        "val_ratio":                 args.val_ratio,
        "test_ratio":                args.test_ratio,
        "min_label_count":           args.min_label_count,
        "seed":                      args.seed,
        "train_label_counts":        label_counts(train_df),
        "val_label_counts":          label_counts(val_df),
        "test_label_counts":         label_counts(test_df),
    }


def run(args: argparse.Namespace) -> int:
    """데이터셋 분리 파이프라인 실행."""
    input_csv      = args.input_csv.resolve()
    train_output   = args.train_output.resolve()
    val_output     = args.val_output.resolve()
    test_output    = args.test_output.resolve()
    dropped_output = args.dropped_output.resolve()
    summary_output = args.summary_output.resolve()

    dataframe     = pd.read_csv(input_csv, encoding="utf-8-sig")
    filtered      = filter_rows(dataframe, args)
    group_columns = resolve_group_columns(filtered, args)
    train_df, val_df, test_df, dropped_df = split_dataframe(filtered, args, group_columns)
    summary = build_summary(filtered, train_df, val_df, test_df, dropped_df, args, group_columns)

    for path in (train_output, val_output, test_output, dropped_output, summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(  train_output,   index=False, encoding="utf-8-sig")
    val_df.to_csv(    val_output,     index=False, encoding="utf-8-sig")
    test_df.to_csv(   test_output,    index=False, encoding="utf-8-sig")
    dropped_df.to_csv(dropped_output, index=False, encoding="utf-8-sig")
    save_json(summary_output, summary)

    print(f"학습 셋 저장: {train_output}")
    print(f"검증 셋 저장: {val_output}")
    print(f"테스트 셋 저장: {test_output}")
    print(f"제외된 라벨 저장: {dropped_output}")
    print(f"분리 요약 저장: {summary_output}")
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
