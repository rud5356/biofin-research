"""사업설명자료 Attention Pooling 분류 학습 CLI.

설치 예시::

    pip install pandas numpy scikit-learn torch transformers accelerate tqdm \
        pypdf pdfplumber olefile beautifulsoup4 lxml

데이터 점검은 pretrained model을 내려받지 않는다::

    python transformer/v1/src/train_attention_classifier.py --dry_run
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_dataset import (
    BudgetDocumentDataset,
    build_metadata_fallback_records,
    discover_documents,
    document_collate_fn,
    extract_matched_documents,
    load_label_data,
    match_documents_to_labels,
)
from document_parser import DocumentParseError, extract_document
from utils import configure_logging, ensure_dir, is_cuda_oom, set_seed, write_csv, write_json


PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_BASE_DIR = PROJECT_DIR / "document"
DEFAULT_DOC_DIR = DEFAULT_BASE_DIR / "2023" / "사업설명자료"
DEFAULT_LABEL_FILE = DEFAULT_BASE_DIR / "2023biofin_label.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "model_results"
LOGGER = logging.getLogger("budget_document_classifier")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="사업설명자료와 예산정보 기반 BIOFIN Attention Pooling 문서분류",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--doc_dir", default=str(DEFAULT_DOC_DIR))
    parser.add_argument("--label_file", default=str(DEFAULT_LABEL_FILE))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model_name", default="klue/roberta-base")
    parser.add_argument("--label_column", default="BIOFIN 1차 카테고리")
    parser.add_argument("--num_labels", type=int, default=10, help="BIOFIN 1차 카테고리 0~9")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128, help="인접 chunk 사이에 겹칠 token 수")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--encoder_chunk_batch_size",
        type=int,
        default=16,
        help="한 번의 transformer 호출에 넣을 chunk 수(문서 batch와 별도)",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--class_weight", action="store_true", help="train 분포 역비례 class weight 적용")
    parser.add_argument(
        "--undersample_majority",
        action="store_true",
        help="split 후 train 세트의 majority label만 동적으로 언더샘플링",
    )
    parser.add_argument("--majority_label", type=int, default=0)
    parser.add_argument("--majority_cap_multiplier", type=float, default=10.0)
    parser.add_argument("--majority_cap_min", type=int, default=1000)
    parser.add_argument(
        "--balanced_sampling",
        action="store_true",
        help="학습 세트에서 클래스별 동일 개수를 매 epoch 복원/비복원 추출",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=100,
        help="balanced sampling에서 한 epoch에 추출할 클래스별 sample 수",
    )
    parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hwp_com", action="store_true", help="OLE/pyhwp 실패 시 한글 COM 자동화 시도")
    parser.add_argument(
        "--document_only",
        action="store_true",
        help="사업설명 본문 추출 성공 건만 학습하고 예산정보 fallback은 제외",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("epochs는 1 이상이어야 합니다")
    if args.batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다")
    if args.max_length < 8:
        raise ValueError("max_length가 너무 작습니다")
    if args.stride < 0 or args.stride >= args.max_length - 2:
        raise ValueError("stride는 0 이상 max_length-2 미만이어야 합니다")
    if not 0 < args.valid_ratio < 1:
        raise ValueError("valid_ratio는 0과 1 사이여야 합니다")
    if not 0 < args.test_ratio < 1:
        raise ValueError("test_ratio는 0과 1 사이여야 합니다")
    if args.valid_ratio + args.test_ratio >= 1:
        raise ValueError("valid_ratio + test_ratio는 1보다 작아야 합니다")
    if args.encoder_chunk_batch_size < 1:
        raise ValueError("encoder_chunk_batch_size는 1 이상이어야 합니다")
    if args.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience는 1 이상이어야 합니다")
    if args.num_labels < 2:
        raise ValueError("num_labels는 2 이상이어야 합니다")
    if args.samples_per_class < 1:
        raise ValueError("samples_per_class는 1 이상이어야 합니다")
    if args.majority_cap_multiplier <= 0:
        raise ValueError("majority_cap_multiplier는 0보다 커야 합니다")
    if args.majority_cap_min < 1:
        raise ValueError("majority_cap_min은 1 이상이어야 합니다")


def save_match_outputs(
    success: pd.DataFrame,
    failed: pd.DataFrame,
    output_dir: Path,
) -> None:
    # 모델 입력용 긴 metadata_text는 감사 CSV에서 제외한다.
    public_success = success.drop(columns=["metadata_text"], errors="ignore")
    write_csv(public_success, output_dir / "dataset_match_success.csv")
    write_csv(failed, output_dir / "dataset_match_failed.csv")
    if success.empty:
        distribution = pd.DataFrame(columns=["label", "count"])
    else:
        distribution = (
            success["label"].value_counts().sort_index().rename_axis("label").reset_index(name="count")
        )
    write_csv(distribution, output_dir / "label_distribution.csv")


def print_distribution(success: pd.DataFrame) -> None:
    if success.empty:
        LOGGER.warning("매칭 성공 데이터가 없습니다.")
        return
    counts = success["label"].value_counts().sort_index()
    LOGGER.info("매칭 데이터 label 분포:\n%s", counts.to_string())
    if len(counts) < 2:
        LOGGER.warning(
            "매칭 문서에 label이 한 종류(%s)뿐입니다. 이 상태로는 분류 경계를 학습할 수 없습니다.",
            counts.index[0],
        )
    for label, count in counts.items():
        if count < 5:
            LOGGER.warning("label %s sample이 %d개뿐이라 안정적인 stratified split이 어렵습니다.", label, count)


def run_dry_run(
    documents: list[Path],
    labels: pd.DataFrame,
    label_files: list[Path],
    matched: pd.DataFrame,
    training_candidates: pd.DataFrame,
    failed: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> int:
    LOGGER.info("[dry-run] 문서 파일: %d개", len(documents))
    LOGGER.info("[dry-run] 레이블 CSV: %d개 / 행: %d개", len(label_files), len(labels))
    fallback_count = int((training_candidates["match_type"] == "METADATA_FALLBACK").sum())
    LOGGER.info(
        "[dry-run] 문서 매칭: %d개 / 예산정보 fallback: %d개 / 실패 로그: %d건",
        len(matched),
        fallback_count,
        len(failed),
    )
    if not failed.empty:
        LOGGER.info("[dry-run] 실패 사유:\n%s", failed["reason"].value_counts().to_string())
    print_distribution(training_candidates)

    # 앞의 파일이 손상됐더라도 성공 본문 3개를 보여주기 위해 최대 30개까지 시도한다.
    samples: list[tuple[dict[str, Any], str]] = []
    sample_failures: list[dict[str, Any]] = []
    for row in matched.head(30).to_dict("records"):
        if len(samples) >= 3:
            break
        try:
            text = extract_document(row["file_path"], use_hwp_com=args.hwp_com)
            samples.append((row, text))
        except DocumentParseError as exc:
            sample_failures.append(
                {
                    "reason": exc.reason,
                    "detail": exc.detail,
                    "year": row["year"],
                    "ministry": row["ministry"],
                    "activity_name": row["activity_name"],
                    "file_path": row["file_path"],
                    "source_file": row["source_file"],
                    "source_row": row["source_row"],
                }
            )
    if sample_failures:
        failed = pd.concat([failed, pd.DataFrame(sample_failures)], ignore_index=True)
        save_match_outputs(training_candidates, failed, output_dir)
    for index, (row, text) in enumerate(samples, start=1):
        preview = text[:700].replace("\n", " | ")
        LOGGER.info(
            "[dry-run] 본문 샘플 %d: %s / %s자\n%s",
            index,
            Path(row["file_path"]).name,
            len(text),
            preview,
        )
    if len(samples) < 3:
        LOGGER.warning("본문 추출 성공 샘플이 %d개뿐입니다. 실패 로그를 확인하세요.", len(samples))
    LOGGER.info("dry-run 완료: %s", output_dir)
    return 0


def normalize_group_component(value: Any) -> str:
    """연도별 표기 차이가 있어도 같은 부처/사업은 동일 그룹으로 묶는다."""

    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def build_business_group_key(record: dict[str, Any]) -> str:
    """소관명+세부사업명을 사업의 불변 KEY로 사용한다.

    회계연도는 의도적으로 제외해 같은 사업의 다른 연도가 train/validation에
    나뉘는 누수를 막는다. 사업명이 없는 비정상 행은 출처 행을 고유 키로 써서
    서로 무관한 행이 한 그룹으로 합쳐지는 것을 피한다.
    """

    ministry = normalize_group_component(record.get("ministry"))
    activity = normalize_group_component(record.get("activity_name"))
    if activity:
        return f"business::{ministry}::{activity}"

    source_file = normalize_group_component(record.get("source_file"))
    source_row = normalize_group_component(record.get("source_row"))
    file_path = normalize_group_component(record.get("file_path"))
    fallback = f"{source_file}::{source_row}" if source_file or source_row else file_path
    if not fallback:
        raise ValueError("사업 그룹 키 생성 실패: activity_name과 출처 식별자가 모두 없습니다")
    return f"missing-activity::{fallback}"


def _split_candidate_score(
    labels: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    valid_ratio: float,
) -> tuple[float, float, float]:
    """검증 크기와 train/validation 클래스 분포가 원본에 가까울수록 우선한다."""

    classes = np.unique(labels)
    class_valid_ratios = np.asarray(
        [
            np.sum(labels[valid_indices] == label) / np.sum(labels == label)
            for label in classes
        ]
    )
    missing_classes = sum(
        int(not np.any(labels[train_indices] == label))
        + int(not np.any(labels[valid_indices] == label))
        for label in classes
    )
    size_error = abs(len(valid_indices) / len(labels) - valid_ratio)
    # 다수 클래스가 점수를 지배하지 않도록 클래스별 valid 배정 비율을 동등하게 본다.
    distribution_error = float(np.abs(class_valid_ratios - valid_ratio).mean())
    # 어느 한쪽에서 클래스가 사라지는 후보는 가능한 한 선택하지 않는다.
    return (
        float(missing_classes),
        distribution_error + size_error,
        size_error,
    )


def split_records(
    records: list[dict[str, Any]], valid_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """사업 그룹을 보존하면서 클래스 분포가 가장 나은 split을 선택한다."""

    from sklearn.model_selection import GroupShuffleSplit

    if len(records) < 2:
        raise ValueError("학습/검증 분리에 최소 2개 문서가 필요합니다")

    labels = np.asarray([int(record["label"]) for record in records])
    groups = np.asarray([build_business_group_key(record) for record in records])
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("사업 그룹이 한 개뿐이라 학습/검증 분리를 할 수 없습니다")

    group_labels: dict[str, set[int]] = {}
    for group, label in zip(groups, labels):
        group_labels.setdefault(str(group), set()).add(int(label))
    conflicting_groups = {
        group: values for group, values in group_labels.items() if len(values) > 1
    }
    if conflicting_groups:
        LOGGER.warning(
            "같은 사업 그룹에 서로 다른 label이 있는 그룹: %d개. 그룹 격리는 유지합니다.",
            len(conflicting_groups),
        )

    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    desired_splits = max(2, int(round(1 / valid_ratio)))
    n_splits = min(desired_splits, len(unique_groups))
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        candidates.extend(
            (np.asarray(train_idx), np.asarray(valid_idx))
            for train_idx, valid_idx in splitter.split(
                np.zeros(len(records)), labels, groups
            )
        )
    except (ImportError, ValueError) as exc:
        LOGGER.warning("StratifiedGroupKFold 후보 생성 실패: %s", exc)

    # 그룹 크기가 크게 다를 때를 위해 목표 valid_ratio에 가까운 후보도 함께 탐색한다.
    shuffle_splitter = GroupShuffleSplit(
        n_splits=min(64, max(16, len(unique_groups))),
        test_size=valid_ratio,
        random_state=seed,
    )
    candidates.extend(
        (np.asarray(train_idx), np.asarray(valid_idx))
        for train_idx, valid_idx in shuffle_splitter.split(
            np.zeros(len(records)), labels, groups
        )
    )
    candidates = [
        (train_idx, valid_idx)
        for train_idx, valid_idx in candidates
        if len(train_idx) and len(valid_idx)
    ]
    if not candidates:
        raise ValueError("유효한 사업 그룹 split 후보를 만들지 못했습니다")

    train_indices, valid_indices = min(
        candidates,
        key=lambda pair: _split_candidate_score(
            labels, pair[0], pair[1], valid_ratio
        ),
    )
    train_groups = set(groups[train_indices])
    valid_groups = set(groups[valid_indices])
    overlap = train_groups & valid_groups
    if overlap:
        raise RuntimeError(f"사업 그룹 split 누수 감지: {len(overlap)}개")

    LOGGER.info(
        "group split: train=%d행/%d사업, valid=%d행/%d사업, overlap=0",
        len(train_indices),
        len(train_groups),
        len(valid_indices),
        len(valid_groups),
    )
    LOGGER.info(
        "group split label 분포: train=%s, valid=%s",
        dict(zip(*np.unique(labels[train_indices], return_counts=True))),
        dict(zip(*np.unique(labels[valid_indices], return_counts=True))),
    )
    train_records = [records[int(index)] for index in train_indices]
    valid_records = [records[int(index)] for index in valid_indices]
    return train_records, valid_records


def split_records_three_way(
    records: list[dict[str, Any]],
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """사업 그룹을 격리한 채 train/validation/test로 나눈다."""

    holdout_ratio = valid_ratio + test_ratio
    train_records, holdout_records = split_records(records, holdout_ratio, seed)
    test_share_of_holdout = test_ratio / holdout_ratio
    valid_records, test_records = split_records(
        holdout_records,
        test_share_of_holdout,
        seed + 1,
    )
    split_groups = {
        "train": {build_business_group_key(row) for row in train_records},
        "valid": {build_business_group_key(row) for row in valid_records},
        "test": {build_business_group_key(row) for row in test_records},
    }
    overlap = (
        (split_groups["train"] & split_groups["valid"])
        | (split_groups["train"] & split_groups["test"])
        | (split_groups["valid"] & split_groups["test"])
    )
    if overlap:
        raise RuntimeError(f"3-way 사업 그룹 split 누수 감지: {len(overlap)}개")
    return train_records, valid_records, test_records


def save_split_outputs(
    train_records: list[dict[str, Any]],
    valid_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """split 배정과 그룹 중복 0건을 사람이 재검증할 수 있게 저장한다."""

    train_groups = {build_business_group_key(record) for record in train_records}
    valid_groups = {build_business_group_key(record) for record in valid_records}
    test_groups = {build_business_group_key(record) for record in test_records}
    overlap = (
        (train_groups & valid_groups)
        | (train_groups & test_groups)
        | (valid_groups & test_groups)
    )
    if overlap:
        raise RuntimeError(f"split 저장 전 사업 그룹 누수 감지: {len(overlap)}개")

    all_records = train_records + valid_records + test_records
    labels_by_group: dict[str, set[int]] = {}
    records_by_group: dict[str, list[dict[str, Any]]] = {}
    for record in all_records:
        group_key = build_business_group_key(record)
        labels_by_group.setdefault(group_key, set()).add(int(record["label"]))
        records_by_group.setdefault(group_key, []).append(record)
    conflicting_keys = {
        key for key, labels in labels_by_group.items() if len(labels) > 1
    }

    rows: list[dict[str, Any]] = []
    for split_name, split_records_value in (
        ("train", train_records),
        ("valid", valid_records),
        ("test", test_records),
    ):
        for record in split_records_value:
            rows.append(
                {
                    "split": split_name,
                    "group_key": build_business_group_key(record),
                    "year": record.get("year", ""),
                    "ministry": record.get("ministry", ""),
                    "activity_name": record.get("activity_name", ""),
                    "label": record.get("label", ""),
                    "source_type": record.get("source_type", ""),
                    "file_path": record.get("file_path", ""),
                }
            )
    write_csv(rows, output_dir / "split_assignments.csv")
    conflict_rows = []
    for group_key in sorted(conflicting_keys):
        group_records = records_by_group[group_key]
        first = group_records[0]
        conflict_rows.append(
            {
                "group_key": group_key,
                "labels": "|".join(
                    str(label) for label in sorted(labels_by_group[group_key])
                ),
                "row_count": len(group_records),
                "years": "|".join(
                    sorted({str(record.get("year", "")) for record in group_records})
                ),
                "ministry": first.get("ministry", ""),
                "activity_name": first.get("activity_name", ""),
            }
        )
    write_csv(
        pd.DataFrame(
            conflict_rows,
            columns=[
                "group_key",
                "labels",
                "row_count",
                "years",
                "ministry",
                "activity_name",
            ],
        ),
        output_dir / "split_conflicting_groups.csv",
    )
    write_json(
        {
            "strategy": "ministry_activity_stratified_group_split_8_1_1",
            "train_rows": len(train_records),
            "valid_rows": len(valid_records),
            "test_rows": len(test_records),
            "train_groups": len(train_groups),
            "valid_groups": len(valid_groups),
            "test_groups": len(test_groups),
            "overlap_groups": len(overlap),
            "conflicting_label_groups": len(conflicting_keys),
            "train_label_counts": {
                str(int(label)): int(count)
                for label, count in sorted(
                    pd.Series([row["label"] for row in train_records])
                    .value_counts()
                    .items()
                )
            },
            "valid_label_counts": {
                str(int(label)): int(count)
                for label, count in sorted(
                    pd.Series([row["label"] for row in valid_records])
                    .value_counts()
                    .items()
                )
            },
            "test_label_counts": {
                str(int(label)): int(count)
                for label, count in sorted(
                    pd.Series([row["label"] for row in test_records])
                    .value_counts()
                    .items()
                )
            },
        },
        output_dir / "split_summary.json",
    )


def make_class_weights(records: list[dict[str, Any]], device: Any, num_labels: int):
    import torch

    counts = np.bincount([int(record["label"]) for record in records], minlength=num_labels)
    weights = np.zeros(num_labels, dtype=np.float32)
    present = counts > 0
    weights[present] = len(records) / (present.sum() * counts[present])
    LOGGER.info("class weights: %s", weights.tolist())
    return torch.tensor(weights, dtype=torch.float32, device=device)


def undersample_majority_records(
    records: list[dict[str, Any]],
    majority_label: int,
    cap_multiplier: float,
    cap_min: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """소수 클래스 중앙값을 기준으로 train의 지정 다수 클래스만 줄인다."""
    indices_by_label: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        indices_by_label.setdefault(int(record["label"]), []).append(index)

    majority_indices = indices_by_label.get(majority_label, [])
    minority_counts = [
        len(indices)
        for label, indices in indices_by_label.items()
        if label != majority_label
    ]
    if not majority_indices or not minority_counts:
        LOGGER.warning("majority undersampling을 적용할 클래스 분포가 충분하지 않습니다.")
        return records, {"applied": False, "reason": "insufficient_class_distribution"}

    minority_median = float(np.median(minority_counts))
    requested_cap = max(cap_min, int(round(minority_median * cap_multiplier)))
    applied_cap = min(len(majority_indices), requested_cap)
    if applied_cap == len(majority_indices):
        LOGGER.info(
            "majority label %d는 %d건으로 cap %d 이하이므로 줄이지 않습니다.",
            majority_label,
            len(majority_indices),
            requested_cap,
        )
        return records, {
            "applied": False,
            "reason": "already_below_cap",
            "minority_median": minority_median,
            "requested_cap": requested_cap,
        }

    rng = np.random.default_rng(seed)
    kept_majority = set(
        int(index)
        for index in rng.choice(majority_indices, size=applied_cap, replace=False)
    )
    sampled = [
        record
        for index, record in enumerate(records)
        if int(record["label"]) != majority_label or index in kept_majority
    ]
    rng.shuffle(sampled)
    summary = {
        "applied": True,
        "majority_label": majority_label,
        "majority_before": len(majority_indices),
        "majority_after": applied_cap,
        "minority_median": minority_median,
        "cap_multiplier": cap_multiplier,
        "cap_min": cap_min,
        "requested_cap": requested_cap,
        "train_before": len(records),
        "train_after": len(sampled),
    }
    LOGGER.info("majority undersampling: %s", summary)
    return sampled, summary


def make_balanced_sampler(
    records: list[dict[str, Any]], samples_per_class: int, seed: int
):
    """각 class에서 정확히 같은 수를 뽑고 epoch마다 다시 섞는 sampler."""
    from torch.utils.data import Sampler

    indices_by_label: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        indices_by_label.setdefault(int(record["label"]), []).append(index)
    present_labels = sorted(indices_by_label)

    class ExactBalancedSampler(Sampler):
        def __init__(self) -> None:
            self.epoch = 0

        def __iter__(self):
            rng = np.random.default_rng(seed + self.epoch)
            sampled: list[int] = []
            for label in present_labels:
                candidates = indices_by_label[label]
                selected = rng.choice(
                    candidates,
                    size=samples_per_class,
                    replace=len(candidates) < samples_per_class,
                )
                sampled.extend(int(index) for index in selected)
            rng.shuffle(sampled)
            self.epoch += 1
            return iter(sampled)

        def __len__(self) -> int:
            return samples_per_class * len(present_labels)

    LOGGER.info(
        "balanced sampling: labels=%s, samples_per_class=%d, samples_per_epoch=%d",
        present_labels,
        samples_per_class,
        samples_per_class * len(present_labels),
    )
    return ExactBalancedSampler()


def _autocast(device: Any, enabled: bool):
    import torch

    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train(args: argparse.Namespace, records: list[dict[str, Any]], output_dir: Path) -> int:
    import torch
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from evaluate import evaluate_model, save_evaluation_outputs
    from model import DocumentAttentionClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.mixed_precision and device.type == "cuda")
    if args.mixed_precision and not amp_enabled:
        LOGGER.warning("CUDA가 없어 mixed precision을 비활성화합니다.")
    LOGGER.info("device=%s, mixed_precision=%s", device, amp_enabled)

    train_records, valid_records, test_records = split_records_three_way(
        records, args.valid_ratio, args.test_ratio, args.seed
    )
    LOGGER.info(
        "train=%d, valid=%d, test=%d",
        len(train_records),
        len(valid_records),
        len(test_records),
    )
    save_split_outputs(train_records, valid_records, test_records, output_dir)
    if args.undersample_majority:
        train_records, undersampling_summary = undersample_majority_records(
            train_records,
            args.majority_label,
            args.majority_cap_multiplier,
            args.majority_cap_min,
            args.seed,
        )
        write_json(
            undersampling_summary,
            output_dir / "train_undersampling_summary.json",
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_dataset = BudgetDocumentDataset(
        train_records, tokenizer, max_length=args.max_length, stride=args.stride
    )
    valid_dataset = BudgetDocumentDataset(
        valid_records, tokenizer, max_length=args.max_length, stride=args.stride
    )
    test_dataset = BudgetDocumentDataset(
        test_records, tokenizer, max_length=args.max_length, stride=args.stride
    )
    train_sampler = (
        make_balanced_sampler(train_records, args.samples_per_class, args.seed)
        if args.balanced_sampling
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=document_collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=document_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=document_collate_fn,
    )

    try:
        model = DocumentAttentionClassifier(
            model_name=args.model_name,
            num_labels=args.num_labels,
            attention_size=args.attention_size,
            dropout=args.dropout,
            encoder_chunk_batch_size=args.encoder_chunk_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
        ).to(device)
    except RuntimeError as exc:
        if is_cuda_oom(exc):
            raise RuntimeError(
                "모델을 GPU에 올리는 중 메모리가 부족합니다. 더 작은 pretrained model을 선택하세요."
            ) from exc
        raise

    if args.balanced_sampling and args.class_weight:
        LOGGER.warning(
            "balanced sampling과 class weight의 이중 보정을 피하기 위해 class weight를 비활성화합니다."
        )
    class_weights = (
        make_class_weights(train_records, device, args.num_labels)
        if args.class_weight and not args.balanced_sampling
        else None
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_score = -float("inf")
    best_path = output_dir / "best_model.pt"
    epochs_without_improvement = 0
    train_log: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            try:
                with _autocast(device, amp_enabled):
                    outputs = model(
                        batch["input_ids"].to(device, non_blocking=True),
                        batch["attention_mask"].to(device, non_blocking=True),
                        batch["chunk_mask"].to(device, non_blocking=True),
                    )
                    labels = batch["labels"].to(device, non_blocking=True)
                    loss = criterion(outputs["logits"], labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"NaN/Inf loss 발생(epoch={epoch}, step={step}, loss={loss.item()})"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                epoch_losses.append(float(loss.item()))
            except RuntimeError as exc:
                if is_cuda_oom(exc):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        "GPU 메모리가 부족합니다. --batch_size 1 또는 더 작은 "
                        "--encoder_chunk_batch_size 값을 사용하세요."
                    ) from exc
                raise
            if step % 20 == 0 or step == len(train_loader):
                LOGGER.info(
                    "epoch %d/%d step %d/%d loss=%.5f",
                    epoch,
                    args.epochs,
                    step,
                    len(train_loader),
                    float(np.mean(epoch_losses[-20:])),
                )

        valid_metrics, _, _ = evaluate_model(
            model,
            valid_loader,
            device,
            criterion=criterion,
            mixed_precision=amp_enabled,
            collect_details=False,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "valid_loss": valid_metrics["loss"],
            "accuracy": valid_metrics["accuracy"],
            "macro_f1": valid_metrics["macro_f1"],
            "weighted_f1": valid_metrics["weighted_f1"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        train_log.append(row)
        write_csv(train_log, output_dir / "train_log.csv")
        LOGGER.info("epoch %d metrics=%s", epoch, row)

        score = valid_metrics["macro_f1"]
        if score > best_score + 1e-8:
            best_score = score
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_name": args.model_name,
                "num_labels": args.num_labels,
                "epoch": epoch,
                "metrics": valid_metrics,
                "args": vars(args),
            }
            temporary_path = output_dir / "best_model.pt.tmp"
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(best_path)
            LOGGER.info("best model 저장: %s", best_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                LOGGER.info("early stopping: %d epoch 연속 개선 없음", epochs_without_improvement)
                break

    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch 2.5 이전 호환
        checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics, _, _ = evaluate_model(
        model,
        valid_loader,
        device,
        criterion=criterion,
        mixed_precision=amp_enabled,
        collect_details=False,
    )
    test_metrics, prediction_rows, attention_rows = evaluate_model(
        model,
        test_loader,
        device,
        criterion=criterion,
        mixed_precision=amp_enabled,
        collect_details=True,
    )
    reported_metrics = save_evaluation_outputs(
        prediction_rows, attention_rows, output_dir, num_labels=args.num_labels
    )
    test_metrics.update(reported_metrics)
    write_json(
        {"validation": validation_metrics, "test": test_metrics},
        output_dir / "metrics.json",
    )
    tokenizer.save_pretrained(output_dir / "tokenizer")
    LOGGER.info("최종 validation metrics=%s", validation_metrics)
    LOGGER.info("최종 test metrics=%s", test_metrics)
    LOGGER.info("학습 완료: %s", output_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    configure_logging(args.debug)
    try:
        validate_arguments(args)
        set_seed(args.seed)
        output_dir = ensure_dir(args.output_dir)
        documents = discover_documents(args.doc_dir)
        labels, label_files = load_label_data(
            args.label_file,
            label_column=args.label_column,
            num_labels=args.num_labels,
        )
        LOGGER.info("문서 %d개, label 행 %d개를 확인했습니다.", len(documents), len(labels))
        matched, failed = match_documents_to_labels(documents, labels)
        if args.document_only:
            training_candidates = matched.copy()
            LOGGER.info(
                "document-only 모드: 예산정보 fallback을 제외하고 매칭 문서만 사용합니다."
            )
        else:
            _, initial_fallback_success = build_metadata_fallback_records(
                labels, failed
            )
            training_candidates = pd.concat(
                [matched, initial_fallback_success], ignore_index=True, sort=False
            )
        save_match_outputs(training_candidates, failed, output_dir)

        if args.dry_run:
            return run_dry_run(
                documents,
                labels,
                label_files,
                matched,
                training_candidates,
                failed,
                args,
                output_dir,
            )

        records: list[dict[str, Any]] = []
        parsed_success = pd.DataFrame()
        parse_failed = pd.DataFrame()
        if not matched.empty:
            LOGGER.info("매칭 성공 문서의 전체 본문 추출을 시작합니다.")
            records, parsed_success, parse_failed = extract_matched_documents(
                matched, use_hwp_com=args.hwp_com
            )
        if not parse_failed.empty:
            failed = pd.concat([failed, parse_failed], ignore_index=True)
        if args.document_only:
            fallback_records: list[dict[str, Any]] = []
            final_success = parsed_success
            LOGGER.info(
                "document-only 모드: 본문 추출 성공 %d건만 학습에 사용합니다.",
                len(records),
            )
        else:
            fallback_records, fallback_success = build_metadata_fallback_records(
                labels, failed
            )
            records.extend(fallback_records)
            final_success = pd.concat(
                [parsed_success, fallback_success], ignore_index=True, sort=False
            )
        save_match_outputs(final_success, failed, output_dir)
        LOGGER.info(
            "본문 사용=%d, 예산정보 fallback=%d, 문서 파싱 실패=%d",
            len(records) - len(fallback_records),
            len(fallback_records),
            len(parse_failed),
        )
        if len(records) < 2:
            raise ValueError("본문 추출 후 학습 가능한 문서가 2개 미만입니다")
        unique_labels = sorted({int(record["label"]) for record in records})
        if len(unique_labels) < 2:
            raise ValueError(
                f"학습 문서의 label이 {unique_labels} 한 종류뿐입니다. "
                f"서로 다른 {args.label_column} 문서가 최소 한 개씩 필요합니다."
            )
        return train(args, records, output_dir)
    except KeyboardInterrupt:
        LOGGER.warning("사용자가 작업을 중단했습니다. 지금까지 생성된 로그는 보존됩니다.")
        return 130
    except Exception as exc:
        LOGGER.exception("파이프라인 실패: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
