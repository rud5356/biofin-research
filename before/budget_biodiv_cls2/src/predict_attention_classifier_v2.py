"""저장된 Attention 문서분류 모델로 새 사업설명자료를 예측한다.

입력 예산 CSV와 ``연도_소관명_세부사업명_긴ID`` 형식의 문서 파일을
매칭하고, 학습 때와 동일하게 예산정보와 문서 본문을 결합해 추론한다.
예측 결과, 문서 조각별 Attention 가중치와 독립 실행형 히트맵 HTML을
한 번에 저장한다. CSV에 기준 라벨이 있으면 참고 평가 지표도 생성한다.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from build_dataset import (
    BudgetDocumentDataset,
    REQUIRED_COLUMNS,
    build_budget_metadata_text,
    discover_documents,
    document_collate_fn,
    normalize_name,
    parse_document_filename,
)
from document_parser import DocumentParseError, extract_document
from generate_attention_heatmap import LABEL_NAMES, generate_attention_heatmap
from utils import configure_logging, ensure_dir, read_csv_flexible, set_seed, write_csv, write_json


LOGGER = logging.getLogger("budget_document_classifier")


class PredictionDocumentDataset(BudgetDocumentDataset):
    """대량 예측 시 토큰 tensor를 전부 메모리에 누적하지 않는 Dataset."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._encode(index)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="저장된 BIOFIN Attention 모델로 새 사업설명자료 예측",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_dir", required=True, help="best_model.pt와 tokenizer가 있는 폴더")
    parser.add_argument("--doc_dir", required=True, help="예측할 HWP/HWPX/PDF/TXT 폴더")
    parser.add_argument("--budget_file", required=True, help="예측 문서와 매칭할 예산 CSV")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label_column", default="biofin_category")
    parser.add_argument(
        "--filename_column",
        default="matched_filename",
        help="문서 파일명과 직접 매칭할 예산 CSV 컬럼",
    )
    parser.add_argument(
        "--id_column",
        default="No.",
        help="파일명 맨 앞 번호와 매칭할 예산 CSV 컬럼",
    )
    parser.add_argument(
        "--exclude_dir_name",
        action="append",
        default=[],
        help="문서 탐색에서 제외할 폴더명, 여러 번 지정 가능",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--encoder_chunk_batch_size", type=int, default=None)
    parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hwp_com", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="앞 N개 문서만 예측, 0은 전체")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heatmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heatmap_output", default="attention_heatmap.html")
    parser.add_argument("--heatmap_max_documents", type=int, default=1000)
    parser.add_argument("--heatmap_min_chunks", type=int, default=1)
    parser.add_argument("--heatmap_errors_only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다")
    if args.num_workers < 0:
        raise ValueError("num_workers는 0 이상이어야 합니다")
    if args.limit < 0:
        raise ValueError("limit은 0 이상이어야 합니다")
    if args.max_length is not None and args.max_length < 8:
        raise ValueError("max_length가 너무 작습니다")
    if args.stride is not None and args.stride < 0:
        raise ValueError("stride는 0 이상이어야 합니다")
    if args.encoder_chunk_batch_size is not None and args.encoder_chunk_batch_size < 1:
        raise ValueError("encoder_chunk_batch_size는 1 이상이어야 합니다")
    if args.heatmap_max_documents < 1:
        raise ValueError("heatmap_max_documents는 1 이상이어야 합니다")
    if args.heatmap_min_chunks < 1:
        raise ValueError("heatmap_min_chunks는 1 이상이어야 합니다")


def load_budget_data(
    budget_file: str | Path,
    label_column: str = "biofin_category",
) -> pd.DataFrame:
    path = Path(budget_file)
    if not path.is_file():
        raise FileNotFoundError(f"예산 CSV가 없습니다: {path}")
    frame = read_csv_flexible(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"예산 CSV 필수 컬럼이 없습니다: {', '.join(missing)}")

    frame = frame.copy()
    # 2024 열린재정 결산 CSV의 명칭을 학습 시 예산편성 입력 명칭으로 보정한다.
    if "회계명" not in frame.columns and "회계코드명" in frame.columns:
        frame["회계명"] = frame["회계코드명"]
    if "국회확정금액(천원)" not in frame.columns and "세출예산금액" in frame.columns:
        amount = pd.to_numeric(
            frame["세출예산금액"].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        frame["국회확정금액(천원)"] = amount / 1000.0
    frame["_source_file"] = str(path.resolve())
    frame["_source_row"] = range(2, len(frame) + 2)
    frame["_year"] = pd.to_numeric(frame["회계연도"], errors="coerce").astype("Int64")
    frame["_ministry_norm"] = frame["소관명"].map(normalize_name)
    frame["_activity_norm"] = frame["세부사업명"].map(normalize_name)
    frame["_activity_account_norm"] = frame["세부사업명"].map(
        lambda value: normalize_name(value, compensate_account=True)
    )
    if label_column in frame.columns:
        numeric = pd.to_numeric(frame[label_column], errors="coerce")
        frame["_true_label"] = numeric.where(
            numeric.notna() & (numeric % 1 == 0) & numeric.between(0, 9)
        ).astype("Int64")
    else:
        frame["_true_label"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    return frame


def _failure_row(
    reason: str,
    detail: str,
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = identity or {}
    return {
        "reason": reason,
        "detail": detail,
        "year": identity.get("year", ""),
        "ministry": identity.get("ministry", ""),
        "activity_name": identity.get("activity_name", ""),
        "file_path": identity.get("file_path", ""),
    }


def match_document_to_budget_row(
    identity: dict[str, Any],
    budget: pd.DataFrame,
    groups: dict[tuple[int, str], list[int]],
) -> tuple[pd.Series | None, str, str]:
    ministry_norm = normalize_name(identity["ministry"])
    activity_norm = normalize_name(identity["activity_name"])
    account_norm = normalize_name(identity["activity_name"], compensate_account=True)
    candidates = groups.get((int(identity["year"]), ministry_norm), [])

    exact = [idx for idx in candidates if budget.at[idx, "_activity_norm"] == activity_norm]
    if exact:
        matched, match_type = exact, "EXACT"
    else:
        account_exact = [
            idx
            for idx in candidates
            if account_norm and budget.at[idx, "_activity_account_norm"] == account_norm
        ]
        if account_exact:
            matched, match_type = account_exact, "ACCOUNT_COMPENSATED"
        else:
            contained = []
            for idx in candidates:
                target = budget.at[idx, "_activity_norm"]
                if min(len(activity_norm), len(target)) >= 4 and (
                    activity_norm in target or target in activity_norm
                ):
                    contained.append(idx)
            matched, match_type = contained, "CONTAINS"

    if not matched:
        return None, "", "연도/소관명/세부사업명 일치 행 없음"
    if len(matched) > 1:
        rows = ",".join(str(int(budget.at[idx, "_source_row"])) for idx in matched[:20])
        return None, "", f"{len(matched)}개 예산 행 일치(source rows: {rows})"
    return budget.loc[matched[0]], match_type, ""


def _build_direct_lookups(
    budget: pd.DataFrame,
    filename_column: str,
    id_column: str,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    filename_lookup: dict[str, list[int]] = defaultdict(list)
    id_lookup: dict[str, list[int]] = defaultdict(list)
    if filename_column in budget.columns:
        for index, value in budget[filename_column].items():
            filename = str(value or "").strip()
            if filename and filename.lower() != "nan":
                filename_lookup[filename.casefold()].append(index)
    if id_column in budget.columns:
        for index, value in budget[id_column].items():
            number = str(value or "").strip()
            if number.endswith(".0"):
                number = number[:-2]
            if number and number.lower() != "nan":
                id_lookup[number].append(index)
    return filename_lookup, id_lookup


def _identity_from_budget_row(row: pd.Series, path: Path) -> dict[str, Any]:
    return {
        "year": int(row["_year"]),
        "ministry": str(row["소관명"]).strip(),
        "activity_name": str(row["세부사업명"]).strip(),
        "file_path": str(path.resolve()),
    }


def _match_by_filename_or_id(
    path: Path,
    budget: pd.DataFrame,
    filename_lookup: dict[str, list[int]],
    id_lookup: dict[str, list[int]],
) -> tuple[pd.Series | None, str, str]:
    matched = filename_lookup.get(path.name.casefold(), [])
    match_type = "MATCHED_FILENAME"
    if not matched:
        number_match = re.match(r"^(\d+)_", path.name)
        if number_match:
            matched = id_lookup.get(number_match.group(1), [])
            match_type = "ROW_ID"
    if not matched:
        return None, "", "matched_filename 및 파일명 앞 번호 일치 행 없음"
    if len(matched) > 1:
        rows = ",".join(str(int(budget.at[idx, "_source_row"])) for idx in matched[:20])
        return None, "", f"{len(matched)}개 예산 행 일치(source rows: {rows})"
    row = budget.loc[matched[0]]
    if pd.isna(row["_year"]):
        return None, "", "예산 행의 회계연도가 비어 있음"
    return row, match_type, ""


def build_prediction_records(
    documents: Sequence[Path],
    budget: pd.DataFrame,
    use_hwp_com: bool = False,
    limit: int = 0,
    filename_column: str = "matched_filename",
    id_column: str = "No.",
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, row in budget.iterrows():
        if pd.notna(row["_year"]) and row["_ministry_norm"]:
            groups[(int(row["_year"]), row["_ministry_norm"])].append(index)
    filename_lookup, id_lookup = _build_direct_lookups(
        budget, filename_column, id_column
    )

    selected = list(documents[:limit] if limit > 0 else documents)
    records: list[dict[str, Any]] = []
    success_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for position, path in enumerate(selected, start=1):
        row, match_type, detail = _match_by_filename_or_id(
            path, budget, filename_lookup, id_lookup
        )
        if row is not None:
            identity = _identity_from_budget_row(row, path)
        else:
            identity = parse_document_filename(path)
            if identity is not None:
                row, match_type, detail = match_document_to_budget_row(
                    identity, budget, groups
                )
            else:
                identity = {"file_path": str(path.resolve())}
        if row is None:
            reason = (
                "MULTIPLE_BUDGET_ROWS"
                if "개 예산 행 일치" in detail
                else "BUDGET_ROW_NOT_FOUND"
            )
            failures.append(_failure_row(reason, detail, identity))
            continue
        try:
            body = extract_document(path, use_hwp_com=use_hwp_com)
        except DocumentParseError as exc:
            failures.append(_failure_row(exc.reason, exc.detail, identity))
            continue

        metadata_text = build_budget_metadata_text(row)
        combined_text = f"{metadata_text}\n\n[사업설명자료 본문]\n{body}"
        true_label = int(row["_true_label"]) if pd.notna(row["_true_label"]) else None
        record = {
            **identity,
            "label": true_label if true_label is not None else 0,
            "true_label": true_label,
            "text": combined_text,
            "source_type": "document",
            "match_type": match_type,
            "source_file": str(row["_source_file"]),
            "source_row": int(row["_source_row"]),
        }
        records.append(record)
        success_rows.append(
            {
                key: value
                for key, value in record.items()
                if key not in {"text", "label"}
            }
            | {"text_length": len(combined_text)}
        )
        if position % 50 == 0 or position == len(selected):
            LOGGER.info(
                "예측 입력 준비 %d/%d (성공=%d, 실패=%d)",
                position,
                len(selected),
                len(records),
                len(failures),
            )
    return records, pd.DataFrame(success_rows), pd.DataFrame(failures)


def _autocast(device: Any, enabled: bool):
    if enabled and device.type == "cuda":
        import torch

        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _load_checkpoint(path: Path, device: Any) -> dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def predict_records(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from model import DocumentAttentionClassifier

    model_dir = Path(args.model_dir)
    checkpoint_path = model_dir / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"모델 체크포인트가 없습니다: {checkpoint_path}")
    tokenizer_dir = model_dir / "tokenizer"
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"tokenizer 폴더가 없습니다: {tokenizer_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _load_checkpoint(checkpoint_path, device)
    checkpoint_args = checkpoint.get("args", {}) or {}
    model_name = str(checkpoint.get("model_name", checkpoint_args.get("model_name", "klue/roberta-base")))
    num_labels = int(checkpoint.get("num_labels", checkpoint_args.get("num_labels", 10)))
    max_length = int(args.max_length or checkpoint_args.get("max_length", 512))
    stride = int(args.stride if args.stride is not None else checkpoint_args.get("stride", 128))
    encoder_chunk_batch_size = int(
        args.encoder_chunk_batch_size
        or checkpoint_args.get("encoder_chunk_batch_size", 16)
    )
    if stride >= max_length - 2:
        raise ValueError("stride는 max_length-2 미만이어야 합니다")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    dataset = PredictionDocumentDataset(
        records,
        tokenizer,
        max_length=max_length,
        stride=stride,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=document_collate_fn,
    )
    model = DocumentAttentionClassifier(
        model_name=model_name,
        num_labels=num_labels,
        attention_size=int(checkpoint_args.get("attention_size", 256)),
        dropout=float(checkpoint_args.get("dropout", 0.1)),
        encoder_chunk_batch_size=encoder_chunk_batch_size,
        gradient_checkpointing=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    amp_enabled = bool(args.mixed_precision and device.type == "cuda")
    record_lookup = {str(record["file_path"]): record for record in records}

    prediction_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_number, batch in enumerate(loader, start=1):
            with _autocast(device, amp_enabled):
                outputs = model(
                    batch["input_ids"].to(device, non_blocking=True),
                    batch["attention_mask"].to(device, non_blocking=True),
                    batch["chunk_mask"].to(device, non_blocking=True),
                )
            probabilities = torch.softmax(outputs["logits"].float(), dim=-1)
            confidence, predictions = probabilities.max(dim=-1)
            weights = outputs["attention_weights"].float().cpu()
            for item_index, meta in enumerate(batch["meta"]):
                record = record_lookup[str(meta["file_path"])]
                pred_label = int(predictions[item_index].item())
                probability = float(confidence[item_index].item())
                row = {
                    "year": meta["year"],
                    "ministry": meta["ministry"],
                    "activity_name": meta["activity_name"],
                    "file_path": meta["file_path"],
                    "source_type": meta.get("source_type", "document"),
                    "pred_label": pred_label,
                    "pred_category": LABEL_NAMES.get(pred_label, "알 수 없음"),
                    "probability": probability,
                }
                if record.get("true_label") is not None:
                    row["true_label"] = int(record["true_label"])
                prediction_rows.append(row)

                previews = meta.get("chunk_text_previews", [])
                valid_chunks = int(batch["chunk_mask"][item_index].sum().item())
                for chunk_index in range(valid_chunks):
                    attention_row = {
                        **row,
                        "chunk_index": chunk_index,
                        "chunk_text_preview": previews[chunk_index] if chunk_index < len(previews) else "",
                        "attention_weight": float(weights[item_index, chunk_index].item()),
                    }
                    attention_rows.append(attention_row)
            if batch_number % 20 == 0 or batch_number == len(loader):
                LOGGER.info("예측 %d/%d batch 완료", batch_number, len(loader))

    write_csv(prediction_rows, output_dir / "predictions.csv")
    write_csv(attention_rows, output_dir / "attention_outputs.csv")
    return prediction_rows, attention_rows, num_labels


def save_optional_evaluation(
    prediction_rows: list[dict[str, Any]],
    output_dir: Path,
    num_labels: int,
) -> dict[str, Any]:
    labeled = [row for row in prediction_rows if "true_label" in row]
    summary: dict[str, Any] = {
        "total_predictions": len(prediction_rows),
        "labeled_predictions": len(labeled),
        "unlabeled_predictions": len(prediction_rows) - len(labeled),
    }
    if not labeled:
        write_json(summary, output_dir / "prediction_summary.json")
        return summary

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    y_true = [int(row["true_label"]) for row in labeled]
    y_pred = [int(row["pred_label"]) for row in labeled]
    summary.update(
        {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
        }
    )
    labels = list(range(num_labels))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[str(label) for label in labels],
        digits=4,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_frame = pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    matrix_frame.index.name = "true_label"
    matrix_frame.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    write_json(summary, output_dir / "prediction_summary.json")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.debug)
    try:
        validate_args(args)
        set_seed(args.seed)
        output_dir = ensure_dir(args.output_dir)
        budget = load_budget_data(args.budget_file, args.label_column)
        documents = discover_documents(args.doc_dir)
        excluded = {name.casefold() for name in args.exclude_dir_name}
        if excluded:
            before = len(documents)
            documents = [
                path
                for path in documents
                if not any(part.casefold() in excluded for part in path.parts)
            ]
            LOGGER.info("제외 폴더 적용: 문서 %d개 제외", before - len(documents))
        if not documents:
            raise ValueError("예측할 사업설명자료가 없습니다")
        LOGGER.info("예측 문서 %d개, 예산 행 %d개를 확인했습니다.", len(documents), len(budget))

        records, success, failed = build_prediction_records(
            documents,
            budget,
            use_hwp_com=args.hwp_com,
            limit=args.limit,
            filename_column=args.filename_column,
            id_column=args.id_column,
        )
        write_csv(success, output_dir / "dataset_match_success.csv")
        write_csv(failed, output_dir / "dataset_match_failed.csv")
        if not records:
            raise ValueError("매칭 및 본문 추출에 성공한 예측 문서가 없습니다")

        predictions, _, num_labels = predict_records(args, records, output_dir)
        summary = save_optional_evaluation(predictions, output_dir, num_labels)
        if args.heatmap:
            heatmap_path = Path(args.heatmap_output)
            if not heatmap_path.is_absolute():
                heatmap_path = output_dir / heatmap_path
            try:
                heatmap_summary = generate_attention_heatmap(
                    output_dir / "attention_outputs.csv",
                    heatmap_path,
                    max_documents=args.heatmap_max_documents,
                    min_chunks=args.heatmap_min_chunks,
                    errors_only=args.heatmap_errors_only,
                )
            except ValueError:
                if args.heatmap_min_chunks <= 1:
                    raise
                LOGGER.warning("다중 조각 문서가 없어 min_chunks=1로 히트맵을 다시 생성합니다.")
                heatmap_summary = generate_attention_heatmap(
                    output_dir / "attention_outputs.csv",
                    heatmap_path,
                    max_documents=args.heatmap_max_documents,
                    min_chunks=1,
                    errors_only=args.heatmap_errors_only,
                )
            summary["heatmap"] = heatmap_summary
            write_json(summary, output_dir / "prediction_summary.json")
            LOGGER.info("Attention 히트맵 저장: %s", heatmap_path.resolve())

        LOGGER.info("예측 완료: %s", output_dir.resolve())
        return 0
    except KeyboardInterrupt:
        LOGGER.warning("사용자가 예측을 중단했습니다.")
        return 130
    except Exception as exc:
        LOGGER.exception("예측 실패: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
