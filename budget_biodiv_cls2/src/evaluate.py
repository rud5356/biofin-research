"""검증 지표 계산과 예측/attention 결과 저장."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from utils import write_csv


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def evaluate_model(
    model: torch.nn.Module,
    data_loader: Any,
    device: torch.device,
    criterion: torch.nn.Module | None = None,
    mixed_precision: bool = False,
    collect_details: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    losses: list[float] = []
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    prediction_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            chunk_mask = batch["chunk_mask"].to(device)
            labels = batch["labels"].to(device)
            with _autocast(device, mixed_precision):
                outputs = model(input_ids, attention_mask, chunk_mask)
                if criterion is not None:
                    losses.append(float(criterion(outputs["logits"], labels).item()))
            probabilities = torch.softmax(outputs["logits"].float(), dim=-1)
            confidence, predictions = probabilities.max(dim=-1)
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predictions.cpu().tolist())

            if not collect_details:
                continue
            weights = outputs["attention_weights"].float().cpu()
            for item_index, meta in enumerate(batch["meta"]):
                true_label = int(labels[item_index].item())
                pred_label = int(predictions[item_index].item())
                probability = float(confidence[item_index].item())
                prediction_rows.append(
                    {
                        "year": meta["year"],
                        "ministry": meta["ministry"],
                        "activity_name": meta["activity_name"],
                        "file_path": meta["file_path"],
                        "source_type": meta.get("source_type", "document"),
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "probability": probability,
                    }
                )
                previews = meta.get("chunk_text_previews", [])
                valid_chunks = int(batch["chunk_mask"][item_index].sum().item())
                for chunk_index in range(valid_chunks):
                    attention_rows.append(
                        {
                            "year": meta["year"],
                            "ministry": meta["ministry"],
                            "activity_name": meta["activity_name"],
                            "file_path": meta["file_path"],
                            "source_type": meta.get("source_type", "document"),
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "probability": probability,
                            "chunk_index": chunk_index,
                            "chunk_text_preview": previews[chunk_index] if chunk_index < len(previews) else "",
                            "attention_weight": float(weights[item_index, chunk_index].item()),
                        }
                    )

    if not true_labels:
        raise ValueError("평가할 validation sample이 없습니다")
    metrics = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(f1_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)
        ),
    }
    return metrics, prediction_rows, attention_rows


def save_evaluation_outputs(
    prediction_rows: list[dict[str, Any]],
    attention_rows: list[dict[str, Any]],
    output_dir: str | Path,
    num_labels: int = 10,
) -> dict[str, float]:
    """최종 검증 결과를 요청된 파일명과 0~10 고정 축으로 저장한다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame(prediction_rows)
    if prediction_frame.empty:
        raise ValueError("저장할 validation prediction이 없습니다")
    y_true = prediction_frame["true_label"].astype(int).tolist()
    y_pred = prediction_frame["pred_label"].astype(int).tolist()
    all_labels = list(range(num_labels))

    report_text = classification_report(
        y_true,
        y_pred,
        labels=all_labels,
        target_names=[str(label) for label in all_labels],
        digits=4,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    matrix = confusion_matrix(y_true, y_pred, labels=all_labels)
    matrix_frame = pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in all_labels],
        columns=[f"pred_{label}" for label in all_labels],
    )
    matrix_frame.index.name = "true_label"
    matrix_frame.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    write_csv(prediction_frame, output_dir / "valid_predictions.csv")
    write_csv(attention_rows, output_dir / "attention_outputs.csv")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
