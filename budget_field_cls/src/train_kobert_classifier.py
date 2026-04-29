from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLASSIFICATION_TEXT_COLUMN,
    DEFAULT_KOBERT_MODEL,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_WEIGHT_DECAY,
    WORKFILE_KOBERT_MODEL_DIR,
    WORKFILE_TEST_SPLIT_PATH,
    WORKFILE_TRAIN_SPLIT_PATH,
    WORKFILE_VAL_SPLIT_PATH,
)
from utils import save_json

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
except ImportError as exc:  # pragma: no cover - dependency error path
    torch = None
    Dataset = object
    DataLoader = object
    _IMPORT_ERROR = exc
else:  # pragma: no cover - import success path
    _IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a KoBERT-style classifier on BIOFIN workfile text splits."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=WORKFILE_TRAIN_SPLIT_PATH,
        help="Training split CSV path.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=WORKFILE_VAL_SPLIT_PATH,
        help="Validation split CSV path.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=WORKFILE_TEST_SPLIT_PATH,
        help="Test split CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKFILE_KOBERT_MODEL_DIR,
        help="Directory to save model checkpoints and reports.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_KOBERT_MODEL,
        help="Hugging Face model name or local checkpoint path.",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_CLASSIFICATION_TEXT_COLUMN,
        help="Column containing the document text.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column containing the label name.",
    )
    parser.add_argument(
        "--label-id-column",
        default="label_id",
        help="Column containing the original label id.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum token length used by the tokenizer.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Optional character-level truncation before tokenization. 0 means no extra truncation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for the linear scheduler.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training even if CUDA is available.",
    )    
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Training device. 'auto' uses CUDA when available, 'cuda' fails if CUDA is unavailable.",    
    )
    return parser.parse_args()


def ensure_runtime_dependencies() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing dependencies for KoBERT training. "
            "Install at least: torch, transformers, sentencepiece"
        ) from _IMPORT_ERROR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.cpu and args.device != "auto":
        raise ValueError("Use either --cpu or --device, not both.")

    requested_device = "cpu" if args.cpu else args.device

    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested with --device cuda, but PyTorch cannot see a CUDA GPU. "
                "Check PyTorch CUDA build, NVIDIA driver, and Docker --gpus option."
            )
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    print("CUDA is not available; falling back to CPU. Use --device cuda to fail instead.")
    return torch.device("cpu")


def print_device_summary(device: torch.device) -> None:
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")


def read_split(path: Path, text_column: str, label_column: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path, encoding="utf-8-sig")
    required_columns = {text_column, label_column}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing_columns)}")

    dataframe = dataframe.copy()
    dataframe[text_column] = dataframe[text_column].fillna("").astype(str)
    dataframe[label_column] = dataframe[label_column].fillna("").astype(str)
    dataframe = dataframe[dataframe[text_column].str.strip() != ""]
    dataframe = dataframe[dataframe[label_column].str.strip() != ""]
    return dataframe.reset_index(drop=True)


def build_label_mapping(
    train_df: pd.DataFrame,
    label_column: str,
    label_id_column: str,
) -> tuple[dict[str, int], dict[int, dict[str, object]]]:
    records: list[tuple[object, str]] = []
    seen_keys: set[tuple[str, str]] = set()

    for row in train_df.to_dict(orient="records"):
        label = str(row[label_column])
        original_label_id = str(row.get(label_id_column, ""))
        key = (label, original_label_id)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        records.append((row.get(label_id_column, ""), label))

    def sort_key(item: tuple[object, str]) -> tuple[int, object]:
        raw_id, label = item
        try:
            return (0, int(raw_id))
        except (TypeError, ValueError):
            return (1, label)

    records.sort(key=sort_key)

    label_to_model_id: dict[str, int] = {}
    model_id_to_meta: dict[int, dict[str, object]] = {}
    for model_label_id, (original_label_id, label) in enumerate(records):
        label_to_model_id[label] = model_label_id
        model_id_to_meta[model_label_id] = {
            "model_label_id": model_label_id,
            "original_label_id": "" if pd.isna(original_label_id) else original_label_id,
            "label": label,
        }

    return label_to_model_id, model_id_to_meta


def attach_model_labels(
    dataframe: pd.DataFrame,
    label_column: str,
    label_to_model_id: dict[str, int],
) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe["model_label_id"] = dataframe[label_column].map(label_to_model_id)
    missing_labels = dataframe[dataframe["model_label_id"].isna()][label_column].unique().tolist()
    if missing_labels:
        raise ValueError(f"Found labels not present in training split: {missing_labels}")
    dataframe["model_label_id"] = dataframe["model_label_id"].astype(int)
    return dataframe


def maybe_truncate_text(text: str, max_chars: int) -> str:
    if max_chars and max_chars > 0:
        return text[:max_chars]
    return text


@dataclass
class TextRecord:
    record_index: int
    text: str
    model_label_id: int
    raw_row: dict[str, object]


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        text_column: str,
        max_chars: int,
    ) -> None:
        self.records: list[TextRecord] = []
        for index, row in enumerate(dataframe.to_dict(orient="records")):
            self.records.append(
                TextRecord(
                    record_index=index,
                    text=maybe_truncate_text(str(row[text_column]), max_chars),
                    model_label_id=int(row["model_label_id"]),
                    raw_row=row,
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TextRecord:
        return self.records[index]


def make_collate_fn(tokenizer, max_length: int):
    def collate_fn(batch: list[TextRecord]) -> dict[str, object]:
        texts = [item.text for item in batch]
        labels = [item.model_label_id for item in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        encoded["records"] = batch
        return encoded

    return collate_fn


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> tuple[dict[str, torch.Tensor], list[TextRecord]]:
    records = batch.pop("records")
    tensor_batch = {key: value.to(device) for key, value in batch.items()}
    return tensor_batch, records


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_meta: dict[int, dict[str, object]]) -> dict[str, object]:
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty evaluation set.")

    supports: list[int] = []
    per_label_metrics: list[dict[str, object]] = []
    f1_scores: list[float] = []
    weighted_f1_sum = 0.0

    for model_label_id, meta in sorted(label_meta.items()):
        true_positive = int(np.sum((y_true == model_label_id) & (y_pred == model_label_id)))
        false_positive = int(np.sum((y_true != model_label_id) & (y_pred == model_label_id)))
        false_negative = int(np.sum((y_true == model_label_id) & (y_pred != model_label_id)))
        support = int(np.sum(y_true == model_label_id))

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        supports.append(support)
        f1_scores.append(f1)
        weighted_f1_sum += f1 * support
        per_label_metrics.append(
            {
                "model_label_id": model_label_id,
                "original_label_id": meta["original_label_id"],
                "label": meta["label"],
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    accuracy = float(np.mean(y_true == y_pred))
    total_support = int(sum(supports))
    macro_f1 = float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0
    weighted_f1 = float(weighted_f1_sum / total_support) if total_support else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "support": total_support,
        "per_label": per_label_metrics,
    }


def evaluate(
    model,
    data_loader: DataLoader,
    device: torch.device,
    label_meta: dict[int, dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in data_loader:
            tensor_batch, records = move_batch_to_device(batch, device)
            outputs = model(**tensor_batch)
            logits = outputs.logits
            total_loss += float(outputs.loss.item())
            total_batches += 1

            probabilities = torch.softmax(logits, dim=-1)
            predicted_ids = torch.argmax(probabilities, dim=-1)
            predicted_scores = torch.max(probabilities, dim=-1).values
            true_ids = tensor_batch["labels"]

            predicted_ids_np = predicted_ids.detach().cpu().numpy()
            predicted_scores_np = predicted_scores.detach().cpu().numpy()
            true_ids_np = true_ids.detach().cpu().numpy()

            y_true.extend(true_ids_np.tolist())
            y_pred.extend(predicted_ids_np.tolist())

            for record, true_id, pred_id, confidence in zip(
                records,
                true_ids_np.tolist(),
                predicted_ids_np.tolist(),
                predicted_scores_np.tolist(),
            ):
                row = dict(record.raw_row)
                row.update(
                    {
                        "true_model_label_id": true_id,
                        "true_label": label_meta[true_id]["label"],
                        "pred_model_label_id": pred_id,
                        "pred_label": label_meta[pred_id]["label"],
                        "confidence": float(confidence),
                        "correct": int(true_id == pred_id),
                    }
                )
                prediction_rows.append(row)

    metrics = compute_metrics(np.array(y_true), np.array(y_pred), label_meta)
    metrics["loss"] = total_loss / total_batches if total_batches else 0.0
    return metrics, prediction_rows


def train_one_epoch(
    model,
    data_loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
) -> float:
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    total_batches = 0

    for step, batch in enumerate(data_loader, start=1):
        tensor_batch, _ = move_batch_to_device(batch, device)
        outputs = model(**tensor_batch)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        if step % gradient_accumulation_steps == 0 or step == len(data_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += float(outputs.loss.item())
        total_batches += 1

    return total_loss / total_batches if total_batches else 0.0


def build_dataloader(
    dataframe: pd.DataFrame,
    tokenizer,
    text_column: str,
    max_chars: int,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
):
    dataset = TextClassificationDataset(dataframe, text_column=text_column, max_chars=max_chars)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collate_fn(tokenizer, max_length=max_length),
    )
    return dataset, data_loader


def run(args: argparse.Namespace) -> int:
    ensure_runtime_dependencies()
    set_seed(args.seed)

    train_csv = args.train_csv.resolve()
    val_csv = args.val_csv.resolve()
    test_csv = args.test_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_split(train_csv, args.text_column, args.label_column)
    val_df = read_split(val_csv, args.text_column, args.label_column)
    test_df = read_split(test_csv, args.text_column, args.label_column)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Train/val/test splits must all contain at least one row.")

    label_to_model_id, label_meta = build_label_mapping(train_df, args.label_column, args.label_id_column)
    train_df = attach_model_labels(train_df, args.label_column, label_to_model_id)
    val_df = attach_model_labels(val_df, args.label_column, label_to_model_id)
    test_df = attach_model_labels(test_df, args.label_column, label_to_model_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_meta),
        id2label={index: meta["label"] for index, meta in label_meta.items()},
        label2id={meta["label"]: index for index, meta in label_meta.items()},
        problem_type="single_label_classification"
    )

    device = resolve_device(args)
    print_device_summary(device)
    model.to(device)


    _, train_loader = build_dataloader(
        train_df,
        tokenizer=tokenizer,
        text_column=args.text_column,
        max_chars=args.max_chars,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    _, val_loader = build_dataloader(
        val_df,
        tokenizer=tokenizer,
        text_column=args.text_column,
        max_chars=args.max_chars,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    _, test_loader = build_dataloader(
        test_df,
        tokenizer=tokenizer,
        text_column=args.text_column,
        max_chars=args.max_chars,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    updates_per_epoch = math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps))
    total_training_steps = max(1, updates_per_epoch * args.epochs)
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    history_rows: list[dict[str, object]] = []
    best_val_macro_f1 = -1.0
    best_model_dir = output_dir / "best_model"
    best_metrics_path = output_dir / "best_val_metrics.json"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            gradient_accumulation_steps=max(1, args.gradient_accumulation_steps),
            max_grad_norm=args.max_grad_norm,
        )
        val_metrics, _ = evaluate(model, val_loader, device, label_meta)
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        history_rows.append(history_row)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            save_json(best_metrics_path, val_metrics)

    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    best_model.to(device)
    test_metrics, test_predictions = evaluate(best_model, test_loader, device, label_meta)

    training_history_path = output_dir / "training_history.csv"
    label_mapping_path = output_dir / "label_mapping.json"
    test_metrics_path = output_dir / "test_metrics.json"
    predictions_path = output_dir / "test_predictions.csv"
    run_config_path = output_dir / "run_config.json"

    pd.DataFrame(history_rows).to_csv(training_history_path, index=False, encoding="utf-8-sig")
    save_json(label_mapping_path, list(label_meta.values()))
    save_json(test_metrics_path, test_metrics)
    pd.DataFrame(test_predictions).to_csv(predictions_path, index=False, encoding="utf-8-sig")
    save_json(
        run_config_path,
        {
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "test_csv": str(test_csv),
            "output_dir": str(output_dir),
            "model_name": args.model_name,
            "text_column": args.text_column,
            "max_length": args.max_length,
            "max_chars": args.max_chars,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "seed": args.seed,
            "device": str(device),
            "num_labels": len(label_meta),
        },
    )

    print(f"Saved best model: {best_model_dir}")
    print(f"Saved training history: {training_history_path}")
    print(f"Saved test metrics: {test_metrics_path}")
    print(f"Saved test predictions: {predictions_path}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
