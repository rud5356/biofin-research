"""
KoBigBird 기반 생물다양성 긴문서 이진 분류 학습 스크립트.

기본적으로 biodiv_document_text_dataset_labeled_v2.csv의 clean_document_text를
최대 2048 토큰까지 입력하고 label_v2(0/1)를 예측합니다.

사용 예:
    python src/train_biodiv_long_cls.py
    python src/train_biodiv_long_cls.py --max-len 4096 --batch-size 1 --grad-accum-steps 8
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    BIODIV_TEXT_LABELED_V2_CSV,
    DOCUMENT_TEXT_COLUMN,
    METADATA_COLUMNS,
    MODEL_DIR,
)


DEFAULT_DATA_CSV = BIODIV_TEXT_LABELED_V2_CSV
DEFAULT_MODEL_NAME = "monologg/kobigbird-bert-base"
DEFAULT_OUTPUT_DIR = MODEL_DIR / "label_v2_long"
DEFAULT_TEXT_COL = DOCUMENT_TEXT_COLUMN
DEFAULT_LABEL_COL = "label_v2"
DEFAULT_UNDERSAMPLE_RATIO = 3.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def undersample(
    texts: list[str],
    labels: list[int],
    ratio: float,
    seed: int,
) -> tuple[list[str], list[int]]:
    rng = random.Random(seed)
    pos_idx = [i for i, lb in enumerate(labels) if lb == 1]
    neg_idx = [i for i, lb in enumerate(labels) if lb == 0]
    keep_neg = min(len(neg_idx), int(len(pos_idx) * ratio))
    sampled_neg = rng.sample(neg_idx, keep_neg)
    indices = pos_idx + sampled_neg
    rng.shuffle(indices)
    return [texts[i] for i in indices], [labels[i] for i in indices]


def clean_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value or "").strip()
    return "" if text.lower() in {"nan", "none"} else text


def build_metadata_fallback_text(row: pd.Series) -> str:
    parts = []
    for col in METADATA_COLUMNS:
        value = clean_cell(row.get(col, ""))
        if value:
            parts.append(f"{col}: {value}")
    return "\n".join(parts)


def build_text(row: pd.Series, text_col: str) -> str:
    text = clean_cell(row.get(text_col, ""))
    if text:
        return text
    return build_metadata_fallback_text(row)


class BiodivDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_len: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[float], float]:
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()
        all_probs.extend(probs if isinstance(probs, list) else [probs])
        all_labels.extend(batch["label"].numpy().astype(int).tolist())

    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return all_labels, all_probs, auc


def metrics_at_threshold(
    labels: list[int],
    probs: list[float],
    threshold: float,
) -> tuple[float, float, float, list[int]]:
    from sklearn.metrics import f1_score, precision_score, recall_score

    preds = [1 if p >= threshold else 0 for p in probs]
    f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
    precision = precision_score(labels, preds, pos_label=1, zero_division=0)
    recall = recall_score(labels, preds, pos_label=1, zero_division=0)
    return f1, precision, recall, preds


def find_best_threshold(
    labels: list[int],
    probs: list[float],
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> tuple[float, float, float, float, list[int]]:
    if threshold_step <= 0:
        raise ValueError("--threshold-step은 0보다 커야 합니다.")
    if not 0 <= threshold_min <= threshold_max <= 1:
        raise ValueError("--threshold-min/max는 0~1 범위에서 min <= max 여야 합니다.")

    best_threshold = threshold_min
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0
    best_preds: list[int] = []

    thresholds = np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step)
    for threshold in thresholds:
        threshold = float(min(threshold, threshold_max))
        f1, precision, recall, preds = metrics_at_threshold(labels, probs, threshold)
        if f1 > best_f1:
            best_threshold = threshold
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_preds = preds

    return best_threshold, best_f1, best_precision, best_recall, best_preds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="생물다양성 긴문서 이진 분류 모델 학습")
    parser.add_argument("--data-csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--text-col", default=DEFAULT_TEXT_COL)
    parser.add_argument("--label-col", default=DEFAULT_LABEL_COL)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="CUDA 사용 시 AMP fp16 학습")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="GPU 메모리를 줄이기 위해 gradient checkpointing 사용",
    )
    parser.add_argument(
        "--attention-type",
        choices=["block_sparse", "original_full"],
        default="block_sparse",
        help="BigBird attention 방식. 긴문서는 block_sparse 권장",
    )
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--num-random-blocks", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--balance-mode",
        choices=["pos_weight", "undersample", "none"],
        default="pos_weight",
        help="클래스 불균형 처리 방식. 기본값: pos_weight",
    )
    parser.add_argument(
        "--undersample-ratio",
        type=float,
        default=DEFAULT_UNDERSAMPLE_RATIO,
        metavar="R",
        help="balance-mode=undersample일 때 음성 샘플을 양성의 R배로 줄임",
    )
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_accum_steps: int,
    use_fp16: bool,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.cuda.amp.autocast(enabled=use_fp16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            loss = criterion(logits, labels)
            loss_for_backward = loss / grad_accum_steps

        scaler.scale(loss_for_backward).backward()
        total_loss += loss.item()

        if step % grad_accum_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / len(loader)


def load_training_frame(args: argparse.Namespace) -> tuple[list[str], list[int], pd.DataFrame]:
    df = pd.read_csv(args.data_csv, encoding="utf-8-sig")
    if args.text_col not in df.columns:
        raise ValueError(f"입력 CSV에 텍스트 컬럼이 없습니다: {args.text_col}")
    if args.label_col not in df.columns:
        raise ValueError(f"입력 CSV에 라벨 컬럼이 없습니다: {args.label_col}")

    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
    df = df[df[args.label_col].isin([0, 1])].reset_index(drop=True)
    df[args.label_col] = df[args.label_col].astype(int)

    n_pos = int(df[args.label_col].sum())
    n_neg = len(df) - n_pos
    print(f"유효 데이터: {len(df)}행  (관련(1): {n_pos}, 비관련(0): {n_neg})")
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"학습하려면 {args.label_col} 0과 1 데이터가 모두 필요합니다.")
    if min(n_pos, n_neg) < 2:
        raise ValueError("stratify 검증 분할을 위해 각 라벨이 최소 2개 이상 필요합니다.")

    if "text_source" in df.columns:
        source_counts = df["text_source"].fillna("").replace("", "unknown").value_counts()
        print("입력 소스 분포: " + ", ".join(f"{name}={count}" for name, count in source_counts.items()))

    texts = [build_text(row, args.text_col) for _, row in df.iterrows()]
    labels = df[args.label_col].astype(int).tolist()
    non_empty = [(text, label) for text, label in zip(texts, labels) if text.strip()]
    dropped_empty = len(texts) - len(non_empty)
    if dropped_empty:
        print(f"빈 텍스트 제외: {dropped_empty}행")
    if not non_empty:
        raise ValueError("학습 가능한 텍스트가 없습니다.")

    texts, labels = map(list, zip(*non_empty))
    return texts, labels, df


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("--batch-size와 --eval-batch-size는 0보다 커야 합니다.")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps는 0보다 커야 합니다.")

    set_seed(args.seed)
    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    use_fp16 = bool(args.fp16 and device.type == "cuda")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"max_len: {args.max_len}, batch_size: {args.batch_size}, grad_accum_steps: {args.grad_accum_steps}")

    texts, labels, _ = load_training_frame(args)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=args.val_ratio,
        stratify=labels,
        random_state=args.seed,
    )
    print(f"학습: {len(train_texts)}행  검증: {len(val_texts)}행")

    if args.balance_mode == "undersample":
        if args.undersample_ratio <= 0:
            raise ValueError("--undersample-ratio는 0보다 커야 합니다.")
        train_texts, train_labels = undersample(
            train_texts, train_labels, args.undersample_ratio, args.seed
        )
        u_pos = sum(train_labels)
        u_neg = len(train_labels) - u_pos
        print(
            f"언더샘플링 후: {len(train_texts)}행  "
            f"(관련(1): {u_pos}, 비관련(0): {u_neg}, ratio={args.undersample_ratio:.1f})"
        )
    else:
        print(f"불균형 처리: {args.balance_mode}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.max_len > getattr(tokenizer, "model_max_length", args.max_len):
        print(f"주의: tokenizer model_max_length={tokenizer.model_max_length}, 요청 max_len={args.max_len}")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        attention_type=args.attention_type,
        block_size=args.block_size,
        num_random_blocks=args.num_random_blocks,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)

    train_ds = BiodivDataset(train_texts, train_labels, tokenizer, args.max_len)
    val_ds = BiodivDataset(val_texts, val_labels, tokenizer, args.max_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )

    train_n_pos = sum(train_labels)
    train_n_neg = len(train_labels) - train_n_pos
    if train_n_pos == 0 or train_n_neg == 0:
        raise ValueError(f"학습 셋에 {args.label_col} 0과 1이 모두 필요합니다.")

    if args.balance_mode == "pos_weight":
        pos_weight = torch.tensor([train_n_neg / train_n_pos], dtype=torch.float).to(device)
        print(f"pos_weight: {pos_weight.item():.2f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch = -1.0, 0
    best_threshold = 0.5
    final_labels: list[int] = []
    final_preds: list[int] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.grad_accum_steps,
            use_fp16,
        )
        v_labels, v_probs, auc = evaluate(model, val_loader, device)
        threshold, f1, precision, recall, v_preds = find_best_threshold(
            v_labels,
            v_probs,
            args.threshold_min,
            args.threshold_max,
            args.threshold_step,
        )
        print(
            f"Epoch {epoch:2d} | loss={train_loss:.4f} | "
            f"thr={threshold:.2f} | P={precision:.4f} | R={recall:.4f} | "
            f"F1={f1:.4f} | AUC={auc:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            best_threshold = threshold
            final_labels = v_labels
            final_preds = v_preds
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            metadata = {
                "best_epoch": int(best_epoch),
                "best_threshold": float(best_threshold),
                "best_f1": float(best_f1),
                "best_precision": float(precision),
                "best_recall": float(recall),
                "best_auc": float(auc),
                "model_name": args.model_name,
                "data_csv": str(args.data_csv),
                "text_col": args.text_col,
                "label_col": args.label_col,
                "max_len": int(args.max_len),
                "batch_size": int(args.batch_size),
                "eval_batch_size": int(args.eval_batch_size),
                "grad_accum_steps": int(args.grad_accum_steps),
                "balance_mode": args.balance_mode,
                "undersample_ratio": float(args.undersample_ratio),
                "attention_type": args.attention_type,
                "block_size": int(args.block_size),
                "num_random_blocks": int(args.num_random_blocks),
                "fp16": bool(use_fp16),
                "gradient_checkpointing": bool(args.gradient_checkpointing),
                "threshold_min": float(args.threshold_min),
                "threshold_max": float(args.threshold_max),
                "threshold_step": float(args.threshold_step),
                "seed": int(args.seed),
            }
            (args.output_dir / "training_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  → 최고 모델 저장 (F1={best_f1:.4f}, threshold={best_threshold:.2f})")

    print(f"\n최고 F1: {best_f1:.4f}  (epoch {best_epoch}, threshold {best_threshold:.2f})")
    if final_labels:
        print("\n=== 최고 모델 검증 결과 ===")
        print(
            classification_report(
                final_labels,
                final_preds,
                target_names=["비관련(0)", "관련(1)"],
                zero_division=0,
            )
        )
    print(f"모델 저장 위치: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
