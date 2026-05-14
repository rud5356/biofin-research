"""
KLUE/RoBERTa-small 기반 생물다양성 다중 분류 학습 스크립트.

clean_document_text 컬럼(문서 본문 또는 메타데이터 fallback)을 사용해
0~(num_labels-1) 범위의 카테고리를 예측합니다.

사용 예:
    python train_biodiv_multiclass_cls.py --label-col category --num-labels 10
    python train_biodiv_multiclass_cls.py --label-col category --num-labels 10 --epochs 15
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
from sklearn.metrics import classification_report, f1_score
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
DEFAULT_MODEL_NAME = "klue/roberta-small"
DEFAULT_OUTPUT_DIR = MODEL_DIR / "label_multiclass"
DEFAULT_TEXT_COL = DOCUMENT_TEXT_COLUMN
DEFAULT_LABEL_COL = "category"
DEFAULT_NUM_LABELS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="생물다양성 다중 분류 모델 학습")
    parser.add_argument("--data-csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--text-col", default=DEFAULT_TEXT_COL)
    parser.add_argument("--label-col", default=DEFAULT_LABEL_COL)
    parser.add_argument("--num-labels", type=int, default=DEFAULT_NUM_LABELS)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--class-weight",
        action="store_true",
        help="클래스 불균형 보정: 각 클래스 빈도의 역수를 가중치로 사용",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].numpy().tolist())
    return all_labels, all_preds


def compute_class_weights(labels: list[int], num_labels: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_labels).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_labels
    return torch.tensor(weights, dtype=torch.float).to(device)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")
    print(f"분류 카테고리 수: {args.num_labels}")

    df = pd.read_csv(args.data_csv, encoding="utf-8-sig")
    if args.text_col not in df.columns:
        raise ValueError(f"입력 CSV에 텍스트 컬럼이 없습니다: {args.text_col}")
    if args.label_col not in df.columns:
        raise ValueError(f"입력 CSV에 라벨 컬럼이 없습니다: {args.label_col}")

    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
    valid_labels = list(range(args.num_labels))
    df = df[df[args.label_col].isin(valid_labels)].reset_index(drop=True)
    df[args.label_col] = df[args.label_col].astype(int)

    label_dist = df[args.label_col].value_counts().sort_index()
    print(f"유효 데이터: {len(df)}행")
    print("클래스 분포:")
    for cls, cnt in label_dist.items():
        print(f"  {cls}: {cnt}행")

    texts = [build_text(row, args.text_col) for _, row in df.iterrows()]
    labels = df[args.label_col].astype(int).tolist()
    non_empty = [(text, label) for text, label in zip(texts, labels) if text.strip()]
    dropped_empty = len(texts) - len(non_empty)
    if dropped_empty:
        print(f"빈 텍스트 제외: {dropped_empty}행")
    if not non_empty:
        raise ValueError("학습 가능한 텍스트가 없습니다.")
    texts, labels = map(list, zip(*non_empty))

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=args.val_ratio,
        stratify=labels,
        random_state=args.seed,
    )
    print(f"학습: {len(train_texts)}행  검증: {len(val_texts)}행")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=args.num_labels
    )
    model.to(device)

    train_ds = BiodivDataset(train_texts, train_labels, tokenizer, args.max_len)
    val_ds = BiodivDataset(val_texts, val_labels, tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    if args.class_weight:
        weight = compute_class_weights(train_labels, args.num_labels, device)
        print(f"클래스 가중치: {weight.cpu().numpy().round(3).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch = -1.0, 0
    final_labels: list[int] = []
    final_preds: list[int] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        v_labels, v_preds = evaluate(model, val_loader, device)
        f1 = f1_score(v_labels, v_preds, average="macro", zero_division=0)
        acc = sum(p == l for p, l in zip(v_preds, v_labels)) / len(v_labels)
        print(
            f"Epoch {epoch:2d} | loss={train_loss:.4f} | "
            f"macro-F1={f1:.4f} | acc={acc:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            final_labels = v_labels
            final_preds = v_preds
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            metadata = {
                "best_epoch": int(best_epoch),
                "best_macro_f1": float(best_f1),
                "best_acc": float(acc),
                "num_labels": args.num_labels,
                "model_name": args.model_name,
                "data_csv": str(args.data_csv),
                "text_col": args.text_col,
                "label_col": args.label_col,
                "max_len": int(args.max_len),
                "class_weight": args.class_weight,
                "seed": int(args.seed),
            }
            (args.output_dir / "training_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  → 최고 모델 저장 (macro-F1={best_f1:.4f})")

    print(f"\n최고 macro-F1: {best_f1:.4f}  (epoch {best_epoch})")
    if final_labels:
        print("\n=== 최고 모델 검증 결과 ===")
        print(
            classification_report(
                final_labels,
                final_preds,
                labels=list(range(args.num_labels)),
                zero_division=0,
            )
        )
    print(f"모델 저장 위치: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
