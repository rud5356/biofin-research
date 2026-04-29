"""
KLUE/RoBERTa-small 기반 생물다양성 이진 분류 학습 스크립트.

4개 텍스트 컬럼(분야명, 부문명, 프로그램명, 세부사업명)을 이어 붙여
biodiv_label(0/1)을 예측합니다. -1(실패) 라벨은 제외합니다.

사용 예:
    python train_biodiv_cls.py
    python train_biodiv_cls.py --epochs 15 --batch-size 32
    python train_biodiv_cls.py --model-name klue/bert-base
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_DATA_CSV = (
    Path(__file__).parents[1] / "data" / "사업별결산세출지출현황_2024년도_biodiv_labeled.csv"
)
DEFAULT_MODEL_NAME = "klue/roberta-small"
DEFAULT_OUTPUT_DIR = Path(__file__).parents[1] / "model"
TEXT_COLUMNS = ["분야명", "부문명", "프로그램명", "세부사업명"]
LABEL_COL = "biodiv_label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="생물다양성 이진 분류 모델 학습")
    parser.add_argument("--data-csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--undersample-ratio",
        type=float,
        default=None,
        metavar="R",
        help="학습 셋 언더샘플링: 음성 샘플을 양성의 R배로 줄임 (예: 3.0 → neg:pos=3:1). 기본값: 미적용",
    )
    return parser.parse_args()


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
    """음성 샘플을 양성의 ratio배 수로 줄인 학습 셋을 반환."""
    rng = random.Random(seed)
    pos_idx = [i for i, lb in enumerate(labels) if lb == 1]
    neg_idx = [i for i, lb in enumerate(labels) if lb == 0]
    keep_neg = min(len(neg_idx), int(len(pos_idx) * ratio))
    sampled_neg = rng.sample(neg_idx, keep_neg)
    indices = pos_idx + sampled_neg
    rng.shuffle(indices)
    return [texts[i] for i in indices], [labels[i] for i in indices]


def build_text(row: pd.Series) -> str:
    parts = [str(row.get(col, "") or "").strip() for col in TEXT_COLUMNS]
    return " | ".join(p for p in parts if p)


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
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
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
) -> tuple[float, float, float, float, list, list, list]:
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

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
    precision = precision_score(all_labels, preds, pos_label=1, zero_division=0)
    recall = recall_score(all_labels, preds, pos_label=1, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return f1, precision, recall, auc, all_labels, preds, all_probs


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")

    df = pd.read_csv(args.data_csv, encoding="utf-8-sig")
    df = df[df[LABEL_COL].isin([0, 1])].reset_index(drop=True)
    n_pos = int(df[LABEL_COL].sum())
    n_neg = len(df) - n_pos
    print(f"유효 데이터: {len(df)}행  (관련(1): {n_pos}, 비관련(0): {n_neg})")
    if n_pos == 0 or n_neg == 0:
        raise ValueError("학습하려면 biodiv_label 0과 1 데이터가 모두 필요합니다.")
    if min(n_pos, n_neg) < 2:
        raise ValueError("stratify 검증 분할을 위해 각 라벨이 최소 2개 이상 필요합니다.")

    texts = [build_text(row) for _, row in df.iterrows()]
    labels = df[LABEL_COL].astype(int).tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=args.val_ratio,
        stratify=labels,
        random_state=args.seed,
    )
    print(f"학습: {len(train_texts)}행  검증: {len(val_texts)}행")

    if args.undersample_ratio is not None:
        train_texts, train_labels = undersample(
            train_texts, train_labels, args.undersample_ratio, args.seed
        )
        u_pos = sum(train_labels)
        u_neg = len(train_labels) - u_pos
        print(
            f"언더샘플링 후: {len(train_texts)}행  "
            f"(관련(1): {u_pos}, 비관련(0): {u_neg}, ratio={args.undersample_ratio:.1f})"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=1
    )
    model.to(device)

    train_ds = BiodivDataset(train_texts, train_labels, tokenizer, args.max_len)
    val_ds = BiodivDataset(val_texts, val_labels, tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # 클래스 불균형 보정: 학습 셋 neg/pos 비율을 pos_weight로 사용
    train_n_pos = sum(train_labels)
    train_n_neg = len(train_labels) - train_n_pos
    pos_weight = torch.tensor([train_n_neg / train_n_pos], dtype=torch.float).to(device)
    print(f"pos_weight: {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch = -1.0, 0
    final_labels: list[int] = []
    final_preds: list[int] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        f1, precision, recall, auc, v_labels, v_preds, _ = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:2d} | loss={train_loss:.4f} | "
            f"P={precision:.4f} | R={recall:.4f} | F1={f1:.4f} | AUC={auc:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            final_labels = v_labels
            final_preds = v_preds
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  → 최고 모델 저장 (F1={best_f1:.4f})")

    print(f"\n최고 F1: {best_f1:.4f}  (epoch {best_epoch})")
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


if __name__ == "__main__":
    main()
