"""
KLUE/RoBERTa-small 기반 생물다양성 다중 분류 모델 학습 스크립트.

clean_document_text 컬럼(문서 본문 또는 메타데이터 fallback)을 사용해
0 ~ (num_labels-1) 범위의 카테고리를 예측합니다.

이진 분류(train_biodiv_cls.py)와 달리 여러 카테고리를 동시에 분류합니다.
손실함수: CrossEntropyLoss (다중 분류 표준)
평가지표: macro-F1 (클래스 불균형 시 각 클래스를 동등하게 평가)

사용 예:
    python train_biodiv_multiclass_cls.py --label-col category --num-labels 10
    python train_biodiv_multiclass_cls.py --label-col category --num-labels 10 --epochs 15
    python train_biodiv_multiclass_cls.py --class-weight  # 클래스 불균형 보정
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


# ─── 기본 설정값 ─────────────────────────────────────────────────────────────
DEFAULT_DATA_CSV    = BIODIV_TEXT_LABELED_V2_CSV
DEFAULT_MODEL_NAME  = "klue/roberta-small"           # 한국어 경량 BERT
DEFAULT_OUTPUT_DIR  = MODEL_DIR / "label_multiclass"
DEFAULT_TEXT_COL    = DOCUMENT_TEXT_COLUMN
DEFAULT_LABEL_COL   = "category"
DEFAULT_NUM_LABELS  = 10


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description="생물다양성 다중 분류 모델 학습")
    parser.add_argument("--data-csv",    type=Path,  default=DEFAULT_DATA_CSV,   help="학습 데이터 CSV")
    parser.add_argument("--text-col",                default=DEFAULT_TEXT_COL,   help="텍스트 컬럼명")
    parser.add_argument("--label-col",               default=DEFAULT_LABEL_COL,  help="라벨 컬럼명")
    parser.add_argument("--num-labels",  type=int,   default=DEFAULT_NUM_LABELS, help="분류 카테고리 수")
    parser.add_argument("--model-name",              default=DEFAULT_MODEL_NAME, help="HuggingFace 모델 이름")
    parser.add_argument("--output-dir",  type=Path,  default=DEFAULT_OUTPUT_DIR, help="모델 저장 폴더")
    parser.add_argument("--max-len",     type=int,   default=512,                help="토크나이저 최대 토큰 수")
    parser.add_argument("--batch-size",  type=int,   default=16,                 help="배치 크기")
    parser.add_argument("--epochs",      type=int,   default=10,                 help="학습 에폭 수")
    parser.add_argument("--lr",          type=float, default=2e-5,               help="학습률")
    parser.add_argument("--val-ratio",   type=float, default=0.15,               help="검증 데이터 비율")
    parser.add_argument("--seed",        type=int,   default=42,                 help="난수 시드")
    parser.add_argument("--no-cuda",     action="store_true",                    help="CPU만 사용")
    parser.add_argument(
        "--class-weight",
        action="store_true",
        help="클래스 불균형 보정: 각 클래스 빈도의 역수를 손실 가중치로 사용",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """재현성을 위해 모든 난수 생성기의 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_cell(value: object) -> str:
    """셀 값을 문자열로 변환하고 NaN/None은 빈 문자열로 반환합니다."""
    if pd.isna(value):
        return ""
    text = str(value or "").strip()
    return "" if text.lower() in {"nan", "none"} else text


def build_metadata_fallback_text(row: pd.Series) -> str:
    """문서 본문이 없을 때 메타데이터 컬럼들을 '{컬럼}: {값}' 형식으로 이어붙입니다."""
    parts = []
    for col in METADATA_COLUMNS:
        value = clean_cell(row.get(col, ""))
        if value:
            parts.append(f"{col}: {value}")
    return "\n".join(parts)


def build_text(row: pd.Series, text_col: str) -> str:
    """지정 컬럼에서 텍스트를 가져오고, 비어있으면 메타데이터로 대체합니다."""
    text = clean_cell(row.get(text_col, ""))
    if text:
        return text
    return build_metadata_fallback_text(row)


class BiodivDataset(Dataset):
    """
    PyTorch Dataset: BERT 토크나이저를 적용한 다중 분류용 데이터셋.

    이진 분류(BiodivDataset in train_biodiv_cls.py)와 달리
    라벨 타입이 long (정수 인덱스)입니다.
    """
    def __init__(
        self,
        texts:     list[str],
        labels:    list[int],
        tokenizer,
        max_len:   int,
    ) -> None:
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

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
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            # CrossEntropyLoss는 정수(long) 타입 라벨 요구
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    """한 에폭 학습을 수행하고 평균 손실을 반환합니다."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """
    검증 셋에서 모델을 평가하고 (실제 라벨, 예측 라벨) 목록을 반환합니다.

    torch.argmax: 가장 높은 확률의 클래스 인덱스를 예측값으로 선택
    """
    model.eval()
    all_preds:  list[int] = []
    all_labels: list[int] = []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        # 가장 높은 로짓 값의 인덱스 = 예측 클래스
        preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].numpy().tolist())
    return all_labels, all_preds


def compute_class_weights(
    labels:     list[int],
    num_labels: int,
    device:     torch.device,
) -> torch.Tensor:
    """
    각 클래스의 샘플 수에 반비례하는 가중치를 계산합니다.

    드문 클래스에 높은 가중치를 부여해 불균형 문제를 완화합니다.
    weights 합이 num_labels 가 되도록 정규화합니다.
    """
    # 각 클래스의 샘플 수 계산 (0이면 1로 대체해 ZeroDivision 방지)
    counts  = np.bincount(labels, minlength=num_labels).astype(float)
    counts  = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / counts
    # 전체 합이 num_labels 가 되도록 정규화
    weights = weights / weights.sum() * num_labels
    return torch.tensor(weights, dtype=torch.float).to(device)


def main() -> int:
    """다중 분류 학습 파이프라인 실행."""
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    print(f"분류 카테고리 수: {args.num_labels}")

    # 데이터 로드 및 유효 라벨 필터링
    df = pd.read_csv(args.data_csv, encoding="utf-8-sig")
    if args.text_col not in df.columns:
        raise ValueError(f"입력 CSV에 텍스트 컬럼이 없습니다: {args.text_col}")
    if args.label_col not in df.columns:
        raise ValueError(f"입력 CSV에 라벨 컬럼이 없습니다: {args.label_col}")

    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
    # 0 ~ num_labels-1 범위 외의 값 제외
    valid_labels = list(range(args.num_labels))
    df = df[df[args.label_col].isin(valid_labels)].reset_index(drop=True)
    df[args.label_col] = df[args.label_col].astype(int)

    label_dist = df[args.label_col].value_counts().sort_index()
    print(f"유효 데이터: {len(df)}행")
    print("클래스 분포:")
    for cls, cnt in label_dist.items():
        print(f"  {cls}: {cnt}행")

    # 텍스트 구성 및 빈 텍스트 제외
    texts  = [build_text(row, args.text_col) for _, row in df.iterrows()]
    labels = df[args.label_col].astype(int).tolist()
    non_empty     = [(text, label) for text, label in zip(texts, labels) if text.strip()]
    dropped_empty = len(texts) - len(non_empty)
    if dropped_empty:
        print(f"빈 텍스트 제외: {dropped_empty}행")
    if not non_empty:
        raise ValueError("학습 가능한 텍스트가 없습니다.")
    texts, labels = map(list, zip(*non_empty))

    # stratify: 클래스 비율을 유지하면서 학습/검증 분리
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=args.val_ratio,
        stratify=labels,
        random_state=args.seed,
    )
    print(f"학습: {len(train_texts)}행  검증: {len(val_texts)}행")

    # 모델/토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=args.num_labels
    )
    model.to(device)

    train_ds     = BiodivDataset(train_texts, train_labels, tokenizer, args.max_len)
    val_ds       = BiodivDataset(val_texts,   val_labels,   tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    # 손실함수: 클래스 불균형 보정 여부에 따라 가중치 적용
    if args.class_weight:
        weight = compute_class_weights(train_labels, args.num_labels, device)
        print(f"클래스 가중치: {weight.cpu().numpy().round(3).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1    = -1.0
    best_epoch = 0
    final_labels: list[int] = []
    final_preds:  list[int] = []

    for epoch in range(1, args.epochs + 1):
        train_loss        = train_epoch(model, train_loader, optimizer, criterion, device)
        v_labels, v_preds = evaluate(model, val_loader, device)

        # macro-F1: 각 클래스의 F1을 동등하게 평균 (클래스 불균형에도 공정한 평가)
        f1  = f1_score(v_labels, v_preds, average="macro", zero_division=0)
        acc = sum(p == l for p, l in zip(v_preds, v_labels)) / len(v_labels)
        print(f"Epoch {epoch:2d} | loss={train_loss:.4f} | macro-F1={f1:.4f} | acc={acc:.4f}")

        # 최고 macro-F1 갱신 시 모델 저장
        if f1 > best_f1:
            best_f1    = f1
            best_epoch = epoch
            final_labels = v_labels
            final_preds  = v_preds
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            metadata = {
                "best_epoch":     int(best_epoch),
                "best_macro_f1":  float(best_f1),
                "best_acc":       float(acc),
                "num_labels":     args.num_labels,
                "model_name":     args.model_name,
                "data_csv":       str(args.data_csv),
                "text_col":       args.text_col,
                "label_col":      args.label_col,
                "max_len":        int(args.max_len),
                "class_weight":   args.class_weight,
                "seed":           int(args.seed),
            }
            (args.output_dir / "training_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            print(f"  → 최고 모델 저장 (macro-F1={best_f1:.4f})")

    print(f"\n최고 macro-F1: {best_f1:.4f}  (epoch {best_epoch})")
    if final_labels:
        print("\n=== 최고 모델 검증 결과 ===")
        print(
            classification_report(
                final_labels, final_preds,
                labels=list(range(args.num_labels)),
                zero_division=0,
            )
        )
    print(f"모델 저장 위치: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
