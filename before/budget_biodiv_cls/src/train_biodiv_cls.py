"""
KLUE/RoBERTa-small 기반 생물다양성 이진 분류 모델 학습 스크립트.

clean_document_text 컬럼(문서 본문 또는 메타데이터 fallback)을 사용해
label_v2(0=비관련, 1=관련)을 예측합니다. -1(라벨링 실패) 값은 제외합니다.

클래스 불균형 처리 방식:
  pos_weight  : 손실함수에 양성 클래스 가중치 부여 (BCEWithLogitsLoss)
  undersample : 다수 클래스(0) 샘플 수를 줄여서 균형 맞춤
  none        : 처리 없음

사용 예:
    python train_biodiv_cls.py
    python train_biodiv_cls.py --epochs 15 --batch-size 32
    python train_biodiv_cls.py --model-name klue/bert-base
    python train_biodiv_cls.py --balance-mode undersample --undersample-ratio 2.0
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
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
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
DEFAULT_DATA_CSV         = BIODIV_TEXT_LABELED_V2_CSV
DEFAULT_MODEL_NAME       = "klue/roberta-small"   # 한국어 특화 경량 BERT
DEFAULT_OUTPUT_DIR       = MODEL_DIR / "label_v2"
DEFAULT_TEXT_COL         = DOCUMENT_TEXT_COLUMN
DEFAULT_LABEL_COL        = "label_v2"
DEFAULT_UNDERSAMPLE_RATIO = 3.0                    # 음성:양성 비율 (기본 3배)


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description="생물다양성 이진 분류 모델 학습")
    parser.add_argument("--data-csv",   type=Path,  default=DEFAULT_DATA_CSV,   help="학습 데이터 CSV")
    parser.add_argument("--text-col",               default=DEFAULT_TEXT_COL,   help="텍스트 컬럼명")
    parser.add_argument("--label-col",              default=DEFAULT_LABEL_COL,  help="라벨 컬럼명")
    parser.add_argument("--model-name",             default=DEFAULT_MODEL_NAME, help="HuggingFace 모델 이름")
    parser.add_argument("--output-dir", type=Path,  default=DEFAULT_OUTPUT_DIR, help="모델 저장 폴더")
    parser.add_argument("--max-len",    type=int,   default=512,                help="토크나이저 최대 토큰 수")
    parser.add_argument("--batch-size", type=int,   default=16,                 help="배치 크기")
    parser.add_argument("--epochs",     type=int,   default=10,                 help="학습 에폭 수")
    parser.add_argument("--lr",         type=float, default=2e-5,               help="학습률")
    parser.add_argument("--val-ratio",  type=float, default=0.15,               help="검증 데이터 비율")
    parser.add_argument("--seed",       type=int,   default=42,                 help="난수 시드")
    parser.add_argument("--no-cuda",    action="store_true",                    help="CPU만 사용")
    parser.add_argument(
        "--balance-mode",
        choices=["pos_weight", "undersample", "none"],
        default="pos_weight",
        help="클래스 불균형 처리 방식 (기본값: pos_weight)",
    )
    parser.add_argument(
        "--undersample-ratio", type=float, default=DEFAULT_UNDERSAMPLE_RATIO, metavar="R",
        help="undersample 모드: 음성 샘플을 양성의 R배로 줄임 (기본: 3.0)",
    )
    parser.add_argument("--threshold-min",  type=float, default=0.05, help="임계값 탐색 최솟값")
    parser.add_argument("--threshold-max",  type=float, default=0.95, help="임계값 탐색 최댓값")
    parser.add_argument("--threshold-step", type=float, default=0.01, help="임계값 탐색 간격")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """재현성을 위해 모든 난수 생성기의 시드를 설정합니다."""
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
    """
    음성 샘플(label=0)을 양성 샘플(label=1)의 ratio배 수로 줄입니다.

    예) ratio=3.0이면 neg:pos = 3:1
    데이터 불균형이 심할 때 과적합 방지에 도움이 됩니다.
    """
    rng      = random.Random(seed)
    pos_idx  = [i for i, lb in enumerate(labels) if lb == 1]
    neg_idx  = [i for i, lb in enumerate(labels) if lb == 0]
    keep_neg = min(len(neg_idx), int(len(pos_idx) * ratio))
    sampled_neg = rng.sample(neg_idx, keep_neg)
    indices  = pos_idx + sampled_neg
    rng.shuffle(indices)
    return [texts[i] for i in indices], [labels[i] for i in indices]


def clean_cell(value: object) -> str:
    """셀 값을 문자열로 변환하고 NaN/None은 빈 문자열로 반환합니다."""
    if pd.isna(value):
        return ""
    text = str(value or "").strip()
    return "" if text.lower() in {"nan", "none"} else text


def build_metadata_fallback_text(row: pd.Series) -> str:
    """
    문서 본문이 없을 때 메타데이터 컬럼들을 대신 사용합니다.

    각 컬럼을 '{컬럼명}: {값}' 형식으로 이어붙입니다.
    """
    parts = []
    for col in METADATA_COLUMNS:
        value = clean_cell(row.get(col, ""))
        if value:
            parts.append(f"{col}: {value}")
    return "\n".join(parts)


def build_text(row: pd.Series, text_col: str) -> str:
    """
    지정된 텍스트 컬럼 값을 반환합니다.
    값이 비어있으면 메타데이터로 대체(fallback)합니다.
    """
    text = clean_cell(row.get(text_col, ""))
    if text:
        return text
    return build_metadata_fallback_text(row)


class BiodivDataset(Dataset):
    """
    PyTorch Dataset: 텍스트 목록과 라벨 목록을 입력받아
    BERT 토크나이저를 적용한 배치를 반환합니다.

    Dataset: DataLoader와 함께 사용하는 PyTorch 기본 추상 클래스
    """
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_len: int,
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
            padding="max_length",   # 짧은 텍스트는 0으로 채움
            truncation=True,        # 긴 텍스트는 max_len에서 자름
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),       # [seq_len]
            "attention_mask": enc["attention_mask"].squeeze(0),   # [seq_len]
            # 이진 분류: BCEWithLogitsLoss를 위해 float 타입 사용
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    한 에폭 학습을 수행하고 평균 손실을 반환합니다.

    model.train(): 드롭아웃, 배치 정규화 등이 학습 모드로 전환됨
    optimizer.zero_grad(): 이전 배치의 기울기 초기화
    loss.backward(): 역전파 (기울기 계산)
    optimizer.step(): 파라미터 업데이트
    """
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        # squeeze(-1): [batch, 1] → [batch] (BCEWithLogitsLoss 입력 형태)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[float], float]:
    """
    검증 셋에서 모델을 평가하고 (실제 라벨, 예측 확률, AUC) 를 반환합니다.

    @torch.no_grad(): 검증 중 기울기 계산 비활성화 → 메모리/속도 절약
    torch.sigmoid(): 로짓(logit)을 0~1 확률로 변환
    """
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
        probs  = torch.sigmoid(logits).cpu().numpy().tolist()
        all_probs.extend(probs if isinstance(probs, list) else [probs])
        all_labels.extend(batch["label"].numpy().astype(int).tolist())

    # ROC-AUC: 1에 가까울수록 좋음. 단일 클래스면 계산 불가
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return all_labels, all_probs, auc


def metrics_at_threshold(
    labels: list[int],
    probs: list[float],
    threshold: float,
) -> tuple[float, float, float, list[int]]:
    """확률을 threshold로 이진화하여 F1, Precision, Recall을 계산합니다."""
    preds     = [1 if p >= threshold else 0 for p in probs]
    f1        = f1_score(labels, preds, pos_label=1, zero_division=0)
    precision = precision_score(labels, preds, pos_label=1, zero_division=0)
    recall    = recall_score(labels, preds, pos_label=1, zero_division=0)
    return f1, precision, recall, preds


def find_best_threshold(
    labels: list[int],
    probs: list[float],
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> tuple[float, float, float, float, list[int]]:
    """
    F1이 가장 높아지는 최적 임계값(threshold)을 탐색합니다.

    이진 분류에서 threshold=0.5 가 항상 최적이 아닙니다.
    데이터 불균형 시 threshold를 낮추면 양성 클래스 recall이 높아지고,
    높이면 precision이 높아집니다. F1 최대화 지점을 그리드 탐색합니다.
    """
    if threshold_step <= 0:
        raise ValueError("--threshold-step 은 0보다 커야 합니다.")
    if not 0 <= threshold_min <= threshold_max <= 1:
        raise ValueError("--threshold-min/max 는 0~1 범위에서 min ≤ max 여야 합니다.")

    best_threshold = threshold_min
    best_f1        = -1.0
    best_precision = 0.0
    best_recall    = 0.0
    best_preds: list[int] = []

    for threshold in np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step):
        threshold = float(min(threshold, threshold_max))
        f1, precision, recall, preds = metrics_at_threshold(labels, probs, threshold)
        if f1 > best_f1:
            best_threshold = threshold
            best_f1        = f1
            best_precision = precision
            best_recall    = recall
            best_preds     = preds

    return best_threshold, best_f1, best_precision, best_recall, best_preds


def main() -> int:
    """학습 파이프라인 실행: 데이터 로드 → 전처리 → 학습 → 최고 모델 저장."""
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    # 데이터 로드 및 유효 라벨(0, 1)만 필터링 (-1 실패 라벨 제외)
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
        raise ValueError("stratify 분할을 위해 각 라벨이 최소 2개 이상 필요합니다.")

    # 입력 텍스트 소스 분포 출력 (디버깅용)
    if "text_source" in df.columns:
        source_counts = df["text_source"].fillna("").replace("", "unknown").value_counts()
        print("입력 소스 분포: " + ", ".join(f"{name}={count}" for name, count in source_counts.items()))

    # 텍스트 구성 + 빈 텍스트 제외
    texts  = [build_text(row, args.text_col) for _, row in df.iterrows()]
    labels = df[args.label_col].astype(int).tolist()
    non_empty     = [(text, label) for text, label in zip(texts, labels) if text.strip()]
    dropped_empty = len(texts) - len(non_empty)
    if dropped_empty:
        print(f"빈 텍스트 제외: {dropped_empty}행")
    if not non_empty:
        raise ValueError("학습 가능한 텍스트가 없습니다.")
    texts, labels = map(list, zip(*non_empty))

    # stratify: 라벨 비율을 유지하면서 학습/검증 분리
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=args.val_ratio,
        stratify=labels,
        random_state=args.seed,
    )
    print(f"학습: {len(train_texts)}행  검증: {len(val_texts)}행")

    # 클래스 불균형 처리: undersample
    if args.balance_mode == "undersample":
        if args.undersample_ratio <= 0:
            raise ValueError("--undersample-ratio 는 0보다 커야 합니다.")
        train_texts, train_labels = undersample(
            train_texts, train_labels, args.undersample_ratio, args.seed
        )
        u_pos = sum(train_labels)
        u_neg = len(train_labels) - u_pos
        print(f"언더샘플링 후: {len(train_texts)}행  (관련(1): {u_pos}, 비관련(0): {u_neg}, ratio={args.undersample_ratio:.1f})")
    else:
        print(f"불균형 처리: {args.balance_mode}")

    # 모델/토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
    model.to(device)

    train_ds     = BiodivDataset(train_texts, train_labels, tokenizer, args.max_len)
    val_ds       = BiodivDataset(val_texts,   val_labels,   tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    train_n_pos = sum(train_labels)
    train_n_neg = len(train_labels) - train_n_pos
    if train_n_pos == 0 or train_n_neg == 0:
        raise ValueError(f"학습 셋에 {args.label_col} 0과 1이 모두 필요합니다.")

    # 클래스 불균형 처리: pos_weight (양성 클래스에 더 높은 손실 가중치 부여)
    if args.balance_mode == "pos_weight":
        # pos_weight = neg 수 / pos 수: 양성이 희귀할수록 가중치 높아짐
        pos_weight = torch.tensor([train_n_neg / train_n_pos], dtype=torch.float).to(device)
        print(f"pos_weight: {pos_weight.item():.2f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # AdamW: BERT 계열 모델에 일반적으로 사용되는 옵티마이저 (weight decay 포함)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1    = -1.0
    best_epoch = 0
    best_threshold  = 0.5
    final_labels: list[int] = []
    final_preds:  list[int] = []

    for epoch in range(1, args.epochs + 1):
        train_loss                          = train_epoch(model, train_loader, optimizer, criterion, device)
        v_labels, v_probs, auc              = evaluate(model, val_loader, device)
        threshold, f1, precision, recall, v_preds = find_best_threshold(
            v_labels, v_probs, args.threshold_min, args.threshold_max, args.threshold_step,
        )
        print(
            f"Epoch {epoch:2d} | loss={train_loss:.4f} | "
            f"thr={threshold:.2f} | P={precision:.4f} | R={recall:.4f} | "
            f"F1={f1:.4f} | AUC={auc:.4f}"
        )

        # 최고 F1 갱신 시 모델 저장
        if f1 > best_f1:
            best_f1        = f1
            best_epoch     = epoch
            best_threshold = threshold
            final_labels   = v_labels
            final_preds    = v_preds
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            # 학습 메타데이터 저장 (나중에 재현이나 추론에 필요한 정보)
            metadata = {
                "best_epoch":         int(best_epoch),
                "best_threshold":     float(best_threshold),
                "best_f1":            float(best_f1),
                "best_precision":     float(precision),
                "best_recall":        float(recall),
                "best_auc":           float(auc),
                "model_name":         args.model_name,
                "data_csv":           str(args.data_csv),
                "text_col":           args.text_col,
                "label_col":          args.label_col,
                "max_len":            int(args.max_len),
                "balance_mode":       args.balance_mode,
                "undersample_ratio":  float(args.undersample_ratio),
                "threshold_min":      float(args.threshold_min),
                "threshold_max":      float(args.threshold_max),
                "threshold_step":     float(args.threshold_step),
                "seed":               int(args.seed),
            }
            (args.output_dir / "training_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            print(f"  → 최고 모델 저장 (F1={best_f1:.4f}, threshold={best_threshold:.2f})")

    print(f"\n최고 F1: {best_f1:.4f}  (epoch {best_epoch}, threshold {best_threshold:.2f})")
    if final_labels:
        print("\n=== 최고 모델 검증 결과 ===")
        print(
            classification_report(
                final_labels, final_preds,
                target_names=["비관련(0)", "관련(1)"],
                zero_division=0,
            )
        )
    print(f"모델 저장 위치: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
