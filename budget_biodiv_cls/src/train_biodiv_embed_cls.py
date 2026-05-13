"""
DB에서 사전 계산된 임베딩을 읽어 MLP 이진 분류 모델을 학습합니다.

biodiv_document_chunks.embedding(real[])을 청크 평균풀링으로 문서 벡터를 만들고
biodiv_documents.label_v2(0/1)를 예측합니다.

사용 예:
    python src/train_biodiv_embed_cls.py --user postgres --password " "
    python src/train_biodiv_embed_cls.py --user postgres --password " " --epochs 30 --hidden-dim 512
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from config import MODEL_DIR
from load_biodiv_csv_to_postgres import connect

try:
    from psycopg import sql
except ImportError as exc:
    raise SystemExit("psycopg is required: pip install psycopg[binary]") from exc


DEFAULT_OUTPUT_DIR = MODEL_DIR / "label_v2_embed"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_UNDERSAMPLE_RATIO = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DB 임베딩 기반 생물다양성 MLP 분류 모델 학습")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--documents-table", default="biodiv_documents")
    parser.add_argument("--chunks-table", default="biodiv_document_chunks")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--label-col", default="label_v2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--balance-mode",
        choices=["pos_weight", "undersample", "none"],
        default="pos_weight",
    )
    parser.add_argument("--undersample-ratio", type=float, default=DEFAULT_UNDERSAMPLE_RATIO)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--host", default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--db-name", default=os.getenv("PGDATABASE", "biofin"))
    parser.add_argument("--user", default=os.getenv("PGUSER", "postgres"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD"))
    parser.add_argument("--no-password-prompt", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_embeddings_from_db(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """DB에서 임베딩을 읽어 문서 단위 평균풀링 후 (embeddings, labels) 반환."""
    query = sql.SQL("""
        SELECT d.id, d.{label_col}, c.embedding
        FROM {docs} d
        JOIN {chunks} c ON c.document_id = d.id
        WHERE d.{label_col} IN (0, 1)
          AND c.embedding_model = %s
        ORDER BY d.id, c.chunk_index
    """).format(
        label_col=sql.Identifier(args.label_col),
        docs=sql.Identifier(args.schema, args.documents_table),
        chunks=sql.Identifier(args.schema, args.chunks_table),
    )

    print("DB에서 임베딩 로딩 중...")
    with connect(args) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (args.embedding_model,))
            rows = cur.fetchall()

    if not rows:
        raise ValueError(
            f"임베딩이 없습니다. embedding_model='{args.embedding_model}' 을 확인하세요."
        )

    # 문서별 청크 수집
    doc_chunks: dict[int, list[list[float]]] = {}
    doc_labels: dict[int, int] = {}
    for doc_id, label, embedding in rows:
        doc_id = int(doc_id)
        if doc_id not in doc_chunks:
            doc_chunks[doc_id] = []
            doc_labels[doc_id] = int(label)
        doc_chunks[doc_id].append(embedding)

    # 청크 평균풀링 → 문서 벡터
    embeddings_list = []
    labels_list = []
    for doc_id in sorted(doc_chunks):
        chunk_matrix = np.array(doc_chunks[doc_id], dtype=np.float32)
        embeddings_list.append(chunk_matrix.mean(axis=0))
        labels_list.append(doc_labels[doc_id])

    embeddings = np.stack(embeddings_list)
    labels = np.array(labels_list, dtype=np.int64)
    print(f"로드 완료 — 문서: {len(labels)}, 임베딩 차원: {embeddings.shape[1]}")
    return embeddings, labels


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        self.embeddings = torch.from_numpy(embeddings)
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {"embedding": self.embeddings[idx], "label": self.labels[idx]}


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def undersample(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    keep_neg = min(len(neg_idx), int(len(pos_idx) * ratio))
    sampled_neg = rng.choice(neg_idx, keep_neg, replace=False)
    indices = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(indices)
    return embeddings[indices], labels[indices]


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
        emb = batch["embedding"].to(device)
        lbl = batch["label"].to(device)
        optimizer.zero_grad()
        loss = criterion(model(emb), lbl)
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
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []
    for batch in loader:
        probs = torch.sigmoid(model(batch["embedding"].to(device))).cpu().numpy().tolist()
        all_probs.extend(probs if isinstance(probs, list) else [probs])
        all_labels.extend(batch["label"].numpy().astype(int).tolist())
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return all_labels, all_probs, auc


def find_best_threshold(
    labels: list[int],
    probs: list[float],
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> tuple[float, float, float, float, list[int]]:
    best = (threshold_min, -1.0, 0.0, 0.0, [])
    for thr in np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step):
        thr = float(min(thr, threshold_max))
        preds = [1 if p >= thr else 0 for p in probs]
        f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        if f1 > best[1]:
            best = (
                thr,
                f1,
                precision_score(labels, preds, pos_label=1, zero_division=0),
                recall_score(labels, preds, pos_label=1, zero_division=0),
                preds,
            )
    return best


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    embeddings, labels = load_embeddings_from_db(args)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    print(f"관련(1): {n_pos}, 비관련(0): {n_neg}")

    train_emb, val_emb, train_lbl, val_lbl = train_test_split(
        embeddings, labels, test_size=args.val_ratio, stratify=labels, random_state=args.seed
    )
    print(f"학습: {len(train_lbl)}행  검증: {len(val_lbl)}행")

    if args.balance_mode == "undersample":
        train_emb, train_lbl = undersample(train_emb, train_lbl, args.undersample_ratio, args.seed)
        u_pos = int((train_lbl == 1).sum())
        print(
            f"언더샘플링 후: {len(train_lbl)}행 "
            f"(관련: {u_pos}, 비관련: {len(train_lbl) - u_pos})"
        )
    else:
        print(f"불균형 처리: {args.balance_mode}")

    input_dim = embeddings.shape[1]
    model = MLP(input_dim, args.hidden_dim, args.dropout).to(device)
    print(f"MLP: {input_dim} → {args.hidden_dim} → {args.hidden_dim // 2} → 1")

    train_n_pos = int((train_lbl == 1).sum())
    train_n_neg = len(train_lbl) - train_n_pos
    if args.balance_mode == "pos_weight":
        pos_weight = torch.tensor([train_n_neg / train_n_pos], dtype=torch.float).to(device)
        print(f"pos_weight: {pos_weight.item():.2f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader = DataLoader(EmbeddingDataset(train_emb, train_lbl), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(EmbeddingDataset(val_emb, val_lbl), batch_size=args.batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch, best_threshold = -1.0, 0, 0.5
    final_labels: list[int] = []
    final_preds: list[int] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        v_labels, v_probs, auc = evaluate(model, val_loader, device)
        threshold, f1, precision, recall, v_preds = find_best_threshold(
            v_labels, v_probs, args.threshold_min, args.threshold_max, args.threshold_step
        )
        print(
            f"Epoch {epoch:2d} | loss={train_loss:.4f} | "
            f"thr={threshold:.2f} | P={precision:.4f} | R={recall:.4f} | "
            f"F1={f1:.4f} | AUC={auc:.4f}"
        )
        if f1 > best_f1:
            best_f1, best_epoch, best_threshold = f1, epoch, threshold
            final_labels, final_preds = v_labels, v_preds
            torch.save(model.state_dict(), args.output_dir / "model.pt")
            (args.output_dir / "training_metadata.json").write_text(
                json.dumps(
                    {
                        "best_epoch": int(best_epoch),
                        "best_threshold": float(best_threshold),
                        "best_f1": float(best_f1),
                        "best_precision": float(precision),
                        "best_recall": float(recall),
                        "best_auc": float(auc),
                        "input_dim": int(input_dim),
                        "hidden_dim": int(args.hidden_dim),
                        "dropout": float(args.dropout),
                        "embedding_model": args.embedding_model,
                        "label_col": args.label_col,
                        "balance_mode": args.balance_mode,
                        "undersample_ratio": float(args.undersample_ratio),
                        "seed": int(args.seed),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
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
