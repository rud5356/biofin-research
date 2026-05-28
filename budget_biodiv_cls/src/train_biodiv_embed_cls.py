"""
PostgreSQL에 저장된 문서 임베딩으로 MLP 이진 분류 모델을 학습합니다.

embed_biodiv_chunks_to_postgres.py 가 생성한 biodiv_document_chunks.embedding(real[]) 을
청크 단위 평균풀링(mean pooling)으로 문서 벡터를 만들고
biodiv_documents.label_v2(0=비관련, 1=관련)를 예측합니다.

텍스트 → BERT 토크나이저 → 학습 방식(train_biodiv_cls.py)과 달리
임베딩 벡터를 입력받아 간단한 MLP(다층 퍼셉트론)로 분류합니다.
(장점: BERT 추론 없이 빠름, 단점: BERT 수준의 정확도는 기대 어려움)

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
    raise SystemExit("psycopg 패키지가 필요합니다: pip install psycopg[binary]") from exc


# ─── 기본 설정값 ─────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR       = MODEL_DIR / "label_v2_embed"
# 임베딩 생성에 사용한 모델 이름 (embed_biodiv_chunks_to_postgres.py 와 일치해야 함)
DEFAULT_EMBEDDING_MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_UNDERSAMPLE_RATIO = 3.0   # 언더샘플링 시 neg:pos 비율


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description="DB 임베딩 기반 생물다양성 MLP 분류 모델 학습")
    # 데이터베이스 테이블 설정
    parser.add_argument("--schema",           default="public",              help="PostgreSQL 스키마")
    parser.add_argument("--documents-table",  default="biodiv_documents",    help="문서 테이블 이름")
    parser.add_argument("--chunks-table",     default="biodiv_document_chunks", help="청크 테이블 이름")
    parser.add_argument("--embedding-model",  default=DEFAULT_EMBEDDING_MODEL,  help="임베딩 모델 이름")
    parser.add_argument("--label-col",        default="label_v2",            help="라벨 컬럼명")
    # 모델 구조
    parser.add_argument("--output-dir",   type=Path,  default=DEFAULT_OUTPUT_DIR, help="모델 저장 폴더")
    parser.add_argument("--hidden-dim",   type=int,   default=256,               help="은닉층 차원")
    parser.add_argument("--dropout",      type=float, default=0.3,               help="드롭아웃 비율")
    # 학습 하이퍼파라미터
    parser.add_argument("--batch-size",   type=int,   default=64,                help="배치 크기")
    parser.add_argument("--epochs",       type=int,   default=20,                help="학습 에폭 수")
    parser.add_argument("--lr",           type=float, default=1e-3,              help="학습률")
    parser.add_argument("--val-ratio",    type=float, default=0.15,              help="검증 데이터 비율")
    parser.add_argument("--seed",         type=int,   default=42,                help="난수 시드")
    parser.add_argument("--no-cuda",      action="store_true",                   help="CPU만 사용")
    # 클래스 불균형 처리
    parser.add_argument(
        "--balance-mode",
        choices=["pos_weight", "undersample", "none"],
        default="pos_weight",
        help="클래스 불균형 처리 방식",
    )
    parser.add_argument("--undersample-ratio",  type=float, default=DEFAULT_UNDERSAMPLE_RATIO, help="언더샘플링 neg:pos 비율")
    # 임계값 탐색 범위
    parser.add_argument("--threshold-min",  type=float, default=0.05, help="임계값 탐색 최솟값")
    parser.add_argument("--threshold-max",  type=float, default=0.95, help="임계값 탐색 최댓값")
    parser.add_argument("--threshold-step", type=float, default=0.01, help="임계값 탐색 간격")
    # PostgreSQL 접속 정보
    parser.add_argument("--database-url",       default=os.getenv("DATABASE_URL"))
    parser.add_argument("--host",               default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port",           type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--db-name",            default=os.getenv("PGDATABASE", "biofin"))
    parser.add_argument("--user",               default=os.getenv("PGUSER", "postgres"))
    parser.add_argument("--password",           default=os.getenv("PGPASSWORD"))
    parser.add_argument("--no-password-prompt", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """재현성을 위해 모든 난수 생성기의 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_embeddings_from_db(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """
    PostgreSQL에서 청크 임베딩을 읽어 문서별 평균풀링 후 (embeddings, labels)를 반환합니다.

    평균풀링(mean pooling): 문서의 여러 청크 임베딩 벡터를 평균내어 하나의 문서 벡터로 만듦.
    예) 3개 청크 [v1, v2, v3] → (v1+v2+v3)/3

    DB 쿼리: biodiv_documents + biodiv_document_chunks JOIN
      - document_id 기준으로 청크 그룹화
      - embedding_model 필터로 지정된 모델 임베딩만 선택
    """
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

    # 문서별 청크 임베딩 수집
    doc_chunks: dict[int, list[list[float]]] = {}
    doc_labels: dict[int, int] = {}
    for doc_id, label, embedding in rows:
        doc_id = int(doc_id)
        if doc_id not in doc_chunks:
            doc_chunks[doc_id] = []
            doc_labels[doc_id] = int(label)
        doc_chunks[doc_id].append(embedding)

    # 각 문서의 청크 임베딩들을 평균하여 문서 벡터 생성
    embeddings_list = []
    labels_list     = []
    for doc_id in sorted(doc_chunks):
        # stack: 리스트 → 2D 배열 [num_chunks, embed_dim]
        chunk_matrix = np.array(doc_chunks[doc_id], dtype=np.float32)
        embeddings_list.append(chunk_matrix.mean(axis=0))  # 행 방향 평균
        labels_list.append(doc_labels[doc_id])

    embeddings = np.stack(embeddings_list)
    labels     = np.array(labels_list, dtype=np.int64)
    print(f"로드 완료 — 문서: {len(labels)}, 임베딩 차원: {embeddings.shape[1]}")
    return embeddings, labels


class EmbeddingDataset(Dataset):
    """임베딩 벡터와 라벨을 PyTorch Tensor로 변환하는 Dataset."""
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        self.embeddings = torch.from_numpy(embeddings)
        self.labels     = torch.from_numpy(labels).float()  # BCEWithLogitsLoss는 float 필요

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {"embedding": self.embeddings[idx], "label": self.labels[idx]}


class MLP(nn.Module):
    """
    다층 퍼셉트론(Multi-Layer Perceptron) 이진 분류 모델.

    구조: input_dim → hidden_dim → hidden_dim//2 → 1
    LayerNorm: 배치 크기에 무관하게 안정적인 학습 (BatchNorm 대비 소규모 데이터에 유리)
    Dropout: 과적합 방지를 위한 뉴런 랜덤 비활성화
    """
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
            nn.Linear(hidden_dim // 2, 1),  # 이진 분류: 출력 1개
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # [batch, 1] → [batch]


def undersample(
    embeddings: np.ndarray,
    labels:     np.ndarray,
    ratio:      float,
    seed:       int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    음성 샘플(label=0)을 양성(label=1)의 ratio배 수로 줄입니다.

    np.random.default_rng: 재현성 있는 NumPy 난수 생성기
    """
    rng     = np.random.default_rng(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    keep_neg    = min(len(neg_idx), int(len(pos_idx) * ratio))
    sampled_neg = rng.choice(neg_idx, keep_neg, replace=False)
    indices     = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(indices)
    return embeddings[indices], labels[indices]


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
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[float], float]:
    """
    검증 셋에서 모델을 평가하고 (실제 라벨, 예측 확률, AUC)를 반환합니다.

    torch.sigmoid: BCEWithLogitsLoss의 출력(로짓)을 확률(0~1)로 변환
    """
    model.eval()
    all_probs:  list[float] = []
    all_labels: list[int]   = []
    for batch in loader:
        probs = torch.sigmoid(model(batch["embedding"].to(device))).cpu().numpy().tolist()
        all_probs.extend(probs if isinstance(probs, list) else [probs])
        all_labels.extend(batch["label"].numpy().astype(int).tolist())
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return all_labels, all_probs, auc


def find_best_threshold(
    labels:         list[int],
    probs:          list[float],
    threshold_min:  float,
    threshold_max:  float,
    threshold_step: float,
) -> tuple[float, float, float, float, list[int]]:
    """F1이 최대가 되는 임계값(threshold)을 그리드 탐색합니다."""
    best = (threshold_min, -1.0, 0.0, 0.0, [])
    for thr in np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step):
        thr   = float(min(thr, threshold_max))
        preds = [1 if p >= thr else 0 for p in probs]
        f1    = f1_score(labels, preds, pos_label=1, zero_division=0)
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
    """MLP 학습 파이프라인 실행: DB 로드 → 전처리 → 학습 → 최고 모델 저장."""
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    embeddings, labels = load_embeddings_from_db(args)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    print(f"관련(1): {n_pos}, 비관련(0): {n_neg}")

    # stratify: 라벨 비율 유지하며 학습/검증 분리
    train_emb, val_emb, train_lbl, val_lbl = train_test_split(
        embeddings, labels, test_size=args.val_ratio, stratify=labels, random_state=args.seed
    )
    print(f"학습: {len(train_lbl)}행  검증: {len(val_lbl)}행")

    if args.balance_mode == "undersample":
        train_emb, train_lbl = undersample(train_emb, train_lbl, args.undersample_ratio, args.seed)
        u_pos = int((train_lbl == 1).sum())
        print(f"언더샘플링 후: {len(train_lbl)}행 (관련: {u_pos}, 비관련: {len(train_lbl) - u_pos})")
    else:
        print(f"불균형 처리: {args.balance_mode}")

    # MLP 모델 생성 (입력 차원은 임베딩 벡터 크기에 자동으로 맞춤)
    input_dim = embeddings.shape[1]
    model     = MLP(input_dim, args.hidden_dim, args.dropout).to(device)
    print(f"MLP: {input_dim} → {args.hidden_dim} → {args.hidden_dim // 2} → 1")

    train_n_pos = int((train_lbl == 1).sum())
    train_n_neg = len(train_lbl) - train_n_pos
    if args.balance_mode == "pos_weight":
        # pos_weight = neg/pos: 양성이 희귀할수록 가중치 높아짐
        pos_weight = torch.tensor([train_n_neg / train_n_pos], dtype=torch.float).to(device)
        print(f"pos_weight: {pos_weight.item():.2f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader = DataLoader(EmbeddingDataset(train_emb, train_lbl), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(EmbeddingDataset(val_emb,   val_lbl),   batch_size=args.batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1    = -1.0
    best_epoch = 0
    best_threshold  = 0.5
    final_labels: list[int] = []
    final_preds:  list[int] = []

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
            # model.pt: PyTorch 가중치만 저장 (HuggingFace 모델과 달리 config 불필요)
            torch.save(model.state_dict(), args.output_dir / "model.pt")
            (args.output_dir / "training_metadata.json").write_text(
                json.dumps(
                    {
                        "best_epoch":       int(best_epoch),
                        "best_threshold":   float(best_threshold),
                        "best_f1":          float(best_f1),
                        "best_precision":   float(precision),
                        "best_recall":      float(recall),
                        "best_auc":         float(auc),
                        "input_dim":        int(input_dim),
                        "hidden_dim":       int(args.hidden_dim),
                        "dropout":          float(args.dropout),
                        "embedding_model":  args.embedding_model,
                        "label_col":        args.label_col,
                        "balance_mode":     args.balance_mode,
                        "undersample_ratio": float(args.undersample_ratio),
                        "seed":             int(args.seed),
                    },
                    ensure_ascii=False, indent=2,
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
