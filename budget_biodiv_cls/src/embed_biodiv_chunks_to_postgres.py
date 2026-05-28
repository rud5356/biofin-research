"""
생물다양성 문서를 청크(chunk)로 나누고 트랜스포머 임베딩을 PostgreSQL에 저장하는 스크립트.

긴 문서를 통째로 임베딩하면 최대 토큰 제한(512~4096)에 걸리고
중요한 정보가 평균되어 희석됩니다.
이 스크립트는 문서를 적절한 크기로 나눈 청크 단위로 임베딩합니다.

임베딩 방법:
    sentence-transformers의 mean pooling을 사용합니다.
    어텐션 마스크로 가중치를 부여해 패딩 토큰의 영향을 제거합니다.
    --no-normalize 옵션이 없으면 L2 정규화를 적용합니다.

저장 방식:
    --embedding-storage array: 표준 PostgreSQL real[] 배열
    --embedding-storage pgvector: pgvector 확장의 vector 타입 (코사인 유사도 인덱스 지원)

사용 예:
    python src/embed_biodiv_chunks_to_postgres.py --user postgres
    python src/embed_biodiv_chunks_to_postgres.py --user postgres --limit 20
    python src/embed_biodiv_chunks_to_postgres.py --user postgres --if-exists replace-model
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

from tqdm import tqdm

try:
    from psycopg import sql
except ImportError as exc:
    raise SystemExit(
        "psycopg is required. Install dependencies with: pip install -r requirements_train.txt"
    ) from exc

# 같은 패키지의 DB 연결 함수들을 재사용
from load_biodiv_csv_to_postgres import connect, qualified_table, table_exists


# ─── 기본 테이블/모델 설정 ─────────────────────────────────────────────────────
DEFAULT_DOCUMENTS_TABLE = "biodiv_documents"
DEFAULT_CHUNKS_TABLE    = "biodiv_document_chunks"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class Chunk:
    """텍스트 청크 하나를 나타냅니다."""
    index:       int    # 문서 내 청크 순서 (0부터 시작)
    text:        str    # 청크 텍스트
    token_count: int    # 청크의 토큰 수


class TransformerEmbedder:
    """
    HuggingFace 트랜스포머 모델을 사용한 텍스트 임베더.

    mean pooling 방식:
        - 모델의 last_hidden_state에서 각 토큰의 벡터를 평균냅니다.
        - 어텐션 마스크로 패딩 토큰(0)의 영향을 제거합니다.
        - 수식: sum(hidden * mask) / sum(mask)

    L2 정규화:
        - 임베딩 벡터의 크기를 1로 맞춥니다.
        - 코사인 유사도를 내적(dot product)으로 계산할 수 있게 됩니다.
    """
    def __init__(self, model_name: str, device: str, normalize: bool) -> None:
        try:
            import numpy as np
            import torch
            import torch.nn.functional as F
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "Embedding dependencies are required. Install them with: "
                "pip install -r requirements_train.txt"
            ) from exc

        self.np         = np
        self.torch      = torch
        self.F          = F
        self.model_name = model_name
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        self.model      = AutoModel.from_pretrained(model_name)
        self.device     = self.resolve_device(device)
        self.normalize  = normalize
        self.model.to(self.device)
        self.model.eval()  # 추론 모드: Dropout/BatchNorm 비활성화

    def resolve_device(self, device: str) -> Any:
        """'auto'이면 GPU 사용 가능 여부를 자동 감지합니다."""
        if device == "auto":
            return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        return self.torch.device(device)

    @property
    def model_max_length(self) -> int:
        """토크나이저의 최대 토큰 길이를 반환합니다 (비정상 값이면 512 반환)."""
        value = getattr(self.tokenizer, "model_max_length", 512)
        if not isinstance(value, int) or value <= 0 or value > 10000:
            return 512
        return value

    def token_ids(self, text: str) -> list[int]:
        """텍스트를 토큰 ID 목록으로 변환합니다 (특수 토큰 제외)."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수를 반환합니다."""
        return len(self.token_ids(text))

    def decode_tokens(self, token_ids: list[int]) -> str:
        """토큰 ID 목록을 텍스트로 변환합니다."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def embed(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """
        텍스트 목록을 임베딩 벡터 목록으로 변환합니다.

        처리 흐름:
        1. 배치 단위로 토크나이징
        2. 모델에 입력해 last_hidden_state(토큰별 벡터) 획득
        3. 어텐션 마스크로 가중 평균 (mean pooling)
        4. L2 정규화 (--no-normalize가 없을 때)
        """
        embeddings: list[list[float]] = []
        # 모델의 최대 길이와 512 중 작은 값 사용
        max_length = min(self.model_max_length, 512)

        with self.torch.no_grad():  # 임베딩 시 그래디언트 계산 불필요
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,    # 배치 내 최장 시퀀스 길이로 패딩
                    truncation=True, # max_length 초과 시 잘라냄
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)

                hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
                # 어텐션 마스크를 hidden과 같은 shape으로 확장
                mask   = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                # 패딩 토큰 제외 mean pooling: clamp로 0 나누기 방지
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

                if self.normalize:
                    # L2 정규화: |v| = 1 (코사인 유사도 = 내적)
                    pooled = self.F.normalize(pooled, p=2, dim=1)

                embeddings.extend(pooled.cpu().numpy().astype(self.np.float32).tolist())

        return embeddings


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="biodiv_documents.clean_document_text를 청크로 나눠 임베딩을 PostgreSQL에 저장합니다."
    )
    parser.add_argument("--schema",          default="public")
    parser.add_argument("--documents-table", default=DEFAULT_DOCUMENTS_TABLE)
    parser.add_argument("--chunks-table",    default=DEFAULT_CHUNKS_TABLE)
    parser.add_argument("--text-column",     default="clean_document_text")
    parser.add_argument("--id-column",       default="id")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    # max-tokens: 청크 하나의 최대 토큰 수
    parser.add_argument("--max-tokens",      type=int, default=384)
    # overlap-tokens: 인접 청크 간 겹치는 토큰 수 (문맥 연속성 유지)
    parser.add_argument("--overlap-tokens",  type=int, default=64)
    # min-chars: 이 문자 수 미만의 청크는 저장하지 않음
    parser.add_argument("--min-chars",       type=int, default=20)
    parser.add_argument("--batch-size",      type=int, default=16)
    # limit: 처리할 문서 수 (0=전체)
    parser.add_argument("--limit",           type=int, default=0)
    parser.add_argument("--device",          default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-normalize",    action="store_true")
    parser.add_argument(
        "--embedding-storage",
        choices=["array", "pgvector"],
        default="array",
        help="array: 표준 PostgreSQL real[], pgvector: vector 타입 (코사인 인덱스 지원)",
    )
    parser.add_argument(
        "--if-exists",
        choices=["skip", "append", "replace-model", "replace-all"],
        default="skip",
        help=(
            "기존 청크 처리 방식. "
            "skip=이미 임베딩된 문서 건너뜀, "
            "append=모든 문서 upsert, "
            "replace-model=해당 모델 청크 삭제 후 재생성, "
            "replace-all=테이블 전체 비우고 재생성"
        ),
    )
    parser.add_argument(
        "--recreate-chunks-table",
        action="store_true",
        help="청크 테이블을 드롭하고 새로 생성합니다.",
    )
    parser.add_argument("--database-url",       default=os.getenv("DATABASE_URL"))
    parser.add_argument("--host",               default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port",               type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--db-name",            default=os.getenv("PGDATABASE", "biofin"))
    parser.add_argument("--user",               default=os.getenv("PGUSER", "postgres"))
    parser.add_argument("--password",           default=os.getenv("PGPASSWORD"))
    parser.add_argument(
        "--no-password-prompt",
        action="store_true",
        help="--password/PGPASSWORD가 없을 때 비밀번호를 묻지 않습니다.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """빈 줄을 제거하고 각 줄의 앞뒤 공백을 정리합니다."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def split_long_text_by_tokens(
    text: str,
    embedder: TransformerEmbedder,
    max_tokens: int,
    overlap_tokens: int,
) -> Iterable[tuple[str, int]]:
    """
    단락 하나가 max_tokens를 초과할 때 토큰 단위로 슬라이딩 윈도우 분할합니다.

    슬라이딩 윈도우:
    - 윈도우 크기: max_tokens
    - 이동 거리(step): max_tokens - overlap_tokens
    - 마지막 윈도우가 경계를 넘으면 루프 종료

    예: max_tokens=384, overlap_tokens=64, step=320
        윈도우1: 토큰 0~383
        윈도우2: 토큰 320~703
        ...
    """
    token_ids = embedder.token_ids(text)
    if len(token_ids) <= max_tokens:
        yield text, len(token_ids)
        return

    step = max_tokens - overlap_tokens
    for start in range(0, len(token_ids), step):
        window     = token_ids[start : start + max_tokens]
        chunk_text = embedder.decode_tokens(window)
        if chunk_text:
            yield chunk_text, len(window)
        if start + max_tokens >= len(token_ids):
            break


def paragraph_segments(text: str) -> list[str]:
    """
    텍스트를 단락 단위로 분리합니다.

    먼저 빈 줄 2개 이상으로 분리를 시도하고,
    단락이 1개뿐이면 줄바꿈 단위로 분리합니다.
    """
    normalized = normalize_text(text)
    if not normalized:
        return []
    parts = re.split(r"\n{2,}", normalized)
    if len(parts) == 1:
        parts = normalized.splitlines()
    return [part.strip() for part in parts if part.strip()]


def chunk_text(
    text: str,
    embedder: TransformerEmbedder,
    max_tokens: int,
    overlap_tokens: int,
    min_chars: int,
) -> list[Chunk]:
    """
    문서 텍스트를 max_tokens 이하의 청크 목록으로 분할합니다.

    알고리즘:
    1. 텍스트를 단락 단위로 분리
    2. 단락이 max_tokens 초과이면 슬라이딩 윈도우로 강제 분할
    3. 단락을 순서대로 쌓다가 max_tokens 초과 직전에 flush (청크 확정)
    4. min_chars 미만 청크는 제외

    이 방식은 단락 경계를 최대한 보존합니다.
    """
    segments      = paragraph_segments(text)
    chunks:       list[Chunk] = []
    current_parts: list[str]  = []
    current_tokens            = 0

    def flush_current() -> None:
        """현재까지 쌓인 단락들을 하나의 청크로 확정합니다."""
        nonlocal current_parts, current_tokens
        chunk = "\n".join(current_parts).strip()
        if len(chunk) >= min_chars:
            chunks.append(Chunk(len(chunks), chunk, current_tokens))
        current_parts  = []
        current_tokens = 0

    for segment in segments:
        segment_tokens = embedder.count_tokens(segment)
        if segment_tokens > max_tokens:
            # 단락 자체가 max_tokens를 초과하면 슬라이딩 윈도우로 분할
            flush_current()
            for split_text, split_tokens in split_long_text_by_tokens(
                segment, embedder, max_tokens, overlap_tokens
            ):
                if len(split_text) >= min_chars:
                    chunks.append(Chunk(len(chunks), split_text, split_tokens))
            continue

        # 현재 청크에 추가하면 max_tokens 초과이면 먼저 flush
        if current_parts and current_tokens + segment_tokens > max_tokens:
            flush_current()

        current_parts.append(segment)
        current_tokens += segment_tokens

    flush_current()  # 마지막에 쌓인 단락들도 청크로 저장
    return chunks


def ensure_source_table(conn: Any, args: argparse.Namespace) -> None:
    """소스 테이블이 없으면 에러를 발생시킵니다. load_biodiv_csv_to_postgres.py를 먼저 실행해야 합니다."""
    if not table_exists(conn, args.schema, args.documents_table):
        raise RuntimeError(
            f"소스 테이블 {args.schema}.{args.documents_table}가 없습니다. "
            "load_biodiv_csv_to_postgres.py를 먼저 실행하세요."
        )


def create_vector_extension(conn: Any) -> None:
    """pgvector 확장을 설치합니다. 설치되어 있지 않으면 에러를 발생시킵니다."""
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception as exc:
        raise RuntimeError(
            "pgvector 확장을 사용할 수 없습니다. "
            "--embedding-storage pgvector 없이 다시 실행하면 real[] 배열로 저장됩니다."
        ) from exc


def create_chunks_table(conn: Any, args: argparse.Namespace, embedding_dim: int) -> None:
    """
    청크 테이블을 생성합니다.

    embedding 컬럼의 타입:
    - pgvector 모드: vector(dim) — pgvector 확장 필요, 코사인 인덱스 지원
    - array 모드: real[] — 표준 PostgreSQL, 인덱스 없이 스캔

    UNIQUE(document_id, chunk_index, embedding_model):
    동일 문서+청크+모델 조합의 중복 삽입을 방지합니다.
    """
    if args.embedding_storage == "pgvector":
        create_vector_extension(conn)
        embedding_type = f"vector({embedding_dim})"
    else:
        embedding_type = "real[]"

    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(args.schema)))
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE {} (
                    id bigserial PRIMARY KEY,
                    document_id bigint NOT NULL REFERENCES {} ({}) ON DELETE CASCADE,
                    chunk_index integer NOT NULL,
                    chunk_text text NOT NULL,
                    token_count integer NOT NULL,
                    char_count integer NOT NULL,
                    embedding_model text NOT NULL,
                    embedding_dim integer NOT NULL,
                    chunk_max_tokens integer NOT NULL,
                    chunk_overlap_tokens integer NOT NULL,
                    embedding {} NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    UNIQUE (document_id, chunk_index, embedding_model)
                )
                """
            ).format(
                qualified_table(args.schema, args.chunks_table),
                qualified_table(args.schema, args.documents_table),
                sql.Identifier(args.id_column),
                sql.SQL(embedding_type),
            )
        )
        # 인덱스: document_id로 문서별 청크 조회, embedding_model로 모델별 조회
        cur.execute(
            sql.SQL("CREATE INDEX {} ON {} (document_id)").format(
                sql.Identifier(f"{args.chunks_table}_document_id_idx"),
                qualified_table(args.schema, args.chunks_table),
            )
        )
        cur.execute(
            sql.SQL("CREATE INDEX {} ON {} (embedding_model)").format(
                sql.Identifier(f"{args.chunks_table}_embedding_model_idx"),
                qualified_table(args.schema, args.chunks_table),
            )
        )


def prepare_chunks_table(conn: Any, args: argparse.Namespace, embedding_dim: int) -> None:
    """
    --if-exists 옵션에 따라 기존 청크 테이블을 처리합니다.

    처리 방식:
    - skip: 아무것도 하지 않음 (fetch_documents에서 이미 임베딩된 문서 제외)
    - append: 아무것도 하지 않음 (upsert로 중복 처리)
    - replace-model: 해당 모델의 청크만 삭제 후 재생성
    - replace-all: 테이블 전체 TRUNCATE 후 재생성
    - recreate-chunks-table: 테이블 자체를 DROP 후 새로 생성
    """
    exists = table_exists(conn, args.schema, args.chunks_table)
    with conn.cursor() as cur:
        if exists and args.recreate_chunks_table:
            cur.execute(sql.SQL("DROP TABLE {}").format(
                qualified_table(args.schema, args.chunks_table)
            ))
            exists = False

    if not exists:
        create_chunks_table(conn, args, embedding_dim)

    with conn.cursor() as cur:
        if args.if_exists == "replace-all":
            # 테이블 전체를 비움 (DELETE보다 빠름)
            cur.execute(sql.SQL("TRUNCATE TABLE {}").format(
                qualified_table(args.schema, args.chunks_table)
            ))
        elif args.if_exists == "replace-model":
            # 해당 모델의 청크만 삭제
            cur.execute(
                sql.SQL("DELETE FROM {} WHERE embedding_model = %s").format(
                    qualified_table(args.schema, args.chunks_table)
                ),
                (args.embedding_model,),
            )


def fetch_documents(conn: Any, args: argparse.Namespace) -> list[tuple[int, str]]:
    """
    임베딩할 문서 목록을 PostgreSQL에서 조회합니다.

    --if-exists skip 모드:
        이미 해당 모델로 임베딩된 문서는 건너뜁니다.
        (NOT EXISTS 서브쿼리로 청크 테이블과 조인)

    --limit > 0이면 앞 N개 문서만 처리합니다.
    """
    where_parts: list[sql.Composed] = [
        sql.SQL("{} IS NOT NULL").format(sql.Identifier(args.text_column)),
        sql.SQL("btrim({}) <> ''").format(sql.Identifier(args.text_column)),
    ]
    params: list[Any] = []

    if args.if_exists == "skip":
        # 이미 임베딩된 문서는 제외 (NOT EXISTS)
        where_parts.append(
            sql.SQL(
                """
                NOT EXISTS (
                    SELECT 1
                    FROM {} c
                    WHERE c.document_id = d.{}
                      AND c.embedding_model = %s
                )
                """
            ).format(
                qualified_table(args.schema, args.chunks_table),
                sql.Identifier(args.id_column),
            )
        )
        params.append(args.embedding_model)

    limit_sql = sql.SQL("")
    if args.limit > 0:
        limit_sql = sql.SQL(" LIMIT %s")
        params.append(args.limit)

    query = sql.SQL("SELECT {}, {} FROM {} d WHERE {} ORDER BY {}{}").format(
        sql.Identifier(args.id_column),
        sql.Identifier(args.text_column),
        qualified_table(args.schema, args.documents_table),
        sql.SQL(" AND ").join(where_parts),
        sql.Identifier(args.id_column),
        limit_sql,
    )

    with conn.cursor() as cur:
        cur.execute(query, params)
        return [(int(row[0]), str(row[1])) for row in cur.fetchall()]


def embedding_value(embedding: list[float], storage: str) -> Any:
    """
    임베딩 벡터를 저장 방식에 맞는 형태로 변환합니다.

    pgvector: "[0.1,0.2,...]" 형태의 문자열 (pgvector가 파싱)
    array: float 목록 (psycopg가 real[]로 변환)
    """
    if storage == "pgvector":
        return "[" + ",".join(f"{value:.8g}" for value in embedding) + "]"
    return embedding


def insert_chunk_records(
    conn: Any,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
) -> None:
    """
    청크 레코드를 PostgreSQL에 일괄 삽입합니다.

    ON CONFLICT ... DO UPDATE (upsert):
        동일 (document_id, chunk_index, embedding_model) 조합이 이미 있으면
        기존 데이터를 새 데이터로 갱신합니다.
        이미 임베딩된 문서를 다시 처리해도 안전합니다.
    """
    if not records:
        return

    columns   = [
        "document_id", "chunk_index", "chunk_text", "token_count",
        "char_count", "embedding_model", "embedding_dim",
        "chunk_max_tokens", "chunk_overlap_tokens", "embedding",
    ]
    statement = sql.SQL(
        """
        INSERT INTO {} ({})
        VALUES ({})
        ON CONFLICT (document_id, chunk_index, embedding_model)
        DO UPDATE SET
            chunk_text           = EXCLUDED.chunk_text,
            token_count          = EXCLUDED.token_count,
            char_count           = EXCLUDED.char_count,
            embedding_dim        = EXCLUDED.embedding_dim,
            chunk_max_tokens     = EXCLUDED.chunk_max_tokens,
            chunk_overlap_tokens = EXCLUDED.chunk_overlap_tokens,
            embedding            = EXCLUDED.embedding,
            created_at           = now()
        """
    ).format(
        qualified_table(args.schema, args.chunks_table),
        sql.SQL(", ").join(sql.Identifier(column) for column in columns),
        sql.SQL(", ").join(sql.Placeholder() for _ in columns),
    )
    values = [tuple(record[column] for column in columns) for record in records]
    with conn.cursor() as cur:
        cur.executemany(statement, values)


def validate_args(args: argparse.Namespace) -> None:
    """인수 유효성 검사."""
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens는 0보다 커야 합니다.")
    if args.overlap_tokens < 0:
        raise ValueError("--overlap-tokens는 0 이상이어야 합니다.")
    if args.overlap_tokens >= args.max_tokens:
        raise ValueError("--overlap-tokens는 --max-tokens보다 작아야 합니다.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size는 0보다 커야 합니다.")
    if args.limit < 0:
        raise ValueError("--limit는 0 이상이어야 합니다.")


def run(args: argparse.Namespace) -> int:
    """임베딩 파이프라인 실행."""
    validate_args(args)

    # 임베더 초기화 및 임베딩 차원 확인 (테이블 생성 전에 필요)
    embedder = TransformerEmbedder(
        args.embedding_model,
        device=args.device,
        normalize=not args.no_normalize,
    )
    # 더미 텍스트로 임베딩 차원 확인
    sample_embedding = embedder.embed(["dimension probe"], batch_size=1)[0]
    embedding_dim    = len(sample_embedding)
    print(f"Embedding model: {args.embedding_model}")
    print(f"Device: {embedder.device}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Storage: {args.embedding_storage}")

    with connect(args) as conn:
        ensure_source_table(conn, args)
        prepare_chunks_table(conn, args, embedding_dim)
        documents = fetch_documents(conn, args)
        print(f"임베딩할 문서 수: {len(documents)}")

        total_chunks = 0
        for document_id, text in tqdm(documents, desc="Embedding documents"):
            # 문서를 청크로 분할
            chunks = chunk_text(
                text, embedder,
                max_tokens=args.max_tokens,
                overlap_tokens=args.overlap_tokens,
                min_chars=args.min_chars,
            )
            if not chunks:
                continue

            # 배치 단위로 임베딩
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings  = embedder.embed(chunk_texts, batch_size=args.batch_size)

            records = []
            for chunk, embedding in zip(chunks, embeddings):
                records.append({
                    "document_id":         document_id,
                    "chunk_index":         chunk.index,
                    "chunk_text":          chunk.text,
                    "token_count":         chunk.token_count,
                    "char_count":          len(chunk.text),
                    "embedding_model":     args.embedding_model,
                    "embedding_dim":       embedding_dim,
                    "chunk_max_tokens":    args.max_tokens,
                    "chunk_overlap_tokens": args.overlap_tokens,
                    "embedding":           embedding_value(embedding, args.embedding_storage),
                })
            insert_chunk_records(conn, args, records)
            total_chunks += len(records)

        conn.commit()

    print(f"완료. {total_chunks}개 청크 삽입/업데이트.")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
