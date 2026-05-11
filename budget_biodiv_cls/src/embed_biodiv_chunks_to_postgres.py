"""
Chunk document text from PostgreSQL and store transformer embeddings.

Default source table:
    public.biodiv_documents

Default target table:
    public.biodiv_document_chunks

Examples:
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
except ImportError as exc:  # pragma: no cover - helpful runtime message
    raise SystemExit(
        "psycopg is required. Install dependencies with: pip install -r requirements_train.txt"
    ) from exc

from load_biodiv_csv_to_postgres import connect, qualified_table, table_exists


DEFAULT_DOCUMENTS_TABLE = "biodiv_documents"
DEFAULT_CHUNKS_TABLE = "biodiv_document_chunks"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class Chunk:
    index: int
    text: str
    token_count: int


class TransformerEmbedder:
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

        self.np = np
        self.torch = torch
        self.F = F
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = self.resolve_device(device)
        self.normalize = normalize
        self.model.to(self.device)
        self.model.eval()

    def resolve_device(self, device: str) -> Any:
        if device == "auto":
            return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        return self.torch.device(device)

    @property
    def model_max_length(self) -> int:
        value = getattr(self.tokenizer, "model_max_length", 512)
        if not isinstance(value, int) or value <= 0 or value > 10000:
            return 512
        return value

    def token_ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def count_tokens(self, text: str) -> int:
        return len(self.token_ids(text))

    def decode_tokens(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def embed(self, texts: list[str], batch_size: int) -> list[list[float]]:
        embeddings: list[list[float]] = []
        max_length = min(self.model_max_length, 512)

        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                if self.normalize:
                    pooled = self.F.normalize(pooled, p=2, dim=1)
                embeddings.extend(pooled.cpu().numpy().astype(self.np.float32).tolist())

        return embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk biodiv_documents.clean_document_text and store embeddings in PostgreSQL."
    )
    parser.add_argument("--schema", default="public")
    parser.add_argument("--documents-table", default=DEFAULT_DOCUMENTS_TABLE)
    parser.add_argument("--chunks-table", default=DEFAULT_CHUNKS_TABLE)
    parser.add_argument("--text-column", default="clean_document_text")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument(
        "--embedding-storage",
        choices=["array", "pgvector"],
        default="array",
        help="array works with plain PostgreSQL. pgvector requires the vector extension.",
    )
    parser.add_argument(
        "--if-exists",
        choices=["skip", "append", "replace-model", "replace-all"],
        default="skip",
        help=(
            "How to handle existing chunk rows. skip=skip docs already embedded for this model, "
            "append=upsert all docs, replace-model=delete this model first, replace-all=truncate chunks."
        ),
    )
    parser.add_argument(
        "--recreate-chunks-table",
        action="store_true",
        help="Drop and recreate the chunks table before embedding.",
    )
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--host", default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--db-name", default=os.getenv("PGDATABASE", "biofin"))
    parser.add_argument("--user", default=os.getenv("PGUSER", "postgres"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD"))
    parser.add_argument(
        "--no-password-prompt",
        action="store_true",
        help="Do not prompt for a password when --password/PGPASSWORD is missing.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def split_long_text_by_tokens(
    text: str,
    embedder: TransformerEmbedder,
    max_tokens: int,
    overlap_tokens: int,
) -> Iterable[tuple[str, int]]:
    token_ids = embedder.token_ids(text)
    if len(token_ids) <= max_tokens:
        yield text, len(token_ids)
        return

    step = max_tokens - overlap_tokens
    for start in range(0, len(token_ids), step):
        window = token_ids[start : start + max_tokens]
        chunk_text = embedder.decode_tokens(window)
        if chunk_text:
            yield chunk_text, len(window)
        if start + max_tokens >= len(token_ids):
            break


def paragraph_segments(text: str) -> list[str]:
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
    segments = paragraph_segments(text)
    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_tokens = 0

    def flush_current() -> None:
        nonlocal current_parts, current_tokens
        chunk = "\n".join(current_parts).strip()
        if len(chunk) >= min_chars:
            chunks.append(Chunk(len(chunks), chunk, current_tokens))
        current_parts = []
        current_tokens = 0

    for segment in segments:
        segment_tokens = embedder.count_tokens(segment)
        if segment_tokens > max_tokens:
            flush_current()
            for split_text, split_tokens in split_long_text_by_tokens(
                segment,
                embedder,
                max_tokens,
                overlap_tokens,
            ):
                if len(split_text) >= min_chars:
                    chunks.append(Chunk(len(chunks), split_text, split_tokens))
            continue

        if current_parts and current_tokens + segment_tokens > max_tokens:
            flush_current()

        current_parts.append(segment)
        current_tokens += segment_tokens

    flush_current()
    return chunks


def ensure_source_table(conn: Any, args: argparse.Namespace) -> None:
    if not table_exists(conn, args.schema, args.documents_table):
        raise RuntimeError(
            f"Source table {args.schema}.{args.documents_table} does not exist. "
            "Run load_biodiv_csv_to_postgres.py first."
        )


def create_vector_extension(conn: Any) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception as exc:
        raise RuntimeError(
            "pgvector extension is not available in this PostgreSQL installation. "
            "Run again without --embedding-storage pgvector to store embeddings as real[] arrays."
        ) from exc


def create_chunks_table(conn: Any, args: argparse.Namespace, embedding_dim: int) -> None:
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
    exists = table_exists(conn, args.schema, args.chunks_table)
    with conn.cursor() as cur:
        if exists and args.recreate_chunks_table:
            cur.execute(sql.SQL("DROP TABLE {}").format(qualified_table(args.schema, args.chunks_table)))
            exists = False

    if not exists:
        create_chunks_table(conn, args, embedding_dim)

    with conn.cursor() as cur:
        if args.if_exists == "replace-all":
            cur.execute(sql.SQL("TRUNCATE TABLE {}").format(qualified_table(args.schema, args.chunks_table)))
        elif args.if_exists == "replace-model":
            cur.execute(
                sql.SQL("DELETE FROM {} WHERE embedding_model = %s").format(
                    qualified_table(args.schema, args.chunks_table)
                ),
                (args.embedding_model,),
            )


def fetch_documents(conn: Any, args: argparse.Namespace) -> list[tuple[int, str]]:
    where_parts = [
        sql.SQL("{} IS NOT NULL").format(sql.Identifier(args.text_column)),
        sql.SQL("btrim({}) <> ''").format(sql.Identifier(args.text_column)),
    ]
    params: list[Any] = []

    if args.if_exists == "skip":
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
    if storage == "pgvector":
        return "[" + ",".join(f"{value:.8g}" for value in embedding) + "]"
    return embedding


def insert_chunk_records(conn: Any, args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    if not records:
        return

    columns = [
        "document_id",
        "chunk_index",
        "chunk_text",
        "token_count",
        "char_count",
        "embedding_model",
        "embedding_dim",
        "chunk_max_tokens",
        "chunk_overlap_tokens",
        "embedding",
    ]
    statement = sql.SQL(
        """
        INSERT INTO {} ({})
        VALUES ({})
        ON CONFLICT (document_id, chunk_index, embedding_model)
        DO UPDATE SET
            chunk_text = EXCLUDED.chunk_text,
            token_count = EXCLUDED.token_count,
            char_count = EXCLUDED.char_count,
            embedding_dim = EXCLUDED.embedding_dim,
            chunk_max_tokens = EXCLUDED.chunk_max_tokens,
            chunk_overlap_tokens = EXCLUDED.chunk_overlap_tokens,
            embedding = EXCLUDED.embedding,
            created_at = now()
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
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be greater than 0")
    if args.overlap_tokens < 0:
        raise ValueError("--overlap-tokens must be 0 or greater")
    if args.overlap_tokens >= args.max_tokens:
        raise ValueError("--overlap-tokens must be smaller than --max-tokens")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.limit < 0:
        raise ValueError("--limit must be 0 or greater")


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    embedder = TransformerEmbedder(
        args.embedding_model,
        device=args.device,
        normalize=not args.no_normalize,
    )
    sample_embedding = embedder.embed(["dimension probe"], batch_size=1)[0]
    embedding_dim = len(sample_embedding)
    print(f"Embedding model: {args.embedding_model}")
    print(f"Device: {embedder.device}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Storage: {args.embedding_storage}")

    with connect(args) as conn:
        ensure_source_table(conn, args)
        prepare_chunks_table(conn, args, embedding_dim)
        documents = fetch_documents(conn, args)
        print(f"Documents to embed: {len(documents)}")

        total_chunks = 0
        for document_id, text in tqdm(documents, desc="Embedding documents"):
            chunks = chunk_text(
                text,
                embedder,
                max_tokens=args.max_tokens,
                overlap_tokens=args.overlap_tokens,
                min_chars=args.min_chars,
            )
            if not chunks:
                continue

            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = embedder.embed(chunk_texts, batch_size=args.batch_size)
            records = []
            for chunk, embedding in zip(chunks, embeddings):
                records.append(
                    {
                        "document_id": document_id,
                        "chunk_index": chunk.index,
                        "chunk_text": chunk.text,
                        "token_count": chunk.token_count,
                        "char_count": len(chunk.text),
                        "embedding_model": args.embedding_model,
                        "embedding_dim": embedding_dim,
                        "chunk_max_tokens": args.max_tokens,
                        "chunk_overlap_tokens": args.overlap_tokens,
                        "embedding": embedding_value(embedding, args.embedding_storage),
                    }
                )
            insert_chunk_records(conn, args, records)
            total_chunks += len(records)

        conn.commit()

    print(f"Done. Inserted/updated {total_chunks} chunks.")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
