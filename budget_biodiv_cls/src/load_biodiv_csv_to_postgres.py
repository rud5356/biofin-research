"""
Load the labeled biodiversity document CSV into PostgreSQL.

Default input:
    data/biodiv_document_text_dataset_labeled_v2.csv

Example:
    python src/load_biodiv_csv_to_postgres.py --user postgres
    python src/load_biodiv_csv_to_postgres.py --database-url postgresql://postgres:pass@localhost:5432/biofin
"""
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import psycopg
    from psycopg import sql
    from psycopg.types.json import Jsonb
except ImportError as exc:  # pragma: no cover - helpful runtime message
    raise SystemExit(
        "psycopg is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc

from config import BIODIV_TEXT_LABELED_V2_CSV


DEFAULT_TABLE = "biodiv_documents"

CSV_TO_DB_COLUMNS = {
    "No.": "source_no",
    "matched_filename": "matched_filename",
    "biodiv_label": "biodiv_label",
    "소관명": "agency_name",
    "분야명": "field_name",
    "부문명": "sector_name",
    "프로그램명": "program_name",
    "단위사업명": "unit_project_name",
    "세부사업명": "detail_project_name",
    "clean_document_text": "clean_document_text",
    "text_source": "text_source",
    "document_status": "document_status",
    "purpose_anchor_found": "purpose_anchor_found",
    "resolved_paths": "resolved_paths",
    "relative_paths": "relative_paths",
    "resolution_statuses": "resolution_statuses",
    "extract_methods": "extract_methods",
    "extract_errors": "extract_errors",
    "clean_text_char_count": "clean_text_char_count",
    "clean_text_word_count": "clean_text_word_count",
    "label_v2": "label_v2",
}

INT_COLUMNS = {
    "biodiv_label",
    "clean_text_char_count",
    "clean_text_word_count",
    "label_v2",
}
BOOL_COLUMNS = {"purpose_anchor_found"}

TABLE_COLUMNS = [
    ("source_no", "text"),
    ("matched_filename", "text"),
    ("biodiv_label", "integer"),
    ("agency_name", "text"),
    ("field_name", "text"),
    ("sector_name", "text"),
    ("program_name", "text"),
    ("unit_project_name", "text"),
    ("detail_project_name", "text"),
    ("clean_document_text", "text"),
    ("text_source", "text"),
    ("document_status", "text"),
    ("purpose_anchor_found", "boolean"),
    ("resolved_paths", "text"),
    ("relative_paths", "text"),
    ("resolution_statuses", "text"),
    ("extract_methods", "text"),
    ("extract_errors", "text"),
    ("clean_text_char_count", "integer"),
    ("clean_text_word_count", "integer"),
    ("label_v2", "integer"),
    ("raw_row", "jsonb"),
]
INSERT_COLUMNS = [name for name, _ in TABLE_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load biodiv_document_text_dataset_labeled_v2.csv into PostgreSQL."
    )
    parser.add_argument("--csv", type=Path, default=BIODIV_TEXT_LABELED_V2_CSV)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument(
        "--if-exists",
        choices=["fail", "append", "replace", "truncate"],
        default="fail",
        help=(
            "What to do when the table already exists. "
            "fail=stop, append=add rows, replace=drop/create, truncate=delete rows."
        ),
    )
    parser.add_argument("--chunksize", type=int, default=1000)
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


def clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def parse_int(value: Any) -> int | None:
    text = clean_text(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_bool(value: Any) -> bool | None:
    text = clean_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in {"true", "t", "1", "yes", "y"}:
        return True
    if lowered in {"false", "f", "0", "no", "n"}:
        return False
    return None


def json_safe(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def normalize_record(row: pd.Series) -> dict[str, Any]:
    record: dict[str, Any] = {column: None for column in INSERT_COLUMNS}

    for csv_column, db_column in CSV_TO_DB_COLUMNS.items():
        value = row.get(csv_column)
        if db_column in INT_COLUMNS:
            record[db_column] = parse_int(value)
        elif db_column in BOOL_COLUMNS:
            record[db_column] = parse_bool(value)
        else:
            record[db_column] = clean_text(value)

    record["raw_row"] = Jsonb({str(key): json_safe(value) for key, value in row.items()})
    return record


def connect(args: argparse.Namespace) -> psycopg.Connection:
    if args.database_url:
        return psycopg.connect(args.database_url)

    password = args.password
    if password is None and not args.no_password_prompt:
        password = getpass.getpass(f"PostgreSQL password for {args.user}@{args.host}: ")

    return psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.db_name,
        user=args.user,
        password=password,
    )


def qualified_table(schema: str, table: str) -> sql.Composed:
    return sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))


def table_exists(conn: psycopg.Connection, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT to_regclass(%s)",
            (f"{schema}.{table}",),
        )
        return cur.fetchone()[0] is not None


def create_table(conn: psycopg.Connection, schema: str, table: str) -> None:
    column_defs = [
        sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(column_type))
        for name, column_type in TABLE_COLUMNS
    ]

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE {} (
                    id bigserial PRIMARY KEY,
                    {},
                    imported_at timestamptz NOT NULL DEFAULT now()
                )
                """
            ).format(
                qualified_table(schema, table),
                sql.SQL(", ").join(column_defs),
            )
        )

        indexes = [
            ("label_v2", "label_v2"),
            ("text_source", "text_source"),
            ("matched_filename", "matched_filename"),
        ]
        for suffix, column_name in indexes:
            cur.execute(
                sql.SQL("CREATE INDEX {} ON {} ({})").format(
                    sql.Identifier(f"{table}_{suffix}_idx"),
                    qualified_table(schema, table),
                    sql.Identifier(column_name),
                )
            )


def prepare_table(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    exists = table_exists(conn, args.schema, args.table)

    if exists and args.if_exists == "fail":
        raise RuntimeError(
            f"Table {args.schema}.{args.table} already exists. "
            "Use --if-exists append, truncate, or replace."
        )

    with conn.cursor() as cur:
        if exists and args.if_exists == "replace":
            cur.execute(sql.SQL("DROP TABLE {}").format(qualified_table(args.schema, args.table)))
            exists = False
        elif exists and args.if_exists == "truncate":
            cur.execute(sql.SQL("TRUNCATE TABLE {}").format(qualified_table(args.schema, args.table)))

    if not exists:
        create_table(conn, args.schema, args.table)


def insert_records(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    records: list[dict[str, Any]],
) -> None:
    if not records:
        return

    statement = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        qualified_table(schema, table),
        sql.SQL(", ").join(sql.Identifier(column) for column in INSERT_COLUMNS),
        sql.SQL(", ").join(sql.Placeholder() for _ in INSERT_COLUMNS),
    )
    values = [tuple(record[column] for column in INSERT_COLUMNS) for record in records]
    with conn.cursor() as cur:
        cur.executemany(statement, values)


def load_csv(conn: psycopg.Connection, args: argparse.Namespace) -> int:
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be greater than 0")

    total = 0
    reader = pd.read_csv(
        args.csv,
        encoding=args.encoding,
        dtype=str,
        keep_default_na=False,
        chunksize=args.chunksize,
    )

    for chunk_index, chunk in enumerate(reader, start=1):
        records = [normalize_record(row) for _, row in chunk.iterrows()]
        insert_records(conn, args.schema, args.table, records)
        total += len(records)
        print(f"Inserted chunk {chunk_index}: {len(records)} rows (total={total})")

    return total


def main() -> int:
    args = parse_args()
    csv_path = args.csv.resolve()
    print(f"CSV: {csv_path}")
    print(f"Target table: {args.schema}.{args.table}")

    with connect(args) as conn:
        prepare_table(conn, args)
        total = load_csv(conn, args)
        conn.commit()

    print(f"Done. Inserted {total} rows into {args.schema}.{args.table}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
