"""
라벨링된 생물다양성 문서 CSV를 PostgreSQL 데이터베이스에 로드합니다.

기본 입력 파일:
    data/biodiv_document_text_dataset_labeled_v2.csv

실행 예:
    python src/load_biodiv_csv_to_postgres.py --user postgres
    python src/load_biodiv_csv_to_postgres.py --database-url postgresql://postgres:pass@localhost:5432/biofin
    python src/load_biodiv_csv_to_postgres.py --if-exists replace   # 테이블 삭제 후 재생성
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
except ImportError as exc:
    raise SystemExit(
        "psycopg 패키지가 필요합니다. pip install -r requirements.txt 로 설치하세요."
    ) from exc

from config import BIODIV_TEXT_LABELED_V2_CSV


# ── 데이터베이스 테이블 기본 이름 ──────────────────────────────────────────────
DEFAULT_TABLE = "biodiv_documents"

# ── CSV 컬럼 → DB 컬럼 이름 매핑 ──────────────────────────────────────────────
# 한국어 CSV 컬럼명을 영문 DB 컬럼명으로 변환합니다.
CSV_TO_DB_COLUMNS = {
    "No.":                  "source_no",
    "matched_filename":     "matched_filename",
    "biodiv_label":         "biodiv_label",
    "소관명":               "agency_name",
    "분야명":               "field_name",
    "부문명":               "sector_name",
    "프로그램명":           "program_name",
    "단위사업명":           "unit_project_name",
    "세부사업명":           "detail_project_name",
    "clean_document_text":  "clean_document_text",
    "text_source":          "text_source",
    "document_status":      "document_status",
    "purpose_anchor_found": "purpose_anchor_found",
    "resolved_paths":       "resolved_paths",
    "relative_paths":       "relative_paths",
    "resolution_statuses":  "resolution_statuses",
    "extract_methods":      "extract_methods",
    "extract_errors":       "extract_errors",
    "clean_text_char_count": "clean_text_char_count",
    "clean_text_word_count": "clean_text_word_count",
    "label_v2":             "label_v2",
}

# ── 타입별 컬럼 집합 ──────────────────────────────────────────────────────────
INT_COLUMNS  = {"biodiv_label", "clean_text_char_count", "clean_text_word_count", "label_v2"}
BOOL_COLUMNS = {"purpose_anchor_found"}

# ── DB 테이블 스키마 정의 (컬럼명, PostgreSQL 타입) ────────────────────────────
TABLE_COLUMNS = [
    ("source_no",           "text"),
    ("matched_filename",    "text"),
    ("biodiv_label",        "integer"),
    ("agency_name",         "text"),
    ("field_name",          "text"),
    ("sector_name",         "text"),
    ("program_name",        "text"),
    ("unit_project_name",   "text"),
    ("detail_project_name", "text"),
    ("clean_document_text", "text"),
    ("text_source",         "text"),
    ("document_status",     "text"),
    ("purpose_anchor_found","boolean"),
    ("resolved_paths",      "text"),
    ("relative_paths",      "text"),
    ("resolution_statuses", "text"),
    ("extract_methods",     "text"),
    ("extract_errors",      "text"),
    ("clean_text_char_count","integer"),
    ("clean_text_word_count","integer"),
    ("label_v2",            "integer"),
    # 원본 CSV 행 전체를 JSONB로 보관 (나중에 분석에 활용 가능)
    ("raw_row",             "jsonb"),
]
INSERT_COLUMNS = [name for name, _ in TABLE_COLUMNS]


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="biodiv_document_text_dataset_labeled_v2.csv를 PostgreSQL에 로드합니다."
    )
    parser.add_argument("--csv",        type=Path, default=BIODIV_TEXT_LABELED_V2_CSV, help="입력 CSV 파일 경로")
    parser.add_argument("--encoding",   default="utf-8-sig",  help="CSV 파일 인코딩")
    parser.add_argument("--schema",     default="public",     help="PostgreSQL 스키마 이름")
    parser.add_argument("--table",      default=DEFAULT_TABLE, help="테이블 이름")
    parser.add_argument(
        "--if-exists",
        choices=["fail", "append", "replace", "truncate"],
        default="fail",
        help="테이블이 이미 존재할 때 처리 방법: fail(중단), append(추가), replace(재생성), truncate(비우고 삽입)",
    )
    parser.add_argument("--chunksize",  type=int, default=1000, help="한 번에 삽입할 행 수")
    # 연결 방법 1: DATABASE_URL 환경변수 또는 --database-url 옵션
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"), help="PostgreSQL 연결 URL")
    # 연결 방법 2: 개별 접속 정보 (PGHOST, PGPORT 등 환경변수 지원)
    parser.add_argument("--host",       default=os.getenv("PGHOST",     "localhost"))
    parser.add_argument("--port",       type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--db-name",    default=os.getenv("PGDATABASE", "biofin"))
    parser.add_argument("--user",       default=os.getenv("PGUSER",     "postgres"))
    parser.add_argument("--password",   default=os.getenv("PGPASSWORD"))
    parser.add_argument(
        "--no-password-prompt",
        action="store_true",
        help="비밀번호 미지정 시 프롬프트를 표시하지 않습니다.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str | None:
    """
    값을 문자열로 변환하고 정제합니다.

    None, NaN, "nan", "none", "null" 문자열은 None으로 반환합니다.
    """
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def parse_int(value: Any) -> int | None:
    """
    값을 정수로 변환합니다. 실패 시 None 반환.

    float → int 경로: CSV에서 정수가 '1.0' 형태로 저장될 수 있음
    """
    text = clean_text(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_bool(value: Any) -> bool | None:
    """
    값을 불리언으로 변환합니다.

    True로 인식: 'true', 't', '1', 'yes', 'y'
    False로 인식: 'false', 'f', '0', 'no', 'n'
    그 외: None 반환
    """
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
    """
    NumPy 스칼라 등 JSON 직렬화 불가 타입을 Python 기본 타입으로 변환합니다.

    hasattr(value, "item"): numpy.int64, numpy.float32 등이 해당
    """
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()   # numpy 스칼라 → Python 기본 타입
    return value


def normalize_record(row: pd.Series) -> dict[str, Any]:
    """
    CSV 한 행을 DB 삽입용 딕셔너리로 변환합니다.

    - CSV 컬럼명을 DB 컬럼명으로 변환
    - 각 컬럼의 데이터 타입에 맞게 변환 (int/bool/text)
    - raw_row 컬럼에 원본 행 전체를 JSONB로 저장
    """
    record: dict[str, Any] = {column: None for column in INSERT_COLUMNS}

    for csv_column, db_column in CSV_TO_DB_COLUMNS.items():
        value = row.get(csv_column)
        if db_column in INT_COLUMNS:
            record[db_column] = parse_int(value)
        elif db_column in BOOL_COLUMNS:
            record[db_column] = parse_bool(value)
        else:
            record[db_column] = clean_text(value)

    # 원본 행 전체를 JSONB로 보관 (추후 분석이나 복구에 활용)
    record["raw_row"] = Jsonb({str(key): json_safe(value) for key, value in row.items()})
    return record


def connect(args: argparse.Namespace) -> psycopg.Connection:
    """
    PostgreSQL에 연결하고 연결 객체를 반환합니다.

    --database-url 이 있으면 URL로, 없으면 개별 접속 정보로 연결합니다.
    비밀번호가 없고 --no-password-prompt 가 False이면 입력 프롬프트를 표시합니다.
    """
    if args.database_url:
        return psycopg.connect(args.database_url)

    password = args.password
    if password is None and not args.no_password_prompt:
        password = getpass.getpass(f"PostgreSQL 비밀번호 ({args.user}@{args.host}): ")

    return psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.db_name,
        user=args.user,
        password=password,
    )


def qualified_table(schema: str, table: str) -> sql.Composed:
    """
    'schema.table' 형태의 SQL 식별자를 안전하게 생성합니다.

    sql.Identifier: SQL 인젝션 방지를 위해 이름을 따옴표로 감쌈
    """
    return sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))


def table_exists(conn: psycopg.Connection, schema: str, table: str) -> bool:
    """테이블이 존재하면 True, 없으면 False를 반환합니다."""
    with conn.cursor() as cur:
        # to_regclass: 테이블이 없으면 NULL 반환
        cur.execute("SELECT to_regclass(%s)", (f"{schema}.{table}",))
        return cur.fetchone()[0] is not None


def create_table(conn: psycopg.Connection, schema: str, table: str) -> None:
    """
    테이블과 인덱스를 생성합니다.

    자동 생성되는 인덱스:
      - label_v2         : 생물다양성 라벨 필터링용
      - text_source      : 텍스트 출처 필터링용
      - matched_filename : 파일명 검색용
    """
    # 컬럼 정의 SQL 목록 생성
    column_defs = [
        sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(column_type))
        for name, column_type in TABLE_COLUMNS
    ]

    with conn.cursor() as cur:
        # 스키마 생성 (없을 경우에만)
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))

        # 테이블 생성 (id: 자동 증가 기본키, imported_at: 삽입 시각 자동 기록)
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

        # 자주 필터링되는 컬럼에 인덱스 생성
        for suffix, column_name in [
            ("label_v2",        "label_v2"),
            ("text_source",     "text_source"),
            ("matched_filename","matched_filename"),
        ]:
            cur.execute(
                sql.SQL("CREATE INDEX {} ON {} ({})").format(
                    sql.Identifier(f"{table}_{suffix}_idx"),
                    qualified_table(schema, table),
                    sql.Identifier(column_name),
                )
            )


def prepare_table(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    """
    --if-exists 옵션에 따라 테이블을 준비합니다.

    fail     : 테이블이 이미 있으면 오류 발생 (기본값)
    append   : 기존 테이블에 행 추가
    replace  : 테이블 삭제 후 재생성
    truncate : 테이블 내용만 삭제 후 삽입
    """
    exists = table_exists(conn, args.schema, args.table)

    if exists and args.if_exists == "fail":
        raise RuntimeError(
            f"테이블 {args.schema}.{args.table} 이 이미 존재합니다. "
            "--if-exists append, truncate, replace 중 하나를 선택하세요."
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
    """
    records 목록을 한 번의 executemany 로 테이블에 삽입합니다.

    sql.Placeholder(): psycopg의 파라미터 바인딩 플레이스홀더 (%s)
    executemany: 여러 행을 한 번에 삽입하여 오버헤드 감소
    """
    if not records:
        return

    statement = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        qualified_table(schema, table),
        sql.SQL(", ").join(sql.Identifier(column) for column in INSERT_COLUMNS),
        sql.SQL(", ").join(sql.Placeholder() for _ in INSERT_COLUMNS),
    )
    # 각 레코드를 INSERT_COLUMNS 순서에 맞는 튜플로 변환
    values = [tuple(record[column] for column in INSERT_COLUMNS) for record in records]
    with conn.cursor() as cur:
        cur.executemany(statement, values)


def load_csv(conn: psycopg.Connection, args: argparse.Namespace) -> int:
    """
    CSV를 청크 단위로 읽어 PostgreSQL에 삽입합니다.

    dtype=str        : 모든 값을 문자열로 읽어 타입 변환 오류 방지
    keep_default_na=False: "NA", "nan" 등을 NaN이 아닌 문자열로 유지
    """
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {args.csv}")
    if args.chunksize <= 0:
        raise ValueError("--chunksize 는 1 이상이어야 합니다.")

    total  = 0
    reader = pd.read_csv(
        args.csv,
        encoding=args.encoding,
        dtype=str,
        keep_default_na=False,
        chunksize=args.chunksize,   # 대용량 파일을 청크 단위로 읽기
    )

    for chunk_index, chunk in enumerate(reader, start=1):
        records = [normalize_record(row) for _, row in chunk.iterrows()]
        insert_records(conn, args.schema, args.table, records)
        total += len(records)
        print(f"삽입 완료: 청크 {chunk_index} ({len(records)}행, 누적 {total}행)")

    return total


def main() -> int:
    """CSV를 PostgreSQL에 로드합니다."""
    args     = parse_args()
    csv_path = args.csv.resolve()
    print(f"CSV: {csv_path}")
    print(f"대상 테이블: {args.schema}.{args.table}")

    with connect(args) as conn:
        prepare_table(conn, args)
        total = load_csv(conn, args)
        conn.commit()   # 모든 삽입 완료 후 트랜잭션 커밋

    print(f"완료: {args.schema}.{args.table} 에 총 {total}행 삽입.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
