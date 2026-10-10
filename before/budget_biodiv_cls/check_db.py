"""
PostgreSQL 데이터베이스 연결 상태와 테이블 내용을 빠르게 확인하는 스크립트.

이 스크립트를 실행하면 현재 DB에 어떤 테이블이 있는지,
핵심 테이블(biodiv_documents, biodiv_document_chunks)에 몇 건의 데이터가 있는지 출력합니다.

사용법:
    python check_db.py

환경 변수로 접속 정보를 설정하지 않으면 기본값(로컬 PostgreSQL)을 사용합니다.
"""

import os

import psycopg

# ─── DB 접속 정보 (환경 변수 우선, 없으면 기본값 사용) ──────────────────────
# os.getenv("변수명", "기본값") : 환경 변수가 설정돼 있으면 그 값을, 없으면 기본값을 사용합니다.
db_host = os.getenv("PGHOST", "localhost")     # DB 서버 주소
db_port = os.getenv("PGPORT", "5432")          # PostgreSQL 기본 포트
db_name = os.getenv("PGDATABASE", "biofin")   # 데이터베이스 이름
db_user = os.getenv("PGUSER", "postgres")      # 접속 계정
db_password = os.getenv("PGPASSWORD", "")      # 비밀번호 (.env 파일에서 설정 권장)

try:
    # DB에 연결합니다. autocommit=True: 각 쿼리를 자동으로 커밋 (읽기 전용 확인에 적합)
    conn = psycopg.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        autocommit=True,
    )

    # public 스키마에 있는 모든 테이블 목록을 가져옵니다.
    # pg_tables: PostgreSQL 내장 시스템 테이블로, 모든 테이블 정보를 담고 있습니다.
    cur = conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
    )
    tables = [row[0] for row in cur.fetchall()]
    print("현재 테이블 목록:", tables)

    # 문서 메인 테이블의 행 수를 확인합니다.
    if "biodiv_documents" in tables:
        cur2 = conn.execute("SELECT COUNT(*) FROM biodiv_documents")
        print("biodiv_documents 행 수:", cur2.fetchone()[0])

    # 청크(문서를 잘게 쪼갠 단위) 테이블의 행 수를 확인합니다.
    if "biodiv_document_chunks" in tables:
        cur3 = conn.execute("SELECT COUNT(*) FROM biodiv_document_chunks")
        print("biodiv_document_chunks 행 수:", cur3.fetchone()[0])

    conn.close()

except Exception as e:
    # 연결 실패나 쿼리 오류 시 오류 메시지를 출력합니다.
    print("DB 연결/조회 오류:", e)
