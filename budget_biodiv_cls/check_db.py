import os
import psycopg

host = os.getenv("PGHOST", "localhost")
port = os.getenv("PGPORT", "5432")
dbname = os.getenv("PGDATABASE", "biofin")
user = os.getenv("PGUSER", "postgres")
password = os.getenv("PGPASSWORD", "")

try:
    conn = psycopg.connect(
        host=host, port=port, dbname=dbname, user=user, password=password,
        autocommit=True,
    )
    cur = conn.execute("SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename")
    tables = [r[0] for r in cur.fetchall()]
    print("Tables:", tables)
    if "biodiv_documents" in tables:
        cur2 = conn.execute("SELECT COUNT(*) FROM biodiv_documents")
        print("biodiv_documents rows:", cur2.fetchone()[0])
    if "biodiv_document_chunks" in tables:
        cur3 = conn.execute("SELECT COUNT(*) FROM biodiv_document_chunks")
        print("biodiv_document_chunks rows:", cur3.fetchone()[0])
    conn.close()
except Exception as e:
    print("Error:", e)
