"""
Applies database/schema.sql to a SQLite file. Idempotent — every statement
in schema.sql is CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS /
INSERT OR IGNORE, so calling this on an already-initialised database is
a no-op.

Nothing in the repo previously created the database in a repeatable way;
every downstream script (risk_governor, shadow_ledger, populate_edges, ...)
just connected and queried, assuming the tables already existed from some
earlier ad hoc run. Centralising it here means any entry point can call
ensure_schema() defensively without duplicating the executescript logic.
"""

import os
import sqlite3

_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'schema.sql')


def ensure_schema(db_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)) or '.', exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        with open(_SCHEMA_PATH) as f:
            conn.executescript(f.read())
        conn.commit()
    finally:
        conn.close()


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DB_PATH
    ensure_schema(DB_PATH)
    print(f"Schema applied to {DB_PATH}")
