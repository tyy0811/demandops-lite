"""Shared SQLite database for auth and quality tracking."""

from __future__ import annotations

import sqlite3
from pathlib import Path

_CURRENT_SCHEMA = """
    CREATE TABLE IF NOT EXISTS api_keys (
        key_hash        TEXT PRIMARY KEY,
        client_name     TEXT NOT NULL UNIQUE,
        created_at      TEXT NOT NULL,
        rate_limit      INTEGER NOT NULL DEFAULT 100 CHECK(rate_limit > 0),
        max_batch_size  INTEGER NOT NULL DEFAULT 10000 CHECK(max_batch_size > 0),
        is_active       BOOLEAN NOT NULL DEFAULT 1
    )
"""


def get_db(db_path: str = "data/demandops.db") -> sqlite3.Connection:
    """Open (or create) the shared SQLite database and ensure tables exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    _create_tables(conn)
    _migrate_api_keys(conn)
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.execute(_CURRENT_SCHEMA)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_log (
            prediction_id   TEXT PRIMARY KEY,
            zone_id         INTEGER NOT NULL,
            hour_ts         TEXT NOT NULL,
            predicted_value REAL NOT NULL,
            actual_value    REAL,
            created_at      TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_zone_ts ON prediction_log(zone_id, hour_ts)")
    conn.commit()


def _migrate_api_keys(conn: sqlite3.Connection) -> None:
    """Migrate legacy api_keys table to add UNIQUE(client_name) and CHECK constraints.

    SQLite doesn't support ALTER TABLE ADD CONSTRAINT, so we use the
    rename-copy-drop pattern. Skips migration if the table already has
    the UNIQUE constraint (detected via CREATE TABLE SQL in sqlite_master).
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='api_keys'"
    ).fetchone()
    if row is None:
        return  # Table doesn't exist yet (will be created fresh)

    create_sql = row[0]
    if "UNIQUE" in create_sql and "CHECK" in create_sql:
        return  # Already migrated

    # Delete rows with non-positive limits (would violate new CHECK constraints)
    conn.execute("DELETE FROM api_keys WHERE rate_limit <= 0 OR max_batch_size <= 0")

    # For duplicate client_names, keep only the most recently created key
    conn.execute("""
        DELETE FROM api_keys WHERE rowid NOT IN (
            SELECT MAX(rowid) FROM api_keys GROUP BY client_name
        )
    """)

    conn.execute("ALTER TABLE api_keys RENAME TO _api_keys_old")
    conn.execute(_CURRENT_SCHEMA)
    conn.execute("""
        INSERT INTO api_keys (key_hash, client_name, created_at, rate_limit, max_batch_size, is_active)
        SELECT key_hash, client_name, created_at, rate_limit, max_batch_size, is_active
        FROM _api_keys_old
    """)
    conn.execute("DROP TABLE _api_keys_old")
    conn.commit()
