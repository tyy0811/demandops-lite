"""Shared SQLite database for auth and quality tracking."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def get_db(db_path: str = "data/demandops.db") -> sqlite3.Connection:
    """Open (or create) the shared SQLite database and ensure tables exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key_hash        TEXT PRIMARY KEY,
            client_name     TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            rate_limit      INTEGER NOT NULL DEFAULT 100,
            max_batch_size  INTEGER NOT NULL DEFAULT 10000,
            is_active       BOOLEAN NOT NULL DEFAULT 1
        )
    """)
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
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pred_zone_ts "
        "ON prediction_log(zone_id, hour_ts)"
    )
    conn.commit()
