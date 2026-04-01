"""Tests for the shared SQLite database module."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from demandops.db import get_db


class TestGetDb:
    def test_creates_database_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = get_db(str(db_path))
        assert db_path.exists()
        conn.close()

    def test_creates_api_keys_table(self, tmp_path: Path) -> None:
        conn = get_db(str(tmp_path / "test.db"))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='api_keys'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_creates_prediction_log_table(self, tmp_path: Path) -> None:
        conn = get_db(str(tmp_path / "test.db"))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_log'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        db_path = tmp_path / "subdir" / "test.db"
        conn = get_db(str(db_path))
        assert db_path.exists()
        conn.close()

    def test_duplicate_client_name_rejected(self, tmp_path: Path) -> None:
        conn = get_db(str(tmp_path / "test.db"))
        conn.execute(
            "INSERT INTO api_keys (key_hash, client_name, created_at) "
            "VALUES ('hash1', 'team_a', '2024-01-01')"
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO api_keys (key_hash, client_name, created_at) "
                "VALUES ('hash2', 'team_a', '2024-01-01')"
            )
        conn.close()

    def test_non_positive_rate_limit_rejected(self, tmp_path: Path) -> None:
        conn = get_db(str(tmp_path / "test.db"))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO api_keys (key_hash, client_name, created_at, rate_limit) "
                "VALUES ('hash1', 'team_a', '2024-01-01', 0)"
            )
        conn.close()

    def test_non_positive_max_batch_size_rejected(self, tmp_path: Path) -> None:
        conn = get_db(str(tmp_path / "test.db"))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO api_keys (key_hash, client_name, created_at, max_batch_size) "
                "VALUES ('hash1', 'team_a', '2024-01-01', 0)"
            )
        conn.close()

    def test_idempotent_on_existing_db(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        conn1 = get_db(db_path)
        conn1.execute(
            "INSERT INTO api_keys (key_hash, client_name, created_at) "
            "VALUES ('abc', 'test', '2024-01-01')"
        )
        conn1.commit()
        conn1.close()

        conn2 = get_db(db_path)
        row = conn2.execute("SELECT client_name FROM api_keys").fetchone()
        assert row[0] == "test"
        conn2.close()


def _create_legacy_db(db_path: str) -> sqlite3.Connection:
    """Create a database with the pre-migration schema (no UNIQUE, no CHECK)."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE api_keys (
            key_hash        TEXT PRIMARY KEY,
            client_name     TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            rate_limit      INTEGER NOT NULL DEFAULT 100,
            max_batch_size  INTEGER NOT NULL DEFAULT 10000,
            is_active       BOOLEAN NOT NULL DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE prediction_log (
            prediction_id   TEXT PRIMARY KEY,
            zone_id         INTEGER NOT NULL,
            hour_ts         TEXT NOT NULL,
            predicted_value REAL NOT NULL,
            actual_value    REAL,
            created_at      TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


class TestSchemaMigration:
    def test_legacy_db_gets_unique_constraint(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "legacy.db")
        legacy = _create_legacy_db(db_path)
        legacy.execute(
            "INSERT INTO api_keys (key_hash, client_name, created_at) "
            "VALUES ('h1', 'team_a', '2024-01-01')"
        )
        legacy.commit()
        legacy.close()

        # Reopen through get_db — should migrate
        conn = get_db(db_path)
        # Existing row preserved
        row = conn.execute(
            "SELECT client_name FROM api_keys WHERE key_hash = 'h1'"
        ).fetchone()
        assert row[0] == "team_a"
        # UNIQUE constraint now enforced
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO api_keys (key_hash, client_name, created_at) "
                "VALUES ('h2', 'team_a', '2024-01-02')"
            )
        conn.close()

    def test_legacy_db_gets_check_constraints(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "legacy.db")
        legacy = _create_legacy_db(db_path)
        legacy.close()

        conn = get_db(db_path)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO api_keys (key_hash, client_name, created_at, rate_limit) "
                "VALUES ('h1', 'team_a', '2024-01-01', 0)"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO api_keys (key_hash, client_name, created_at, max_batch_size) "
                "VALUES ('h1', 'team_a', '2024-01-01', -1)"
            )
        conn.close()

    def test_migration_deduplicates_client_names(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "legacy.db")
        legacy = _create_legacy_db(db_path)
        # Insert two keys for same client (possible in legacy schema)
        legacy.execute(
            "INSERT INTO api_keys (key_hash, client_name, created_at) "
            "VALUES ('old_hash', 'team_a', '2024-01-01')"
        )
        legacy.execute(
            "INSERT INTO api_keys (key_hash, client_name, created_at) "
            "VALUES ('new_hash', 'team_a', '2024-01-02')"
        )
        legacy.commit()
        legacy.close()

        conn = get_db(db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM api_keys WHERE client_name = 'team_a'"
        ).fetchone()[0]
        assert count == 1
        # Most recent key (highest rowid) should survive
        row = conn.execute(
            "SELECT key_hash FROM api_keys WHERE client_name = 'team_a'"
        ).fetchone()
        assert row[0] == "new_hash"
        conn.close()

    def test_migration_removes_invalid_limits(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "legacy.db")
        legacy = _create_legacy_db(db_path)
        legacy.execute(
            "INSERT INTO api_keys (key_hash, client_name, created_at, rate_limit) "
            "VALUES ('h1', 'zero_rate', '2024-01-01', 0)"
        )
        legacy.execute(
            "INSERT INTO api_keys (key_hash, client_name, created_at, max_batch_size) "
            "VALUES ('h2', 'neg_batch', '2024-01-01', -5)"
        )
        legacy.execute(
            "INSERT INTO api_keys (key_hash, client_name, created_at, rate_limit) "
            "VALUES ('h3', 'valid', '2024-01-01', 100)"
        )
        legacy.commit()
        legacy.close()

        conn = get_db(db_path)
        rows = conn.execute("SELECT client_name FROM api_keys").fetchall()
        names = [r[0] for r in rows]
        assert "valid" in names
        assert "zero_rate" not in names
        assert "neg_batch" not in names
        conn.close()

    def test_migration_idempotent(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "legacy.db")
        legacy = _create_legacy_db(db_path)
        legacy.execute(
            "INSERT INTO api_keys (key_hash, client_name, created_at) "
            "VALUES ('h1', 'team_a', '2024-01-01')"
        )
        legacy.commit()
        legacy.close()

        # Migrate once
        conn1 = get_db(db_path)
        conn1.close()
        # Migrate again — should be a no-op
        conn2 = get_db(db_path)
        row = conn2.execute(
            "SELECT client_name FROM api_keys WHERE key_hash = 'h1'"
        ).fetchone()
        assert row[0] == "team_a"
        conn2.close()
