"""Tests for the shared SQLite database module."""

from __future__ import annotations

from pathlib import Path

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
