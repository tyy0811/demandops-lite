"""Tests for the API key management CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from demandops.db import get_db
from demandops.manage_keys import create_key, list_keys, revoke_key
from demandops.security.auth import hash_key


@pytest.fixture
def cli_db(tmp_path: Path):
    conn = get_db(str(tmp_path / "cli_test.db"))
    yield conn
    conn.close()


class TestCreateKey:
    def test_returns_raw_key(self, cli_db) -> None:
        raw_key = create_key(cli_db, client_name="team_a", rate_limit=100)
        assert len(raw_key) > 20  # token_urlsafe(32) is 43 chars

    def test_key_stored_as_hash(self, cli_db) -> None:
        raw_key = create_key(cli_db, client_name="team_a", rate_limit=100)
        expected_hash = hash_key(raw_key)
        row = cli_db.execute(
            "SELECT key_hash FROM api_keys WHERE client_name = ?", ("team_a",)
        ).fetchone()
        assert row[0] == expected_hash

    def test_key_is_active_by_default(self, cli_db) -> None:
        create_key(cli_db, client_name="team_a", rate_limit=100)
        row = cli_db.execute(
            "SELECT is_active FROM api_keys WHERE client_name = ?", ("team_a",)
        ).fetchone()
        assert row[0] == 1

    def test_custom_rate_limit(self, cli_db) -> None:
        create_key(cli_db, client_name="team_a", rate_limit=50)
        row = cli_db.execute(
            "SELECT rate_limit FROM api_keys WHERE client_name = ?", ("team_a",)
        ).fetchone()
        assert row[0] == 50

    def test_rotation_replaces_old_key(self, cli_db) -> None:
        old_key = create_key(cli_db, client_name="team_a", rate_limit=100)
        new_key = create_key(cli_db, client_name="team_a", rate_limit=100)
        assert old_key != new_key
        # Only one row for team_a
        count = cli_db.execute(
            "SELECT COUNT(*) FROM api_keys WHERE client_name = ?", ("team_a",)
        ).fetchone()[0]
        assert count == 1
        # New key hash is stored
        row = cli_db.execute(
            "SELECT key_hash FROM api_keys WHERE client_name = ?", ("team_a",)
        ).fetchone()
        assert row[0] == hash_key(new_key)

    def test_zero_rate_limit_rejected(self, cli_db) -> None:
        with pytest.raises(ValueError, match="rate_limit must be positive"):
            create_key(cli_db, client_name="team_a", rate_limit=0)

    def test_negative_rate_limit_rejected(self, cli_db) -> None:
        with pytest.raises(ValueError, match="rate_limit must be positive"):
            create_key(cli_db, client_name="team_a", rate_limit=-5)

    def test_zero_max_batch_size_rejected(self, cli_db) -> None:
        with pytest.raises(ValueError, match="max_batch_size must be positive"):
            create_key(cli_db, client_name="team_a", rate_limit=100, max_batch_size=0)


class TestListKeys:
    def test_lists_created_keys(self, cli_db) -> None:
        create_key(cli_db, client_name="team_a", rate_limit=100)
        create_key(cli_db, client_name="team_b", rate_limit=50)
        keys = list_keys(cli_db)
        names = [k["client_name"] for k in keys]
        assert "team_a" in names
        assert "team_b" in names

    def test_never_includes_hash(self, cli_db) -> None:
        create_key(cli_db, client_name="team_a", rate_limit=100)
        keys = list_keys(cli_db)
        for k in keys:
            assert "key_hash" not in k

    def test_empty_when_no_keys(self, cli_db) -> None:
        assert list_keys(cli_db) == []


class TestRevokeKey:
    def test_revokes_by_client_name(self, cli_db) -> None:
        create_key(cli_db, client_name="team_a", rate_limit=100)
        revoke_key(cli_db, client_name="team_a")
        row = cli_db.execute(
            "SELECT is_active FROM api_keys WHERE client_name = ?", ("team_a",)
        ).fetchone()
        assert row[0] == 0

    def test_revoke_nonexistent_raises(self, cli_db) -> None:
        with pytest.raises(ValueError, match="not found"):
            revoke_key(cli_db, client_name="nonexistent")
