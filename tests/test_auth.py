"""Tests for API key authentication and rate limiting."""

from __future__ import annotations

import hashlib
import secrets
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from demandops.db import get_db
from demandops.security.auth import RateLimiter, hash_key, requires_auth


@pytest.fixture
def auth_db(tmp_path: Path):
    conn = get_db(str(tmp_path / "auth_test.db"))
    yield conn
    conn.close()


@pytest.fixture
def raw_key() -> str:
    return secrets.token_urlsafe(32)


@pytest.fixture
def active_client(auth_db, raw_key) -> dict:
    """Insert an active API key and return client info."""
    key_hash = hash_key(raw_key)
    auth_db.execute(
        "INSERT INTO api_keys (key_hash, client_name, created_at, rate_limit, max_batch_size, is_active) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (key_hash, "test_client", "2024-01-01T00:00:00", 100, 10000, True),
    )
    auth_db.commit()
    return {
        "raw_key": raw_key,
        "client_name": "test_client",
        "rate_limit": 100,
        "max_batch_size": 10000,
    }


@pytest.fixture
def auth_app(auth_db) -> FastAPI:
    """Minimal FastAPI app with auth dependency for testing."""
    app = FastAPI()
    app.state.db = auth_db
    app.state.rate_limiter = RateLimiter()

    @app.get("/protected")
    async def protected(client: dict = pytest.importorskip("fastapi").Depends(requires_auth)):
        return {"client_name": client["client_name"]}

    return app


@pytest.fixture
def auth_client(auth_app) -> TestClient:
    return TestClient(auth_app)


class TestHashKey:
    def test_deterministic(self) -> None:
        assert hash_key("test123") == hash_key("test123")

    def test_returns_sha256_hex(self) -> None:
        result = hash_key("test123")
        assert len(result) == 64
        assert result == hashlib.sha256(b"test123").hexdigest()


class TestRequiresAuth:
    def test_valid_key_returns_200(self, auth_client, active_client) -> None:
        resp = auth_client.get(
            "/protected",
            headers={"Authorization": f"Bearer {active_client['raw_key']}"},
        )
        assert resp.status_code == 200
        assert resp.json()["client_name"] == "test_client"

    def test_invalid_key_returns_401(self, auth_client) -> None:
        resp = auth_client.get(
            "/protected",
            headers={"Authorization": "Bearer invalid_key_here"},
        )
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid or inactive API key"

    def test_revoked_key_returns_401(self, auth_client, auth_db, active_client) -> None:
        auth_db.execute(
            "UPDATE api_keys SET is_active = 0 WHERE client_name = ?",
            ("test_client",),
        )
        auth_db.commit()
        resp = auth_client.get(
            "/protected",
            headers={"Authorization": f"Bearer {active_client['raw_key']}"},
        )
        assert resp.status_code == 401

    def test_missing_header_returns_401(self, auth_client) -> None:
        resp = auth_client.get("/protected")
        assert resp.status_code == 401

    def test_malformed_header_returns_401(self, auth_client) -> None:
        resp = auth_client.get(
            "/protected",
            headers={"Authorization": "Basic abc123"},
        )
        assert resp.status_code == 401

    def test_lowercase_bearer_accepted(self, auth_client, active_client) -> None:
        """RFC 7235: auth scheme is case-insensitive."""
        resp = auth_client.get(
            "/protected",
            headers={"Authorization": f"bearer {active_client['raw_key']}"},
        )
        assert resp.status_code == 200

    def test_same_error_message_for_invalid_and_revoked(
        self, auth_client, auth_db, active_client
    ) -> None:
        """Intentionally vague 401 — same message prevents oracle attacks."""
        resp_invalid = auth_client.get(
            "/protected",
            headers={"Authorization": "Bearer nonexistent"},
        )
        auth_db.execute(
            "UPDATE api_keys SET is_active = 0 WHERE client_name = ?",
            ("test_client",),
        )
        auth_db.commit()
        resp_revoked = auth_client.get(
            "/protected",
            headers={"Authorization": f"Bearer {active_client['raw_key']}"},
        )
        assert resp_invalid.json()["detail"] == resp_revoked.json()["detail"]


class TestRateLimiter:
    def test_allows_under_limit(self) -> None:
        limiter = RateLimiter()
        for _ in range(5):
            assert limiter.check("client_a", rate_limit=10)

    def test_blocks_over_limit(self) -> None:
        limiter = RateLimiter()
        for _ in range(10):
            limiter.check("client_a", rate_limit=10)
        assert not limiter.check("client_a", rate_limit=10)

    def test_independent_per_client(self) -> None:
        limiter = RateLimiter()
        for _ in range(10):
            limiter.check("client_a", rate_limit=10)
        # client_b should still be allowed
        assert limiter.check("client_b", rate_limit=10)

    def test_rate_limit_returns_429(self, auth_app, auth_db, raw_key) -> None:
        key_hash = hash_key(raw_key)
        auth_db.execute(
            "INSERT INTO api_keys (key_hash, client_name, created_at, rate_limit, is_active) "
            "VALUES (?, ?, ?, ?, ?)",
            (key_hash, "rate_test", "2024-01-01", 3, True),
        )
        auth_db.commit()

        client = TestClient(auth_app)
        headers = {"Authorization": f"Bearer {raw_key}"}
        for _ in range(3):
            resp = client.get("/protected", headers=headers)
            assert resp.status_code == 200

        resp = client.get("/protected", headers=headers)
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
