# ML Monitoring & Data Security — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add data drift detection, prediction quality monitoring, and API key authentication to demandops-lite.

**Architecture:** Three new modules (`demandops/monitoring/drift_detector.py`, `demandops/monitoring/quality_tracker.py`, `demandops/security/auth.py`) plus a shared SQLite database (`data/demandops.db`). Auth gates prediction and actuals endpoints via a `requires_auth` FastAPI dependency. Drift detection accumulates feature vectors in a bounded deque and computes PSI/KS/correlation on demand. Quality tracking logs predictions to SQLite and matches ground truth when actuals arrive.

**Tech Stack:** scipy (KS test), numpy (PSI, correlation), sqlite3 (stdlib), hashlib/secrets (stdlib), threading (stdlib). No new packages beyond scipy.

**Design doc:** `docs/plans/2026-04-01-monitoring-security-design.md`

---

### Task 1: Foundation — scipy dependency, database module, security package

**Files:**
- Modify: `pyproject.toml:6-25`
- Create: `demandops/db.py`
- Create: `demandops/security/__init__.py`
- Test: `tests/test_db.py`

**Step 1: Add scipy to dependencies**

In `pyproject.toml`, add `scipy>=1.11.0` to the dependencies list:

```python
# After "scikit-learn>=1.3.0",
"scipy>=1.11.0",
```

**Step 2: Create the database module**

```python
# demandops/db.py
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
```

**Step 3: Create security package**

```python
# demandops/security/__init__.py
```

(Empty `__init__.py` — just creates the package.)

**Step 4: Write the test**

```python
# tests/test_db.py
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
```

**Step 5: Run tests**

Run: `pytest tests/test_db.py -v`
Expected: all 5 tests PASS

**Step 6: Commit**

```bash
git add pyproject.toml demandops/db.py demandops/security/__init__.py tests/test_db.py
git commit -m "feat: add shared SQLite database module and security package

Foundation for auth and quality tracking. Single DB (data/demandops.db)
with api_keys and prediction_log tables. WAL journal mode for concurrent
read/write. Adds scipy dependency for KS test."
```

---

### Task 2: Auth module — key hashing, lookup, requires_auth dependency, rate limiter

**Files:**
- Create: `demandops/security/auth.py`
- Test: `tests/test_auth.py`

**Step 1: Write the failing tests**

```python
# tests/test_auth.py
"""Tests for API key authentication and rate limiting."""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
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
    return {"raw_key": raw_key, "client_name": "test_client", "rate_limit": 100, "max_batch_size": 10000}


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
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_auth.py -v`
Expected: FAIL — `demandops.security.auth` does not exist

**Step 3: Write the implementation**

```python
# demandops/security/auth.py
"""API key authentication and per-client rate limiting."""

from __future__ import annotations

import hashlib
import threading
import time

from fastapi import HTTPException, Request


def hash_key(raw_key: str) -> str:
    """SHA-256 hash of an API key. Never store or log the raw key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


class RateLimiter:
    """In-memory sliding window rate limiter. Resets on restart."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._windows: dict[str, list[float]] = {}

    def check(self, client_name: str, rate_limit: int) -> bool:
        """Return True if the request is allowed, False if rate limit exceeded."""
        now = time.time()
        with self._lock:
            timestamps = self._windows.get(client_name, [])
            timestamps = [t for t in timestamps if now - t < 60]
            if len(timestamps) >= rate_limit:
                self._windows[client_name] = timestamps
                return False
            timestamps.append(now)
            self._windows[client_name] = timestamps
            return True


async def requires_auth(request: Request) -> dict:
    """FastAPI dependency: validate Bearer token, check rate limit.

    Returns dict with client_name, rate_limit, max_batch_size.
    Raises 401 for invalid/inactive keys, 429 for rate limit exceeded.
    """
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    raw_key = auth_header[7:]
    key_hash = hash_key(raw_key)

    db = request.app.state.db
    row = db.execute(
        "SELECT client_name, rate_limit, max_batch_size, is_active "
        "FROM api_keys WHERE key_hash = ?",
        (key_hash,),
    ).fetchone()

    if row is None or not row[3]:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    client_name, rate_limit, max_batch_size = row[0], row[1], row[2]

    limiter: RateLimiter = request.app.state.rate_limiter
    if not limiter.check(client_name, rate_limit):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"},
        )

    return {
        "client_name": client_name,
        "rate_limit": rate_limit,
        "max_batch_size": max_batch_size,
    }
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_auth.py -v`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add demandops/security/auth.py tests/test_auth.py
git commit -m "feat: add API key auth with SHA-256 hashing and rate limiting

requires_auth FastAPI dependency validates Bearer tokens against SQLite,
returns 401 (intentionally vague) for invalid/revoked keys. In-memory
sliding window rate limiter per client, returns 429 with Retry-After."
```

---

### Task 3: Key management CLI

**Files:**
- Create: `demandops/manage_keys.py`
- Test: `tests/test_manage_keys.py`

**Step 1: Write the failing tests**

```python
# tests/test_manage_keys.py
"""Tests for the API key management CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_manage_keys.py -v`
Expected: FAIL — `demandops.manage_keys` does not exist

**Step 3: Write the implementation**

```python
# demandops/manage_keys.py
"""API key management CLI.

Usage:
    python -m demandops.manage_keys create --client "analytics_team" --rate-limit 100
    python -m demandops.manage_keys list
    python -m demandops.manage_keys revoke --client "analytics_team"
"""

from __future__ import annotations

import argparse
import secrets
import sqlite3
from datetime import datetime, timezone

from demandops.db import get_db
from demandops.security.auth import hash_key


def create_key(
    db: sqlite3.Connection,
    client_name: str,
    rate_limit: int = 100,
    max_batch_size: int = 10000,
) -> str:
    """Create a new API key. Returns the raw key (shown only once)."""
    raw_key = secrets.token_urlsafe(32)
    key_hash = hash_key(raw_key)
    db.execute(
        "INSERT INTO api_keys (key_hash, client_name, created_at, rate_limit, max_batch_size, is_active) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (key_hash, client_name, datetime.now(timezone.utc).isoformat(), rate_limit, max_batch_size, True),
    )
    db.commit()
    return raw_key


def list_keys(db: sqlite3.Connection) -> list[dict]:
    """List all API keys (never includes hashes)."""
    rows = db.execute(
        "SELECT client_name, created_at, rate_limit, max_batch_size, is_active FROM api_keys"
    ).fetchall()
    return [
        {
            "client_name": r[0],
            "created_at": r[1],
            "rate_limit": r[2],
            "max_batch_size": r[3],
            "is_active": bool(r[4]),
        }
        for r in rows
    ]


def revoke_key(db: sqlite3.Connection, client_name: str) -> None:
    """Revoke an API key by client name."""
    row = db.execute(
        "SELECT key_hash FROM api_keys WHERE client_name = ?", (client_name,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Client '{client_name}' not found")
    db.execute(
        "UPDATE api_keys SET is_active = 0 WHERE client_name = ?", (client_name,)
    )
    db.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage API keys")
    sub = parser.add_subparsers(dest="command", required=True)

    create_parser = sub.add_parser("create")
    create_parser.add_argument("--client", required=True)
    create_parser.add_argument("--rate-limit", type=int, default=100)
    create_parser.add_argument("--max-batch-size", type=int, default=10000)
    create_parser.add_argument("--db", default="data/demandops.db")

    list_parser = sub.add_parser("list")
    list_parser.add_argument("--db", default="data/demandops.db")

    revoke_parser = sub.add_parser("revoke")
    revoke_parser.add_argument("--client", required=True)
    revoke_parser.add_argument("--db", default="data/demandops.db")

    args = parser.parse_args()
    db = get_db(args.db)

    if args.command == "create":
        raw_key = create_key(db, args.client, args.rate_limit, args.max_batch_size)
        print(f"API key created for '{args.client}':")
        print(f"  Key: {raw_key}")
        print("  (This key will not be shown again)")
    elif args.command == "list":
        keys = list_keys(db)
        if not keys:
            print("No API keys found.")
        for k in keys:
            status = "active" if k["is_active"] else "REVOKED"
            print(f"  {k['client_name']}: rate_limit={k['rate_limit']}, "
                  f"max_batch={k['max_batch_size']}, {status}, created={k['created_at']}")
    elif args.command == "revoke":
        revoke_key(db, args.client)
        print(f"Key revoked for '{args.client}'")

    db.close()


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_manage_keys.py -v`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add demandops/manage_keys.py tests/test_manage_keys.py
git commit -m "feat: add API key management CLI (create/list/revoke)

python -m demandops.manage_keys create --client 'name' --rate-limit 100
Raw key shown once at creation. Keys stored as SHA-256 hashes. List
never exposes hashes. Revoke sets is_active=0."
```

---

### Task 4: Wire auth into prediction routes + update existing tests

**Files:**
- Modify: `demandops/serving/schemas.py:24-31` — add `prediction_id` to `PredictResponse`
- Modify: `demandops/serving/routes.py:1-10,58-131,133-224` — add auth + batch size check
- Modify: `demandops/serving/app.py:1-95` — init DB, rate limiter, quality tracker, drift detector on startup
- Modify: `tests/conftest.py:135-255` — add auth fixtures, update `test_app`
- Modify: `tests/test_serving.py` — add auth headers to all requests

**Step 1: Update PredictResponse schema**

Add `prediction_id` field to `PredictResponse` in `demandops/serving/schemas.py`:

```python
# In PredictResponse, add after zone_id:
class PredictResponse(BaseModel):
    prediction_id: str
    zone_id: int
    zone_name: str
    hour_ts: datetime
    predicted_count: float = Field(ge=0.0)
    model_name: str
    model_version: str
    metadata: PredictionMetadata
```

Add monitoring schemas at the end of `demandops/serving/schemas.py`:

```python
class ActualSubmission(BaseModel):
    prediction_id: str | None = None
    zone_id: int | None = None
    hour_ts: datetime | None = None
    actual_value: float


class ActualsRequest(BaseModel):
    actuals: list[ActualSubmission] = Field(min_length=1)


class ActualsResponse(BaseModel):
    matched_count: int
    unmatched_count: int
    warnings: list[str] = Field(default_factory=list)
```

**Step 2: Update routes.py — add auth to /predict and /predict/batch**

In `demandops/serving/routes.py`:

- Add imports: `from fastapi import Depends` (already in fastapi), add `from demandops.security.auth import requires_auth`
- Add `client: dict = Depends(requires_auth)` parameter to `predict()` and `predict_batch()`
- Generate `prediction_id` via `uuid.uuid4()` (already imported) and include in response
- Add batch size check against `client["max_batch_size"]` in `predict_batch()`

Key changes to `predict()` (line 58):
```python
@router.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request, client: dict = Depends(requires_auth)):
    request_id = str(uuid.uuid4())
    prediction_id = str(uuid.uuid4())
    # ... existing code ...
    return PredictResponse(
        prediction_id=prediction_id,
        # ... rest unchanged ...
    )
```

Key changes to `predict_batch()` (line 133):
```python
@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(body: BatchPredictRequest, request: Request, client: dict = Depends(requires_auth)):
    start = time.perf_counter()

    # Per-key batch size enforcement
    if len(body.requests) > client["max_batch_size"]:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {len(body.requests)} exceeds limit {client['max_batch_size']}",
        )
    # ... rest of existing code, add prediction_id to each PredictResponse ...
```

**Step 3: Update app.py — init DB and rate limiter at startup**

In `demandops/serving/app.py`, add to startup:

```python
from demandops.db import get_db
from demandops.security.auth import RateLimiter

# Inside startup(), before configure():
db = get_db(config.get("db", {}).get("path", "data/demandops.db"))
app.state.db = db
app.state.rate_limiter = RateLimiter()
```

**Step 4: Update test fixtures in conftest.py**

Add to `tests/conftest.py`:

```python
from demandops.db import get_db
from demandops.security.auth import RateLimiter, hash_key


@pytest.fixture
def test_db(tmp_path: Path):
    conn = get_db(str(tmp_path / "test.db"))
    yield conn
    conn.close()


@pytest.fixture
def api_key(test_db) -> str:
    """Create a test API key, return the raw key."""
    raw_key = "test-api-key-for-unit-tests-1234567890"
    key_hash = hash_key(raw_key)
    test_db.execute(
        "INSERT INTO api_keys (key_hash, client_name, created_at, rate_limit, max_batch_size, is_active) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (key_hash, "test_client", "2024-01-01T00:00:00", 1000, 10000, True),
    )
    test_db.commit()
    return raw_key
```

Update the `test_app` fixture to include DB and rate limiter:

```python
@pytest.fixture
def test_app(mock_feature_service, mock_model, test_db, api_key):
    from fastapi import FastAPI
    from demandops.serving.routes import configure, router

    app = FastAPI()
    app.include_router(router)
    app.state.db = test_db
    app.state.rate_limiter = RateLimiter()
    configure(
        app,
        mock_feature_service,
        mock_model,
        "lightgbm",
        time.time(),
        model_artifact_loaded=True,
        model_objective="regression",
        model_version="lightgbm-regression",
    )
    return app
```

Update `test_client` to include auth header:

```python
@pytest.fixture
def test_client(test_app, api_key):
    from fastapi.testclient import TestClient

    client = TestClient(test_app)
    client.headers["Authorization"] = f"Bearer {api_key}"
    return client
```

**Step 5: Update test_serving.py**

Most existing tests should pass unchanged since the `test_client` fixture now includes auth headers. But verify by running:

Run: `pytest tests/test_serving.py -v`

Add auth-specific serving tests at the end of `tests/test_serving.py`:

```python
class TestAuthOnPredictionEndpoints:
    def test_predict_without_auth_returns_401(self, test_app) -> None:
        from fastapi.testclient import TestClient

        client = TestClient(test_app)  # No auth header
        resp = client.post(
            "/predict",
            json={"zone_id": 1, "hour_ts": "2024-02-01T12:00:00"},
        )
        assert resp.status_code == 401

    def test_batch_without_auth_returns_401(self, test_app) -> None:
        from fastapi.testclient import TestClient

        client = TestClient(test_app)
        resp = client.post(
            "/predict/batch",
            json={"requests": [{"zone_id": 1, "hour_ts": "2024-02-01T12:00:00"}]},
        )
        assert resp.status_code == 401

    def test_health_without_auth_returns_200(self, test_app) -> None:
        from fastapi.testclient import TestClient

        client = TestClient(test_app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_predict_response_includes_prediction_id(self, test_client) -> None:
        resp = test_client.post(
            "/predict",
            json={"zone_id": 1, "hour_ts": "2024-02-01T12:00:00"},
        )
        assert resp.status_code == 200
        assert "prediction_id" in resp.json()

    def test_batch_size_enforcement(self, test_app, test_db) -> None:
        from fastapi.testclient import TestClient
        from demandops.security.auth import hash_key

        # Create a key with max_batch_size=5
        raw_key = "limited-batch-key-1234567890"
        test_db.execute(
            "INSERT INTO api_keys (key_hash, client_name, created_at, rate_limit, max_batch_size, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (hash_key(raw_key), "batch_limited", "2024-01-01", 1000, 5, True),
        )
        test_db.commit()

        client = TestClient(test_app)
        client.headers["Authorization"] = f"Bearer {raw_key}"
        resp = client.post(
            "/predict/batch",
            json={"requests": [{"zone_id": 1, "hour_ts": "2024-02-01T12:00:00"}] * 6},
        )
        assert resp.status_code == 413
```

**Step 6: Run all serving tests**

Run: `pytest tests/test_serving.py tests/test_auth.py -v`
Expected: all PASS

**Step 7: Run full test suite to check nothing else broke**

Run: `pytest tests/ -v`
Expected: all PASS

**Step 8: Commit**

```bash
git add demandops/serving/schemas.py demandops/serving/routes.py demandops/serving/app.py \
    tests/conftest.py tests/test_serving.py
git commit -m "feat: wire auth into prediction routes, add prediction_id

/predict and /predict/batch now require Bearer token. Per-key batch size
enforcement returns 413. prediction_id (UUID) added to PredictResponse
for actuals matching. Health/metrics stay unauthenticated."
```

---

### Task 5: Drift detection — pure functions (PSI, KS, correlation shift)

**Files:**
- Create: `demandops/monitoring/drift_detector.py`
- Test: `tests/test_drift.py`

**Step 1: Write the failing tests**

```python
# tests/test_drift.py
"""Tests for data drift detection."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from demandops.features import FEATURE_COLUMNS


@pytest.fixture
def reference_distributions(tmp_path: Path) -> Path:
    """Build a synthetic reference distribution artifact."""
    rng = np.random.RandomState(42)
    n_samples = 5000

    ref = {"features": {}, "metadata": {"ks_subsample_size": n_samples, "n_bins": 10}}

    for feature_name in FEATURE_COLUMNS:
        if feature_name == "zone_id":
            values = rng.choice([1, 2, 3, 4, 5], size=n_samples).astype(float)
        elif feature_name == "hour_of_day":
            values = rng.choice(range(24), size=n_samples).astype(float)
        elif feature_name == "day_of_week":
            values = rng.choice(range(7), size=n_samples).astype(float)
        elif feature_name == "is_weekend":
            values = rng.choice([0, 1], size=n_samples, p=[5 / 7, 2 / 7]).astype(float)
        elif feature_name == "month":
            values = rng.choice(range(1, 13), size=n_samples).astype(float)
        else:
            values = rng.exponential(5, size=n_samples)

        quantiles = np.linspace(0, 100, 11)
        boundaries = np.percentile(values, quantiles).tolist()
        bin_counts = np.histogram(values, bins=boundaries)[0].tolist()

        ref["features"][feature_name] = {
            "decile_boundaries": boundaries,
            "bin_counts": bin_counts,
            "ks_subsample": values.tolist(),
        }

    # Correlation matrix on continuous features only
    cont_features = [c for c in FEATURE_COLUMNS if c != "zone_id"]
    cont_indices = [FEATURE_COLUMNS.index(c) for c in cont_features]
    # Build a full feature matrix to compute correlation
    full_matrix = np.column_stack([
        np.array(ref["features"][f]["ks_subsample"]) for f in FEATURE_COLUMNS
    ])
    cont_matrix = full_matrix[:, cont_indices]
    ref["correlation_matrix"] = np.corrcoef(cont_matrix, rowvar=False).tolist()
    ref["correlation_features"] = cont_features

    path = tmp_path / "reference_distributions.json"
    path.write_text(json.dumps(ref))
    return path


class TestPSI:
    def test_psi_triggers_on_shifted_data(self, reference_distributions) -> None:
        from demandops.monitoring.drift_detector import compute_psi

        ref = json.loads(reference_distributions.read_text())
        feature_ref = ref["features"]["hour_of_day"]

        # Shift: all values in 0-6 (night hours only)
        shifted = np.random.RandomState(99).choice(range(0, 7), size=500).astype(float)
        psi = compute_psi(
            feature_ref["decile_boundaries"],
            np.array(feature_ref["bin_counts"]),
            shifted,
        )
        assert psi > 0.25, f"Expected PSI > 0.25 for shifted data, got {psi}"

    def test_psi_low_on_same_distribution(self, reference_distributions) -> None:
        from demandops.monitoring.drift_detector import compute_psi

        ref = json.loads(reference_distributions.read_text())
        feature_ref = ref["features"]["hour_of_day"]

        # Same distribution as reference
        same = np.random.RandomState(99).choice(range(24), size=500).astype(float)
        psi = compute_psi(
            feature_ref["decile_boundaries"],
            np.array(feature_ref["bin_counts"]),
            same,
        )
        assert psi < 0.1, f"Expected PSI < 0.1 for same distribution, got {psi}"


class TestKS:
    def test_ks_triggers_on_shifted_data(self, reference_distributions) -> None:
        from demandops.monitoring.drift_detector import compute_ks

        ref = json.loads(reference_distributions.read_text())
        ref_sample = np.array(ref["features"]["lag_1h"]["ks_subsample"])

        # Different distribution
        shifted = np.random.RandomState(99).normal(50, 1, size=500)
        _, p_value = compute_ks(ref_sample, shifted)
        assert p_value < 0.05

    def test_ks_no_alert_on_same_distribution(self, reference_distributions) -> None:
        from demandops.monitoring.drift_detector import compute_ks

        ref = json.loads(reference_distributions.read_text())
        ref_sample = np.array(ref["features"]["lag_1h"]["ks_subsample"])

        # Subsample from same reference
        same = np.random.RandomState(99).choice(ref_sample, size=500, replace=True)
        _, p_value = compute_ks(ref_sample, same)
        assert p_value > 0.05


class TestCorrelationShift:
    def test_detects_altered_correlation(self, reference_distributions) -> None:
        from demandops.monitoring.drift_detector import compute_correlation_shift

        ref = json.loads(reference_distributions.read_text())
        ref_corr = np.array(ref["correlation_matrix"])

        # Build samples with flipped correlation between lag features
        rng = np.random.RandomState(42)
        n = 500
        n_cont = ref_corr.shape[0]
        # Random data with identity-ish correlation
        samples = rng.randn(n, n_cont)
        # Flip sign of one column to alter correlations
        samples[:, -1] = -samples[:, -1]

        shift = compute_correlation_shift(ref_corr, samples)
        assert shift > 0.05, f"Expected correlation shift > 0.05, got {shift}"

    def test_low_shift_on_matching_data(self, reference_distributions) -> None:
        from demandops.monitoring.drift_detector import compute_correlation_shift

        ref = json.loads(reference_distributions.read_text())
        ref_corr = np.array(ref["correlation_matrix"])
        cont_features = [c for c in FEATURE_COLUMNS if c != "zone_id"]
        cont_indices = [FEATURE_COLUMNS.index(c) for c in cont_features]

        # Use the reference data itself
        full_matrix = np.column_stack([
            np.array(ref["features"][f]["ks_subsample"]) for f in FEATURE_COLUMNS
        ])
        cont_samples = full_matrix[:, cont_indices]

        shift = compute_correlation_shift(ref_corr, cont_samples)
        assert shift < 0.05, f"Expected low correlation shift, got {shift}"
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_drift.py -v`
Expected: FAIL — imports fail

**Step 3: Write the implementation**

```python
# demandops/monitoring/drift_detector.py
"""Data drift detection: PSI, KS test, correlation shift.

Accumulates feature vectors in a bounded deque. Computes drift metrics
on demand when /monitoring/drift is called — no background threads.
"""

from __future__ import annotations

import collections
import json
import threading
from pathlib import Path

import numpy as np
from scipy import stats

from demandops.features import FEATURE_COLUMNS

CONTINUOUS_FEATURES = [c for c in FEATURE_COLUMNS if c != "zone_id"]
CONTINUOUS_INDICES = [FEATURE_COLUMNS.index(c) for c in CONTINUOUS_FEATURES]

PSI_WARNING = 0.1
PSI_ALERT = 0.25
KS_ALPHA = 0.05
CORRELATION_WARNING = 0.1


def compute_psi(
    decile_boundaries: list[float],
    reference_bin_counts: np.ndarray,
    current_values: np.ndarray,
) -> float:
    """Population Stability Index between reference and current distributions."""
    current_bin_counts = np.histogram(current_values, bins=decile_boundaries)[0]
    eps = 1e-6
    n_bins = len(reference_bin_counts)
    ref_pct = (reference_bin_counts + eps) / (reference_bin_counts.sum() + eps * n_bins)
    cur_pct = (current_bin_counts + eps) / (current_bin_counts.sum() + eps * n_bins)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_ks(
    reference_sample: np.ndarray, current_values: np.ndarray
) -> tuple[float, float]:
    """KS two-sample test. Returns (statistic, p_value)."""
    stat, p_value = stats.ks_2samp(reference_sample, current_values)
    return float(stat), float(p_value)


def compute_correlation_shift(
    reference_corr: np.ndarray, current_continuous: np.ndarray
) -> float:
    """Frobenius norm of correlation matrix difference, normalized by feature pairs."""
    current_corr = np.corrcoef(current_continuous, rowvar=False)
    diff = current_corr - reference_corr
    n = reference_corr.shape[0]
    n_pairs = n * (n - 1) / 2
    return float(np.linalg.norm(diff, "fro") / max(n_pairs, 1))


class DriftAccumulator:
    """Thread-safe bounded buffer for feature vectors."""

    def __init__(self, maxlen: int = 1000, min_samples: int = 100) -> None:
        self._lock = threading.Lock()
        self._buffer: collections.deque[list[float]] = collections.deque(maxlen=maxlen)
        self.min_samples = min_samples
        self.maxlen = maxlen

    def add(self, feature_vector: list[float]) -> None:
        with self._lock:
            self._buffer.append(feature_vector)

    def add_batch(self, feature_vectors: list[list[float]]) -> None:
        with self._lock:
            for v in feature_vectors:
                self._buffer.append(v)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def get_samples(self) -> np.ndarray | None:
        """Return accumulated samples as numpy array, or None if below minimum."""
        with self._lock:
            if len(self._buffer) < self.min_samples:
                return None
            return np.array(list(self._buffer))


class DriftDetector:
    """Computes drift metrics against training reference distributions."""

    def __init__(
        self,
        reference_path: Path,
        maxlen: int = 1000,
        min_samples: int = 100,
    ) -> None:
        self.accumulator = DriftAccumulator(maxlen=maxlen, min_samples=min_samples)
        ref = json.loads(reference_path.read_text())
        self._reference = ref
        self._ref_corr = np.array(ref["correlation_matrix"])

    def compute_drift(self) -> dict:
        """Compute drift metrics on accumulated samples. On-demand only."""
        samples = self.accumulator.get_samples()
        if samples is None:
            return {
                "status": "insufficient_samples",
                "collected": self.accumulator.count,
                "required": self.accumulator.min_samples,
            }

        result: dict = {
            "status": "ok",
            "collected": len(samples),
            "features": {},
        }

        for i, feature_name in enumerate(FEATURE_COLUMNS):
            feature_ref = self._reference["features"][feature_name]
            current_values = samples[:, i]

            psi = compute_psi(
                feature_ref["decile_boundaries"],
                np.array(feature_ref["bin_counts"]),
                current_values,
            )
            ks_stat, ks_pvalue = compute_ks(
                np.array(feature_ref["ks_subsample"]), current_values
            )

            if psi > PSI_ALERT or ks_pvalue < KS_ALPHA:
                verdict = "alert"
            elif psi > PSI_WARNING:
                verdict = "warning"
            else:
                verdict = "ok"

            result["features"][feature_name] = {
                "psi": round(psi, 6),
                "ks_statistic": round(ks_stat, 6),
                "ks_pvalue": round(ks_pvalue, 6),
                "verdict": verdict,
            }

        # Correlation shift on continuous features only
        continuous_samples = samples[:, CONTINUOUS_INDICES]
        corr_shift = compute_correlation_shift(self._ref_corr, continuous_samples)
        result["correlation_shift"] = round(corr_shift, 6)

        # Overall status
        verdicts = [f["verdict"] for f in result["features"].values()]
        if "alert" in verdicts or corr_shift > CORRELATION_WARNING:
            result["status"] = "alert"
        elif "warning" in verdicts:
            result["status"] = "warning"

        return result
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_drift.py -v`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add demandops/monitoring/drift_detector.py tests/test_drift.py
git commit -m "feat: add drift detection — PSI, KS test, correlation shift

PSI with decile binning (>0.1 warning, >0.25 alert). KS two-sample test
per feature (p<0.05 = drift). Frobenius norm of correlation matrix diff
on 8 continuous features (zone_id excluded). DriftAccumulator with
thread-safe bounded deque and min-sample gate."
```

---

### Task 6: Drift accumulator boundary + concurrency tests

**Files:**
- Modify: `tests/test_drift.py` — add accumulator and DriftDetector tests

**Step 1: Add accumulator and detector tests to test_drift.py**

```python
# Append to tests/test_drift.py

class TestDriftAccumulator:
    def test_insufficient_samples_returns_none(self) -> None:
        from demandops.monitoring.drift_detector import DriftAccumulator

        acc = DriftAccumulator(maxlen=1000, min_samples=100)
        for _ in range(50):
            acc.add([1.0] * 9)
        assert acc.get_samples() is None
        assert acc.count == 50

    def test_returns_samples_above_threshold(self) -> None:
        from demandops.monitoring.drift_detector import DriftAccumulator

        acc = DriftAccumulator(maxlen=1000, min_samples=100)
        for _ in range(100):
            acc.add([1.0] * 9)
        samples = acc.get_samples()
        assert samples is not None
        assert samples.shape == (100, 9)

    def test_deque_boundary_evicts_oldest(self) -> None:
        from demandops.monitoring.drift_detector import DriftAccumulator

        acc = DriftAccumulator(maxlen=1000, min_samples=10)
        # Push 1200 samples with value = index
        for i in range(1200):
            acc.add([float(i)] * 9)

        assert acc.count == 1000
        samples = acc.get_samples()
        assert samples is not None
        # Oldest 200 should be evicted; first sample should be index 200
        assert samples[0, 0] == 200.0
        assert samples[-1, 0] == 1199.0

    def test_concurrent_writes_no_corruption(self) -> None:
        from demandops.monitoring.drift_detector import DriftAccumulator

        acc = DriftAccumulator(maxlen=1000, min_samples=10)

        def writer(thread_id: int) -> None:
            for i in range(100):
                acc.add([float(thread_id * 1000 + i)] * 9)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert acc.count == 1000

        samples = acc.get_samples()
        assert samples is not None
        # Every row should have 9 identical values (no partial vectors)
        for row in samples:
            assert len(set(row)) == 1, f"Corrupted vector: {row}"


class TestDriftDetector:
    def test_insufficient_samples_response(self, reference_distributions) -> None:
        from demandops.monitoring.drift_detector import DriftDetector

        detector = DriftDetector(reference_distributions, min_samples=100)
        for _ in range(50):
            detector.accumulator.add([1.0] * 9)

        result = detector.compute_drift()
        assert result["status"] == "insufficient_samples"
        assert result["collected"] == 50
        assert result["required"] == 100

    def test_no_drift_on_reference_data(self, reference_distributions) -> None:
        from demandops.monitoring.drift_detector import DriftDetector

        ref = json.loads(reference_distributions.read_text())
        detector = DriftDetector(reference_distributions, min_samples=50)

        # Feed reference data back
        rng = np.random.RandomState(42)
        for _ in range(200):
            vector = []
            for feature_name in FEATURE_COLUMNS:
                subsample = ref["features"][feature_name]["ks_subsample"]
                vector.append(rng.choice(subsample))
            detector.accumulator.add(vector)

        result = detector.compute_drift()
        assert result["status"] == "ok"
        for feature_name, metrics in result["features"].items():
            assert metrics["verdict"] == "ok", (
                f"False alarm on {feature_name}: PSI={metrics['psi']}, "
                f"KS p={metrics['ks_pvalue']}"
            )

    def test_detects_shifted_feature(self, reference_distributions) -> None:
        from demandops.monitoring.drift_detector import DriftDetector

        ref = json.loads(reference_distributions.read_text())
        detector = DriftDetector(reference_distributions, min_samples=50)

        rng = np.random.RandomState(42)
        hour_idx = FEATURE_COLUMNS.index("hour_of_day")

        for _ in range(200):
            vector = []
            for j, feature_name in enumerate(FEATURE_COLUMNS):
                if j == hour_idx:
                    vector.append(float(rng.choice(range(0, 3))))  # Extreme shift
                else:
                    subsample = ref["features"][feature_name]["ks_subsample"]
                    vector.append(rng.choice(subsample))
            detector.accumulator.add(vector)

        result = detector.compute_drift()
        assert result["features"]["hour_of_day"]["verdict"] == "alert"
        assert result["status"] in ("alert", "warning")
```

**Step 2: Run tests**

Run: `pytest tests/test_drift.py -v`
Expected: all tests PASS

**Step 3: Commit**

```bash
git add tests/test_drift.py
git commit -m "test: add accumulator boundary, concurrency, and detector tests

Deque evicts oldest at maxlen=1000. 10 threads x 100 writes produce
no partial vectors. DriftDetector returns insufficient_samples below
threshold, no false alarms on reference data, detects shifted features."
```

---

### Task 7: Reference distribution generation

**Files:**
- Create: `demandops/training/reference_distributions.py`
- Modify: `demandops/training/train.py:20-78` — call reference generation
- Test: `tests/test_reference_distributions.py`

**Step 1: Write the failing tests**

```python
# tests/test_reference_distributions.py
"""Tests for reference distribution artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from demandops.features import FEATURE_COLUMNS


@pytest.fixture
def training_features() -> np.ndarray:
    """Synthetic training feature matrix (1000 rows x 9 features)."""
    rng = np.random.RandomState(42)
    n = 1000
    return np.column_stack([
        rng.choice(range(24), size=n),        # hour_of_day
        rng.choice(range(7), size=n),          # day_of_week
        rng.choice([0, 1], size=n),            # is_weekend
        rng.choice(range(1, 13), size=n),      # month
        rng.choice([1, 2, 3], size=n),         # zone_id
        rng.exponential(5, size=n),            # lag_1h
        rng.exponential(5, size=n),            # lag_24h
        rng.exponential(5, size=n),            # lag_168h
        rng.exponential(5, size=n),            # rolling_mean_24h
    ]).astype(float)


class TestGenerateReferenceDistributions:
    def test_creates_artifact_file(self, tmp_path: Path, training_features) -> None:
        from demandops.training.reference_distributions import (
            generate_reference_distributions,
        )

        output_path = tmp_path / "ref.json"
        generate_reference_distributions(training_features, output_path)
        assert output_path.exists()

    def test_contains_all_features(self, tmp_path: Path, training_features) -> None:
        from demandops.training.reference_distributions import (
            generate_reference_distributions,
        )

        output_path = tmp_path / "ref.json"
        generate_reference_distributions(training_features, output_path)
        ref = json.loads(output_path.read_text())
        for feature_name in FEATURE_COLUMNS:
            assert feature_name in ref["features"]

    def test_decile_boundaries_have_11_values(
        self, tmp_path: Path, training_features
    ) -> None:
        from demandops.training.reference_distributions import (
            generate_reference_distributions,
        )

        output_path = tmp_path / "ref.json"
        generate_reference_distributions(training_features, output_path)
        ref = json.loads(output_path.read_text())
        for feature_name in FEATURE_COLUMNS:
            assert len(ref["features"][feature_name]["decile_boundaries"]) == 11

    def test_ks_subsample_capped(self, tmp_path: Path, training_features) -> None:
        from demandops.training.reference_distributions import (
            generate_reference_distributions,
        )

        output_path = tmp_path / "ref.json"
        generate_reference_distributions(
            training_features, output_path, ks_subsample_size=500
        )
        ref = json.loads(output_path.read_text())
        for feature_name in FEATURE_COLUMNS:
            assert len(ref["features"][feature_name]["ks_subsample"]) == 500

    def test_correlation_matrix_8x8(self, tmp_path: Path, training_features) -> None:
        from demandops.training.reference_distributions import (
            generate_reference_distributions,
        )

        output_path = tmp_path / "ref.json"
        generate_reference_distributions(training_features, output_path)
        ref = json.loads(output_path.read_text())
        corr = np.array(ref["correlation_matrix"])
        assert corr.shape == (8, 8)

    def test_metadata_records_subsample_size(
        self, tmp_path: Path, training_features
    ) -> None:
        from demandops.training.reference_distributions import (
            generate_reference_distributions,
        )

        output_path = tmp_path / "ref.json"
        generate_reference_distributions(
            training_features, output_path, ks_subsample_size=500
        )
        ref = json.loads(output_path.read_text())
        assert ref["metadata"]["ks_subsample_size"] == 500
        assert ref["metadata"]["n_training_rows"] == 1000
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_reference_distributions.py -v`
Expected: FAIL — module does not exist

**Step 3: Write the implementation**

```python
# demandops/training/reference_distributions.py
"""Generate reference distribution artifact for drift detection.

Computes per-feature decile boundaries, KS subsamples, and the
correlation matrix on continuous features. Saved to JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from demandops.features import FEATURE_COLUMNS

CONTINUOUS_FEATURES = [c for c in FEATURE_COLUMNS if c != "zone_id"]
CONTINUOUS_INDICES = [FEATURE_COLUMNS.index(c) for c in CONTINUOUS_FEATURES]


def generate_reference_distributions(
    X_train: np.ndarray,
    output_path: Path,
    ks_subsample_size: int = 5000,
    n_bins: int = 10,
    seed: int = 42,
) -> None:
    """Generate and save reference distributions from training data."""
    rng = np.random.RandomState(seed)
    ref: dict = {
        "features": {},
        "metadata": {
            "ks_subsample_size": ks_subsample_size,
            "n_bins": n_bins,
            "n_training_rows": len(X_train),
        },
    }

    for i, feature_name in enumerate(FEATURE_COLUMNS):
        col = X_train[:, i]

        quantiles = np.linspace(0, 100, n_bins + 1)
        boundaries = np.percentile(col, quantiles).tolist()
        bin_counts = np.histogram(col, bins=boundaries)[0].tolist()

        sample_size = min(ks_subsample_size, len(col))
        subsample_idx = rng.choice(len(col), sample_size, replace=False)
        subsample = col[subsample_idx].tolist()

        ref["features"][feature_name] = {
            "decile_boundaries": boundaries,
            "bin_counts": bin_counts,
            "ks_subsample": subsample,
        }

    # Correlation matrix on continuous features only (zone_id excluded)
    cont_data = X_train[:, CONTINUOUS_INDICES]
    ref["correlation_matrix"] = np.corrcoef(cont_data, rowvar=False).tolist()
    ref["correlation_features"] = CONTINUOUS_FEATURES

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ref, indent=2))
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_reference_distributions.py -v`
Expected: all tests PASS

**Step 5: Integrate into training pipeline**

In `demandops/training/train.py`, add to `train_model()` after `_save_feature_schema()` (around line 49):

```python
from demandops.training.reference_distributions import generate_reference_distributions

# Inside train_model(), after _save_feature_schema(feature_schema_path):
ref_path = models_dir.parent / "reference_distributions.json"
generate_reference_distributions(X_train, ref_path)
logger.info("reference_distributions_saved", path=str(ref_path))
```

**Step 6: Run full test suite**

Run: `pytest tests/ -v`
Expected: all PASS

**Step 7: Commit**

```bash
git add demandops/training/reference_distributions.py demandops/training/train.py \
    tests/test_reference_distributions.py
git commit -m "feat: generate reference distributions during training

Per-feature decile boundaries (10 bins), 5000-point KS subsample, and
8x8 correlation matrix (zone_id excluded). Saved to
artifacts/reference_distributions.json. Integrated into train pipeline."
```

---

### Task 8: Quality tracker — prediction logging, actuals matching, metrics

**Files:**
- Create: `demandops/monitoring/quality_tracker.py`
- Test: `tests/test_quality.py`

**Step 1: Write the failing tests**

```python
# tests/test_quality.py
"""Tests for prediction quality monitoring."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from demandops.db import get_db


@pytest.fixture
def quality_db(tmp_path: Path):
    conn = get_db(str(tmp_path / "quality_test.db"))
    yield conn
    conn.close()


class TestPredictionLogging:
    def test_log_returns_prediction_id(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        pid = tracker.log_prediction(
            zone_id=1, hour_ts="2024-02-01T12:00:00", predicted_value=42.5
        )
        assert isinstance(pid, str)
        assert len(pid) == 36  # UUID format

    def test_logged_prediction_in_db(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        pid = tracker.log_prediction(
            zone_id=1, hour_ts="2024-02-01T12:00:00", predicted_value=42.5
        )
        row = quality_db.execute(
            "SELECT zone_id, predicted_value, actual_value FROM prediction_log WHERE prediction_id = ?",
            (pid,),
        ).fetchone()
        assert row[0] == 1
        assert row[1] == 42.5
        assert row[2] is None  # No actual yet

    def test_concurrent_logging_no_errors(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        errors = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(20):
                    tracker.log_prediction(
                        zone_id=thread_id,
                        hour_ts=f"2024-02-01T{i:02d}:00:00",
                        predicted_value=float(i),
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent writes failed: {errors}"
        count = quality_db.execute(
            "SELECT COUNT(*) FROM prediction_log"
        ).fetchone()[0]
        assert count == 200


class TestActualsSubmission:
    def test_match_by_prediction_id(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        pid = tracker.log_prediction(1, "2024-02-01T12:00:00", 42.5)

        result = tracker.submit_actuals([
            {"prediction_id": pid, "actual_value": 40.0},
        ])
        assert result["matched_count"] == 1
        assert result["unmatched_count"] == 0

        row = quality_db.execute(
            "SELECT actual_value FROM prediction_log WHERE prediction_id = ?",
            (pid,),
        ).fetchone()
        assert row[0] == 40.0

    def test_match_by_zone_ts(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        tracker.log_prediction(1, "2024-02-01T12:00:00", 42.5)

        result = tracker.submit_actuals([
            {"zone_id": 1, "hour_ts": "2024-02-01T12:00:00", "actual_value": 40.0},
        ])
        assert result["matched_count"] == 1

    def test_ambiguous_match_uses_most_recent(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        pid1 = tracker.log_prediction(1, "2024-02-01T12:00:00", 30.0)
        pid2 = tracker.log_prediction(1, "2024-02-01T12:00:00", 42.5)

        tracker.submit_actuals([
            {"zone_id": 1, "hour_ts": "2024-02-01T12:00:00", "actual_value": 40.0},
        ])

        # Most recent (pid2) should be matched
        row1 = quality_db.execute(
            "SELECT actual_value FROM prediction_log WHERE prediction_id = ?",
            (pid1,),
        ).fetchone()
        row2 = quality_db.execute(
            "SELECT actual_value FROM prediction_log WHERE prediction_id = ?",
            (pid2,),
        ).fetchone()
        assert row1[0] is None
        assert row2[0] == 40.0

    def test_unmatched_returns_warning(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        result = tracker.submit_actuals([
            {"prediction_id": "nonexistent-id", "actual_value": 40.0},
        ])
        assert result["unmatched_count"] == 1
        assert len(result["warnings"]) == 1


class TestQualityComputation:
    def _seed_matched_pairs(self, tracker, pairs: list[tuple[float, float]]) -> None:
        """Helper: log predictions and submit actuals for known pairs."""
        pids = []
        for pred, _ in pairs:
            pid = tracker.log_prediction(1, "2024-02-01T12:00:00", pred)
            pids.append(pid)
        actuals = [
            {"prediction_id": pid, "actual_value": actual}
            for pid, (_, actual) in zip(pids, pairs)
        ]
        tracker.submit_actuals(actuals)

    def test_mae_computation(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        # MAE = mean(|10-12|, |20-18|, |30-33|, ...) — need at least 10 pairs
        pairs = [(10, 12), (20, 18), (30, 33), (40, 38), (50, 52),
                 (60, 58), (70, 73), (80, 78), (90, 92), (100, 98)]
        self._seed_matched_pairs(tracker, pairs)

        result = tracker.compute_quality(window="7d")
        assert result["status"] == "ok"

        expected_mae = np.mean([abs(p - a) for p, a in pairs])
        assert abs(result["mae"] - expected_mae) < 0.001

    def test_rmse_computation(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        pairs = [(10, 12), (20, 18), (30, 33), (40, 38), (50, 52),
                 (60, 58), (70, 73), (80, 78), (90, 92), (100, 98)]
        self._seed_matched_pairs(tracker, pairs)

        result = tracker.compute_quality(window="7d")
        expected_rmse = np.sqrt(np.mean([(p - a) ** 2 for p, a in pairs]))
        assert abs(result["rmse"] - expected_rmse) < 0.001

    def test_smape_bounded_with_zeros(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        # Include zero actuals that would make MAPE infinite
        pairs = [(10, 12), (5, 0), (0, 0), (20, 18), (30, 33),
                 (40, 38), (50, 52), (60, 58), (70, 73), (80, 78)]
        self._seed_matched_pairs(tracker, pairs)

        result = tracker.compute_quality(window="7d")
        assert result["status"] == "ok"
        assert result["smape"] < 200  # sMAPE is bounded [0, 200]
        assert result["zero_denominator_pairs_excluded"] >= 1  # (0, 0) pair

    def test_insufficient_pairs_gate(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        pairs = [(10, 12), (20, 18)]
        self._seed_matched_pairs(tracker, pairs)

        result = tracker.compute_quality(window="7d")
        assert result["status"] == "insufficient_matched_pairs"
        assert result["matched"] == 2
        assert result["required"] == 10

    def test_window_filtering(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)

        # Log 10 old predictions (outside window) manually
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        for i in range(10):
            quality_db.execute(
                "INSERT INTO prediction_log (prediction_id, zone_id, hour_ts, predicted_value, actual_value, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (f"old-{i}", 1, "2024-01-01T12:00:00", 10.0, 12.0, old_time),
            )
        quality_db.commit()

        # Log 5 recent predictions (inside window)
        pairs = [(10, 12), (20, 18), (30, 33), (40, 38), (50, 52)]
        self._seed_matched_pairs(tracker, pairs)

        # window=1d should only see the 5 recent ones
        result = tracker.compute_quality(window="1d")
        assert result["status"] == "insufficient_matched_pairs"
        assert result["matched"] == 5
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_quality.py -v`
Expected: FAIL — module does not exist

**Step 3: Write the implementation**

```python
# demandops/monitoring/quality_tracker.py
"""Prediction quality monitoring: log predictions, match actuals, compute metrics.

Predictions are logged to SQLite with a UUID. When ground truth arrives
(with a lag), actuals are matched by prediction_id or (zone_id, hour_ts).
Quality metrics (MAE, RMSE, sMAPE) computed over a rolling window.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np


class QualityTracker:
    """Logs predictions and computes quality metrics against actuals."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db
        self._lock = threading.Lock()

    def log_prediction(
        self, zone_id: int, hour_ts: str, predicted_value: float
    ) -> str:
        """Log a prediction. Returns the prediction_id (UUID)."""
        prediction_id = str(uuid.uuid4())
        with self._lock:
            self._db.execute(
                "INSERT INTO prediction_log "
                "(prediction_id, zone_id, hour_ts, predicted_value, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    prediction_id,
                    zone_id,
                    hour_ts,
                    predicted_value,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._db.commit()
        return prediction_id

    def submit_actuals(self, actuals: list[dict]) -> dict:
        """Match actuals to logged predictions. Returns match summary."""
        matched = 0
        unmatched = 0
        warnings: list[str] = []

        with self._lock:
            for actual in actuals:
                if "prediction_id" in actual and actual["prediction_id"] is not None:
                    row = self._db.execute(
                        "SELECT prediction_id FROM prediction_log WHERE prediction_id = ?",
                        (actual["prediction_id"],),
                    ).fetchone()
                    if row:
                        self._db.execute(
                            "UPDATE prediction_log SET actual_value = ? "
                            "WHERE prediction_id = ?",
                            (actual["actual_value"], actual["prediction_id"]),
                        )
                        matched += 1
                    else:
                        unmatched += 1
                        warnings.append(
                            f"prediction_id {actual['prediction_id']} not found"
                        )
                else:
                    # Match by (zone_id, hour_ts) — most recent prediction
                    row = self._db.execute(
                        "SELECT prediction_id FROM prediction_log "
                        "WHERE zone_id = ? AND hour_ts = ? "
                        "ORDER BY created_at DESC LIMIT 1",
                        (actual["zone_id"], actual["hour_ts"]),
                    ).fetchone()
                    if row:
                        self._db.execute(
                            "UPDATE prediction_log SET actual_value = ? "
                            "WHERE prediction_id = ?",
                            (actual["actual_value"], row[0]),
                        )
                        matched += 1
                    else:
                        unmatched += 1
                        warnings.append(
                            f"No prediction found for zone_id={actual['zone_id']}, "
                            f"hour_ts={actual['hour_ts']}"
                        )
            self._db.commit()

        return {
            "matched_count": matched,
            "unmatched_count": unmatched,
            "warnings": warnings,
        }

    def compute_quality(self, window: str = "7d") -> dict:
        """Compute quality metrics over matched pairs in the given window."""
        days = int(window.rstrip("d"))
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        rows = self._db.execute(
            "SELECT predicted_value, actual_value FROM prediction_log "
            "WHERE actual_value IS NOT NULL AND created_at >= ?",
            (cutoff,),
        ).fetchall()

        if len(rows) < 10:
            return {
                "status": "insufficient_matched_pairs",
                "matched": len(rows),
                "required": 10,
            }

        preds = np.array([r[0] for r in rows])
        actuals = np.array([r[1] for r in rows])

        mae = float(np.mean(np.abs(preds - actuals)))
        rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))

        # sMAPE: bounded [0, 200], handles zeros
        denominator = np.abs(preds) + np.abs(actuals)
        nonzero_mask = denominator > 0
        zero_count = int(np.sum(~nonzero_mask))
        if nonzero_mask.any():
            smape = float(
                np.mean(
                    2 * np.abs(preds[nonzero_mask] - actuals[nonzero_mask])
                    / denominator[nonzero_mask]
                )
                * 100
            )
        else:
            smape = 0.0

        return {
            "status": "ok",
            "matched_pairs": len(rows),
            "window": window,
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "smape": round(smape, 6),
            "zero_denominator_pairs_excluded": zero_count,
        }
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_quality.py -v`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add demandops/monitoring/quality_tracker.py tests/test_quality.py
git commit -m "feat: add quality tracker — prediction logging, actuals, sMAPE

Log predictions with UUID to SQLite. Match actuals by prediction_id or
(zone_id, hour_ts) — most recent wins on ambiguity. sMAPE replaces MAPE
(bounded, handles zeros). Min 10 matched pairs gate. Thread-safe."
```

---

### Task 9: Monitoring routes — /monitoring/drift, /monitoring/quality, /monitoring/actuals

**Files:**
- Create: `demandops/serving/monitoring_routes.py`
- Modify: `demandops/serving/app.py` — include monitoring router, init drift detector + quality tracker
- Test: `tests/test_monitoring_routes.py`

**Step 1: Write the failing tests**

```python
# tests/test_monitoring_routes.py
"""Tests for monitoring API endpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from demandops.db import get_db
from demandops.features import FEATURE_COLUMNS
from demandops.security.auth import RateLimiter, hash_key


@pytest.fixture
def monitoring_reference(tmp_path: Path) -> Path:
    """Minimal reference distributions for testing."""
    rng = np.random.RandomState(42)
    n = 5000
    ref = {"features": {}, "metadata": {"ks_subsample_size": n, "n_bins": 10}}

    for feature_name in FEATURE_COLUMNS:
        if feature_name == "zone_id":
            values = rng.choice([1, 2, 3], size=n).astype(float)
        elif feature_name == "hour_of_day":
            values = rng.choice(range(24), size=n).astype(float)
        elif feature_name == "day_of_week":
            values = rng.choice(range(7), size=n).astype(float)
        elif feature_name == "is_weekend":
            values = rng.choice([0, 1], size=n).astype(float)
        elif feature_name == "month":
            values = rng.choice(range(1, 13), size=n).astype(float)
        else:
            values = rng.exponential(5, size=n)

        quantiles = np.linspace(0, 100, 11)
        boundaries = np.percentile(values, quantiles).tolist()
        bin_counts = np.histogram(values, bins=boundaries)[0].tolist()
        ref["features"][feature_name] = {
            "decile_boundaries": boundaries,
            "bin_counts": bin_counts,
            "ks_subsample": values.tolist(),
        }

    cont_features = [c for c in FEATURE_COLUMNS if c != "zone_id"]
    cont_indices = [FEATURE_COLUMNS.index(c) for c in cont_features]
    full_matrix = np.column_stack([
        np.array(ref["features"][f]["ks_subsample"]) for f in FEATURE_COLUMNS
    ])
    ref["correlation_matrix"] = np.corrcoef(
        full_matrix[:, cont_indices], rowvar=False
    ).tolist()
    ref["correlation_features"] = cont_features

    path = tmp_path / "reference_distributions.json"
    path.write_text(json.dumps(ref))
    return path


@pytest.fixture
def monitoring_app(tmp_path: Path, monitoring_reference) -> FastAPI:
    from demandops.monitoring.drift_detector import DriftDetector
    from demandops.monitoring.quality_tracker import QualityTracker
    from demandops.serving.monitoring_routes import monitoring_router

    db = get_db(str(tmp_path / "monitoring_test.db"))
    detector = DriftDetector(monitoring_reference, min_samples=10)
    tracker = QualityTracker(db)

    app = FastAPI()
    app.state.db = db
    app.state.rate_limiter = RateLimiter()
    app.state.drift_detector = detector
    app.state.quality_tracker = tracker
    app.include_router(monitoring_router)

    # Create an API key for actuals endpoint
    raw_key = "monitoring-test-key"
    db.execute(
        "INSERT INTO api_keys (key_hash, client_name, created_at, rate_limit, is_active) "
        "VALUES (?, ?, ?, ?, ?)",
        (hash_key(raw_key), "test", "2024-01-01", 1000, True),
    )
    db.commit()
    app.state._test_api_key = raw_key

    return app


@pytest.fixture
def monitoring_client(monitoring_app) -> TestClient:
    return TestClient(monitoring_app)


class TestDriftEndpoint:
    def test_insufficient_samples(self, monitoring_client) -> None:
        resp = monitoring_client.get("/monitoring/drift")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "insufficient_samples"
        assert data["collected"] == 0

    def test_returns_drift_after_samples(self, monitoring_app, monitoring_client) -> None:
        ref = json.loads(
            monitoring_app.state.drift_detector._reference["features"]["hour_of_day"]["ks_subsample"].__class__.__name__
            # Just feed reference-like data
        ) if False else None

        detector = monitoring_app.state.drift_detector
        rng = np.random.RandomState(42)
        ref_data = detector._reference
        for _ in range(50):
            vector = []
            for f in FEATURE_COLUMNS:
                vector.append(rng.choice(ref_data["features"][f]["ks_subsample"]))
            detector.accumulator.add(vector)

        resp = monitoring_client.get("/monitoring/drift")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "features" in data
        assert "correlation_shift" in data

    def test_no_auth_required(self, monitoring_client) -> None:
        resp = monitoring_client.get("/monitoring/drift")
        assert resp.status_code == 200  # No auth header, still works


class TestQualityEndpoint:
    def test_insufficient_pairs(self, monitoring_client) -> None:
        resp = monitoring_client.get("/monitoring/quality")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "insufficient_matched_pairs"

    def test_with_matched_data(self, monitoring_app, monitoring_client) -> None:
        tracker = monitoring_app.state.quality_tracker
        for i in range(10):
            pid = tracker.log_prediction(1, "2024-02-01T12:00:00", float(i * 10))
            tracker.submit_actuals([{"prediction_id": pid, "actual_value": float(i * 10 + 2)}])

        resp = monitoring_client.get("/monitoring/quality?window=7d")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "mae" in data
        assert "rmse" in data
        assert "smape" in data

    def test_no_auth_required(self, monitoring_client) -> None:
        resp = monitoring_client.get("/monitoring/quality")
        assert resp.status_code == 200


class TestActualsEndpoint:
    def test_requires_auth(self, monitoring_client) -> None:
        resp = monitoring_client.post(
            "/monitoring/actuals",
            json={"actuals": [{"prediction_id": "abc", "actual_value": 10.0}]},
        )
        assert resp.status_code == 401

    def test_submit_actuals_with_auth(self, monitoring_app, monitoring_client) -> None:
        tracker = monitoring_app.state.quality_tracker
        pid = tracker.log_prediction(1, "2024-02-01T12:00:00", 42.5)

        resp = monitoring_client.post(
            "/monitoring/actuals",
            json={"actuals": [{"prediction_id": pid, "actual_value": 40.0}]},
            headers={"Authorization": f"Bearer {monitoring_app.state._test_api_key}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched_count"] == 1
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_monitoring_routes.py -v`
Expected: FAIL — `monitoring_routes` does not exist

**Step 3: Write the implementation**

```python
# demandops/serving/monitoring_routes.py
"""Monitoring API routes: /monitoring/drift, /monitoring/quality, /monitoring/actuals."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from demandops.security.auth import requires_auth
from demandops.serving.schemas import ActualsRequest, ActualsResponse

monitoring_router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@monitoring_router.get("/drift")
async def drift_status(request: Request):
    """Return current drift status per feature. No auth required."""
    detector = request.app.state.drift_detector
    return detector.compute_drift()


@monitoring_router.get("/quality")
async def quality_status(request: Request, window: str = "7d"):
    """Return quality metrics over the specified window. No auth required.

    Includes drift-quality correlation when MAE exceeds threshold.
    """
    tracker = request.app.state.quality_tracker
    result = tracker.compute_quality(window=window)

    # Drift-quality correlation: if quality is degraded, include drift status
    if result.get("status") == "ok":
        mae_threshold = 3.20  # From regression gate
        margin = 1.2
        if result.get("mae", 0) > mae_threshold * margin:
            detector = request.app.state.drift_detector
            drift = detector.compute_drift()
            result["drift_correlation"] = {
                "drift_status": drift.get("status"),
                "note": "Quality degradation detected alongside drift — may indicate retraining needed",
            }

    return result


@monitoring_router.post("/actuals", response_model=ActualsResponse)
async def submit_actuals(
    body: ActualsRequest,
    request: Request,
    client: dict = Depends(requires_auth),
):
    """Submit ground truth actuals to match against logged predictions."""
    tracker = request.app.state.quality_tracker
    actuals = [a.model_dump() for a in body.actuals]
    result = tracker.submit_actuals(actuals)
    return ActualsResponse(**result)
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_monitoring_routes.py -v`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add demandops/serving/monitoring_routes.py tests/test_monitoring_routes.py
git commit -m "feat: add monitoring routes — drift, quality, actuals endpoints

GET /monitoring/drift — on-demand drift computation, no auth.
GET /monitoring/quality — sMAPE/MAE/RMSE with drift-quality correlation.
POST /monitoring/actuals — auth-gated actuals submission."
```

---

### Task 10: Wire drift accumulator + quality logging into prediction flow

**Files:**
- Modify: `demandops/serving/app.py` — init drift detector, quality tracker on startup; include monitoring router
- Modify: `demandops/serving/routes.py` — feed features to drift accumulator, log predictions to quality tracker
- Modify: `demandops/serving/schemas.py` — add `ActualSubmission` model if not done in Task 4
- Modify: `tests/conftest.py` — add drift detector + quality tracker to test fixtures
- Add: `tests/test_integration.py`

**Step 1: Update app.py startup**

In `demandops/serving/app.py`, add drift detector and quality tracker initialization. After the existing `configure()` call:

```python
from demandops.monitoring.drift_detector import DriftDetector
from demandops.monitoring.quality_tracker import QualityTracker
from demandops.serving.monitoring_routes import monitoring_router

# In create_app(), after app.include_router(router):
app.include_router(monitoring_router)

# In startup(), after configure() call:
# Initialize drift detector (graceful degradation if reference missing)
ref_path = Path(config["artifacts"].get("reference_distributions_path",
    "artifacts/reference_distributions.json"))
if ref_path.exists():
    drift_detector = DriftDetector(ref_path)
    logger.info("drift_detector_loaded", path=str(ref_path))
else:
    drift_detector = None
    logger.warning("drift_detector_skipped", path=str(ref_path))
app.state.drift_detector = drift_detector

# Initialize quality tracker
quality_tracker = QualityTracker(db)
app.state.quality_tracker = quality_tracker
```

**Step 2: Update routes.py to feed drift accumulator and log predictions**

In `predict()`, after successful prediction (around line 95):

```python
# Feed drift accumulator
if hasattr(request.app.state, "drift_detector") and request.app.state.drift_detector:
    feature_vector = [features[col] for col in features]
    request.app.state.drift_detector.accumulator.add(feature_vector)

# Log to quality tracker
if hasattr(request.app.state, "quality_tracker") and request.app.state.quality_tracker:
    request.app.state.quality_tracker.log_prediction(
        zone_id=body.zone_id,
        hour_ts=body.hour_ts.isoformat(),
        predicted_value=predicted_count,
    )
```

Similar for `predict_batch()` — feed all feature vectors and log all predictions.

**Step 3: Update conftest.py**

Add drift detector and quality tracker to test fixtures:

```python
@pytest.fixture
def test_app(mock_feature_service, mock_model, test_db, api_key, tmp_path):
    # ... existing setup ...
    from demandops.monitoring.quality_tracker import QualityTracker

    app.state.quality_tracker = QualityTracker(test_db)
    app.state.drift_detector = None  # No reference distributions in unit tests
    return app
```

**Step 4: Write integration test**

```python
# tests/test_integration.py
"""Integration tests: auth + prediction + drift accumulation + quality logging."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from demandops.db import get_db
from demandops.features import FEATURE_COLUMNS
from demandops.security.auth import RateLimiter, hash_key


@pytest.fixture
def integration_app(
    mock_feature_service, mock_model, tmp_path: Path
) -> tuple[FastAPI, str]:
    """Full app with auth, drift detector, and quality tracker."""
    from demandops.monitoring.drift_detector import DriftDetector
    from demandops.monitoring.quality_tracker import QualityTracker
    from demandops.serving.monitoring_routes import monitoring_router
    from demandops.serving.routes import configure, router

    # Build reference distributions
    rng = np.random.RandomState(42)
    n = 5000
    ref = {"features": {}, "metadata": {"ks_subsample_size": n, "n_bins": 10}}
    for f in FEATURE_COLUMNS:
        values = rng.exponential(5, size=n) if f not in ("zone_id", "hour_of_day", "day_of_week", "is_weekend", "month") else rng.choice(range(24), size=n).astype(float)
        quantiles = np.linspace(0, 100, 11)
        boundaries = np.percentile(values, quantiles).tolist()
        ref["features"][f] = {
            "decile_boundaries": boundaries,
            "bin_counts": np.histogram(values, bins=boundaries)[0].tolist(),
            "ks_subsample": values.tolist(),
        }
    cont = [c for c in FEATURE_COLUMNS if c != "zone_id"]
    full = np.column_stack([np.array(ref["features"][f]["ks_subsample"]) for f in FEATURE_COLUMNS])
    ref["correlation_matrix"] = np.corrcoef(full[:, [FEATURE_COLUMNS.index(c) for c in cont]], rowvar=False).tolist()
    ref_path = tmp_path / "ref.json"
    ref_path.write_text(json.dumps(ref))

    db = get_db(str(tmp_path / "integration.db"))
    raw_key = "integration-test-key"
    db.execute(
        "INSERT INTO api_keys (key_hash, client_name, created_at, rate_limit, is_active) VALUES (?, ?, ?, ?, ?)",
        (hash_key(raw_key), "integration", "2024-01-01", 1000, True),
    )
    db.commit()

    import time
    app = FastAPI()
    app.state.db = db
    app.state.rate_limiter = RateLimiter()
    app.state.drift_detector = DriftDetector(ref_path, min_samples=5)
    app.state.quality_tracker = QualityTracker(db)
    app.include_router(router)
    app.include_router(monitoring_router)

    configure(app, mock_feature_service, mock_model, "lightgbm", time.time(),
              model_artifact_loaded=True, model_objective="regression",
              model_version="lightgbm-regression")

    return app, raw_key


class TestPredictionToDriftPipeline:
    def test_predictions_accumulate_in_drift_detector(self, integration_app) -> None:
        app, key = integration_app
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {key}"}

        for _ in range(10):
            resp = client.post(
                "/predict",
                json={"zone_id": 1, "hour_ts": "2024-02-01T12:00:00"},
                headers=headers,
            )
            assert resp.status_code == 200

        drift_resp = client.get("/monitoring/drift")
        data = drift_resp.json()
        assert data["collected"] >= 5  # min_samples=5, should have 10

    def test_predictions_logged_to_quality_tracker(self, integration_app) -> None:
        app, key = integration_app
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {key}"}

        resp = client.post(
            "/predict",
            json={"zone_id": 1, "hour_ts": "2024-02-01T12:00:00"},
            headers=headers,
        )
        prediction_id = resp.json()["prediction_id"]

        # Submit actual
        actuals_resp = client.post(
            "/monitoring/actuals",
            json={"actuals": [{"prediction_id": prediction_id, "actual_value": 40.0}]},
            headers=headers,
        )
        assert actuals_resp.json()["matched_count"] == 1
```

**Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: all PASS

**Step 6: Commit**

```bash
git add demandops/serving/app.py demandops/serving/routes.py tests/conftest.py \
    tests/test_integration.py
git commit -m "feat: wire drift accumulator and quality logging into prediction flow

Every prediction feeds feature vector to drift accumulator and logs to
SQLite via quality tracker. prediction_id returned in response enables
actuals matching. Graceful degradation when reference artifact missing."
```

---

### Task 11: Documentation — DECISIONS.md, README, config

**Files:**
- Modify: `DECISIONS.md` — add 7 new decisions (#24–#30)
- Modify: `README.md` — add Monitoring & Security section
- Modify: `configs/default.yaml` — add db and monitoring config

**Step 1: Add DECISIONS.md entries**

Append to `DECISIONS.md`:

```markdown
## 24. PSI + KS + Correlation Matrix (Not Autoencoder)

**Decision:** Use three complementary drift detection methods: Population Stability Index (PSI), Kolmogorov-Smirnov two-sample test, and Frobenius norm of correlation matrix difference. An autoencoder-based anomaly detector was considered and rejected.

**Why:** PSI is the industry standard for drift detection but is bin-based, making it insensitive to tail changes. KS complements with distribution-wide sensitivity but operates per-feature. The correlation matrix catches joint distribution shifts invisible to per-feature tests (e.g., novel feature combinations where individual features look normal). An autoencoder was rejected because with 9 features and a bottleneck of 8, the compression ratio is too low for meaningful anomaly detection — reconstruction error would be dominated by noise, not distributional novelty.

## 25. sMAPE Over MAPE

**Decision:** Use Symmetric Mean Absolute Percentage Error (sMAPE) instead of MAPE for quality monitoring.

**Why:** Demand data contains genuine zeros (zones with no trips in a given hour). MAPE divides by actual values, so a single zero actual produces infinite MAPE that poisons the aggregate. sMAPE is bounded [0, 200] and handles the zero case gracefully. The count of zero-denominator pairs excluded from sMAPE is reported alongside, so the impact is transparent.

## 26. zone_id Excluded from Correlation Matrix

**Decision:** The correlation matrix for drift detection is computed on the 8 continuous features only. `zone_id` is excluded.

**Why:** `zone_id` is categorical (261 zones in NYC taxi, 802 in TfL, 2,144 in Citibike). Pearson correlation between a categorical variable and continuous features is not meaningful — the Frobenius norm would be dominated by noise in that column.

## 27. SQLite for Prediction Logging and Auth

**Decision:** Use a single SQLite database (`data/demandops.db`) for both prediction logging (quality tracking) and API key storage.

**Why:** No external infrastructure required, portable, sufficient for demo scale. SQLite's WAL mode supports the concurrent read/write pattern (prediction logging during serving, actuals submission, quality queries). A single database avoids managing two files while the tables have no schema overlap.

## 28. Intentionally Vague 401 Response

**Decision:** Authentication failures return `{"detail": "Invalid or inactive API key"}` regardless of whether the key doesn't exist or has been revoked.

**Why:** Distinguishing between "key not found" and "key revoked" enables oracle attacks — an attacker can probe to determine which keys exist. A single vague message prevents this information leak.

## 29. KS Test Uses 5,000-Point Subsample

**Decision:** The KS two-sample test compares incoming data against a stratified subsample of 5,000 training values per feature, not the full training set.

**Why:** `ks_2samp` needs the actual training sample values (not just summary statistics like PSI's quantile boundaries). Storing the full training feature vectors would bloat the reference artifact significantly (3,207 series × thousands of rows). A 5,000-point subsample is sufficient for a stable KS test while keeping the artifact under 5MB. The subsample size is recorded in artifact metadata for reproducibility.

## 30. On-Demand Drift Computation

**Decision:** Drift metrics are computed only when `GET /monitoring/drift` is called. Feature vectors accumulate passively in a bounded deque (maxlen=1000) — no background threads, no periodic timers.

**Why:** Continuous monitoring with background threads adds concurrency complexity (thread-safe accumulation, periodic scheduling, timer management) to a serving layer that is otherwise stateless. The on-demand approach gives the same cross-request drift picture without the complexity. The deque accumulates samples passively from all prediction requests; the drift endpoint is the computation trigger. Simpler to test, no concurrency bugs beyond the deque's own thread-safe append.
```

**Step 2: Add to configs/default.yaml**

```yaml
# After the existing serving section:
db:
  path: data/demandops.db

monitoring:
  drift:
    min_samples: 100
    maxlen: 1000
  quality:
    min_matched_pairs: 10
    mae_alert_margin: 1.2
```

And add to the artifacts section:
```yaml
  reference_distributions_path: artifacts/reference_distributions.json
```

**Step 3: Update README.md**

Add a "Monitoring & Security" section after the existing API Documentation section. Include:
- Drift detection methodology (PSI + KS + correlation matrix)
- Quality monitoring workflow
- Auth setup instructions (`python -m demandops.manage_keys create`)
- Endpoint table

**Step 4: Run full test suite**

Run: `pytest tests/ -v`
Expected: all PASS

**Step 5: Run linter**

Run: `ruff check demandops/ tests/`
Run: `ruff format --check demandops/ tests/`
Expected: both PASS (fix any issues before committing)

**Step 6: Commit**

```bash
git add DECISIONS.md README.md configs/default.yaml
git commit -m "docs: add DECISIONS #24-30, monitoring/security README section

Seven new design decisions: PSI+KS+correlation (not autoencoder),
sMAPE over MAPE, zone_id exclusion from correlation, SQLite choice,
vague 401, KS subsample, on-demand drift. README documents monitoring
methodology, quality workflow, and auth setup."
```
