"""API key authentication, per-client rate limiting, and usage tracking."""

from __future__ import annotations

import collections
import hashlib
import sqlite3
import threading
import time
from datetime import date

from fastapi import HTTPException, Request


def hash_key(raw_key: str) -> str:
    """SHA-256 hash of an API key. Never store or log the raw key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


class RateLimiter:
    """In-memory sliding window rate limiter. Resets on restart."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._windows: dict[str, collections.deque[float]] = {}

    def check(self, client_name: str, rate_limit: int) -> bool:
        """Return True if the request is allowed, False if rate limit exceeded."""
        now = time.time()
        cutoff = now - 60
        with self._lock:
            dq = self._windows.get(client_name)
            if dq is None:
                dq = collections.deque()
                self._windows[client_name] = dq
            # Evict expired timestamps from the left (oldest first)
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= rate_limit:
                return False
            dq.append(now)
            return True


async def requires_auth(request: Request) -> dict:
    """FastAPI dependency: validate Bearer token, check rate limit.

    Returns dict with client_name, rate_limit, max_batch_size.
    Raises 401 for invalid/inactive keys, 429 for rate limit exceeded.
    """
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    raw_key = auth_header[7:]  # len("bearer ") == len("Bearer ") == 7
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


def log_usage(
    db: sqlite3.Connection,
    client_name: str,
    endpoint: str,
    record_count: int = 1,
) -> None:
    """Increment usage counters for a client. Called after successful prediction."""
    today = date.today().isoformat()
    db.execute(
        "INSERT INTO usage_log (client_name, endpoint, date, request_count, total_records) "
        "VALUES (?, ?, ?, 1, ?) "
        "ON CONFLICT(client_name, endpoint, date) DO UPDATE SET "
        "request_count = request_count + 1, total_records = total_records + excluded.total_records",
        (client_name, endpoint, today, record_count),
    )
    db.commit()


def get_usage(
    db: sqlite3.Connection,
    client_name: str | None = None,
) -> list[dict]:
    """Return usage stats, optionally filtered by client."""
    if client_name:
        rows = db.execute(
            "SELECT client_name, endpoint, date, request_count, total_records "
            "FROM usage_log WHERE client_name = ? ORDER BY date DESC",
            (client_name,),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT client_name, endpoint, date, request_count, total_records "
            "FROM usage_log ORDER BY date DESC"
        ).fetchall()
    return [
        {
            "client_name": r[0],
            "endpoint": r[1],
            "date": r[2],
            "request_count": r[3],
            "total_records": r[4],
        }
        for r in rows
    ]
