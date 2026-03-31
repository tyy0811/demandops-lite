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
