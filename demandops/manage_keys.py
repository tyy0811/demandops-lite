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
    """Create a new API key. Returns the raw key (shown only once).

    If a key already exists for this client, the old key is revoked first
    (one active key per client). Raises ValueError for non-positive limits.
    """
    if rate_limit <= 0:
        raise ValueError(f"rate_limit must be positive, got {rate_limit}")
    if max_batch_size <= 0:
        raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")

    # Revoke any existing key for this client (rotation)
    existing = db.execute(
        "SELECT key_hash FROM api_keys WHERE client_name = ?", (client_name,)
    ).fetchone()
    if existing:
        db.execute("DELETE FROM api_keys WHERE client_name = ?", (client_name,))

    raw_key = secrets.token_urlsafe(32)
    key_hash = hash_key(raw_key)
    db.execute(
        "INSERT INTO api_keys (key_hash, client_name, created_at, rate_limit, max_batch_size, is_active) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            key_hash,
            client_name,
            datetime.now(timezone.utc).isoformat(),
            rate_limit,
            max_batch_size,
            True,
        ),
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
    db.execute("UPDATE api_keys SET is_active = 0 WHERE client_name = ?", (client_name,))
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
            print(
                f"  {k['client_name']}: rate_limit={k['rate_limit']}, "
                f"max_batch={k['max_batch_size']}, {status}, created={k['created_at']}"
            )
    elif args.command == "revoke":
        revoke_key(db, args.client)
        print(f"Key revoked for '{args.client}'")

    db.close()


if __name__ == "__main__":
    main()
