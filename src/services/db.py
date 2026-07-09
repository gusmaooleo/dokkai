"""
Postgres connection layer for Dokkai.

Owns a single asyncpg connection pool built from ``DATABASE_URL`` and applies
pending SQL migrations from ``src/db/migrations/`` on boot. Conversation
history and future DB-backed features go through :func:`get_pool`.

Environment variables
----------------------
DATABASE_URL - Postgres connection string
               (default ``postgresql://dokkai:dokkai@localhost:5432/dokkai``)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path

import asyncpg

logger = logging.getLogger(__name__)

_DEFAULT_DATABASE_URL = "postgresql://dokkai:dokkai@localhost:5432/dokkai"
_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "db" / "migrations"

_pool: asyncpg.Pool | None = None
_init_lock: asyncio.Lock = asyncio.Lock()


class DatabaseUnavailableError(RuntimeError):
    """Raised when there is no usable Postgres connection pool."""


def _database_url() -> str:
    return os.getenv("DATABASE_URL", _DEFAULT_DATABASE_URL)


def _display_url() -> str:
    """The connection URL with any password masked, safe for error text."""
    return re.sub(r"://([^:/@]+):[^@]*@", r"://\1:***@", _database_url())


async def apply_migrations(conn: asyncpg.Connection) -> None:
    """
    Apply every unapplied ``*.sql`` file under ``src/db/migrations/``, in
    lexical order, each inside its own transaction.

    Tracks applied versions in a ``schema_migrations`` table (created here if
    missing) keyed by filename, so re-running on a later boot is a no-op.
    """
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )

    applied = {
        row["version"]
        for row in await conn.fetch("SELECT version FROM schema_migrations")
    }

    for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        if path.name in applied:
            continue
        sql = path.read_text(encoding="utf-8")
        async with conn.transaction():
            await conn.execute(sql)
            await conn.execute(
                "INSERT INTO schema_migrations (version) VALUES ($1)", path.name
            )
        logger.info("applied migration %s", path.name)


async def init_db() -> None:
    """
    Connect to Postgres and apply pending migrations.

    Sets the module-level pool on success. Raises whatever asyncpg raises
    (e.g. connection refused) on failure — callers decide how to degrade.
    """
    global _pool
    pool = await asyncpg.create_pool(_database_url())
    try:
        async with pool.acquire() as conn:
            await apply_migrations(conn)
    except Exception:
        # A broken migration must not leak the freshly created pool.
        await pool.close()
        raise
    _pool = pool


async def get_pool() -> asyncpg.Pool:
    """
    Return the active connection pool.

    If no pool is set (Postgres was down at boot, or a previous attempt
    failed), makes one lazy reconnect attempt before giving up. Raises
    :class:`DatabaseUnavailableError` when Postgres still can't be reached.
    """
    global _pool
    if _pool is None:
        async with _init_lock:  # concurrent cold starts must not double-create
            if _pool is None:
                try:
                    await init_db()
                except Exception as e:
                    logger.warning("lazy reconnect to Postgres failed: %s", e)

    if _pool is None:
        raise DatabaseUnavailableError(
            f"cannot connect to Postgres at {_display_url()} — conversation "
            "history is unavailable; start it with 'docker compose up -d'"
        )

    return _pool
