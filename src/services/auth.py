"""
Authentication service: password hashing, DB-backed sessions, and the FastAPI
dependencies that protect routes.

Auth is always enabled (decision 6b). On boot — and lazily before login —
:func:`ensure_seeded` makes sure an admin user exists: it seeds ``admin``/
``admin`` when the ``users`` table is empty, and re-applies ``DOKKAI_ROOT_USER``
/ ``DOKKAI_ROOT_PASSWORD`` (when set) onto user id=1 if they differ from its
current credentials, invalidating that user's sessions (decision 6j).

Passwords are hashed with stdlib ``hashlib.scrypt`` (decision 6h) using a
self-describing stored format: ``scrypt$n$r$p$<salt-b64>$<hash-b64>``. Session
tokens are opaque (``secrets.token_urlsafe``), stored SHA-256-hashed, 30-day
TTL (decisions 6a/6g).

Endpoints live in ``controllers/auth.py``; this module only provides the
service functions and dependencies they and the rest of the API build on.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from services.db import DatabaseUnavailableError, get_pool

_SCRYPT_N = 16384
_SCRYPT_R = 8
_SCRYPT_P = 1
_SALT_BYTES = 16
_DKLEN = 64

_SESSION_TTL = timedelta(days=30)

_bearer_scheme = HTTPBearer(auto_error=False)


# -----------------------------------------------------------------------
# Password hashing
# -----------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Hash *password* with scrypt, returning a self-describing stored string."""
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_DKLEN
    )
    return (
        f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}"
        f"${base64.b64encode(salt).decode('ascii')}${base64.b64encode(digest).decode('ascii')}"
    )


def verify_password(password: str, stored: str) -> bool:
    """
    Check *password* against a stored scrypt hash. Never raises — a
    malformed/tampered *stored* value is treated as a non-match.
    """
    try:
        scheme, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        digest = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected)
        )
        return hmac.compare_digest(digest, expected)
    except Exception:
        return False


# -----------------------------------------------------------------------
# Admin seeding (decisions 6b, 6j)
# -----------------------------------------------------------------------

async def ensure_seeded() -> None:
    """
    Idempotent admin bootstrap. Safe to call on every boot and lazily before
    every login.

    - Empty ``users`` table: insert one admin user, using
      ``DOKKAI_ROOT_USER``/``DOKKAI_ROOT_PASSWORD`` when set, else the
      ``admin``/``admin`` fallback.
    - Non-empty table + either env var set: if user id=1's username or
      password differs from the env values, update it and delete its
      sessions (old tokens die with the old password).
    """
    pool = await get_pool()
    root_user = os.getenv("DOKKAI_ROOT_USER")
    root_password = os.getenv("DOKKAI_ROOT_PASSWORD")

    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT count(*) FROM users")
        if count == 0:
            username = root_user or "admin"
            password = root_password or "admin"
            # ON CONFLICT: two concurrent first-boot seeds must not 500 —
            # the loser of the unique(username) race becomes a no-op.
            await conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES ($1, $2, 'admin') "
                "ON CONFLICT (username) DO NOTHING",
                username,
                hash_password(password),
            )
            return

        if not root_user and not root_password:
            return

        row = await conn.fetchrow("SELECT username, password_hash FROM users WHERE id = 1")
        if row is None:
            return

        new_username = root_user or row["username"]
        username_changed = new_username != row["username"]
        password_changed = bool(root_password) and not verify_password(root_password, row["password_hash"])
        if not username_changed and not password_changed:
            return

        new_hash = hash_password(root_password) if password_changed else row["password_hash"]
        async with conn.transaction():  # rotate credentials and sessions atomically
            await conn.execute(
                "UPDATE users SET username = $1, password_hash = $2, updated_at = now() WHERE id = 1",
                new_username,
                new_hash,
            )
            await conn.execute("DELETE FROM sessions WHERE user_id = 1")


async def is_default_admin_active() -> bool:
    """True if any user named ``admin`` still verifies against password ``admin``."""
    pool = await get_pool()
    rows = await pool.fetch("SELECT password_hash FROM users WHERE username = 'admin'")
    return any(verify_password("admin", row["password_hash"]) for row in rows)


# -----------------------------------------------------------------------
# Sessions
# -----------------------------------------------------------------------

def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def create_session(user_id: int) -> tuple[str, datetime]:
    """Create a session for *user_id*, returning the plaintext token (given
    out once) and its expiry."""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + _SESSION_TTL
    pool = await get_pool()
    await pool.execute(
        "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES ($1, $2, $3)",
        _hash_token(token),
        user_id,
        expires_at,
    )
    return token, expires_at


async def resolve_session(token: str) -> dict | None:
    """Look up the user for a bearer *token*, or ``None`` if missing/expired."""
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        SELECT u.id, u.username, u.role, s.expires_at
        FROM sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.token_hash = $1
        """,
        _hash_token(token),
    )
    if row is None:
        return None
    if row["expires_at"] < datetime.now(timezone.utc):
        return None
    return dict(row)


async def delete_session(token: str) -> bool:
    """Delete the session for *token*. Returns whether a row was deleted."""
    pool = await get_pool()
    result = await pool.execute("DELETE FROM sessions WHERE token_hash = $1", _hash_token(token))
    return result.split(" ")[-1] != "0"


async def purge_expired() -> None:
    """Delete all expired sessions. Called opportunistically at login."""
    pool = await get_pool()
    await pool.execute("DELETE FROM sessions WHERE expires_at < now()")


async def delete_user_sessions(user_id: int) -> None:
    """Delete all sessions belonging to *user_id*."""
    pool = await get_pool()
    await pool.execute("DELETE FROM sessions WHERE user_id = $1", user_id)


# -----------------------------------------------------------------------
# User CRUD primitives
# -----------------------------------------------------------------------

async def list_users() -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch("SELECT id, username, role, created_at, updated_at FROM users ORDER BY id")
    return [dict(row) for row in rows]


async def create_user(username: str, password: str, role: str = "user") -> dict:
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO users (username, password_hash, role)
        VALUES ($1, $2, $3)
        RETURNING id, username, role, created_at, updated_at
        """,
        username,
        hash_password(password),
        role,
    )
    return dict(row)


async def delete_user(user_id: int) -> bool:
    pool = await get_pool()
    result = await pool.execute("DELETE FROM users WHERE id = $1", user_id)
    return result.split(" ")[-1] != "0"


async def get_user_by_username(username: str) -> dict | None:
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT id, username, password_hash, role FROM users WHERE username = $1", username
    )
    return dict(row) if row else None


async def update_password(user_id: int, new_password: str, keep_token: str | None = None) -> None:
    """
    Set a new password for *user_id*.

    Drops all of the user's sessions except the one identified by
    *keep_token* (if given) — used when a user changes their own password:
    every OTHER session dies, but the session making the request stays
    valid (decision 6d).
    """
    pool = await get_pool()
    await pool.execute(
        "UPDATE users SET password_hash = $1, updated_at = now() WHERE id = $2",
        hash_password(new_password),
        user_id,
    )
    if keep_token is None:
        await delete_user_sessions(user_id)
    else:
        await pool.execute(
            "DELETE FROM sessions WHERE user_id = $1 AND token_hash != $2",
            user_id,
            _hash_token(keep_token),
        )


# -----------------------------------------------------------------------
# FastAPI dependencies
# -----------------------------------------------------------------------

async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict:
    """
    Resolve the bearer token into a user, attaching it to ``request.state.user``.

    401 (with ``WWW-Authenticate: Bearer``) on a missing or invalid/expired
    token. A dead Postgres connection surfaces as 503 here, since this runs
    as a router-level dependency before any controller gets a chance to map
    :class:`DatabaseUnavailableError` itself.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=401, detail="missing bearer token", headers={"WWW-Authenticate": "Bearer"}
        )
    try:
        user = await resolve_session(credentials.credentials)
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if user is None:
        raise HTTPException(
            status_code=401, detail="invalid or expired session", headers={"WWW-Authenticate": "Bearer"}
        )
    request.state.user = user
    request.state.token = credentials.credentials
    return user


def require_role(*roles: str):
    """Dependency factory: 403 unless the authenticated user's role is in *roles*."""

    async def _check(user: dict = Depends(require_auth)) -> dict:
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail=f"role '{user['role']}' cannot perform this action")
        return user

    return _check
