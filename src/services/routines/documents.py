"""
Postgres persistence for playbooks and skills (``playbooks`` + ``skills``
tables, migration 007) — the two document types a routine run can attach to
its prompt (decision 9d):

  - Playbooks are PUSHED into every run of the routine kinds listed in their
    ``routines`` column.
  - Skills are PULLED by the model at its own discretion, choosing from a
    catalog of ``{name, description}`` by relevance.

Both are markdown with OPTIONAL YAML frontmatter (a leading
``---\\n...\\n---\\n`` block). This module does NOT parse that frontmatter —
name/description/routines come from the API fields on the functions below,
not from frontmatter, which is stored and returned verbatim as user
convenience/metadata. :func:`split_frontmatter` only *detects and
separates* it, so a later prompt-building step can feed the model the BODY
alone when frontmatter is present.

Same async/sync split as ``services.routines.store`` (see that module's
docstring for the full reasoning):

  - The async functions (``list_*``, ``get_*``, ``create_*``, ``update_*``,
    ``delete_*``) run on the server's MAIN event loop — used by the future
    documents controller (B2). They use the shared app pool
    (``services.db.get_pool()``). A duplicate ``name`` raises
    ``asyncpg.UniqueViolationError`` uncaught (mirroring
    ``controllers/auth.py``'s ``create_user`` — the controller maps it to a
    409).
  - The sync ``fetch_*_sync`` readers run inside a routine's worker THREAD
    (the engine, wired in a later step) via a short-lived direct
    ``asyncpg.connect()`` (the ``_direct_conn`` pattern), so they never touch
    the main loop's pool.
"""

from __future__ import annotations

import asyncio
import logging
import re

import asyncpg

from services.db import _database_url, get_pool

logger = logging.getLogger(__name__)

_MAX_CONTENT_BYTES = 16_384
_MAX_NAME_LEN = 128

_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?\r?\n)?---\r?\n", re.DOTALL)


def split_frontmatter(content: str) -> tuple[str | None, str]:
    """
    Split a leading YAML frontmatter block (``---\\n...\\n---\\n``) off
    *content*, returning ``(frontmatter_text, body)``.

    ``frontmatter_text`` is the raw text BETWEEN the ``---`` markers (not
    parsed as YAML) — or ``None`` when *content* has no leading frontmatter
    block, in which case ``body`` is *content* unchanged. An empty block
    (``---\\n---\\n``) yields ``frontmatter_text == ""`` (frontmatter is
    present, just blank).
    """
    match = _FRONTMATTER_RE.match(content)
    if match is None:
        return None, content
    return match.group(1) or "", content[match.end():]


_VALID_ROUTINES = ("review", "bughunt")


def _validate_routines(routines: list[str]) -> None:
    """Reject unknown routine kinds — a typo would otherwise be stored
    silently and simply never match at playbook-injection time."""
    for r in routines:
        if r not in _VALID_ROUTINES:
            raise ValueError(
                f"unknown routine kind '{r}' — valid kinds: {', '.join(_VALID_ROUTINES)}"
            )


def _validate_name(name: str) -> None:
    if not name or not name.strip():
        raise ValueError("name must not be empty")
    if name != name.strip():
        raise ValueError("name must not have leading/trailing whitespace")
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(f"name exceeds the {_MAX_NAME_LEN}-character limit ({len(name)} chars)")
    # '/' and '\' break the REST GET/PATCH/DELETE-by-name path param (FastAPI
    # won't match a literal slash there, even URL-encoded as %2F); control
    # characters are similarly hostile to a path segment. Reject both so a
    # name is always reachable after creation.
    if any(c in ("/", "\\") or ord(c) < 32 or ord(c) == 127 for c in name):
        raise ValueError("name must not contain slashes or control characters")


def _validate_content(content: str, label: str) -> None:
    size = len(content.encode("utf-8"))
    if size > _MAX_CONTENT_BYTES:
        raise ValueError(f"{label} content exceeds the 16KB limit ({size} bytes)")


def _validate_description(description: str) -> None:
    if not description or not description.strip():
        raise ValueError("description must not be empty")


def _row_to_playbook_dict(row: asyncpg.Record) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "routines": list(row["routines"]),
        "content": row["content"],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }


def _row_to_skill_dict(row: asyncpg.Record) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "content": row["content"],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }


# -------------------------------------------------------------------------
# Async API — main event loop (future documents controller, B2)
# -------------------------------------------------------------------------

async def list_playbooks() -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch("SELECT * FROM playbooks ORDER BY name")
    return [_row_to_playbook_dict(r) for r in rows]


async def get_playbook(name: str) -> dict | None:
    pool = await get_pool()
    row = await pool.fetchrow("SELECT * FROM playbooks WHERE name = $1", name)
    return _row_to_playbook_dict(row) if row is not None else None


async def create_playbook(name: str, content: str, routines: list[str] | None = None) -> dict:
    _validate_name(name)
    _validate_content(content, "playbook")
    pool = await get_pool()
    if routines is not None:
        _validate_routines(routines)
    if routines is None:
        row = await pool.fetchrow(
            "INSERT INTO playbooks (name, content) VALUES ($1, $2) RETURNING *",
            name,
            content,
        )
    else:
        row = await pool.fetchrow(
            "INSERT INTO playbooks (name, content, routines) VALUES ($1, $2, $3) RETURNING *",
            name,
            content,
            routines,
        )
    return _row_to_playbook_dict(row)


async def update_playbook(
    name: str, content: str | None = None, routines: list[str] | None = None
) -> dict | None:
    """Update the given fields of the playbook named *name*. Fields left
    ``None`` are left unchanged. Returns ``None`` if no playbook has that
    name."""
    if content is None and routines is None:
        return await get_playbook(name)
    if content is not None:
        _validate_content(content, "playbook")
    pool = await get_pool()
    sets = ["updated_at = now()"]
    args: list = []
    if content is not None:
        args.append(content)
        sets.append(f"content = ${len(args)}")
    if routines is not None:
        _validate_routines(routines)
        args.append(routines)
        sets.append(f"routines = ${len(args)}")
    args.append(name)
    row = await pool.fetchrow(
        f"UPDATE playbooks SET {', '.join(sets)} WHERE name = ${len(args)} RETURNING *",
        *args,
    )
    return _row_to_playbook_dict(row) if row is not None else None


async def delete_playbook(name: str) -> bool:
    pool = await get_pool()
    result = await pool.execute("DELETE FROM playbooks WHERE name = $1", name)
    return result == "DELETE 1"


async def list_skills() -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch("SELECT * FROM skills ORDER BY name")
    return [_row_to_skill_dict(r) for r in rows]


async def get_skill(name: str) -> dict | None:
    pool = await get_pool()
    row = await pool.fetchrow("SELECT * FROM skills WHERE name = $1", name)
    return _row_to_skill_dict(row) if row is not None else None


async def create_skill(name: str, description: str, content: str) -> dict:
    _validate_name(name)
    _validate_description(description)
    _validate_content(content, "skill")
    pool = await get_pool()
    row = await pool.fetchrow(
        "INSERT INTO skills (name, description, content) VALUES ($1, $2, $3) RETURNING *",
        name,
        description,
        content,
    )
    return _row_to_skill_dict(row)


async def update_skill(
    name: str, description: str | None = None, content: str | None = None
) -> dict | None:
    """Update the given fields of the skill named *name*. Fields left
    ``None`` are left unchanged. Returns ``None`` if no skill has that
    name."""
    if description is None and content is None:
        return await get_skill(name)
    if description is not None:
        _validate_description(description)
    if content is not None:
        _validate_content(content, "skill")
    pool = await get_pool()
    sets = ["updated_at = now()"]
    args: list = []
    if description is not None:
        args.append(description)
        sets.append(f"description = ${len(args)}")
    if content is not None:
        args.append(content)
        sets.append(f"content = ${len(args)}")
    args.append(name)
    row = await pool.fetchrow(
        f"UPDATE skills SET {', '.join(sets)} WHERE name = ${len(args)} RETURNING *",
        *args,
    )
    return _row_to_skill_dict(row) if row is not None else None


async def delete_skill(name: str) -> bool:
    pool = await get_pool()
    result = await pool.execute("DELETE FROM skills WHERE name = $1", name)
    return result == "DELETE 1"


# -------------------------------------------------------------------------
# Sync API — worker thread (direct connection per call), for the engine
# -------------------------------------------------------------------------

async def _direct_conn() -> asyncpg.Connection | None:
    try:
        return await asyncpg.connect(_database_url())
    except Exception as e:
        logger.warning("routine documents: could not reach Postgres: %s", e)
        return None


async def _fetch_playbooks_async(names: list[str]) -> list[dict]:
    conn = await _direct_conn()
    if conn is None:
        return []
    try:
        rows = await conn.fetch("SELECT * FROM playbooks WHERE name = ANY($1::text[])", names)
        by_name = {r["name"]: _row_to_playbook_dict(r) for r in rows}
        return [by_name[n] for n in names if n in by_name]
    except Exception as e:
        logger.warning("routine documents: failed to fetch playbooks %s: %s", names, e)
        return []
    finally:
        await conn.close()


def fetch_playbooks_sync(names: list[str]) -> list[dict]:
    """
    Fetch the named playbooks, in the SAME order as *names* — selection
    order is priority order (decision 9d). Unknown names are silently
    skipped.
    """
    if not names:
        return []
    return asyncio.run(_fetch_playbooks_async(names))


async def _fetch_skills_catalog_async() -> list[dict]:
    conn = await _direct_conn()
    if conn is None:
        return []
    try:
        rows = await conn.fetch("SELECT name, description FROM skills ORDER BY name")
        return [{"name": r["name"], "description": r["description"]} for r in rows]
    except Exception as e:
        logger.warning("routine documents: failed to fetch skills catalog: %s", e)
        return []
    finally:
        await conn.close()


def fetch_skills_catalog_sync() -> list[dict]:
    """The full ``{name, description}`` catalog, for the model to pull from
    by relevance — never includes ``content`` (see :func:`fetch_skill_sync`
    for that)."""
    return asyncio.run(_fetch_skills_catalog_async())


async def _fetch_skill_async(name: str) -> dict | None:
    conn = await _direct_conn()
    if conn is None:
        return None
    try:
        row = await conn.fetchrow("SELECT * FROM skills WHERE name = $1", name)
        return _row_to_skill_dict(row) if row is not None else None
    except Exception as e:
        logger.warning("routine documents: failed to fetch skill %s: %s", name, e)
        return None
    finally:
        await conn.close()


def fetch_skill_sync(name: str) -> dict | None:
    return asyncio.run(_fetch_skill_async(name))
