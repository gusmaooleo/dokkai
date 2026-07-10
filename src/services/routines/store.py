"""
Postgres persistence for code review routines (``routine_runs`` + ``findings``).

Two write paths, same reasoning as ``services.jobs`` (see that module's
docstring):

  - The async functions (``create_run``, ``get_run``, ``list_runs``,
    ``delete_run``) run on the server's MAIN event loop — used by the
    controller (A6) and by ``services.routines.engine.submit_routine``,
    which is itself awaited from the main loop. They use the shared app
    pool (``services.db.get_pool()``).
  - The sync functions (``mark_run_running``, ``mark_run_done``,
    ``mark_run_failed``, ``insert_findings``) run inside a job's worker
    THREAD, which owns its own private event loop per call
    (``asyncio.run``, mirroring ``services.jobs._persist_job_direct`` /
    ``_persist_job_from_worker``) and must never touch the main loop's
    pool — each opens a short-lived direct ``asyncpg.connect()``.

Both the sync functions and their async internals are best-effort: a
Postgres failure is logged as a warning and swallowed (``insert_findings``
returns ``False`` instead, so the caller can decide how to react — see
``services.routines.engine._make_runner`` for the failure semantics built
on top of that).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

import asyncpg

from services.db import _database_url, get_pool

logger = logging.getLogger(__name__)


def _row_to_run_dict(row: asyncpg.Record) -> dict:
    params = row["params"]
    stats = row["stats"]
    return {
        "id": str(row["id"]),
        "job_id": str(row["job_id"]),
        "kind": row["kind"],
        "project": row["project"],
        "status": row["status"],
        "params": params if isinstance(params, dict) else json.loads(params) if params is not None else {},
        "summary": row["summary"],
        "stats": stats if isinstance(stats, dict) or stats is None else json.loads(stats),
        "error": row["error"],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }


# -------------------------------------------------------------------------
# Async API — main event loop (controller / submit_routine)
# -------------------------------------------------------------------------

async def create_run(job_id: str, kind: str, project: str, params: dict) -> dict:
    """
    Insert a new ``routine_runs`` row (status ``queued``) and return it.

    Called from ``routines.engine.submit_routine`` BEFORE the job is
    registered — a Postgres failure here raises and propagates uncaught, so
    the controller (A6) can fail loud with a 503 (6i pattern): no job is
    ever created for a run that couldn't be recorded.
    """
    pool = await get_pool()
    run_id = str(uuid.uuid4())
    row = await pool.fetchrow(
        """
        INSERT INTO routine_runs (id, job_id, kind, project, status, params)
        VALUES ($1, $2, $3, $4, 'queued', $5::jsonb)
        RETURNING *
        """,
        uuid.UUID(run_id),
        uuid.UUID(job_id),
        kind,
        project,
        json.dumps(params or {}),
    )
    return _row_to_run_dict(row)


async def get_run(run_id: str) -> dict | None:
    pool = await get_pool()
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        return None
    row = await pool.fetchrow("SELECT * FROM routine_runs WHERE id = $1", run_uuid)
    return _row_to_run_dict(row) if row is not None else None


async def list_runs(project: str | None = None, kind: str | None = None, limit: int = 50) -> list[dict]:
    pool = await get_pool()
    clauses = []
    args: list = []
    if project is not None:
        args.append(project)
        clauses.append(f"project = ${len(args)}")
    if kind is not None:
        args.append(kind)
        clauses.append(f"kind = ${len(args)}")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    args.append(limit)
    rows = await pool.fetch(
        f"SELECT * FROM routine_runs {where} ORDER BY created_at DESC LIMIT ${len(args)}",
        *args,
    )
    return [_row_to_run_dict(r) for r in rows]


async def delete_run(run_id: str) -> bool:
    """Delete a run, cascade-wiping its findings (FK ON DELETE CASCADE)."""
    pool = await get_pool()
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        return False
    result = await pool.execute("DELETE FROM routine_runs WHERE id = $1", run_uuid)
    return result == "DELETE 1"


# -------------------------------------------------------------------------
# Sync API — worker thread (direct connection per call)
# -------------------------------------------------------------------------

async def _direct_conn() -> asyncpg.Connection | None:
    try:
        return await asyncpg.connect(_database_url())
    except Exception as e:
        logger.warning("routine run: could not reach Postgres: %s", e)
        return None


async def _mark_run_running_async(run_id: str) -> None:
    conn = await _direct_conn()
    if conn is None:
        return
    try:
        await conn.execute(
            "UPDATE routine_runs SET status = 'running', updated_at = now() WHERE id = $1",
            uuid.UUID(run_id),
        )
    except Exception as e:
        logger.warning("routine run: failed to mark %s running: %s", run_id, e)
    finally:
        await conn.close()


def mark_run_running(run_id: str) -> None:
    asyncio.run(_mark_run_running_async(run_id))


async def _mark_run_done_async(run_id: str, summary: str | None, stats: dict | None) -> None:
    conn = await _direct_conn()
    if conn is None:
        return
    try:
        await conn.execute(
            """
            UPDATE routine_runs
            SET status = 'done', summary = $2, stats = $3::jsonb, updated_at = now()
            WHERE id = $1
            """,
            uuid.UUID(run_id),
            summary,
            json.dumps(stats) if stats is not None else None,
        )
    except Exception as e:
        logger.warning("routine run: failed to mark %s done: %s", run_id, e)
    finally:
        await conn.close()


def mark_run_done(run_id: str, summary: str | None = None, stats: dict | None = None) -> None:
    asyncio.run(_mark_run_done_async(run_id, summary, stats))


async def _mark_run_failed_async(run_id: str, error: str) -> None:
    conn = await _direct_conn()
    if conn is None:
        return
    try:
        await conn.execute(
            "UPDATE routine_runs SET status = 'failed', error = $2, updated_at = now() WHERE id = $1",
            uuid.UUID(run_id),
            error,
        )
    except Exception as e:
        logger.warning("routine run: failed to mark %s failed: %s", run_id, e)
    finally:
        await conn.close()


def mark_run_failed(run_id: str, error: str) -> None:
    asyncio.run(_mark_run_failed_async(run_id, error))


async def _insert_findings_async(run_id: str, findings: list[dict]) -> bool:
    conn = await _direct_conn()
    if conn is None:
        return False
    try:
        run_uuid = uuid.UUID(run_id)
        rows = [
            (
                run_uuid,
                f["file_path"],
                f.get("start_line"),
                f.get("end_line"),
                f["severity"],
                f["category"],
                f["title"],
                f["body"],
                f.get("suggestion"),
                json.dumps(f["evidence"]) if f.get("evidence") is not None else None,
                f.get("anchored", True),
            )
            for f in findings
        ]
        await conn.executemany(
            """
            INSERT INTO findings
                (run_id, file_path, start_line, end_line, severity, category,
                 title, body, suggestion, evidence, anchored)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11)
            """,
            rows,
        )
        return True
    except Exception as e:
        logger.warning("routine run: failed to insert findings for %s: %s", run_id, e)
        return False
    finally:
        await conn.close()


def insert_findings(run_id: str, findings: list[dict]) -> bool:
    """
    Batched insert of a run's findings. Returns ``True`` if there was
    nothing to insert or the insert succeeded, ``False`` on any Postgres
    failure (unreachable, or a bad row — e.g. a constraint violation) so the
    caller can react (see ``routines.engine._make_runner``).
    """
    if not findings:
        return True
    return asyncio.run(_insert_findings_async(run_id, findings))


async def sweep_interrupted_routine_runs() -> None:
    """
    Mark any routine run left 'queued' or 'running' from a previous server
    run as failed — a crash or restart killed its worker, so it will never
    reach a terminal state on its own. Call once at boot, after init_db()
    succeeds, alongside ``services.jobs.sweep_interrupted_jobs``.
    """
    pool = await get_pool()
    await pool.execute(
        """
        UPDATE routine_runs
        SET status = 'failed',
            error = 'interrupted by server restart',
            updated_at = now()
        WHERE status IN ('queued', 'running')
        """
    )
