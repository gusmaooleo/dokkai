"""
Postgres persistence for code review routines (``routine_runs`` + ``findings``).

Started with just the boot-sweep helper for now — the review/bughunt engine
and its endpoints land in later steps and will grow this module.
"""

from __future__ import annotations

from services.db import get_pool


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
