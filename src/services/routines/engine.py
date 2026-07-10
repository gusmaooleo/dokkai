"""
Routine execution engine — the generic plumbing shared by the ``review`` and
``bughunt`` job kinds, built on top of the existing job system.

Dependency direction: this module imports ``services.jobs`` (``Job``, the
event helper, the generic external-job submit path) and
``services.routines.store`` (routine_runs/findings persistence);
``services.jobs`` imports neither — routines knows about jobs, never the
other way around. The mechanism chosen for "least invasive" (per the A3
plan) is a **submit-with-runner API**: ``jobs.submit_external`` takes a
plain sync ``runner(job, emit)`` callable built here rather than a
kind→callable registry — ``jobs.py`` stays fully agnostic of what
"review"/"bughunt" even mean, managing only generic Job lifecycle exactly as
it already does for pipeline/refresh/graph.

The actual review/bug-hunt logic lands in A4/A5; this module only provides
the plumbing that any ``RoutineCallable`` plugs into (see
``scripts/test_routine_engine.py`` for a callable exercising it end to end).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Awaitable, Callable

from services import jobs
from services.jobs import Job
from services.routines import store

logger = logging.getLogger(__name__)

# The actual routine body: given the job, its run id, the launch params, and
# an ``emit(stage, message)`` callback for step events, does the work and
# returns a result dict — optionally with "summary" (str), "stats" (dict),
# and "findings" (list[dict], shaped for ``store.insert_findings``) keys,
# all optional.
RoutineCallable = Callable[[Job, str, dict, Callable[[str, str], None]], Awaitable[dict | None]]


def _make_runner(run_id: str, core: RoutineCallable, params: dict) -> jobs.ExternalRunner:
    """
    Build the sync ``runner(job, emit)`` handed to ``jobs.submit_external``.

    Owns the ``routine_runs`` bookkeeping around ``core``: marks the run
    running before, done/failed after. Failure semantics:

      - ``core`` raising is a genuine routine failure: the run is marked
        failed with that error and the exception is RE-RAISED, so
        ``jobs._run_external_sync`` also fails the job — the job must
        reflect a real failure of the routine logic.
      - Postgres being unreachable for the running/done/failed bookkeeping
        calls themselves degrades with a logged warning (inside
        ``store``'s sync functions) and does NOT fail the job — a
        persistence hiccup on the run row must not mask a routine that
        actually completed.
      - The one deliberate exception: if ``core`` succeeds but its findings
        can't be persisted, the RUN is marked failed (the findings are the
        routine's actual deliverable — losing them is a real failure of the
        run's output) but the JOB is left to succeed, since the routine
        logic itself did not error and its result still rides along in the
        job's own terminal snapshot for debugging.
    """

    def _runner(job: Job, emit: Callable[[str, str], None]) -> None:
        store.mark_run_running(run_id)
        try:
            result = asyncio.run(core(job, run_id, params, emit))
        except Exception as e:
            store.mark_run_failed(run_id, f"{type(e).__name__}: {e}")
            raise

        result = result or {}
        findings = result.get("findings") or []
        if store.insert_findings(run_id, findings):
            store.mark_run_done(run_id, summary=result.get("summary"), stats=result.get("stats"))
        else:
            store.mark_run_failed(run_id, "findings could not be persisted to Postgres")
        job.result = result

    return _runner


async def submit_routine(
    kind: str, project: str, repo_path: str, params: dict, runner: RoutineCallable
) -> dict:
    """
    Launch a routine (``review``/``bughunt``) as a job.

    Creates the ``routine_runs`` row FIRST — a Postgres failure here raises
    and propagates to the caller uncaught, so the controller (A6) can fail
    loud with a 503 (6i pattern): no job is ever created for a run that
    couldn't be recorded. Only once that row exists is the job registered
    with ``services.jobs`` under the SAME per-project lock as
    pipeline/refresh/graph (9g, same ``ProjectJobConflict`` / 409 message).
    A lock conflict there means no job will ever run for this run row, so
    the now-orphaned queued row is deleted before the error propagates.

    Returns ``{"run_id": ..., "job_id": ...}``.
    """
    job_id = str(uuid.uuid4())
    run = await store.create_run(job_id, kind, project, params)
    run_id = run["id"]

    sync_runner = _make_runner(run_id, runner, params)
    try:
        job = await jobs.submit_external(kind, project, repo_path, job_id, sync_runner)
    except jobs.ProjectJobConflict:
        try:
            await store.delete_run(run_id)
        except Exception as e:
            logger.warning(
                "routine run: failed to clean up orphaned run %s after lock conflict: %s", run_id, e
            )
        raise

    return {"run_id": run_id, "job_id": job.id}
