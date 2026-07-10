"""
Background job manager for the ingestion pipeline.

The pipeline (cgr → chunk → describe → upsert) runs for minutes to hours on a
large repo, so it must not block an HTTP request.
``POST /instances/pipeline`` enqueues a :class:`Job` and returns its id;
progress and result are polled via ``GET /instances/jobs/{id}``.

The pipeline mixes blocking work (the ``cgr`` subprocess, file reads, the sync
Weaviate client) with async work (Ollama description calls), so each job runs
in a **worker thread** with its own event loop — the server's main loop stays
free to serve status polls. Jobs are held in memory (the source of truth for
live jobs) and write-through to Postgres on lifecycle transitions only
(created/running/terminal — never per progress tick) so history survives a
restart; see the module docstring notes on ``_persist_job_via_pool`` and
``_persist_job_direct`` for why two different DB-access paths are used.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Callable

import asyncpg

from services.db import DatabaseUnavailableError, _database_url, get_pool
from services.ingest import ingestByLocalRepository
from services.vectorize import process_and_store, refresh_descriptions

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectJobConflict(Exception):
    """Raised when a job is submitted for a project that already has a
    queued or running job (of either kind)."""

    def __init__(self, project: str) -> None:
        self.project = project
        super().__init__(f"another job is running for project '{project}'")


@dataclass
class Job:
    id: str
    repo_path: str
    recreate: bool = False
    describe: bool = True
    kind: str = "pipeline"   # pipeline | refresh | graph | review | bughunt
    project: str = ""        # refresh jobs only
    force: bool = False      # refresh jobs only
    status: str = "queued"   # queued | running | succeeded | failed
    stage: str = ""          # cgr | chunk | describe | upsert | update | done
    done: int = 0
    total: int = 0
    stage_progress: dict | None = None  # {"stage": ..., "done": N, "total": N}
    result: dict | None = None
    error: str | None = None
    events: list[dict] | None = None  # routine (review/bughunt) step log only — see add_job_event
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)


_MAX_JOB_EVENTS = 200


def add_job_event(job: Job, stage: str, message: str) -> None:
    """
    Append a bounded, monotonic-seq progress event to a job's in-memory event
    log. Used by routine (review/bughunt) runners to stream step-by-step
    progress — built-in kinds (pipeline/refresh/graph) never call this, so
    their ``events`` stays ``None`` and their payload doesn't grow.

    Only the last ``_MAX_JOB_EVENTS`` events are kept (oldest dropped first),
    but ``seq`` keeps incrementing across drops so a client can detect a gap
    by seeing consecutive seqs jump by more than 1.

    In-memory only — does NOT persist to Postgres itself. Events ride along
    in the job's JSONB snapshot only at the existing running/terminal
    write-through points (see module docstring), so there are no per-tick DB
    writes. Bumps ``updated_at`` so the existing SSE stream
    (``GET /instances/jobs/{id}/events``, which polls on that field) picks
    up new events live without any change to that endpoint.
    """
    events = job.events
    if events is None:
        events = []
        job.events = events
    next_seq = (events[-1]["seq"] + 1) if events else 0
    events.append({"seq": next_seq, "stage": stage, "message": message, "ts": _now()})
    if len(events) > _MAX_JOB_EVENTS:
        del events[: len(events) - _MAX_JOB_EVENTS]
    job.updated_at = _now()


class JobStore:
    """In-memory registry of ingestion jobs (singleton, resets on restart)."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self, repo_path: str, recreate: bool = False, describe: bool = True) -> Job:
        project = os.path.basename(os.path.normpath(repo_path))
        job = Job(
            id=str(uuid.uuid4()),
            repo_path=repo_path,
            recreate=recreate,
            describe=describe,
            project=project,
        )
        self._jobs[job.id] = job
        return job

    def create_describe_refresh(self, project: str, force: bool = False) -> Job:
        job = Job(id=str(uuid.uuid4()), repo_path="", kind="refresh", project=project, force=force)
        self._jobs[job.id] = job
        return job

    def create_graph_only(self, repo_path: str) -> Job:
        project = os.path.basename(os.path.normpath(repo_path))
        job = Job(id=str(uuid.uuid4()), repo_path=repo_path, kind="graph", project=project)
        self._jobs[job.id] = job
        return job

    def create_external(self, job_id: str, kind: str, project: str, repo_path: str) -> Job:
        """
        Create a job for an externally-driven kind (``review``/``bughunt`` —
        see ``services.routines.engine.submit_routine``). Unlike the other
        ``create_*`` methods, the id is supplied by the caller: a
        ``routine_runs`` row referencing it must already exist before this
        job starts running.
        """
        job = Job(id=job_id, repo_path=repo_path, kind=kind, project=project)
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def active_job_for_project(self, project: str) -> Job | None:
        """The queued/running job (any kind) for *project*, if any."""
        for job in self._jobs.values():
            if job.project == project and job.status in ("queued", "running"):
                return job
        return None


_job_store = JobStore()
_background_tasks: set[asyncio.Task] = set()


def get_job_store() -> JobStore:
    return _job_store


# -----------------------------------------------------------------------
# Job history persistence (best-effort write-through to Postgres)
# -----------------------------------------------------------------------
#
# Two write paths, chosen for the same reason: asyncpg pools/connections are
# bound to the event loop that created them.
#
#   - _persist_job_via_pool: used for the "created" (queued) transition,
#     which happens in submit_* on the server's MAIN event loop — safe to
#     use the shared app pool (services.db.get_pool()) and awaited before the
#     worker thread is spawned, so the queued row always lands before any
#     running/terminal write from that job's worker.
#   - _persist_job_direct: used for the "running" and "terminal" transitions,
#     which happen inside a job worker THREAD's own private loop
#     (asyncio.run in _run_*_sync). That loop must never touch the main
#     loop's pool, so it opens a short-lived direct asyncpg.connect() and
#     closes it right after. Cheap enough since it only fires twice per job.
#
# Both paths are best-effort: any Postgres failure is logged as a warning
# and swallowed — the in-memory job is unaffected, persistence is a nice-to-have.

async def _write_job_row(job: Job, conn: asyncpg.Connection) -> None:
    await conn.execute(
        """
        INSERT INTO jobs (id, project, status, created_at, updated_at, data)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
        ON CONFLICT (id) DO UPDATE
        SET status = EXCLUDED.status,
            updated_at = EXCLUDED.updated_at,
            data = EXCLUDED.data
        """,
        uuid.UUID(job.id),
        job.project,
        job.status,
        datetime.fromisoformat(job.created_at),
        datetime.fromisoformat(job.updated_at),
        json.dumps(job.to_dict()),
    )


async def _persist_job_via_pool(job: Job) -> None:
    """Write-through via the app pool. Main event loop only — see note above."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await _write_job_row(job, conn)
    except DatabaseUnavailableError as e:
        logger.warning("job history: Postgres unavailable, skipping persist for job %s: %s", job.id, e)
    except Exception as e:
        logger.warning("job history: failed to persist job %s: %s", job.id, e)


async def _persist_job_direct(job: Job) -> None:
    """Write-through via a short-lived direct connection. Worker threads only — see note above."""
    try:
        conn = await asyncpg.connect(_database_url())
    except Exception as e:
        logger.warning("job history: could not reach Postgres to persist job %s: %s", job.id, e)
        return
    try:
        await _write_job_row(job, conn)
    except Exception as e:
        logger.warning("job history: failed to persist job %s: %s", job.id, e)
    finally:
        await conn.close()


def _persist_job_from_worker(job: Job) -> None:
    """Sync wrapper so worker threads (running plain functions, not coroutines) can call _persist_job_direct."""
    asyncio.run(_persist_job_direct(job))


async def list_jobs() -> list[dict]:
    """
    Merge in-memory (live) jobs with Postgres-persisted history for
    ``GET /instances/jobs``. Memory wins on id collisions. Falls back to
    memory-only, quietly, when Postgres is unreachable — job listings for
    live jobs must keep working regardless of DB health.
    """
    memory_jobs = {job.id: job.to_dict() for job in get_job_store().list()}

    db_jobs: dict[str, dict] = {}
    try:
        pool = await get_pool()
        rows = await pool.fetch("SELECT id, data FROM jobs")
        for row in rows:
            raw = row["data"]
            db_jobs[str(row["id"])] = json.loads(raw) if isinstance(raw, str) else raw
    except DatabaseUnavailableError as e:
        logger.warning("job history: Postgres unavailable, listing live jobs only: %s", e)
    except Exception as e:
        logger.warning("job history: failed to read job history: %s", e)

    merged = {**db_jobs, **memory_jobs}
    return sorted(merged.values(), key=lambda j: j["created_at"], reverse=True)


async def get_job_from_db(job_id: str) -> dict | None:
    """DB-only job lookup, used as a fallback when a job isn't in memory."""
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        return None
    try:
        pool = await get_pool()
        row = await pool.fetchrow("SELECT data FROM jobs WHERE id = $1", job_uuid)
    except DatabaseUnavailableError as e:
        logger.warning("job history: Postgres unavailable, cannot fetch job %s: %s", job_id, e)
        return None
    except Exception as e:
        logger.warning("job history: failed to read job %s: %s", job_id, e)
        return None
    if row is None:
        return None
    raw = row["data"]
    return json.loads(raw) if isinstance(raw, str) else raw


async def sweep_interrupted_jobs() -> None:
    """
    Mark any job left 'queued' or 'running' from a previous server run as
    failed — a crash or restart killed its worker thread, so it will never
    reach a terminal state on its own. Call once at boot, after init_db()
    succeeds. Patches both the status column and the JSONB data payload so
    reads stay Job.to_dict()-shaped.
    """
    pool = await get_pool()
    await pool.execute(
        """
        UPDATE jobs
        SET status = 'failed',
            updated_at = now(),
            data = jsonb_set(
                jsonb_set(
                    jsonb_set(data, '{status}', '"failed"'),
                    '{error}', to_jsonb('interrupted by server restart'::text)
                ),
                '{updated_at}', to_jsonb(now()::text)
            )
        WHERE status IN ('queued', 'running')
        """
    )


def _run_pipeline_sync(job: Job) -> None:
    """
    Execute the full pipeline for a job, in a worker thread. Updates ``job`` in
    place as it progresses. Blocking + async work is driven by a single private
    event loop (``asyncio.run``) owned by this thread.
    """
    job.status = "running"
    job.updated_at = _now()
    _persist_job_from_worker(job)

    def report(stage: str, done: int, total: int) -> None:
        job.stage, job.done, job.total = stage, done, total
        job.stage_progress = {"stage": stage, "done": done, "total": total}
        job.updated_at = _now()

    async def _pipeline() -> tuple[str, dict]:
        report("cgr", 0, 0)
        ingest_result = await ingestByLocalRepository(job.repo_path)
        output_json = ingest_result["output_json"]
        vectorize_result = await process_and_store(
            output_json,
            source_root=job.repo_path,
            recreate=job.recreate,
            describe=job.describe,
            progress=report,
        )
        return output_json, vectorize_result

    try:
        output_json, vectorize_result = asyncio.run(_pipeline())
        job.result = {
            "ingest": {"output_json": output_json},
            "vectorize": vectorize_result,
        }
        job.status = "succeeded"
        job.stage = "done"
    except Exception as e:  # surface a readable error to the poller
        job.status = "failed"
        job.error = f"{type(e).__name__}: {e}"
    finally:
        job.updated_at = _now()
        _persist_job_from_worker(job)


async def submit_pipeline(repo_path: str, recreate: bool = False, describe: bool = True) -> Job:
    """
    Create a job and spawn it on a worker thread. Returns immediately with the
    queued job. The task reference is retained so it isn't garbage-collected
    mid-run.

    Raises ``ProjectJobConflict`` if a job is already queued/running for the
    same project. The conflict check and job creation are plain synchronous
    code with no ``await`` in between, so — running on the single asyncio
    event loop — no other submit can interleave between them; this makes the
    check-then-create atomic without needing an explicit lock.
    """
    store = get_job_store()
    project = os.path.basename(os.path.normpath(repo_path))
    active = store.active_job_for_project(project)
    if active is not None:
        raise ProjectJobConflict(project)
    job = store.create(repo_path, recreate=recreate, describe=describe)
    await _persist_job_via_pool(job)
    task = asyncio.create_task(asyncio.to_thread(_run_pipeline_sync, job))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return job


def _run_describe_refresh_sync(job: Job) -> None:
    """
    Execute a describe-refresh job in a worker thread, mirroring
    ``_run_pipeline_sync``'s thread/asyncio.run/status pattern.
    """
    job.status = "running"
    job.updated_at = _now()
    _persist_job_from_worker(job)

    def report(stage: str, done: int, total: int) -> None:
        job.stage, job.done, job.total = stage, done, total
        job.stage_progress = {"stage": stage, "done": done, "total": total}
        job.updated_at = _now()

    try:
        result = asyncio.run(
            refresh_descriptions(job.project, force=job.force, progress=report)
        )
        job.result = result
        job.status = "succeeded"
        job.stage = "done"
    except Exception as e:  # surface a readable error to the poller
        job.status = "failed"
        job.error = f"{type(e).__name__}: {e}"
    finally:
        job.updated_at = _now()
        _persist_job_from_worker(job)


async def submit_describe_refresh(project: str, force: bool = False) -> Job:
    """
    Create a describe-refresh job and spawn it on a worker thread. Returns
    immediately with the queued job.

    Raises ``ProjectJobConflict`` if a job is already queued/running for the
    same project — see ``submit_pipeline`` for why the check-then-create is
    race-free on the event loop.
    """
    store = get_job_store()
    active = store.active_job_for_project(project)
    if active is not None:
        raise ProjectJobConflict(project)
    job = store.create_describe_refresh(project, force=force)
    await _persist_job_via_pool(job)
    task = asyncio.create_task(asyncio.to_thread(_run_describe_refresh_sync, job))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return job


def _run_graph_only_sync(job: Job) -> None:
    """
    Execute a graph-only job in a worker thread, mirroring
    ``_run_pipeline_sync``'s thread/asyncio.run/status pattern. Runs ONLY
    ``ingestByLocalRepository`` — no describe, no vectorize, no Weaviate
    (decisions 1b-arch/1b-graph).
    """
    job.status = "running"
    job.updated_at = _now()
    _persist_job_from_worker(job)

    def report(stage: str, done: int, total: int) -> None:
        job.stage, job.done, job.total = stage, done, total
        job.stage_progress = {"stage": stage, "done": done, "total": total}
        job.updated_at = _now()

    async def _graph() -> dict:
        report("cgr", 0, 0)
        return await ingestByLocalRepository(job.repo_path)

    try:
        ingest_result = asyncio.run(_graph())
        job.result = {"ingest": {"output_json": ingest_result["output_json"]}}
        job.status = "succeeded"
        job.stage = "done"
    except Exception as e:  # surface a readable error to the poller
        job.status = "failed"
        job.error = f"{type(e).__name__}: {e}"
    finally:
        job.updated_at = _now()
        _persist_job_from_worker(job)


async def submit_graph_only(repo_path: str) -> Job:
    """
    Create a graph-only job and spawn it on a worker thread. Returns
    immediately with the queued job.

    Raises ``ProjectJobConflict`` if a job is already queued/running for the
    same project (pipeline, refresh, or graph) — see ``submit_pipeline`` for
    why the check-then-create is race-free on the event loop. The lock is
    shared with the other job kinds so a graph run and a pipeline/refresh run
    on the same project can't clobber each other's graph JSON concurrently.
    """
    store = get_job_store()
    project = os.path.basename(os.path.normpath(repo_path))
    active = store.active_job_for_project(project)
    if active is not None:
        raise ProjectJobConflict(project)
    job = store.create_graph_only(repo_path)
    await _persist_job_via_pool(job)
    task = asyncio.create_task(asyncio.to_thread(_run_graph_only_sync, job))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return job


# -----------------------------------------------------------------------
# External job kinds (review / bughunt — services.routines.engine)
# -----------------------------------------------------------------------
#
# This module intentionally does NOT import services.routines: the
# dependency runs the other way (routines imports jobs). The mechanism is a
# submit-with-runner API rather than a kind→callable registry — the caller
# (routines.engine.submit_routine) hands this module a plain sync callable
# that fully owns the actual routine work; this module only ever manages
# generic Job lifecycle (status, timestamps, write-through, the per-project
# lock) exactly like it does for pipeline/refresh/graph, without knowing
# anything about what "review" or "bughunt" mean.

ExternalRunner = Callable[[Job, Callable[[str, str], None]], None]


def _run_external_sync(job: Job, runner: ExternalRunner) -> None:
    """
    Execute an externally-driven job (review/bughunt) in a worker thread,
    mirroring ``_run_pipeline_sync``'s thread/status/persist pattern. Unlike
    the built-in kinds, the actual work is fully owned by ``runner`` — this
    function only manages Job lifecycle plumbing (status, timestamps,
    write-through) and hands the runner an ``emit(stage, message)`` closure
    for step events (see ``add_job_event``).
    """
    job.status = "running"
    job.updated_at = _now()
    _persist_job_from_worker(job)

    def emit(stage: str, message: str) -> None:
        add_job_event(job, stage, message)

    try:
        runner(job, emit)
        job.status = "succeeded"
        job.stage = "done"
    except Exception as e:  # surface a readable error to the poller
        job.status = "failed"
        job.error = f"{type(e).__name__}: {e}"
    finally:
        job.updated_at = _now()
        _persist_job_from_worker(job)


async def submit_external(
    kind: str, project: str, repo_path: str, job_id: str, runner: ExternalRunner
) -> Job:
    """
    Create a job for an externally-driven kind (``review``/``bughunt``) with
    a caller-supplied id and spawn it on a worker thread. Returns
    immediately with the queued job.

    ``job_id`` is supplied by the caller (``services.routines.engine``)
    because a ``routine_runs`` row referencing it must already exist before
    this job starts — see ``submit_routine`` there.

    Raises ``ProjectJobConflict`` if a job is already queued/running for the
    same project (any kind, 9g) — see ``submit_pipeline`` for why the
    check-then-create is race-free on the event loop.
    """
    store = get_job_store()
    active = store.active_job_for_project(project)
    if active is not None:
        raise ProjectJobConflict(project)
    job = store.create_external(job_id, kind, project, repo_path)
    await _persist_job_via_pool(job)
    task = asyncio.create_task(asyncio.to_thread(_run_external_sync, job, runner))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return job
