"""
Background job manager for the ingestion pipeline.

The pipeline (cgr → chunk → describe → upsert) runs for minutes to hours on a
large repo, so it must not block an HTTP request.
``POST /instances/pipeline`` enqueues a :class:`Job` and returns its id;
progress and result are polled via ``GET /instances/jobs/{id}``.

The pipeline mixes blocking work (the ``cgr`` subprocess, file reads, the sync
Weaviate client) with async work (Ollama description calls), so each job runs
in a **worker thread** with its own event loop — the server's main loop stays
free to serve status polls. Jobs are held in memory and reset on restart.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from services.ingest import ingestByLocalRepository
from services.vectorize import process_and_store, refresh_descriptions


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
    kind: str = "pipeline"   # pipeline | refresh | graph
    project: str = ""        # refresh jobs only
    force: bool = False      # refresh jobs only
    status: str = "queued"   # queued | running | succeeded | failed
    stage: str = ""          # cgr | chunk | describe | upsert | update | done
    done: int = 0
    total: int = 0
    stage_progress: dict | None = None  # {"stage": ..., "done": N, "total": N}
    result: dict | None = None
    error: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)


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


def _run_pipeline_sync(job: Job) -> None:
    """
    Execute the full pipeline for a job, in a worker thread. Updates ``job`` in
    place as it progresses. Blocking + async work is driven by a single private
    event loop (``asyncio.run``) owned by this thread.
    """
    job.status = "running"
    job.updated_at = _now()

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


def submit_pipeline(repo_path: str, recreate: bool = False, describe: bool = True) -> Job:
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


def submit_describe_refresh(project: str, force: bool = False) -> Job:
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


def submit_graph_only(repo_path: str) -> Job:
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
    task = asyncio.create_task(asyncio.to_thread(_run_graph_only_sync, job))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return job
