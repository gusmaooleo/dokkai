"""
Background job manager for the ingestion pipeline.

The pipeline (graph extraction → chunking → descriptions → embedding) runs for
minutes to hours on a large repo, so it must not block an HTTP request.
``POST /instances/pipeline`` enqueues a :class:`Job` and returns its id;
progress and result are polled via ``GET /instances/jobs/{id}``.

The pipeline mixes blocking work (the ``cgr`` subprocess, file reads, the sync
Weaviate client) with async work (Ollama description calls), so each job runs
in a **worker thread** with its own event loop — the server's main loop stays
free to serve status polls. Jobs are held in memory and reset on restart.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from services.ingest import ingestByLocalRepository
from services.vectorize import process_and_store


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    repo_path: str
    recreate: bool = False
    describe: bool = True
    status: str = "queued"   # queued | running | succeeded | failed
    stage: str = ""          # chunking | describing | embedding | done
    done: int = 0
    total: int = 0
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
        job = Job(id=str(uuid.uuid4()), repo_path=repo_path, recreate=recreate, describe=describe)
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)


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
        job.updated_at = _now()

    async def _pipeline() -> tuple[str, dict]:
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
    """
    job = get_job_store().create(repo_path, recreate=recreate, describe=describe)
    task = asyncio.create_task(asyncio.to_thread(_run_pipeline_sync, job))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return job
