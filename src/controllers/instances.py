import asyncio
import json
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from models.dtos.receive import DescribeRefreshRequest, GraphOnlyRequest, IngestRequest
from services.auth import require_auth, require_role
from services.describe import DescriptorError, ensure_descriptor_available
from services.jobs import (
    ProjectJobConflict,
    get_job_from_db,
    get_job_store,
    list_jobs,
    submit_describe_refresh,
    submit_graph_only,
    submit_pipeline,
)
from services.vectorize import find_latest_graph_json, no_graph_json_message
from services.weaviate_client import get_client, project_has_chunks

router = APIRouter(prefix="/instances", dependencies=[Depends(require_auth)])

# Ingestion/refresh routes mutate state — role 'viewer' is read-only (6k).
require_write = require_role("admin", "user")


# Only OS cruft in a folder still counts as "empty".
_IGNORABLE_ENTRIES = {".DS_Store", "Thumbs.db", ".localized"}


def _require_ingestable_dir(repo_path: str) -> None:
    """400 if repo_path is blank, missing, not a directory, or empty — before
    any job is created. The message surfaces via the New ingestion dialog's
    existing error toast; the browser can't stat the server's filesystem, so
    this check must live server-side."""
    path = (repo_path or "").strip()
    try:
        entries = os.listdir(path) if path and os.path.isdir(path) else None
    except OSError:
        entries = None
    if entries is None or not any(e not in _IGNORABLE_ENTRIES for e in entries):
        raise HTTPException(
            status_code=400,
            detail=f"Repository path is empty or does not exist on the server: "
            f"'{path}'. Check the path and try again.",
        )


@router.post("/pipeline", dependencies=[Depends(require_write)])
async def runPipeline(data: IngestRequest):
    """
    Enqueue the ingestion pipeline (graph → chunk → describe → embed) as a
    background job and return its id immediately. Poll ``/instances/jobs/{id}``
    for progress and the final result.

    Descriptions are on by default; the descriptor model is checked *before*
    creating the job, so a missing/unavailable descriptor fails the request
    immediately with no job created. Set ``describe: false`` to opt out and
    skip that pre-flight — the pipeline then runs without descriptions.
    """
    _require_ingestable_dir(data.repo_path)
    if data.describe:
        try:
            await ensure_descriptor_available()
        except DescriptorError as e:
            raise HTTPException(status_code=400, detail=str(e))
    try:
        job = await submit_pipeline(data.repo_path, recreate=data.recreate, describe=data.describe)
    except ProjectJobConflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"job_id": job.id, "status": job.status}


@router.post("/graph", dependencies=[Depends(require_write)])
async def runGraphOnly(data: GraphOnlyRequest):
    """
    Enqueue a graph-ONLY run (decision 1b-graph) as a background job and
    return its id immediately. Poll ``/instances/jobs/{id}`` for progress
    and the final result.

    Runs code-graph-rag and nothing else — no LLM, no vectorization, no
    Weaviate — so there is no descriptor pre-flight. The canonical JSON
    promotion (timestamped output → ``ingested/<project>.json``) already
    happens inside ``ingestByLocalRepository``.
    """
    _require_ingestable_dir(data.repo_path)
    try:
        job = await submit_graph_only(data.repo_path)
    except ProjectJobConflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"job_id": job.id, "status": job.status}


@router.post("/{project}/describe", dependencies=[Depends(require_write)])
async def refreshDescriptions(project: str, data: DescribeRefreshRequest | None = None):
    """
    Re-run ONLY the describe pass over a project's already-ingested chunks
    (no re-chunk-graph-rag, no re-embed of unchanged text) as a background
    job. Poll ``/instances/jobs/{id}`` for progress and the final result.

    ``force`` (default false) bypasses the description cache, regenerating
    every eligible entity. Pre-flight, in order: the project must already
    have chunks in Weaviate (404), its latest graph JSON must still be
    present in ``ingested/`` (409), and the descriptor model must be
    available (400) — each fails the request immediately with no job
    created.
    """
    force = data.force if data is not None else False

    client = get_client()
    try:
        has_chunks = project_has_chunks(client, project)
    finally:
        client.close()
    if not has_chunks:
        raise HTTPException(
            status_code=404,
            detail=(
                f"project '{project}' has no chunks — ingest it first with "
                "POST /instances/pipeline"
            ),
        )

    if find_latest_graph_json(project) is None:
        raise HTTPException(status_code=409, detail=no_graph_json_message(project))

    try:
        await ensure_descriptor_available()
    except DescriptorError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        job = await submit_describe_refresh(project, force=force)
    except ProjectJobConflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"job_id": job.id, "status": job.status}


@router.get("/jobs")
async def listJobs():
    """
    Merges live (in-memory) jobs with Postgres-persisted job history —
    memory wins on id collisions, sorted by ``created_at`` descending. If
    Postgres is unreachable, history is quietly absent and only live jobs
    are returned (never a 503 — live job status must keep working).
    """
    return await list_jobs()


@router.get("/jobs/{job_id}")
async def getJob(job_id: str):
    job = get_job_store().get(job_id)
    if job is not None:
        return job.to_dict()
    db_job = await get_job_from_db(job_id)
    if db_job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return db_job


@router.get("/jobs/{job_id}/events")
async def streamJobEvents(job_id: str):
    """
    Stream a job's progress as Server-Sent Events, mirroring ``POST /chat``'s
    SSE framing (``StreamingResponse`` over ``text/event-stream``, same
    ``event: <type>\\ndata: <json>\\n\\n`` lines, same no-cache headers).

    Emits the current job state immediately on connect, then again whenever
    ``updated_at`` changes (polled internally every ~0.5s). Event type is
    ``job`` for in-progress updates and ``done`` for the terminal state —
    reusing chat's ``done`` name for "stream is now finished" — after which
    the stream closes. Unknown job id is a plain 404, not a stream.

    A job that only exists in Postgres (not in memory — the server restarted
    since it ran) is, by construction, always terminal after the boot sweep;
    its stream is a single ``done`` event with the stored payload.
    """
    job = get_job_store().get(job_id)
    if job is None:
        db_job = await get_job_from_db(job_id)
        if db_job is None:
            raise HTTPException(status_code=404, detail="job not found")

        async def db_event_stream():
            yield f"event: done\ndata: {json.dumps(db_job)}\n\n"

        return StreamingResponse(
            db_event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def event_stream():
        last_updated_at = None
        while True:
            if job.updated_at != last_updated_at:
                last_updated_at = job.updated_at
                terminal = job.status in ("succeeded", "failed")
                event_type = "done" if terminal else "job"
                data = json.dumps(job.to_dict())
                yield f"event: {event_type}\ndata: {data}\n\n"
                if terminal:
                    break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
