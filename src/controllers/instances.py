from fastapi import APIRouter, HTTPException
from models.dtos.receive import DescribeRefreshRequest, IngestRequest
from services.describe import DescriptorError, ensure_descriptor_available
from services.jobs import get_job_store, submit_describe_refresh, submit_pipeline
from services.vectorize import find_latest_graph_json, no_graph_json_message
from services.weaviate_client import get_client, project_has_chunks

router = APIRouter(prefix="/instances")


@router.post("/pipeline")
async def runPipeline(data: IngestRequest):
    """
    Enqueue the ingestion pipeline (graph → chunk → describe → embed) as a
    background job and return its id immediately. Poll ``/instances/jobs/{id}``
    for progress and the final result.

    Descriptions are always on via this endpoint (the no-describe mode is an
    internal-only opt-in until feature 04's CLI/API UX), so the descriptor
    model is checked *before* creating the job: a missing/unavailable
    descriptor fails the request immediately with no job created.
    """
    try:
        await ensure_descriptor_available()
    except DescriptorError as e:
        raise HTTPException(status_code=400, detail=str(e))
    job = submit_pipeline(data.repo_path, recreate=data.recreate)
    return {"job_id": job.id, "status": job.status}


@router.post("/{project}/describe")
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

    job = submit_describe_refresh(project, force=force)
    return {"job_id": job.id, "status": job.status}


@router.get("/jobs")
async def listJobs():
    return [job.to_dict() for job in get_job_store().list()]


@router.get("/jobs/{job_id}")
async def getJob(job_id: str):
    job = get_job_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_dict()
