from fastapi import APIRouter, HTTPException
from models.dtos.receive import IngestRequest
from services.jobs import get_job_store, submit_pipeline

router = APIRouter(prefix="/instances")


@router.post("/pipeline")
async def runPipeline(data: IngestRequest):
    """
    Enqueue the ingestion pipeline (graph → chunk → describe → embed) as a
    background job and return its id immediately. Poll ``/instances/jobs/{id}``
    for progress and the final result.
    """
    job = submit_pipeline(data.repo_path, recreate=data.recreate)
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
