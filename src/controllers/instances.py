from fastapi import APIRouter
from models.dtos.receive import IngestRequest
from services.ingest import ingestByLocalRepository

router = APIRouter(prefix="/instances")

@router.post("/pipeline")
async def runPipeline(data: IngestRequest):
    result = await ingestByLocalRepository(data.repo_path, data.output_json)
    return { "message": "Pipeline started", "result": result }
