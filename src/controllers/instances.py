from fastapi import APIRouter, HTTPException
from models.dtos.receive import IngestRequest
from services.ingest import ingestByLocalRepository
from services.vectorize import process_and_store

router = APIRouter(prefix="/instances")


@router.post("/pipeline")
async def runPipeline(data: IngestRequest):
    ingest_result = await ingestByLocalRepository(data.repo_path)

    output_json = ingest_result["output_json"]

    try:
        vectorize_result = await process_and_store(output_json, source_root=data.repo_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Ingestion succeeded but vectorization failed",
                "ingest": ingest_result,
                "error": str(e),
            },
        )

    return {
        "message": "Pipeline completed",
        "ingest": {
            "output_json": output_json,
        },
        "vectorize": vectorize_result,
    }
