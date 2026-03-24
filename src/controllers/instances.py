from fastapi import APIRouter

router = APIRouter(prefix="/instances")

@router.post("/pipeline")
async def runPipeline():
    return {"message": "Pipeline started"}
