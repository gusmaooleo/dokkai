from fastapi import FastAPI
from controllers import instances

app = FastAPI(title="Dokkai API")

app.include_router(instances.router)

@app.get("/")
async def root():
    return {"status": "online"}
