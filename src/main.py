from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controllers import instances, chat, config
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Dokkai API")

# CORS — allow the Next.js frontend and any local dev tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(instances.router)
app.include_router(chat.router)
app.include_router(config.router)

@app.get("/")
async def root():
    return {"status": "online"}
