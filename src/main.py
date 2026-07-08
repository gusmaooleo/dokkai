import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controllers import instances, chat, config, graph
from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # If OLLAMA_CHAT_MODEL is set, preconfigure the local LLM and warm it on
    # boot — so the first /chat isn't a cold start and the model stays resident
    # (keep_alive). Seeding the config also means you don't have to re-POST
    # /config/llm after every restart. Warming runs in the background and is
    # best-effort, so the server still starts if Ollama isn't up yet.
    model = os.getenv("OLLAMA_CHAT_MODEL")
    if model:
        from services.llm_config import get_config_store, LLMConfig
        from services.llm_provider import OllamaProvider

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        get_config_store().set_llm_config(
            LLMConfig(is_local=True, provider_name="ollama", model=model, base_url=base_url)
        )

        async def _warm() -> None:
            try:
                await OllamaProvider(base_url=base_url).warmup(model)
            except Exception:
                pass  # Ollama may not be up yet; the first real request will load it

        asyncio.create_task(_warm())

    yield


app = FastAPI(title="Dokkai API", lifespan=lifespan)

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
app.include_router(graph.router)

@app.get("/")
async def root():
    return {"status": "online"}
