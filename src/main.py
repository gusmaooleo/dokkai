import os
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controllers import auth, instances, chat, config, graph
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to Postgres and apply pending migrations. Best-effort: if
    # Postgres isn't up yet, log a clear degraded-mode warning and keep
    # serving — services.db.get_pool() will retry lazily on first access.
    from services.db import init_db

    try:
        await init_db()
    except Exception as e:
        logger.warning(
            "cannot connect to Postgres — conversation history is "
            "unavailable; start it with 'docker compose up -d' (%s)", e,
        )
    else:
        # Any job still 'queued'/'running' in the DB belongs to a previous
        # server run that died before it could reach a terminal state —
        # mark it failed so job history doesn't show it stuck forever.
        from services.jobs import sweep_interrupted_jobs

        try:
            await sweep_interrupted_jobs()
        except Exception as e:
            logger.warning("job history: failed to sweep interrupted jobs on boot: %s", e)

        # Same idea, for routine runs (code review/bughunt) — a run stuck
        # 'queued'/'running' belongs to a worker that died before finishing.
        from services.routines.store import sweep_interrupted_routine_runs

        try:
            await sweep_interrupted_routine_runs()
        except Exception as e:
            logger.warning("routines: failed to sweep interrupted runs on boot: %s", e)

        # Seed the admin user (or apply DOKKAI_ROOT_* overrides) now that
        # Postgres is confirmed reachable. Best-effort like the rest of this
        # block — a failure here still leaves the API serving, and login's
        # own lazy ensure_seeded() call will retry.
        from services.auth import ensure_seeded, is_default_admin_active

        try:
            await ensure_seeded()
            if await is_default_admin_active():
                logger.warning(
                    "default admin credentials (admin/admin) are still active — "
                    "log in and change the password, or set DOKKAI_ROOT_USER/"
                    "DOKKAI_ROOT_PASSWORD"
                )
        except Exception as e:
            logger.warning("auth: failed to seed admin user on boot: %s", e)

    # The persisted LLM config (Postgres) takes precedence over the
    # OLLAMA_CHAT_MODEL env seed — the DB row only exists once someone has
    # POSTed /config/llm, at which point it should survive restarts. If
    # there's no row yet (or Postgres is unreachable), fall back to seeding
    # from OLLAMA_CHAT_MODEL as before. load_persisted_config() never raises
    # (it swallows and logs any Postgres error itself), so unlike init_db()
    # above this call doesn't need its own try/except to keep boot alive.
    from services.llm_config import get_config_store, load_persisted_config

    persisted_config = await load_persisted_config()
    if persisted_config is not None:
        get_config_store().set_llm_config(persisted_config)
    else:
        # If OLLAMA_CHAT_MODEL is set, preconfigure the local LLM and warm it
        # on boot — so the first /chat isn't a cold start and the model stays
        # resident (keep_alive). Seeding the config also means you don't have
        # to re-POST /config/llm after every restart. Warming runs in the
        # background and is best-effort, so the server still starts if
        # Ollama isn't up yet.
        model = os.getenv("OLLAMA_CHAT_MODEL")
        if model:
            from services.llm_config import LLMConfig
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

# CORS — restrict to the configured frontend origin(s). DOKKAI_CORS_ORIGINS is a
# comma-separated list of allowed origins (default: the Next.js dev server on
# localhost:3000). We authenticate with bearer headers, not cookies, so
# allow_credentials is False.
_cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "DOKKAI_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(instances.router)
app.include_router(chat.router)
app.include_router(config.router)
app.include_router(graph.router)

@app.get("/")
async def root():
    return {"status": "online"}
