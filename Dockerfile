# Multi-stage build for the Dokkai API (Python 3.14 + uv + FastAPI).
# Local orchestration only — see docker-compose.yml `full` profile / `dokkai up --full`.

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
# Dependency layer cached independently of source changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

FROM python:3.14-slim AS runtime
WORKDIR /app
# git: routines/ops run real git commands (branches, diffs) against mounted
# repos. bash: shell/run_cgr.sh. curl: container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git bash curl \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --system dokkai && useradd --system --gid dokkai --create-home dokkai

COPY --from=builder /app/.venv ./.venv
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY shell/ ./shell/
ENV PATH="/app/.venv/bin:${PATH}"

# code-graph-rag writes graph JSONs to ./ingested/ (relative to repo root,
# i.e. /app here — see services/ingest.py, services/vectorize.py); the LLM
# description cache lives under ./data/ (services/describe.py). Both must
# survive container restarts — named volumes, declared in docker-compose.yml.
RUN mkdir -p /app/ingested /app/data && chown -R dokkai:dokkai /app
VOLUME ["/app/ingested", "/app/data"]

USER dokkai
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=5 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
