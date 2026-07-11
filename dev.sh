#!/usr/bin/env bash
#
# Run the Dokkai API in development mode (auto-reload).
#
# Usage:
#   ./dev.sh              # serve on port 8000
#   PORT=9000 ./dev.sh    # serve on a custom port
#
# Why --app-dir src: the app uses absolute imports (controllers, services,
# models) rooted at src/, so src/ must be on sys.path and the app is "main:app".
#
# MEMGRAPH_HOST defaults to localhost: with `docker compose up -d` running,
# shell/run_cgr.sh connects to the resident `memgraph` compose service
# instead of spinning up an ephemeral container per graph ingestion (saves
# the ~5s startup wait). Unset it to fall back to the legacy ephemeral
# container.
set -euo pipefail
cd "$(dirname "$0")"
export MEMGRAPH_HOST="${MEMGRAPH_HOST:-localhost}"
exec uv run uvicorn main:app --reload --port "${PORT:-8000}" --app-dir src
