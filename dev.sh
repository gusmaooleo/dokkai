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
set -euo pipefail
cd "$(dirname "$0")"
exec uv run uvicorn main:app --reload --port "${PORT:-8000}" --app-dir src
