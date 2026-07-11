#!/bin/bash

set -e

PROJECT_PATH="$1"
OUTPUT_JSON="$2"

if [ -z "$PROJECT_PATH" ] || [ -z "$OUTPUT_JSON" ]; then
  echo "Usage: $0 <project_path> <output_json>"
  exit 1
fi

MEMGRAPH_IMAGE="${MEMGRAPH_IMAGE:-memgraph/memgraph:latest}"

if [ -n "$MEMGRAPH_HOST" ]; then
  # Connect mode: point cgr at a resident Memgraph instance (e.g. the
  # `memgraph` compose service) instead of spinning up an ephemeral
  # container per run. cgr reads MEMGRAPH_HOST/MEMGRAPH_PORT itself (a
  # pydantic-settings field, inherited from this process's env — no flag
  # needed on `cgr start`); this branch only needs to skip the docker
  # run/cleanup and pre-probe the host so a misconfigured MEMGRAPH_HOST
  # fails with an actionable message instead of cgr's raw connection error.
  # --clean still wipes the graph at the start of the run, which is what
  # isolates back-to-back runs on a shared instance.
  PROBE_PORT="${MEMGRAPH_PORT:-7687}"
  if ! (exec 3<>"/dev/tcp/${MEMGRAPH_HOST}/${PROBE_PORT}") 2>/dev/null; then
    echo "Memgraph is not reachable at ${MEMGRAPH_HOST}:${PROBE_PORT} — start it with docker compose up -d, or unset MEMGRAPH_HOST to use the legacy ephemeral container" >&2
    exit 1
  fi
  exec 3<&- 3>&-

  echo "Running CGR (connect mode: ${MEMGRAPH_HOST}:${PROBE_PORT})..."
  cgr start \
    --repo-path "$PROJECT_PATH" \
    --update-graph \
    --clean \
    -o "$OUTPUT_JSON"

  echo "Done. Output: $OUTPUT_JSON"
  exit 0
fi

CONTAINER_NAME="tmp-memgraph-$$"

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

trap cleanup EXIT

echo "Starting Memgraph..."
docker run -d \
  --name "$CONTAINER_NAME" \
  -p 7687:7687 \
  "$MEMGRAPH_IMAGE" >/dev/null

echo "Waiting Memgraph to be ready..."
sleep 5

echo "Running CGR..."
cgr start \
  --repo-path "$PROJECT_PATH" \
  --update-graph \
  --clean \
  -o "$OUTPUT_JSON"

echo "Done. Output: $OUTPUT_JSON"
