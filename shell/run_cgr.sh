set -e

PROJECT_PATH="$1"
OUTPUT_JSON="$2"

if [ -z "$PROJECT_PATH" ] || [ -z "$OUTPUT_JSON" ]; then
  echo "Usage: $0 <project_path> <output_json>"
  exit 1
fi

CONTAINER_NAME="tmp-memgraph-$$"
MEMGRAPH_IMAGE="${MEMGRAPH_IMAGE:-memgraph/memgraph:latest}"

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
