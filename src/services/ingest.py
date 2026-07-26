import json
import subprocess
import os
import tempfile
import threading
from datetime import datetime, timezone

from services.ignore_rules import filter_graph
from services.vectorize import graph_json_candidates

# Global mutex around the cgr subprocess. All ingestion paths that touch the
# graph (pipeline and graph-only jobs) call ingestByLocalRepository, so this
# is the single seam every cgr run passes through. cgr's `--clean` wipes the
# whole graph at the start of a run, and every run — ephemeral-container or
# connect-mode — shares one Memgraph instance across ports; two cgr runs in
# flight at once would --clean each other mid-run. Today that's masked by an
# incidental failure (both ephemeral containers fight over host port 7687);
# this lock makes the limitation explicit instead of relying on that
# collision. A second job simply WAITS for the lock rather than failing fast:
# jobs already run async in background worker threads, so the wait is
# invisible to the polling client.
_cgr_lock = threading.Lock()


def _filter_and_promote_inplace(output_json: str, repo_path: str) -> dict:
    """
    Filter cgr's raw *output_json* in place (gitignore-aware, feature 21).

    Atomic: the filtered payload is written to a sibling temp file, then
    ``os.replace()``'d over *output_json* — the file at *output_json* is
    never observed half-written.

    On ANY failure (bad JSON, :func:`filter_graph` raising, an I/O error
    writing the temp file, ...) BOTH the temp file and the still-unfiltered
    *output_json* are removed before re-raising: *output_json* already
    matches the ``<folder>-<timestamp>.json`` naming convention the moment
    cgr writes it, so leaving it behind on failure would let it silently win
    as the "newest candidate" for :func:`services.vectorize.find_latest_graph_json`
    (e.g. served by ``GET /graph/...``) until the next successful ingest.

    Returns the filter stats dict (see :func:`filter_graph`).
    """
    tmp_json = output_json + ".filtering.tmp"
    try:
        with open(output_json, "r", encoding="utf-8") as fh:
            graph_data = json.load(fh)
        filtered_data, filter_stats = filter_graph(graph_data, repo_path)
        with open(tmp_json, "w", encoding="utf-8") as fh:
            json.dump(filtered_data, fh)
        os.replace(tmp_json, output_json)
    except Exception:
        for path in (tmp_json, output_json):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        raise
    return filter_stats


async def ingestByLocalRepository(repo_path: str) -> dict:
    """
    Run code-graph-rag against *repo_path* and promote its output to the
    canonical graph JSON for the project.

    code-graph-rag writes a timestamped file,
    ``./ingested/<folder_name>-<timestamp>.json``, where ``<folder_name>``
    is the last component of *repo_path*. Before promotion, the graph is
    filtered (see :func:`services.ignore_rules.filter_graph`) — gitignored
    and curated-default paths (build output, minified bundles, ...) are
    stripped so they never reach chunking. The filtered graph is then
    atomically promoted (renamed) to ``./ingested/<folder_name>.json`` — the
    canonical path returned as ``output_json`` — and any other leftover
    graph JSONs for the same project are deleted.
    """
    folder_name = os.path.basename(os.path.normpath(repo_path))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_filename = f"{folder_name}-{timestamp}.json"

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ingested_dir = os.path.join(base_dir, "ingested")
    os.makedirs(ingested_dir, exist_ok=True)

    output_json = os.path.join(ingested_dir, output_filename)

    script_path = os.path.join(base_dir, "shell", "run_cgr.sh")

    # Run code-graph-rag from an isolated working directory. Its config is a
    # strict pydantic-settings model (env_file=".env", extra forbidden); if it
    # ran from the project root it would read — and reject — this project's own
    # .env (COLLECTION_NAME, WEAVIATE_*, EMBED_MODEL, ...). Inherited env vars
    # are fine: the settings source ignores ones that aren't its fields.
    with tempfile.TemporaryDirectory() as cgr_cwd:
        with _cgr_lock:
            result = subprocess.run(
                [script_path, repo_path, output_json],
                capture_output=True,
                text=True,
                cwd=cgr_cwd,
            )

    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)

    # Gitignore-aware filtering (feature 21) — applied to cgr's raw output
    # BEFORE promotion, so the FILTERED graph is what becomes the canonical
    # file every downstream consumer (chunker, graph_store, the UI) reads.
    filter_stats = _filter_and_promote_inplace(output_json, repo_path)

    # Promote the timestamped output to the canonical <project>.json — same
    # directory, so this is an atomic rename — then drop every other
    # leftover graph JSON for this project (e.g. from pre-promotion installs
    # or earlier runs). The candidate matcher is anchored, so a project name
    # that prefixes another (e.g. "foo" vs "foo-bar") is never touched.
    canonical_json = os.path.join(ingested_dir, f"{folder_name}.json")
    os.replace(output_json, canonical_json)

    for candidate in graph_json_candidates(folder_name):
        candidate_path = str(candidate)
        try:
            if os.path.samefile(candidate_path, canonical_json):
                continue
            os.remove(candidate_path)
        except FileNotFoundError:
            # A candidate vanishing mid-loop is exactly what the cleanup
            # wants anyway; the canonical file was already promoted above.
            pass

    return {
        "success": True,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output_json": canonical_json,
        "filter": filter_stats,
    }
