import subprocess
import os
import tempfile
from datetime import datetime, timezone

from services.vectorize import graph_json_candidates


async def ingestByLocalRepository(repo_path: str) -> dict:
    """
    Run code-graph-rag against *repo_path* and promote its output to the
    canonical graph JSON for the project.

    code-graph-rag writes a timestamped file,
    ``./ingested/<folder_name>-<timestamp>.json``, where ``<folder_name>``
    is the last component of *repo_path*. On success it is atomically
    promoted (renamed) to ``./ingested/<folder_name>.json`` — the canonical
    path returned as ``output_json`` — and any other leftover graph JSONs
    for the same project are deleted.
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
        result = subprocess.run(
            [script_path, repo_path, output_json],
            capture_output=True,
            text=True,
            cwd=cgr_cwd,
        )

    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)

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
    }
