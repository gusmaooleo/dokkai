import subprocess
import os
from datetime import datetime, timezone


async def ingestByLocalRepository(repo_path: str) -> dict:
    """
    Run code-graph-rag against *repo_path* and return the path to the
    auto-named output JSON.

    The file is saved as ``./ingested/<folder_name>-<timestamp>.json``
    where ``<folder_name>`` is the last component of *repo_path*.
    """
    folder_name = os.path.basename(os.path.normpath(repo_path))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_filename = f"{folder_name}-{timestamp}.json"

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ingested_dir = os.path.join(base_dir, "ingested")
    os.makedirs(ingested_dir, exist_ok=True)

    output_json = os.path.join(ingested_dir, output_filename)

    script_path = os.path.join(base_dir, "shell", "run_cgr.sh")

    result = subprocess.run(
        [script_path, repo_path, output_json],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)

    return {
        "success": True,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output_json": output_json,
    }
