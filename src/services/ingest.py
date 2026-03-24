import subprocess
import os


async def ingestByLocalRepository(repo_path: str, output_json: str = "./ingested/output.json"):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    result = subprocess.run(
        [str("/Users/leonardo/projects/dokkai/shell/run_cgr.sh"), repo_path, output_json],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)

    return {
        "success": True,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output_json": output_json,
    }
