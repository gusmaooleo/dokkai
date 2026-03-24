import subprocess

class IngestService:
    def __init__(self):
        pass

    def ingestByLocalRepository(self, repo_path: str, output_json: str = f"../injested/output.json"):
        result = subprocess.run(
            ["../../shell/run_cgr.sh", repo_path, output_json],
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
