from pydantic import BaseModel

class IngestRequest(BaseModel):
    repo_path: str
    output_json: str = "./ingested/output.json"
