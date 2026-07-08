from pydantic import BaseModel


class IngestRequest(BaseModel):
    repo_path: str
    # Drop & rebuild the collection before inserting — needed once after a
    # schema change (e.g. adopting named vectors). Wipes ALL projects.
    recreate: bool = False


class DescribeRefreshRequest(BaseModel):
    # Bypass the description cache — every eligible entity is regenerated.
    force: bool = False
