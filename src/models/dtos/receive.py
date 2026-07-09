from pydantic import BaseModel


class IngestRequest(BaseModel):
    repo_path: str
    # Drop & rebuild the collection before inserting — needed once after a
    # schema change (e.g. adopting named vectors). Wipes ALL projects.
    recreate: bool = False
    # Skip per-entity LLM descriptions for this run. Opt-in only — the
    # descriptor pre-flight (and its 400) is skipped only when explicitly
    # set to false.
    describe: bool = True


class DescribeRefreshRequest(BaseModel):
    # Bypass the description cache — every eligible entity is regenerated.
    force: bool = False


class GraphOnlyRequest(BaseModel):
    repo_path: str
