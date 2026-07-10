"""
Routines API endpoints — code review / bug hunt runs.

POST   /routines/runs           — launch a review run as a background job
                                   (bughunt: 400, not yet — later release)
GET    /routines/runs           — list runs, each with severity_counts
GET    /routines/runs/{run_id}  — a run's full detail, findings included
DELETE /routines/runs/{run_id}  — delete a run (cascades to its findings)
GET    /routines/git/branches   — a project's local branches + default base
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from models.dtos.routines import (
    BranchInfo,
    FindingDTO,
    LaunchRoutineRequest,
    LaunchRoutineResponse,
    RoutineRunDetail,
    RoutineRunSummary,
)
from services.auth import require_auth, require_role
from services.db import DatabaseUnavailableError
from services.git_repo import GitError, default_base, is_git_repo, list_branches, resolve_ref
from services.graph_store import GraphNotFoundError, _derive_repo_path, get_graph
from services.jobs import ProjectJobConflict
from services.routines import store
from services.routines.engine import submit_routine
from services.routines.review import _resolve_llm, run_review

router = APIRouter(prefix="/routines", dependencies=[Depends(require_auth)])

# Launching/deleting runs mutate state — role 'viewer' is read-only (6k/12p).
require_write = require_role("admin", "user")


def _resolve_repo_path(project: str) -> str:
    """
    Resolve *project*'s repo root from its ingested graph.

    A leaner accessor than ``graph_store.list_graphs()`` (which loads every
    ingested project's graph and pulls Weaviate chunk counts just to list
    them) — a routine launch only ever needs the one project's repo_path.
    Reuses ``get_graph``'s verbatim not-found message (the same "no graph
    found for project ... — run POST /instances/pipeline" text every other
    graph-backed endpoint already surfaces) and the same per-project
    derivation ``list_graphs`` uses (``graph_store._derive_repo_path``).
    """
    try:
        graph = get_graph(project)
    except GraphNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    repo_path = _derive_repo_path(graph)
    if repo_path is None:
        raise HTTPException(
            status_code=400,
            detail=f"project '{project}' has an ingested graph but no derivable repo_path "
            "(no File node carries both a relative path and an absolute_path) "
            "— re-ingest with POST /instances/pipeline",
        )
    return repo_path


@router.post("/runs", dependencies=[Depends(require_write)], status_code=202)
async def launchRoutine(data: LaunchRoutineRequest) -> LaunchRoutineResponse:
    """
    Launch a routine run as a background job. Poll
    ``GET /routines/runs/{run_id}`` (or the shared ``GET /instances/jobs/{job_id}``)
    for progress and the final result.

    ``kind='bughunt'`` is honest-400ed — that routine arrives in a later
    release. For ``kind='review'``, pre-flights run BEFORE the job is
    created, in order, so a bad request fails synchronously with an
    actionable message rather than only surfacing inside the job's result:
    the project must have an ingested graph (404), its repo_path must be a
    git repository (400), ``base_ref``/``target_ref`` must resolve (400),
    and the LLM (from the launch payload or the active config, 12h) must be
    resolvable and healthy (400) — the same resolution
    ``services.routines.review.run_review`` itself does, run here too so its
    failure message reaches the caller synchronously instead of only inside
    a failed job/run.
    """
    if data.kind == "bughunt":
        raise HTTPException(status_code=400, detail="bug hunt routines arrive in a later release")

    repo_path = _resolve_repo_path(data.project)

    try:
        if not is_git_repo(repo_path):
            raise GitError(f"'{repo_path}' is not a git repository — check the project's repo_path")
        base = data.base_ref or default_base(repo_path)
        resolve_ref(repo_path, base)
        resolve_ref(repo_path, data.target_ref)
    except GitError as e:
        raise HTTPException(status_code=400, detail=str(e))

    params = {
        "repo_path": repo_path,
        "target_ref": data.target_ref,
        "base_ref": data.base_ref,
        "model": data.model,
        "provider": data.provider,
    }
    try:
        await _resolve_llm(params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = await submit_routine("review", data.project, repo_path, params, run_review)
    except ProjectJobConflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return LaunchRoutineResponse(**result)


@router.get("/runs", response_model=list[RoutineRunSummary])
async def listRoutineRuns(
    project: str | None = Query(default=None),
    kind: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    """List routine runs, most recent first, each with per-severity finding counts."""
    try:
        runs = await store.list_runs(project=project, kind=kind, limit=limit)
        counts = await store.severity_counts_for_runs([r["id"] for r in runs])
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return [
        RoutineRunSummary(
            id=r["id"],
            kind=r["kind"],
            project=r["project"],
            status=r["status"],
            params=r["params"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            severity_counts=counts.get(r["id"], {}),
        )
        for r in runs
    ]


@router.get("/runs/{run_id}", response_model=RoutineRunDetail)
async def getRoutineRun(run_id: str):
    """A routine run's full detail, findings included."""
    try:
        run = await store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"routine run '{run_id}' not found")
        findings = await store.list_findings(run_id)
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return RoutineRunDetail(
        id=run["id"],
        kind=run["kind"],
        project=run["project"],
        status=run["status"],
        params=run["params"],
        summary=run["summary"],
        stats=run["stats"],
        error=run["error"],
        created_at=run["created_at"],
        updated_at=run["updated_at"],
        findings=[FindingDTO(**f) for f in findings],
    )


@router.delete("/runs/{run_id}", dependencies=[Depends(require_write)])
async def deleteRoutineRun(run_id: str):
    """Delete a routine run, cascading to its findings."""
    try:
        deleted = await store.delete_run(run_id)
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail=f"routine run '{run_id}' not found")
    return {"message": "Routine run deleted", "run_id": run_id}


@router.get("/git/branches")
async def getBranches(project: str = Query(...)):
    """A project's local branches plus its resolved default base branch."""
    repo_path = _resolve_repo_path(project)
    try:
        branches = list_branches(repo_path)
        base = default_base(repo_path)
    except GitError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "branches": [BranchInfo(**b) for b in branches],
        "default_base": base,
    }
