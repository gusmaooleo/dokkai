"""
Routines API endpoints — code review / bug hunt runs, plus their attachable
playbooks and skills documents (decision 9d).

POST   /routines/runs                 — launch a review run as a background
                                         job (bughunt: 400, not yet — later
                                         release)
GET    /routines/runs                 — list runs, each with severity_counts
GET    /routines/runs/{run_id}        — a run's full detail, findings included
DELETE /routines/runs/{run_id}        — delete a run (cascades to its findings)
GET    /routines/git/branches         — a project's local branches + default base
GET    /routines/playbooks            — list playbooks (no content, with content_bytes)
GET    /routines/playbooks/{name}     — a playbook's full detail (with content)
POST   /routines/playbooks            — create a playbook
PATCH  /routines/playbooks/{name}     — update a playbook's content/routines
DELETE /routines/playbooks/{name}     — delete a playbook
GET    /routines/skills               — list skills (no content, with content_bytes)
GET    /routines/skills/{name}        — a skill's full detail (with content)
POST   /routines/skills               — create a skill
PATCH  /routines/skills/{name}        — update a skill's description/content
DELETE /routines/skills/{name}        — delete a skill
"""

from __future__ import annotations

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query

from models.dtos.routines import (
    BranchInfo,
    CreatePlaybookRequest,
    CreateSkillRequest,
    FindingDTO,
    LaunchRoutineRequest,
    LaunchRoutineResponse,
    PlaybookDTO,
    PlaybookSummary,
    RoutineRunDetail,
    RoutineRunSummary,
    SkillDTO,
    SkillSummary,
    UpdatePlaybookRequest,
    UpdateSkillRequest,
)
from services.auth import require_auth, require_role
from services.db import DatabaseUnavailableError
from services.git_repo import GitError, default_base, is_git_repo, list_branches, resolve_ref
from services.graph_store import GraphNotFoundError, _derive_repo_path, get_graph
from services.jobs import ProjectJobConflict
from services.routines import documents, store
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
    every named playbook (decision 9d) must exist and apply to 'review' with
    at most 4 selected (400), and the LLM (from the launch payload or the
    active config, 12h) must be resolvable and healthy (400) — the same
    resolution ``services.routines.review.run_review`` itself does, run here
    too so its failure message reaches the caller synchronously instead of
    only inside a failed job/run.
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

    playbook_names = data.playbooks or []
    if len(playbook_names) > 4:
        raise HTTPException(status_code=400, detail="a run can use at most 4 playbooks")
    try:
        for name in playbook_names:
            playbook = await documents.get_playbook(name)
            if playbook is None:
                raise HTTPException(status_code=400, detail=f"playbook '{name}' not found")
            if "review" not in playbook["routines"]:
                raise HTTPException(
                    status_code=400, detail=f"playbook '{name}' does not apply to review routines"
                )
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))

    params = {
        "repo_path": repo_path,
        "target_ref": data.target_ref,
        "base_ref": data.base_ref,
        "model": data.model,
        "provider": data.provider,
        "playbooks": data.playbooks,
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


# -------------------------------------------------------------------------
# Playbooks / skills (decision 9d) — GETs are read-only (viewer allowed),
# POST/PATCH/DELETE mutate state and require role admin/user (6k/12p).
# -------------------------------------------------------------------------


@router.get("/playbooks", response_model=list[PlaybookSummary])
async def listPlaybooks():
    """List playbooks. Omits ``content`` in favor of ``content_bytes`` — use
    ``GET /routines/playbooks/{name}`` for the full content."""
    try:
        rows = await documents.list_playbooks()
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return [
        PlaybookSummary(
            id=r["id"],
            name=r["name"],
            routines=r["routines"],
            content_bytes=len(r["content"].encode("utf-8")),
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]


@router.get("/playbooks/{name}", response_model=PlaybookDTO)
async def getPlaybook(name: str):
    """A playbook's full detail, content included."""
    try:
        row = await documents.get_playbook(name)
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if row is None:
        raise HTTPException(status_code=404, detail=f"playbook '{name}' not found")
    return PlaybookDTO(**row)


@router.post(
    "/playbooks", response_model=PlaybookDTO, status_code=201, dependencies=[Depends(require_write)]
)
async def createPlaybook(data: CreatePlaybookRequest):
    """Create a playbook."""
    try:
        row = await documents.create_playbook(data.name, data.content, data.routines)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except asyncpg.UniqueViolationError:
        raise HTTPException(status_code=409, detail=f"playbook '{data.name}' already exists")
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return PlaybookDTO(**row)


@router.patch(
    "/playbooks/{name}", response_model=PlaybookDTO, dependencies=[Depends(require_write)]
)
async def updatePlaybook(name: str, data: UpdatePlaybookRequest):
    """Update a playbook's content and/or routines. Fields left unset are
    left unchanged."""
    try:
        row = await documents.update_playbook(name, data.content, data.routines)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if row is None:
        raise HTTPException(status_code=404, detail=f"playbook '{name}' not found")
    return PlaybookDTO(**row)


@router.delete("/playbooks/{name}", dependencies=[Depends(require_write)])
async def deletePlaybook(name: str):
    """Delete a playbook."""
    try:
        deleted = await documents.delete_playbook(name)
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail=f"playbook '{name}' not found")
    return {"message": "Playbook deleted", "name": name}


@router.get("/skills", response_model=list[SkillSummary])
async def listSkills():
    """List skills. Omits ``content`` in favor of ``content_bytes`` — use
    ``GET /routines/skills/{name}`` for the full content."""
    try:
        rows = await documents.list_skills()
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return [
        SkillSummary(
            id=r["id"],
            name=r["name"],
            description=r["description"],
            content_bytes=len(r["content"].encode("utf-8")),
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]


@router.get("/skills/{name}", response_model=SkillDTO)
async def getSkill(name: str):
    """A skill's full detail, content included."""
    try:
        row = await documents.get_skill(name)
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if row is None:
        raise HTTPException(status_code=404, detail=f"skill '{name}' not found")
    return SkillDTO(**row)


@router.post(
    "/skills", response_model=SkillDTO, status_code=201, dependencies=[Depends(require_write)]
)
async def createSkill(data: CreateSkillRequest):
    """Create a skill."""
    try:
        row = await documents.create_skill(data.name, data.description, data.content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except asyncpg.UniqueViolationError:
        raise HTTPException(status_code=409, detail=f"skill '{data.name}' already exists")
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return SkillDTO(**row)


@router.patch("/skills/{name}", response_model=SkillDTO, dependencies=[Depends(require_write)])
async def updateSkill(name: str, data: UpdateSkillRequest):
    """Update a skill's description and/or content. Fields left unset are
    left unchanged."""
    try:
        row = await documents.update_skill(name, data.description, data.content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if row is None:
        raise HTTPException(status_code=404, detail=f"skill '{name}' not found")
    return SkillDTO(**row)


@router.delete("/skills/{name}", dependencies=[Depends(require_write)])
async def deleteSkill(name: str):
    """Delete a skill."""
    try:
        deleted = await documents.delete_skill(name)
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail=f"skill '{name}' not found")
    return {"message": "Skill deleted", "name": name}
