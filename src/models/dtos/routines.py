"""
DTOs for the routines endpoints (code review / bug hunt runs).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator


class LaunchRoutineRequest(BaseModel):
    kind: Literal["review", "bughunt"]
    project: str
    # Required for kind='review' (validated below); unused for 'bughunt' in
    # sub-part A.
    target_ref: str | None = None
    base_ref: str | None = None
    # bughunt only — unused in sub-part A (bughunt itself 400s at launch).
    scope: str | None = None
    # Override the active LLM config (12h) for this run — see
    # services.routines.review._resolve_llm.
    model: str | None = None
    provider: str | None = None
    # Playbooks PUSHED into this run's analyze prompt (decision 9d) — names,
    # selection order = priority order. Validated at launch (controller): at
    # most 4, every name must exist, and each must apply to this run's
    # routine kind.
    playbooks: list[str] | None = None

    @model_validator(mode="after")
    def _require_target_ref_for_review(self) -> "LaunchRoutineRequest":
        if self.kind == "review" and not self.target_ref:
            raise ValueError("target_ref is required for kind='review'")
        return self


class LaunchRoutineResponse(BaseModel):
    run_id: str
    job_id: str


class RoutineRunSummary(BaseModel):
    id: str
    kind: str
    project: str
    status: str
    params: dict
    created_at: str
    updated_at: str
    # severity -> count, computed via one GROUP BY query over the run's
    # findings (see services.routines.store.severity_counts_for_runs).
    severity_counts: dict[str, int]


class FindingDTO(BaseModel):
    id: int
    file_path: str
    start_line: int | None = None
    end_line: int | None = None
    severity: str
    category: str
    title: str
    body: str
    suggestion: str | None = None
    evidence: dict | None = None
    anchored: bool
    created_at: str


class RoutineRunDetail(BaseModel):
    id: str
    kind: str
    project: str
    status: str
    params: dict
    summary: str | None = None
    stats: dict | None = None
    error: str | None = None
    created_at: str
    updated_at: str
    findings: list[FindingDTO]


class BranchInfo(BaseModel):
    name: str
    is_current: bool
    last_commit_date: str


# -------------------------------------------------------------------------
# Playbooks / skills (decision 9d) — B2
# -------------------------------------------------------------------------


class PlaybookSummary(BaseModel):
    """List-item shape: omits ``content`` (up to 16KB each) in favor of
    ``content_bytes``, so listing every playbook stays cheap. Full content
    is only on the GET-by-name detail (:class:`PlaybookDTO`)."""

    id: int
    name: str
    routines: list[str]
    content_bytes: int
    created_at: str
    updated_at: str


class PlaybookDTO(BaseModel):
    id: int
    name: str
    routines: list[str]
    content: str
    created_at: str
    updated_at: str


class CreatePlaybookRequest(BaseModel):
    name: str
    content: str
    routines: list[str] | None = None


class UpdatePlaybookRequest(BaseModel):
    content: str | None = None
    routines: list[str] | None = None


class SkillSummary(BaseModel):
    """List-item shape: omits ``content`` in favor of ``content_bytes`` —
    see :class:`PlaybookSummary`. Full content is only on the GET-by-name
    detail (:class:`SkillDTO`)."""

    id: int
    name: str
    description: str
    content_bytes: int
    created_at: str
    updated_at: str


class SkillDTO(BaseModel):
    id: int
    name: str
    description: str
    content: str
    created_at: str
    updated_at: str


class CreateSkillRequest(BaseModel):
    name: str
    description: str
    content: str


class UpdateSkillRequest(BaseModel):
    description: str | None = None
    content: str | None = None
