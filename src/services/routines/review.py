"""
Review routine — the ``run_review`` :class:`~services.routines.engine.RoutineCallable`
plugged into ``services.routines.engine.submit_routine`` for ``kind="review"``.

This module owns the whole review pipeline; it currently implements only the
first two of its stages:

  1. **diff** — resolve base/target refs, compute the unified diff, parse it,
     and classify each file as reviewable or skipped (binary/deleted/
     mode-only), guarding against absurdly large or empty ranges (9h).
  2. **context** — for each reviewable file, assemble a
     :class:`ReviewFileContext`: line-numbered excerpts of the target-ref
     content around the changed ranges, the graph entities whose line span
     overlaps those ranges, and — for a capped subset of those entities —
     their Weaviate description plus cheap 1-hop call-graph neighbors.

Stages 3+ (analyze/summarize, A5) land as further private ``_stage_*``
helpers called from :func:`run_review`; ``_stage_diff``/``_stage_context``
are written to be reusable by that code (they return plain data, not
Job/DB-coupled state) rather than folded inline into ``run_review``.

Everything here runs synchronously inside the job's worker thread (the sync
Weaviate client and ``git_repo``'s subprocess calls are fine there — see
``services.jobs``'s module docstring); ``run_review`` itself is a coroutine
only because that's the shape ``routines.engine`` expects (``asyncio.run`` on
a dedicated per-job loop), not because it awaits anything yet — A5's LLM
calls are expected to be the first real ``await`` in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from services import graph_store
from services.chunker import CODE_ENTITY_LABELS
from services.git_repo import (
    FileDiff,
    GitError,
    default_base,
    diff,
    diff_stat,
    is_git_repo,
    merge_base,
    parse_unified_diff,
    resolve_ref,
    show_file,
)
from services.jobs import Job
from services.retriever import Retriever
from services.weaviate_client import get_client

# Budgets (9h) — module constants so A5/the gate step can tune them without
# hunting through the stage bodies.
MAX_FILES = 60                    # changed-files guard: raise above this
CONTEXT_CHARS_PER_FILE = 6_000    # soft cap on target_content_excerpts per file
MAX_RETRIEVED_PER_FILE = 5        # cap on Weaviate lookups per file
_EXCERPT_PAD_LINES = 10           # lines of context padded around each changed range


class ReviewError(RuntimeError):
    """Raised for a review-routine-level failure (as opposed to a raw git
    failure, which surfaces as :class:`services.git_repo.GitError`)."""


@dataclass
class ReviewFileContext:
    """Assembled context for one reviewable file, built by the 'context' stage."""

    path: str
    target_content_excerpts: list[str] = field(default_factory=list)
    graph_entities: list[str] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    truncated: bool = False


# ---------------------------------------------------------------------------
# Stage 1: diff
# ---------------------------------------------------------------------------


def _classify(file_diffs: list[FileDiff]) -> tuple[list[FileDiff], dict[str, int]]:
    """
    Split *file_diffs* into ``(reviewable, skip_counts)``.

    Skipped: binary files, deleted files (nothing to review on the new
    side), and mode/rename-only entries (no hunks — no content changed).
    """
    reviewable: list[FileDiff] = []
    skip_counts = {"binary": 0, "deleted": 0, "mode_only": 0}
    for fd in file_diffs:
        if fd.is_binary:
            skip_counts["binary"] += 1
        elif fd.is_deleted:
            skip_counts["deleted"] += 1
        elif not fd.hunks:
            skip_counts["mode_only"] += 1
        else:
            reviewable.append(fd)
    return reviewable, skip_counts


def _stage_diff(
    repo_path: str, base_ref: str | None, target_ref: str, emit: Callable[[str, str], None]
) -> tuple[str, str, list[FileDiff], dict]:
    """
    Run the 'diff' stage: resolve refs, diff, parse, classify, guard.

    Returns ``(base, target, reviewable_file_diffs, stats)``. Raises
    :class:`GitError` for git-level failures (bad repo/ref) and
    :class:`ReviewError` for the empty-diff / too-many-files guards (9h).
    """
    if not is_git_repo(repo_path):
        raise GitError(f"'{repo_path}' is not a git repository — check the project's repo_path")

    base = base_ref or default_base(repo_path)
    target = target_ref

    base_sha = resolve_ref(repo_path, base)
    target_sha = resolve_ref(repo_path, target)
    emit(
        "diff",
        f"resolving refs: base='{base}' ({base_sha[:8]}) target='{target}' ({target_sha[:8]})",
    )
    merge_base(repo_path, base, target)  # validates the pair resolves to a common ancestor

    diff_text = diff(repo_path, base, target)
    file_diffs = parse_unified_diff(diff_text)

    if not file_diffs:
        raise ReviewError(f"no changes between '{base}' and '{target}'")
    if len(file_diffs) > MAX_FILES:
        raise ReviewError(
            f"{len(file_diffs)} files changed between '{base}' and '{target}' — "
            f"exceeds the {MAX_FILES}-file review limit; narrow the range "
            "(review against a more recent base, or split the change into smaller PRs)"
        )

    stat = diff_stat(repo_path, base, target)
    # len(file_diffs), not numstat's count — the parser keeps mode-only
    # entries that numstat omits, and the persisted stats use the parser.
    emit(
        "diff",
        f"diffstat: {len(file_diffs)} files changed, "
        f"+{stat['insertions']} -{stat['deletions']}",
    )

    reviewable, skip_counts = _classify(file_diffs)
    for fd in file_diffs:
        if fd.is_binary:
            emit("diff", f"skipping binary file: {fd.path}")
        elif fd.is_deleted:
            emit("diff", f"skipping deleted file (nothing to review on the new side): {fd.path}")
        elif not fd.hunks:
            emit("diff", f"skipping mode/rename-only file (no content change): {fd.path}")

    stats = {
        "files_changed": len(file_diffs),
        "files_reviewable": len(reviewable),
        "files_skipped": sum(skip_counts.values()),
        "files_skipped_binary": skip_counts["binary"],
        "files_skipped_deleted": skip_counts["deleted"],
        "files_skipped_mode_only": skip_counts["mode_only"],
        "hunks": sum(len(fd.hunks) for fd in file_diffs),
        "insertions": stat["insertions"],
        "deletions": stat["deletions"],
    }
    return base, target, reviewable, stats


# ---------------------------------------------------------------------------
# Stage 2: context
# ---------------------------------------------------------------------------


def _normalize_path(path: str) -> str:
    """Light normalization for exact-match comparison against graph paths."""
    p = path.strip()
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent (inclusive) line ranges, sorted by start."""
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged = [ordered[0]]
    for s, e in ordered[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + 1:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def _render_ranges(lines: list[str], ranges: list[tuple[int, int]]) -> list[str]:
    return [
        "\n".join(f"{ln}: {lines[ln - 1]}" for ln in range(s, e + 1))
        for s, e in ranges
    ]


def _build_excerpts(content: str, changed_ranges: list[tuple[int, int]]) -> tuple[list[str], bool]:
    """
    Line-numbered excerpts of *content* covering each of *changed_ranges*
    padded by :data:`_EXCERPT_PAD_LINES` on each side, overlapping/adjacent
    ranges merged. If the total size exceeds :data:`CONTEXT_CHARS_PER_FILE`,
    every range is shrunk proportionally (trimmed evenly from both ends) and
    the second return value is ``True``.
    """
    if not changed_ranges:
        return [], False
    # Split on "\n" only — parse_unified_diff numbers lines the same way;
    # splitlines() would also break on form feeds etc. and drift the numbering.
    lines = content.split("\n")
    n = len(lines)
    if n == 0:
        return [], False

    padded = [(max(1, s - _EXCERPT_PAD_LINES), min(n, e + _EXCERPT_PAD_LINES)) for s, e in changed_ranges]
    merged = _merge_ranges(padded)

    excerpts = _render_ranges(lines, merged)
    total = sum(len(x) for x in excerpts)
    if total <= CONTEXT_CHARS_PER_FILE:
        return excerpts, False

    scale = CONTEXT_CHARS_PER_FILE / total
    trimmed: list[tuple[int, int]] = []
    for s, e in merged:
        length = e - s + 1
        new_length = max(1, round(length * scale))
        excess = length - new_length
        trim_start = excess // 2
        new_s = s + trim_start
        new_e = new_s + new_length - 1
        trimmed.append((new_s, new_e))

    return _render_ranges(lines, trimmed), True


def _match_entities(
    graph: graph_store.Graph, file_path: str, changed_ranges: list[tuple[int, int]]
) -> list[dict]:
    """
    Code-entity nodes in *graph* whose ``path`` matches *file_path* and whose
    line span overlaps any of *changed_ranges*. Sorted by ``start_line``.
    """
    target = _normalize_path(file_path)
    matches = []
    for node in graph.nodes:
        if node["labels"][0] not in CODE_ENTITY_LABELS:
            continue
        props = node.get("properties", {})
        node_path = props.get("path")
        if not node_path or _normalize_path(node_path) != target:
            continue
        start, end = props.get("start_line"), props.get("end_line")
        if start is None or end is None:
            continue
        if any(start <= re_ and end >= rs for rs, re_ in changed_ranges):
            matches.append(node)
    matches.sort(key=lambda n: n.get("properties", {}).get("start_line") or 0)
    return matches


def _call_neighbors(graph: graph_store.Graph, node_id: int) -> list[dict]:
    """1-hop CALLS neighbors of *node_id* — who it calls and who calls it — as
    cheap ``{"qualified_name", "kind", "relation"}`` entries."""
    neighbors = []
    for rtype, other_id in graph.outgoing.get(node_id, []):
        if rtype != "CALLS":
            continue
        other = graph.node_by_id.get(other_id)
        if other is not None:
            neighbors.append(
                {
                    "qualified_name": other.get("properties", {}).get("qualified_name"),
                    "kind": other["labels"][0],
                    "relation": "calls",
                }
            )
    for rtype, other_id in graph.incoming.get(node_id, []):
        if rtype != "CALLS":
            continue
        other = graph.node_by_id.get(other_id)
        if other is not None:
            neighbors.append(
                {
                    "qualified_name": other.get("properties", {}).get("qualified_name"),
                    "kind": other["labels"][0],
                    "relation": "called_by",
                }
            )
    return neighbors


def _build_file_context(
    repo_path: str,
    target: str,
    fd: FileDiff,
    graph: graph_store.Graph | None,
    retriever: Retriever | None,
    project: str,
    emit: Callable[[str, str], None],
) -> ReviewFileContext:
    ctx = ReviewFileContext(path=fd.path)

    try:
        content = show_file(repo_path, target, fd.path)
    except GitError as e:
        emit("context", f"context: {fd.path}: failed to read target content ({e}) — skipping excerpts")
        content = None

    if content is None:
        emit("context", f"context: {fd.path}: target content unavailable — skipping excerpts")
    else:
        excerpts, truncated = _build_excerpts(content, fd.changed_line_ranges)
        ctx.target_content_excerpts = excerpts
        ctx.truncated = truncated
        if truncated:
            emit(
                "context",
                f"context: {fd.path}: excerpts truncated to fit the {CONTEXT_CHARS_PER_FILE}-char budget",
            )

    if graph is None:
        return ctx

    matches = _match_entities(graph, fd.path, fd.changed_line_ranges)
    ctx.graph_entities = [
        qn
        for n in matches
        if (qn := n.get("properties", {}).get("qualified_name"))
    ]

    if retriever is None:
        return ctx

    for node in matches[:MAX_RETRIEVED_PER_FILE]:
        qn = node.get("properties", {}).get("qualified_name")
        if not qn:
            continue
        description = None
        try:
            chunk = retriever.get_by_qualified_name(qn, project)
        except Exception as e:
            emit(
                "context",
                f"context: {fd.path}: weaviate lookup failed for '{qn}' ({e}) — continuing without it",
            )
            chunk = None
        if chunk is not None:
            description = chunk.description or None
            if not description and chunk.chunk_text:
                description = chunk.chunk_text[:500]
        ctx.retrieved.append(
            {
                "qualified_name": qn,
                "kind": node["labels"][0],
                "description": description,
                "neighbors": _call_neighbors(graph, node["node_id"]),
            }
        )

    return ctx


def _stage_context(
    repo_path: str,
    target: str,
    reviewable: list[FileDiff],
    project: str,
    emit: Callable[[str, str], None],
) -> tuple[list[ReviewFileContext], int, int]:
    """
    Run the 'context' stage over every reviewable file.

    Returns ``(contexts, context_chars_total, entities_matched)``. Degrades
    gracefully — never raises — when the project graph or Weaviate is
    unavailable: the diff is the routine's core input, context is an
    enhancement (see module docstring).
    """
    n = len(reviewable)

    try:
        graph: graph_store.Graph | None = graph_store.get_graph(project)
    except graph_store.GraphNotFoundError as e:
        emit(
            "context",
            f"context: no graph available for project '{project}' ({e}) — "
            "continuing without graph/entity enrichment",
        )
        graph = None

    client = None
    if graph is not None:
        try:
            client = get_client()
        except Exception as e:
            emit(
                "context",
                f"context: Weaviate unreachable ({e}) — continuing without retrieved descriptions",
            )
    retriever = Retriever(client) if client is not None else None

    contexts: list[ReviewFileContext] = []
    context_chars_total = 0
    entities_matched = 0
    try:
        for i, fd in enumerate(reviewable, 1):
            emit("context", f"context: {fd.path} ({i}/{n})")
            ctx = _build_file_context(repo_path, target, fd, graph, retriever, project, emit)
            contexts.append(ctx)
            context_chars_total += sum(len(x) for x in ctx.target_content_excerpts)
            entities_matched += len(ctx.graph_entities)
    finally:
        if client is not None:
            client.close()

    return contexts, context_chars_total, entities_matched


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_review(job: Job, run_id: str, params: dict, emit: Callable[[str, str], None]) -> dict:
    """
    The review routine's :class:`~services.routines.engine.RoutineCallable`.

    params: ``{repo_path, base_ref?, target_ref, model?, provider?}`` —
    ``model``/``provider`` are carried through for the analyze/summarize
    stages (A5) but unused here.

    For now (stages 1-2 only) returns a placeholder result: no findings, a
    summary noting analysis isn't implemented yet, and stats reflecting the
    real diff/context work done.
    """
    repo_path = params["repo_path"]
    target_ref = params["target_ref"]
    base_ref = params.get("base_ref")

    base, target, reviewable, diff_stats = _stage_diff(repo_path, base_ref, target_ref, emit)
    _contexts, context_chars_total, entities_matched = _stage_context(
        repo_path, target, reviewable, job.project, emit
    )

    stats = {
        **diff_stats,
        "entities_matched": entities_matched,
        "context_chars_total": context_chars_total,
    }

    return {
        "summary": "analysis stages not yet implemented",
        "findings": [],
        "stats": stats,
    }
