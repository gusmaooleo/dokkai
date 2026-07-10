"""
Review routine — the ``run_review`` :class:`~services.routines.engine.RoutineCallable`
plugged into ``services.routines.engine.submit_routine`` for ``kind="review"``.

This module owns the whole review pipeline, in five stages:

  1. **diff** — resolve base/target refs, compute the unified diff, parse it,
     and classify each file as reviewable or skipped (binary/deleted/
     mode-only), guarding against absurdly large or empty ranges (9h).
  2. **context** — for each reviewable file, assemble a
     :class:`ReviewFileContext`: line-numbered excerpts of the target-ref
     content around the changed ranges, the graph entities whose line span
     overlaps those ranges, and — for a capped subset of those entities —
     their Weaviate description plus cheap 1-hop call-graph neighbors.
  3. **playbooks** — only when ``params['playbooks']`` names are given
     (decision 9d, sub-part B): fetch them (priority = selection order) and
     inject their bodies into the analyze system prompt (frontmatter
     stripped), capped at :data:`PLAYBOOK_INJECTION_MAX_CHARS`. A no-op
     stage otherwise — the prompt is byte-identical to sub-part A.
  4. **analyze** — one chat call per reviewable file: a strict JSON-array
     findings contract, parsed with a one-shot retry on invalid JSON, then
     validated/anchored against the file's changed-line ranges (9a/9h).
  5. **summarize** — one chat call turning the diffstat + validated findings
     into the run's markdown summary. Playbooks are NOT injected here — this
     stage judges findings, not code, so it stays lean.

``_stage_diff``/``_stage_context`` return plain data (not Job/DB-coupled
state) so they're reusable outside ``run_review`` (see
``scripts/test_review_stages.py``); the analyze/summarize helpers below
follow the same shape.

Everything here runs synchronously inside the job's worker thread (the sync
Weaviate client and ``git_repo``'s subprocess calls are fine there — see
``services.jobs``'s module docstring) EXCEPT the analyze/summarize LLM calls,
which are real ``await``s (``provider.chat()``) — ``run_review`` is a
coroutine for exactly that reason.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
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
from services.llm_config import get_config_store
from services.llm_provider import LLMMessage, LLMProvider, get_provider
from services.retriever import Retriever
from services.routines.documents import fetch_playbooks_sync, split_frontmatter
from services.weaviate_client import get_client

# Budgets (9h) — module constants so A5/the gate step can tune them without
# hunting through the stage bodies.
MAX_FILES = 60                    # changed-files guard: raise above this
CONTEXT_CHARS_PER_FILE = 6_000    # soft cap on target_content_excerpts per file
MAX_RETRIEVED_PER_FILE = 5        # cap on Weaviate lookups per file
_EXCERPT_PAD_LINES = 10           # lines of context padded around each changed range

MAX_USER_PROMPT_CHARS = 24_000    # final guard on the analyze per-file user prompt (num_ctx 8192 budget)
PLAYBOOK_INJECTION_MAX_CHARS = 12_000  # cap on total playbook body chars injected into the analyze
                                        # system prompt (9h/9d) — 4 playbooks x 16KB raw (64KB) would
                                        # swamp an 8192-token num_ctx; the system prompt already
                                        # competes with a 24k-char user prompt, so 12k here leaves
                                        # headroom. Lowest-priority (last-selected) playbooks are
                                        # truncated first when the combined bodies exceed this.
ANALYZE_TEMPERATURE = 0.15        # low temperature — consistency over creativity for findings
SUMMARIZE_TEMPERATURE = 0.2
MAX_TOKENS_ANALYZE = 4096         # provider default; a file's findings array rarely needs more
MAX_TOKENS_SUMMARIZE = 1024       # the summary is a short markdown blurb, not another findings dump
ANCHOR_TOLERANCE_LINES = 2        # ±N lines of slack when checking a finding's start_line against a changed range


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
# LLM resolution (9a-3 / 12h)
# ---------------------------------------------------------------------------


async def _resolve_llm(params: dict, *, noun: str = "review") -> tuple[LLMProvider, str, str]:
    """
    Resolve ``(provider, model, provider_name)`` for the analyze/summarize
    stages: ``params['model']``/``params['provider']`` override the ACTIVE
    LLM config (12h) when given, falling back to it field-by-field
    otherwise. A remote-provider override reuses the active config's
    key/base_url only when it targets the SAME provider — a cross-provider
    override with no stored credentials for that provider fails with
    ``get_provider``'s own actionable message (e.g. "OpenAI requires an API
    key"), which is left to propagate verbatim rather than being caught here
    (see module docstring / A5 point 3: the run just fails with that
    message).

    Also runs a one-time ``health_check(model)`` preflight (D-N2 discipline,
    mirroring ``services.describe._require_provider``) so a bad model
    override fails with an actionable message up front rather than as a
    raw HTTP error mid-analyze.

    *noun* names the routine kind in the "no provider configured" message
    below (default ``"review"`` — this function's original/only caller for a
    long time, so the review-path message stays byte-identical; documented
    API behavior other tooling depends on). ``services.routines.bughunt``
    passes ``noun="bug hunt"`` when it reuses this same resolution chain.

    Raises :class:`ValueError` if neither the params nor the active config
    supply a provider+model, or if the resolved model isn't actually
    available.
    """
    config = get_config_store().get_llm_config()
    provider_name = (params.get("provider") or (config.provider_name if config else None) or "").strip()
    model = params.get("model") or (config.model if config else None)
    if not provider_name or not model:
        raise ValueError(
            f"No LLM provider configured for this {noun}. "
            "POST to /config/llm to set one up, or pass model/provider in the launch payload."
        )

    same_provider = config is not None and config.provider_name == provider_name
    key = config.key if same_provider else ""
    base_url = config.base_url if same_provider else None
    provider = get_provider(provider_name, api_key=key, base_url=base_url)

    available, reason = await provider.health_check(model)
    if not available:
        if provider_name == "ollama":
            resolved_base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            raise ValueError(
                f"model '{model}' is not available in Ollama at {resolved_base_url} — "
                f"pull it with 'ollama pull {model}'"
            )
        raise ValueError(f"model '{model}' unavailable for provider '{provider_name}': {reason}")

    return provider, model, provider_name


# ---------------------------------------------------------------------------
# Stage 3: playbooks (decision 9d, sub-part B) — PUSHED project rules
# ---------------------------------------------------------------------------

_PLAYBOOK_MARKER = (
    "<!-- PLAYBOOK INJECTION POINT (sub-part B): project-specific review "
    "playbooks/rules land here. Empty in sub-part A. -->"
)


async def _stage_playbooks(names: list[str], emit: Callable[[str, str], None]) -> list[dict]:
    """
    Run the 'playbooks' stage: fetch *names* (selection order = priority,
    decision 9d) via ``documents.fetch_playbooks_sync``, emitting
    ``'loaded playbook <name>'`` per fetched doc (mirrors the future 'loaded
    skill' observability). A no-op returning ``[]`` when *names* is empty —
    the stage simply doesn't run.

    Uses ``asyncio.to_thread`` rather than calling ``fetch_playbooks_sync``
    directly: that helper does its own internal ``asyncio.run()``, and
    ``run_review`` is itself already executing inside an event loop (the
    worker thread's own ``asyncio.run(core(...))`` in
    ``services.routines.engine``) — nesting ``asyncio.run()`` in the same
    thread raises. Running it in a fresh thread sidesteps that.
    """
    if not names:
        return []
    playbooks = await asyncio.to_thread(fetch_playbooks_sync, names)
    for pb in playbooks:
        emit("playbooks", f"loaded playbook {pb['name']}")
    return playbooks


def _build_analyze_system_prompt(playbooks: list[dict]) -> tuple[str, bool]:
    """
    Inject *playbooks* (in priority order) into :data:`_ANALYZE_SYSTEM_PROMPT`
    at :data:`_PLAYBOOK_MARKER`, under a '## House rules and focus areas
    (user-provided)' section, each playbook as '### <name>' followed by its
    BODY (frontmatter stripped via :func:`services.routines.documents.split_frontmatter`).

    Total injected chars are capped at :data:`PLAYBOOK_INJECTION_MAX_CHARS`
    — playbooks are included greedily in priority order, and the LAST
    (lowest-priority) ones are truncated/dropped first once the budget is
    spent. Returns ``(system_prompt, truncated)``; with no playbooks, returns
    the prompt UNCHANGED (byte-identical to sub-part A).
    """
    if not playbooks:
        return _ANALYZE_SYSTEM_PROMPT, False

    included: list[str] = []
    used = 0
    truncated = False
    for pb in playbooks:
        _frontmatter, body = split_frontmatter(pb["content"])
        body = body.strip()
        remaining = PLAYBOOK_INJECTION_MAX_CHARS - used
        if remaining <= 0:
            truncated = True
            continue
        if len(body) > remaining:
            body = body[:remaining]
            truncated = True
        included.append(f"### {pb['name']}\n{body}")
        used += len(body)

    section = "## House rules and focus areas (user-provided)\n\n" + "\n\n".join(included)
    prompt = _ANALYZE_SYSTEM_PROMPT.replace(_PLAYBOOK_MARKER, section)
    return prompt, truncated


# ---------------------------------------------------------------------------
# Stage 4: analyze
# ---------------------------------------------------------------------------

_ANALYZE_SYSTEM_PROMPT = """You are a senior software engineer performing a focused code review of one changed file.

Output contract (STRICT — follow it exactly):
Respond with ONLY a JSON array. No prose, no markdown code fences, nothing before or after the array. If there is nothing to report for this file, respond with exactly: []

Each element of the array is a JSON object with these fields:
  - "file_path": string — the path of the file under review (given below)
  - "start_line": integer or null — first line (in the target file) the finding refers to
  - "end_line": integer or null — last line the finding refers to
  - "severity": one of "critical" | "high" | "medium" | "low" | "info"
      critical = certain to break production, or a real exploitable security vulnerability
      high     = a likely bug or serious risk — should be fixed before merge
      medium   = a real issue, but not urgent, or a moderate risk
      low      = a minor issue, safe to defer
      info     = an observation or suggestion — not a problem
  - "category": one of "bug" | "security" | "performance" | "maintainability" | "style" | "other"
  - "title": short summary, a few words
  - "body": 1-4 sentences explaining the issue
  - "suggestion": a concrete fix, or null if you don't have one

Focus ONLY on the changed lines shown in the diff hunks below — that is what is being reviewed. Use the target file excerpts and related entities to understand the surrounding context, but do NOT report style nits on unchanged code you were not asked to review.

<!-- PLAYBOOK INJECTION POINT (sub-part B): project-specific review playbooks/rules land here. Empty in sub-part A. -->
"""

# Import-time guard: if this marker ever drifts out of _ANALYZE_SYSTEM_PROMPT
# (e.g. a future edit to the prompt text above), _build_analyze_system_prompt's
# str.replace becomes a silent no-op — playbooks would vanish from the prompt
# while stats.playbooks_used still lists them. Fail loudly at import instead.
assert _PLAYBOOK_MARKER in _ANALYZE_SYSTEM_PROMPT, (
    "_PLAYBOOK_MARKER not found in _ANALYZE_SYSTEM_PROMPT — playbook injection would silently no-op"
)

_ANALYZE_RETRY_MESSAGE = (
    "Your previous output was invalid JSON. Reply again with ONLY the JSON array described "
    "in the system prompt above — no prose, no markdown fences."
)


def _strip_fences(text: str) -> str:
    """Strip a leading/trailing markdown code fence, mirroring the fallback
    parsing pattern in ``scripts/mcp_harness.py``."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t[:4].lower() == "json":
            t = t[4:].strip()
    return t


def _parse_json_array(raw: str) -> list | None:
    """
    Parse *raw* as a JSON array, tolerating a markdown fence and (as a
    fallback, since models sometimes add prose despite the strict-JSON-only
    instruction) prose wrapped around a single top-level ``[...]`` block.
    Returns ``None`` if no JSON array could be extracted.
    """
    text = _strip_fences(raw)
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _nearest_hunk(fd: FileDiff, line: int | None):
    """The hunk of *fd* whose new-side range is closest to *line* (or the
    first hunk if *line* is ``None``)."""
    if not fd.hunks:
        return None
    if line is None:
        return fd.hunks[0]

    def distance(h) -> int:
        lo, hi = h.new_start, h.new_start + max(h.new_count, 1) - 1
        if lo <= line <= hi:
            return 0
        return min(abs(line - lo), abs(line - hi))

    return min(fd.hunks, key=distance)


def _hunk_excerpt(fd: FileDiff, line: int | None) -> str:
    """
    ``evidence.hunk_excerpt``: roughly ±5 diff lines of the hunk nearest
    *line* (or the first hunk's opening lines when there's no anchor).
    """
    hunk = _nearest_hunk(fd, line)
    if hunk is None:
        return ""
    lines = hunk.text.split("\n")
    if line is None:
        return "\n".join(lines[:11])

    lo = hunk.new_start
    hi = hunk.new_start + max(hunk.new_count, 1) - 1
    span = max(hi - lo, 1)
    frac = min(max((line - lo) / span, 0.0), 1.0)
    center = int(frac * (len(lines) - 1))
    start = max(0, center - 5)
    end = min(len(lines), center + 6)
    return "\n".join(lines[start:end])


def _format_entities(ctx: ReviewFileContext) -> str:
    if not ctx.retrieved:
        if ctx.graph_entities:
            return "Touched entities (no further detail available): " + ", ".join(ctx.graph_entities)
        return "(none)"

    parts = []
    for r in ctx.retrieved:
        neighbors = r.get("neighbors") or []
        neighbor_str = ", ".join(
            f"{n['relation']} {n['qualified_name']}" for n in neighbors if n.get("qualified_name")
        ) or "(none)"
        desc = r.get("description") or "(no description)"
        parts.append(f"- {r['qualified_name']} ({r['kind']}): {desc}\n  neighbors: {neighbor_str}")

    retrieved_names = {r["qualified_name"] for r in ctx.retrieved}
    extra = [qn for qn in ctx.graph_entities if qn not in retrieved_names]
    if extra:
        parts.append("Other touched entities (no further detail): " + ", ".join(extra))
    return "\n".join(parts)


def _build_analyze_user_prompt(fd: FileDiff, ctx: ReviewFileContext) -> tuple[str, bool]:
    """Returns ``(prompt, truncated)`` — *truncated* is True iff the prompt
    was cut to :data:`MAX_USER_PROMPT_CHARS`."""
    hunks_text = "\n\n".join(h.text for h in fd.hunks) or "(no hunks)"
    excerpts_text = "\n\n".join(ctx.target_content_excerpts) if ctx.target_content_excerpts else "(unavailable)"
    entities_text = _format_entities(ctx)

    prompt = (
        f"File under review: {fd.path}\n\n"
        f"=== DIFF HUNKS (unified diff — the changed lines are the review target) ===\n{hunks_text}\n\n"
        f"=== TARGET FILE EXCERPTS (line-numbered, for context) ===\n{excerpts_text}\n\n"
        f"=== RELATED ENTITIES ===\n{entities_text}\n"
    )

    if len(prompt) > MAX_USER_PROMPT_CHARS:
        return prompt[:MAX_USER_PROMPT_CHARS] + "\n\n[...truncated to fit the prompt budget...]", True
    return prompt, False


async def _analyze_file(
    provider: LLMProvider,
    model: str,
    fd: FileDiff,
    ctx: ReviewFileContext,
    system_prompt: str,
    emit: Callable[[str, str], None],
) -> tuple[list, int, int, bool]:
    """
    One (plus a possible retry) chat call analyzing *fd*.

    *system_prompt* is :data:`_ANALYZE_SYSTEM_PROMPT`, or that prompt with
    playbooks injected (see :func:`_build_analyze_system_prompt`) — built
    once per run, not per file.

    Returns ``(raw_findings, prompt_chars, llm_calls, parse_failed)`` —
    ``raw_findings`` is unvalidated model output (``[]`` when parsing failed
    even after the retry, in which case ``parse_failed`` is True and the
    caller counts it in ``stats.parse_failures``; a run never crashes on
    model noise).
    """
    user_prompt, truncated = _build_analyze_user_prompt(fd, ctx)
    if truncated:
        emit("analyze", f"analyze: {fd.path}: prompt truncated to {MAX_USER_PROMPT_CHARS} chars")

    messages = [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user_prompt),
    ]
    raw = await provider.chat(
        messages, model=model, temperature=ANALYZE_TEMPERATURE, max_tokens=MAX_TOKENS_ANALYZE
    )
    parsed = _parse_json_array(raw)
    llm_calls = 1

    if parsed is None:
        messages.append(LLMMessage(role="assistant", content=raw))
        messages.append(LLMMessage(role="user", content=_ANALYZE_RETRY_MESSAGE))
        raw = await provider.chat(
            messages, model=model, temperature=ANALYZE_TEMPERATURE, max_tokens=MAX_TOKENS_ANALYZE
        )
        llm_calls += 1
        parsed = _parse_json_array(raw)
        if parsed is None:
            emit("analyze", f"parse_failed: {fd.path}")
            return [], len(user_prompt), llm_calls, True

    return parsed, len(user_prompt), llm_calls, False


_VALID_SEVERITIES = {"critical", "high", "medium", "low", "info"}
_VALID_CATEGORIES = {"bug", "security", "performance", "maintainability", "style", "other"}


def _coerce_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_anchored(start_line: int | None, changed_ranges: list[tuple[int, int]]) -> bool:
    if start_line is None:
        return False
    return any(
        (rs - ANCHOR_TOLERANCE_LINES) <= start_line <= (re_ + ANCHOR_TOLERANCE_LINES)
        for rs, re_ in changed_ranges
    )


def _validate_finding(
    raw, fd: FileDiff, ctx: ReviewFileContext, model: str, provider_name: str
) -> tuple[dict | None, list[str]]:
    """
    Validate/anchor one raw finding from the model against *fd*.

    Returns ``(finding_or_None, notes)`` — *notes* are human-readable
    validation events the caller emits either way; the finding is ``None``
    when it must be dropped (missing title/body, or a ``file_path`` clearly
    foreign to *fd* with no plausible coercion).
    """
    if not isinstance(raw, dict):
        return None, ["dropped: finding is not a JSON object"]

    notes: list[str] = []

    title, body = raw.get("title"), raw.get("body")
    if not title or not body:
        return None, ["dropped: missing required title/body"]

    raw_path = raw.get("file_path")
    target_norm = _normalize_path(fd.path)
    if not raw_path or _normalize_path(str(raw_path)) == target_norm:
        file_path = fd.path
    elif _normalize_path(str(raw_path)).rsplit("/", 1)[-1] == target_norm.rsplit("/", 1)[-1]:
        # Same basename, different (or missing) directory prefix — the model
        # echoed a plausible variant of the path it was given; coerce.
        file_path = fd.path
        notes.append(f"coerced foreign file_path '{raw_path}' to '{fd.path}' (basename matched)")
    else:
        # Different basename entirely — the model hallucinated a finding
        # against some other file; not safe to attribute to this one.
        return None, [f"dropped: foreign file_path '{raw_path}' (expected '{fd.path}')"]

    severity = str(raw.get("severity") or "").strip().lower()
    if severity not in _VALID_SEVERITIES:
        notes.append(f"unknown severity {raw.get('severity')!r} coerced to 'info'")
        severity = "info"

    category = str(raw.get("category") or "").strip().lower()
    if category not in _VALID_CATEGORIES:
        notes.append(f"unknown category {raw.get('category')!r} coerced to 'other'")
        category = "other"

    start_line = _coerce_int(raw.get("start_line"))
    end_line = _coerce_int(raw.get("end_line"))

    anchored = _is_anchored(start_line, fd.changed_line_ranges)
    if not anchored and start_line is not None:
        # The output contract carries no code-excerpt field to attempt a
        # snippet-match rescue from (9a-1) — start_line/end_line are the
        # only positional signal the model gives us, so an out-of-range
        # start_line just means unanchored.
        notes.append(
            f"unanchored: start_line {start_line} is not within a changed range "
            f"(±{ANCHOR_TOLERANCE_LINES}) — no code-excerpt field in the output "
            "contract to attempt a snippet-match rescue"
        )

    suggestion = raw.get("suggestion")
    if not isinstance(suggestion, str) or not suggestion.strip():
        suggestion = None

    finding = {
        "file_path": file_path,
        "start_line": start_line,
        "end_line": end_line,
        "severity": severity,
        "category": category,
        "title": str(title),
        "body": str(body),
        "suggestion": suggestion,
        "evidence": {
            "hunk_excerpt": _hunk_excerpt(fd, start_line),
            "entities": list(ctx.graph_entities),
            "model": model,
            "provider": provider_name,
        },
        "anchored": anchored,
    }
    return finding, notes


async def _stage_analyze(
    provider: LLMProvider,
    model: str,
    provider_name: str,
    reviewable: list[FileDiff],
    contexts: list[ReviewFileContext],
    system_prompt: str,
    emit: Callable[[str, str], None],
) -> tuple[list[dict], dict]:
    """Run the 'analyze' stage over every reviewable file (*contexts* in the
    same order, from :func:`_stage_context`). *system_prompt* — see
    :func:`_analyze_file`. Returns ``(findings, stats)``."""
    n = len(reviewable)
    findings: list[dict] = []
    llm_calls = 0
    parse_failures = 0
    prompt_chars_total = 0
    started = time.monotonic()

    for i, (fd, ctx) in enumerate(zip(reviewable, contexts), 1):
        emit("analyze", f"analyzing {fd.path} ({i}/{n})")
        raw_findings, prompt_chars, calls, failed = await _analyze_file(
            provider, model, fd, ctx, system_prompt, emit
        )
        llm_calls += calls
        prompt_chars_total += prompt_chars
        if failed:
            parse_failures += 1
            continue
        for raw in raw_findings:
            finding, notes = _validate_finding(raw, fd, ctx, model, provider_name)
            for note in notes:
                emit("analyze", f"analyze: {fd.path}: {note}")
            if finding is not None:
                findings.append(finding)

    stats = {
        "findings_total": len(findings),
        "findings_anchored": sum(1 for f in findings if f["anchored"]),
        "parse_failures": parse_failures,
        "llm_calls": llm_calls,
        "prompt_chars_total": prompt_chars_total,
        "wall_seconds_analyze": round(time.monotonic() - started, 2),
    }
    return findings, stats


# ---------------------------------------------------------------------------
# Stage 5: summarize
# ---------------------------------------------------------------------------

_SUMMARIZE_SYSTEM_PROMPT = (
    "You are a senior software engineer writing the summary for a finished code review. "
    "You are given the diff statistics and the list of findings already produced by the "
    "review. Write a concise markdown summary with:\n"
    "  - an overall assessment (1-2 sentences)\n"
    "  - key risks, as a bullet list, most severe first\n"
    "  - a one-line count of findings by severity\n"
    "Base the summary ONLY on the findings given below — do not invent new issues. "
    "Keep it under ~300 words."
)


def _build_summarize_user_prompt(diff_stats: dict, findings: list[dict]) -> str:
    lines = [
        f"Diffstat: {diff_stats['files_reviewable']} file(s) reviewed, "
        f"+{diff_stats['insertions']} -{diff_stats['deletions']}, {diff_stats['hunks']} hunk(s).",
        "",
        f"Findings ({len(findings)}):",
    ]
    if not findings:
        lines.append("(none)")
    else:
        lines.extend(f"- [{f['severity']}] {f['file_path']}: {f['title']}" for f in findings)
    return "\n".join(lines)


async def _stage_summarize(
    provider: LLMProvider,
    model: str,
    diff_stats: dict,
    findings: list[dict],
    emit: Callable[[str, str], None],
) -> tuple[str, dict]:
    """Run the 'summarize' stage: one chat call turning the diffstat +
    validated findings into the run's markdown summary."""
    emit("summarize", "summarize: generating run summary")
    started = time.monotonic()
    user_prompt = _build_summarize_user_prompt(diff_stats, findings)
    messages = [
        LLMMessage(role="system", content=_SUMMARIZE_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_prompt),
    ]
    summary = await provider.chat(
        messages, model=model, temperature=SUMMARIZE_TEMPERATURE, max_tokens=MAX_TOKENS_SUMMARIZE
    )
    summary = summary.strip()
    emit("summarize", "summarize: done")

    stats = {
        "llm_calls": 1,
        "prompt_chars_total": len(user_prompt),
        "wall_seconds_summarize": round(time.monotonic() - started, 2),
    }
    return summary, stats


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_review(job: Job, run_id: str, params: dict, emit: Callable[[str, str], None]) -> dict:
    """
    The review routine's :class:`~services.routines.engine.RoutineCallable`.

    params: ``{repo_path, base_ref?, target_ref, model?, provider?, playbooks?}``
    — ``model``/``provider`` override the active LLM config (12h) for the
    analyze/summarize stages (see :func:`_resolve_llm`); ``playbooks`` is a
    list of playbook names (selection order = priority, decision 9d)
    injected into the analyze system prompt only (see
    :func:`_stage_playbooks` / :func:`_build_analyze_system_prompt`) — the
    summarize stage never sees them.

    Resolves + health-checks the LLM FIRST, before any diff/context work —
    a missing/broken model config fails the run immediately rather than
    after the (potentially expensive) diff+context stages have already run.

    Runs all five stages (resolve, diff, context, playbooks, analyze,
    summarize) and returns ``{"summary", "findings", "stats"}`` —
    ``findings`` are validated/anchored dicts shaped for
    ``store.insert_findings``; ``stats`` merges every stage's counters plus
    the resolved ``model``/``provider`` and ``playbooks_used``.
    """
    repo_path = params["repo_path"]
    target_ref = params["target_ref"]
    base_ref = params.get("base_ref")

    provider, model, provider_name = await _resolve_llm(params)

    base, target, reviewable, diff_stats = _stage_diff(repo_path, base_ref, target_ref, emit)
    contexts, context_chars_total, entities_matched = _stage_context(
        repo_path, target, reviewable, job.project, emit
    )

    playbooks = await _stage_playbooks(params.get("playbooks") or [], emit)
    system_prompt, playbooks_truncated = _build_analyze_system_prompt(playbooks)
    if playbooks_truncated:
        emit("playbooks", "playbook content truncated to fit the context budget")

    findings, analyze_stats = await _stage_analyze(
        provider, model, provider_name, reviewable, contexts, system_prompt, emit
    )
    summary, summarize_stats = await _stage_summarize(provider, model, diff_stats, findings, emit)

    llm_calls = analyze_stats.pop("llm_calls") + summarize_stats.pop("llm_calls")
    prompt_chars_total = analyze_stats.pop("prompt_chars_total") + summarize_stats.pop("prompt_chars_total")

    stats = {
        **diff_stats,
        "entities_matched": entities_matched,
        "context_chars_total": context_chars_total,
        **analyze_stats,
        **summarize_stats,
        "llm_calls": llm_calls,
        "prompt_chars_total": prompt_chars_total,
        "model": model,
        "provider": provider_name,
        "playbooks_used": [pb["name"] for pb in playbooks],
    }

    return {
        "summary": summary,
        "findings": findings,
        "stats": stats,
    }
