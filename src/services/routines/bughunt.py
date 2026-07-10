"""
Bug-hunt routine — the ``run_bughunt`` :class:`~services.routines.engine.RoutineCallable`
plugged into ``services.routines.engine.submit_routine`` for ``kind="bughunt"``
(sub-part B, step B5).

Unlike ``review`` (diff-anchored, one analyze call per changed file), bug
hunt has no diff to anchor against: it is a free-text ``scope`` investigated
by the in-process agentic tool loop (``services.routines.agent_loop``,
step B4) against the WHOLE project graph/Weaviate corpus. Guardrail (P-07):
this routine is agent+retrieval driven — it NEVER scores or ranks anything
by graph metrics (centrality, fan-in/out, etc.); every finding traces back
to the model's own tool-gathered evidence.

Five stages, mirroring ``review``'s shape:

  1. **resolve** — :func:`services.routines.review._resolve_llm` (REUSED
     verbatim — same override -> active-config -> error chain, same
     ``health_check`` preflight), PLUS an additional bughunt-only
     constraint: the resolved provider must be ``"ollama"`` (decision 9a —
     the agent loop is Ollama-native only in this release). A non-ollama
     resolution fails the run immediately with the same message
     ``agent_loop.run_agent_loop`` would itself raise
     (:func:`_non_ollama_error`) — the controller ALSO preflights this at
     launch time (see ``controllers/routines.py``) so the caller gets a
     synchronous 400 rather than only a failed run.
  2. **playbooks** — :func:`services.routines.review._stage_playbooks`
     (REUSED verbatim: fetch + 'loaded playbook <name>' events). Their
     bodies are injected into THIS module's own bug-hunter system prompt
     (:func:`_build_hunt_system_prompt`) — NOT into
     ``agent_loop.SYSTEM_PROMPT`` (unused here; bughunt supplies its own
     role prompt to ``run_agent_loop``'s ``system_prompt`` argument). The
     loop appends the skills catalog to whatever system prompt it's given
     (``agent_loop.build_system_prompt``) — playbooks and the skills
     catalog land in two clearly separated sections of the same prompt,
     never duplicated.
  3. **hunt** — one ``run_agent_loop`` run (12 rounds, decision 9h) with
     tools bound to the project via ONE Weaviate client (closed in
     ``finally``). The loop's final answer must be a strict JSON object
     ``{"findings": [...], "notes": str}`` (same finding shape as review's
     analyze contract); parsed with fence-stripping
     (:func:`services.routines.review._strip_fences`, reused) plus ONE
     retry — a plain (non-tool) chat call appending
     :data:`_HUNT_RETRY_MESSAGE`. If it's STILL not parseable, the run does
     NOT fail (P-07/9a spirit: a confused model is not a routine failure) —
     it DEGRADES: zero findings, ``stats.parse_failures = 1``, and the raw
     agent answer is preserved verbatim in ``run.summary``, prefixed
     ``"agent answer (unparsed): "`` so it's never silently lost.
  4. **findings** — validated/anchored against the project graph (there is
     no diff to anchor against, unlike review): see
     :func:`_validate_finding` for the exact rule.
  5. **summarize** — one plain chat call turning the agent's own notes plus
     the validated findings into the run's markdown summary. A bughunt
     analogue of ``review._stage_summarize`` (same shape, different prompt
     inputs — scope/notes instead of a diffstat, so NOT literally reused).

Sharing with ``review.py`` (imported, not copied): ``_resolve_llm``,
``_stage_playbooks``, ``_strip_fences``, ``_normalize_path``, ``_coerce_int``,
``_VALID_SEVERITIES``/``_VALID_CATEGORIES``, and the
``ANALYZE_TEMPERATURE``/``SUMMARIZE_TEMPERATURE``/``MAX_TOKENS_SUMMARIZE``/
``PLAYBOOK_INJECTION_MAX_CHARS`` budget constants — all of these are either
exact-semantics utilities or tuning knobs with no review-specific meaning.
NOT shared: ``_validate_finding`` (file-anchoring rule differs completely —
review anchors against a known target file's changed-line ranges; bughunt
anchors against project-graph entity spans across arbitrary files) and
``_stage_summarize`` (review's prompt is diffstat-shaped; bughunt's is
scope/notes-shaped) — each gets its own small implementation here.

Same async-inside-worker-thread shape as ``review.run_review``: this module
runs synchronously inside the job's worker thread EXCEPT the LLM/agent-loop
calls, which are real ``await``s — ``run_bughunt`` is a coroutine for that
reason (see ``services.routines.engine._make_runner`` /
``services.jobs``'s module docstring).
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable

from services import graph_store
from services.chunker import CODE_ENTITY_LABELS
from services.graph_store import GraphNotFoundError, resolve_file
from services.jobs import Job
from services.llm_provider import LLMMessage, LLMProvider
from services.routines.agent_loop import build_tools, run_agent_loop
from services.routines.documents import split_frontmatter
from services.routines.review import (
    ANALYZE_TEMPERATURE,
    MAX_TOKENS_SUMMARIZE,
    PLAYBOOK_INJECTION_MAX_CHARS,
    SUMMARIZE_TEMPERATURE,
    _coerce_int,
    _normalize_path,
    _resolve_llm,
    _stage_playbooks,
    _strip_fences,
    _VALID_CATEGORIES,
    _VALID_SEVERITIES,
)
from services.weaviate_client import get_client

# Budgets (9h) — module constants, not hardcoded inline. Bughunt-specific;
# the loop's own budgets (MAX_ROUNDS, tool payload chars, skills cap) live
# in agent_loop.py and are reused via run_agent_loop's defaults.
MAX_TOKENS_HUNT_RETRY = 4096  # the JSON-reformat retry call — as generous as review's analyze budget


class BughuntError(RuntimeError):
    """Raised for a bug-hunt-routine-level failure (mirrors
    ``services.routines.review.ReviewError``)."""


def _non_ollama_error(provider_name: str) -> ValueError:
    """The exact message ``agent_loop.run_agent_loop`` itself raises for a
    non-Ollama provider (decision 9a) — shared verbatim between this
    module's 'resolve' stage (a clean run failure) and the controller's
    launch-time preflight (a synchronous 400), so both surfaces say the
    same thing."""
    return ValueError(
        "bug hunt requires Ollama-native tool calling in this release — "
        f"provider '{provider_name}' is not supported yet"
    )


# ---------------------------------------------------------------------------
# Stage 1: resolve (9a-3 / 12h, plus the bughunt-only ollama constraint)
# ---------------------------------------------------------------------------


async def _stage_resolve(params: dict, emit: Callable[[str, str], None]) -> tuple[LLMProvider, str, str]:
    emit("resolve", "resolve: resolving LLM provider/model")
    provider, model, provider_name = await _resolve_llm(params, noun="bug hunt")
    if provider_name != "ollama":
        raise _non_ollama_error(provider_name)
    emit("resolve", f"resolve: using {provider_name}/{model}")
    return provider, model, provider_name


# ---------------------------------------------------------------------------
# Stage 2: playbooks (reuses review._stage_playbooks verbatim) — injected
# into THIS module's own bug-hunter system prompt.
# ---------------------------------------------------------------------------

_HUNT_PLAYBOOK_MARKER = (
    "<!-- PLAYBOOK INJECTION POINT (bughunt): project-specific bug-hunt "
    "playbooks/rules land here. -->"
)

_HUNT_SYSTEM_PROMPT = f"""You are a senior software engineer hunting for bugs in a codebase. You have no knowledge of this code beyond what the tools return — use them. Investigate the scope described in the user message using search/get_entity/neighbors/get_file/grep_project; call load_skill for any skill clearly relevant to the investigation (at most 3, see the catalog below).

Gather EVIDENCE for every issue: cite the exact file path and line numbers from what get_entity()/get_file() actually returned — never invent or guess a file path or line number. When you call a tool, output ONLY the JSON for that one call and nothing else — no explanation, no example result. You do not have real results until the system gives them back to you.

Once your investigation is complete, STOP calling tools and reply with your final answer as ONLY a single strict JSON object — no prose, no markdown fences, nothing before or after it — of this exact shape:

{{
  "findings": [
    {{
      "file_path": string,
      "start_line": integer or null,
      "end_line": integer or null,
      "severity": "critical" | "high" | "medium" | "low" | "info",
      "category": "bug" | "security" | "performance" | "maintainability" | "style" | "other",
      "title": string,
      "body": string,
      "suggestion": string or null
    }}
  ],
  "notes": string
}}

Severity: critical = certain to break production or an exploitable security vulnerability; high = a likely bug or serious risk; medium = a real issue, not urgent; low = a minor issue, safe to defer; info = an observation, not a problem.
If you found nothing, reply with {{"findings": [], "notes": "..."}}.

{_HUNT_PLAYBOOK_MARKER}
"""

# Import-time guard — same reasoning as review._ANALYZE_SYSTEM_PROMPT's own
# guard: if the marker ever drifts out of the prompt text above,
# _build_hunt_system_prompt's str.replace becomes a silent no-op.
assert _HUNT_PLAYBOOK_MARKER in _HUNT_SYSTEM_PROMPT, (
    "_HUNT_PLAYBOOK_MARKER not found in _HUNT_SYSTEM_PROMPT — playbook injection would silently no-op"
)


def _build_hunt_system_prompt(playbooks: list[dict]) -> tuple[str, bool]:
    """
    Inject *playbooks* (priority order = selection order, decision 9d) into
    :data:`_HUNT_SYSTEM_PROMPT` at :data:`_HUNT_PLAYBOOK_MARKER` — same
    injection shape as ``review._build_analyze_system_prompt`` (own small
    implementation here rather than a shared helper: the base prompt/marker
    differ and the algorithm is ~15 lines, not worth threading a generic
    injector through both modules). Total injected chars capped at
    :data:`PLAYBOOK_INJECTION_MAX_CHARS` (reused from review — same budget
    reasoning applies here). Returns ``(system_prompt, truncated)``; with no
    playbooks, returns the prompt UNCHANGED.
    """
    if not playbooks:
        return _HUNT_SYSTEM_PROMPT, False

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
    prompt = _HUNT_SYSTEM_PROMPT.replace(_HUNT_PLAYBOOK_MARKER, section)
    return prompt, truncated


def _build_hunt_user_prompt(scope: str, path_prefix: str | None) -> str:
    prompt = f"Investigation scope: {scope}"
    if path_prefix:
        prompt += (
            f"\n\nOnly report findings whose file_path is under '{path_prefix}' — "
            "you may still investigate related code elsewhere for context."
        )
    return prompt


# ---------------------------------------------------------------------------
# Stage 3: hunt — the agent loop + strict-JSON verdict parsing
# ---------------------------------------------------------------------------

_HUNT_RETRY_MESSAGE = "reply with ONLY the JSON object"


def _parse_hunt_answer(raw: str) -> dict | None:
    """
    Parse *raw* as the loop's final-answer JSON object contract
    (``{"findings": [...], "notes": ...}``), tolerating a markdown fence
    (:func:`services.routines.review._strip_fences`, reused) and, as a
    fallback, prose wrapped around a single top-level ``{...}`` block
    (mirrors ``review._parse_json_array``'s tolerance, adapted to an
    object). Returns ``None`` if no such object could be extracted.
    """
    text = _strip_fences(raw)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("findings"), list):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict) and isinstance(obj.get("findings"), list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
    return None


async def _parse_hunt_answer_with_retry(
    provider: LLMProvider, model: str, answer: str, emit: Callable[[str, str], None]
) -> tuple[dict | None, int]:
    """
    ``_parse_hunt_answer`` plus ONE retry: a plain (non-tool) chat call
    appending :data:`_HUNT_RETRY_MESSAGE` to the agent's own final answer.

    Returns ``(parsed_or_None, llm_calls)`` — *llm_calls* is ``1`` iff the
    retry chat call was actually made (i.e. the agent's own answer wasn't
    already valid JSON), ``0`` otherwise; the caller folds this into
    ``stats.llm_calls`` on BOTH terminal shapes (success and degrade) so the
    stat's key-set doesn't depend on which path was taken. ``parsed`` is
    ``None`` if parsing still fails after the retry — the caller
    (:func:`run_bughunt`) degrades rather than failing the run (see module
    docstring).
    """
    parsed = _parse_hunt_answer(answer)
    if parsed is not None:
        return parsed, 0

    messages = [
        LLMMessage(role="assistant", content=answer),
        LLMMessage(role="user", content=_HUNT_RETRY_MESSAGE),
    ]
    retry_raw = await provider.chat(
        messages, model=model, temperature=ANALYZE_TEMPERATURE, max_tokens=MAX_TOKENS_HUNT_RETRY
    )
    parsed = _parse_hunt_answer(retry_raw)
    if parsed is None:
        emit("hunt", "parse_failed: agent's final answer was not valid JSON, even after the retry")
        return None, 1
    return parsed, 1


# ---------------------------------------------------------------------------
# Stage 4: findings — validation/anchoring against the project graph
# ---------------------------------------------------------------------------


def _relative_to_repo(file_path: str, repo_path: str) -> str:
    """Best-effort repo-relative form of *file_path*, for the
    ``path_prefix`` constraint check — tool results often echo an absolute
    path (``_format_hit_lines`` prefers ``absolute_path``), while
    ``path_prefix`` is naturally given repo-relative (e.g.
    ``src/features/alarms``). Returns *file_path* unchanged when it's
    already relative, or when it's absolute but not under *repo_path*."""
    if not os.path.isabs(file_path):
        return file_path
    try:
        rel = os.path.relpath(file_path, repo_path)
    except ValueError:
        return file_path
    return file_path if rel.startswith("..") else rel


def _find_entity_for_line(project: str, file_path: str, line: int) -> dict | None:
    """
    The code-entity graph node (see ``services.chunker.CODE_ENTITY_LABELS``)
    on *file_path* whose ``[start_line, end_line]`` span contains *line*, or
    ``None``. Mirrors ``review._match_entities``'s path-matching (same
    ``_normalize_path`` exact-match rule) narrowed to a single line instead
    of a range — bughunt has no diff-derived changed-range to anchor
    against, so a finding's own ``start_line`` against an entity's span is
    the only anchor signal available (see :func:`_validate_finding`).
    """
    graph = graph_store.get_graph(project)
    target = _normalize_path(file_path)
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
        if start <= line <= end:
            return {"qualified_name": props.get("qualified_name")}
    return None


def _validate_finding(
    raw, project: str, repo_path: str, path_prefix: str | None, model: str, provider_name: str
) -> tuple[dict | None, list[str]]:
    """
    Validate/anchor one raw finding from the agent's JSON verdict.

    Anchoring rule (no diff to anchor against, unlike review):

      1. ``file_path`` must resolve to something real, checked in two
         steps: (a) an exact match against a File/Module node in the
         project graph (:func:`services.graph_store.resolve_file` — the
         SAME traversal guard the ``get_file`` tool itself uses); failing
         that, (b) a cheap ``os.path.isfile`` check under *repo_path*. A
         ``file_path`` that matches NEITHER is dropped as a pure
         hallucination — there is nothing to attribute the finding to.
      2. When the file IS in the graph and the finding carries a
         ``start_line``, ``anchored = True`` iff that line falls within
         the span of a code-entity node on that file
         (:func:`_find_entity_for_line`) — i.e. the model's cited location
         actually corresponds to something in the graph, not just a
         plausible-looking number. Every other case (file only found on
         disk, no start_line, or a start_line outside every entity's span)
         is kept but ``anchored = False`` — never dropped for that reason
         alone; only a fully hallucinated ``file_path`` is dropped.

    Returns ``(finding_or_None, notes)`` — *notes* are human-readable
    validation events the caller emits either way.
    """
    if not isinstance(raw, dict):
        return None, ["dropped: finding is not a JSON object"]

    notes: list[str] = []

    title, body = raw.get("title"), raw.get("body")
    if not title or not body:
        return None, ["dropped: missing required title/body"]

    raw_path = raw.get("file_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, ["dropped: missing file_path"]
    file_path = raw_path.strip()

    if path_prefix:
        rel_path = _normalize_path(_relative_to_repo(file_path, repo_path))
        prefix_norm = _normalize_path(path_prefix).rstrip("/")
        if rel_path != prefix_norm and not rel_path.startswith(prefix_norm + "/"):
            return None, [f"dropped: file_path '{file_path}' outside path_prefix '{path_prefix}'"]

    try:
        node = resolve_file(project, file_path)
    except GraphNotFoundError:
        node = None
    in_graph = node is not None

    if not in_graph:
        # Confine the disk-existence fallback strictly under repo_path — an
        # absolute file_path (e.g. '/etc/passwd') or a '../' escape must NOT
        # be treated as "exists on disk", or this fallback becomes an
        # arbitrary-file existence oracle. os.path.join discards repo_path
        # entirely when file_path is itself absolute, so realpath-ing the
        # JOINED path and checking it's still under repo_path's realpath
        # catches both an absolute file_path and a '../' traversal in one
        # check (Python's os.path.join/realpath semantics — no bespoke
        # absolute-path special-casing needed).
        repo_real = os.path.realpath(repo_path)
        candidate = os.path.realpath(os.path.join(repo_path, file_path))
        escaped = os.path.commonpath([repo_real, candidate]) != repo_real
        if escaped or not os.path.isfile(candidate):
            return None, [f"dropped: file_path '{file_path}' hallucinated — not in the project graph or on disk"]
        notes.append(f"file_path '{file_path}' exists on disk but is not a node in the project graph")

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

    anchored = False
    anchor_entity_qn: str | None = None
    if in_graph and start_line is not None:
        entity = _find_entity_for_line(project, file_path, start_line)
        if entity is not None:
            anchored = True
            anchor_entity_qn = entity.get("qualified_name")
    if not anchored:
        notes.append(
            f"unanchored: start_line {start_line} does not fall within a graph entity's span on '{file_path}'"
            if in_graph and start_line is not None
            else "unanchored: no graph entity span to confirm this finding's location against"
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
            "entities": [anchor_entity_qn] if anchor_entity_qn else [],
            "model": model,
            "provider": provider_name,
        },
        "anchored": anchored,
    }
    return finding, notes


def _stage_findings(
    raw_findings: list,
    project: str,
    repo_path: str,
    path_prefix: str | None,
    model: str,
    provider_name: str,
    emit: Callable[[str, str], None],
) -> tuple[list[dict], dict]:
    """Run the 'findings' stage over the agent's raw verdict list."""
    emit("findings", f"findings: validating {len(raw_findings)} raw finding(s)")
    started = time.monotonic()
    findings: list[dict] = []
    dropped = 0
    for raw in raw_findings:
        finding, notes = _validate_finding(raw, project, repo_path, path_prefix, model, provider_name)
        for note in notes:
            emit("findings", f"findings: {note}")
        if finding is not None:
            findings.append(finding)
        else:
            dropped += 1

    stats = {
        "findings_total": len(findings),
        "findings_anchored": sum(1 for f in findings if f["anchored"]),
        "findings_dropped": dropped,
        "wall_seconds_findings": round(time.monotonic() - started, 2),
    }
    return findings, stats


# ---------------------------------------------------------------------------
# Stage 5: summarize
# ---------------------------------------------------------------------------

_SUMMARIZE_SYSTEM_PROMPT = (
    "You are a senior software engineer writing the summary for a finished bug-hunt "
    "investigation. You are given the investigation scope, the agent's own notes, and "
    "the list of findings already produced. Write a concise markdown summary with:\n"
    "  - an overall assessment (1-2 sentences)\n"
    "  - key risks, as a bullet list, most severe first\n"
    "  - a one-line count of findings by severity\n"
    "Base the summary ONLY on the findings given below — do not invent new issues. "
    "Keep it under ~300 words."
)


def _build_summarize_user_prompt(scope: str, notes: str, findings: list[dict]) -> str:
    lines = [f"Investigation scope: {scope}"]
    if notes:
        lines += ["", f"Agent notes: {notes}"]
    lines += ["", f"Findings ({len(findings)}):"]
    if not findings:
        lines.append("(none)")
    else:
        lines.extend(f"- [{f['severity']}] {f['file_path']}: {f['title']}" for f in findings)
    return "\n".join(lines)


async def _stage_summarize(
    provider: LLMProvider,
    model: str,
    scope: str,
    notes: str,
    findings: list[dict],
    emit: Callable[[str, str], None],
) -> tuple[str, dict]:
    """Run the 'summarize' stage: one chat call turning the scope + agent
    notes + validated findings into the run's markdown summary. A bughunt
    analogue of ``review._stage_summarize`` — not literally reused, since
    the prompt inputs differ (scope/notes here, a diffstat there)."""
    emit("summarize", "summarize: generating run summary")
    started = time.monotonic()
    user_prompt = _build_summarize_user_prompt(scope, notes, findings)
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
        "wall_seconds_summarize": round(time.monotonic() - started, 2),
    }
    return summary, stats


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_bughunt(job: Job, run_id: str, params: dict, emit: Callable[[str, str], None]) -> dict:
    """
    The bug-hunt routine's :class:`~services.routines.engine.RoutineCallable`.

    params: ``{repo_path, scope, path_prefix?, model?, provider?, playbooks?}``
    — ``scope`` is a required free-text query describing what to
    investigate; ``path_prefix`` optionally restricts REPORTED findings to
    files under that prefix (the agent may still investigate elsewhere for
    context); ``model``/``provider``/``playbooks`` mirror review's params
    (see :func:`services.routines.review._resolve_llm` /
    :func:`services.routines.review._stage_playbooks`).

    Resolves + health-checks the LLM FIRST (and additionally requires
    ``"ollama"``, decision 9a) before any playbook/agent-loop work — see
    :func:`_stage_resolve`.

    Runs all five stages (resolve, playbooks, hunt, findings, summarize) and
    returns ``{"summary", "findings", "stats"}`` in the same shape
    ``services.routines.engine._make_runner`` expects. On a JSON-verdict
    parse failure (even after the one retry), DEGRADES rather than raising:
    zero findings, ``stats.parse_failures = 1``, and ``summary`` is the raw
    agent answer prefixed ``"agent answer (unparsed): "`` (see module
    docstring).
    """
    repo_path = params["repo_path"]
    scope = params["scope"]
    path_prefix = params.get("path_prefix")
    project = job.project

    resolve_started = time.monotonic()
    provider, model, provider_name = await _stage_resolve(params, emit)
    wall_seconds_resolve = round(time.monotonic() - resolve_started, 2)

    playbooks = await _stage_playbooks(params.get("playbooks") or [], emit)
    system_prompt, playbooks_truncated = _build_hunt_system_prompt(playbooks)
    if playbooks_truncated:
        emit("playbooks", "playbook content truncated to fit the context budget")

    user_prompt = _build_hunt_user_prompt(scope, path_prefix)

    emit(
        "hunt",
        f"hunt: investigating scope '{scope}'"
        + (f" — restricted to path_prefix '{path_prefix}'" if path_prefix else ""),
    )
    hunt_started = time.monotonic()
    client = get_client()
    try:
        tools = build_tools(project, client)
        loop_result = await run_agent_loop(provider, model, system_prompt, user_prompt, tools, emit)
    finally:
        client.close()
    wall_seconds_hunt = round(time.monotonic() - hunt_started, 2)
    emit(
        "hunt",
        f"hunt: agent finished after {loop_result['rounds']} round(s), "
        f"{loop_result['tool_calls_made']} tool call(s)",
    )

    base_stats = {
        "rounds": loop_result["rounds"],
        "tool_calls_made": loop_result["tool_calls_made"],
        "tool_payload_chars": loop_result["tool_payload_chars"],
        "loaded_skills": loop_result["loaded_skills"],
        "model": model,
        "provider": provider_name,
        "playbooks_used": [pb["name"] for pb in playbooks],
        "wall_seconds_resolve": wall_seconds_resolve,
        "wall_seconds_hunt": wall_seconds_hunt,
    }

    parsed, parse_retry_llm_calls = await _parse_hunt_answer_with_retry(
        provider, model, loop_result["answer"], emit
    )

    if parsed is None:
        # Same key-set as the success shape below (llm_calls included here
        # too — just the retry call, since summarize never runs on this
        # path) so a caller doesn't need to branch on parse_failures to know
        # which stats keys exist.
        stats = {
            **base_stats,
            "findings_total": 0,
            "findings_anchored": 0,
            "findings_dropped": 0,
            "parse_failures": 1,
            "llm_calls": parse_retry_llm_calls,
            "wall_seconds_findings": 0.0,
            "wall_seconds_summarize": 0.0,
        }
        summary = "agent answer (unparsed): " + loop_result["answer"]
        return {"summary": summary, "findings": [], "stats": stats}

    findings, findings_stats = _stage_findings(
        parsed["findings"], project, repo_path, path_prefix, model, provider_name, emit
    )

    notes = parsed.get("notes") or ""
    summary, summarize_stats = await _stage_summarize(provider, model, scope, notes, findings, emit)
    llm_calls = parse_retry_llm_calls + summarize_stats.pop("llm_calls")

    stats = {
        **base_stats,
        **findings_stats,
        "parse_failures": 0,
        "llm_calls": llm_calls,
        **summarize_stats,
    }
    return {"summary": summary, "findings": findings, "stats": stats}
