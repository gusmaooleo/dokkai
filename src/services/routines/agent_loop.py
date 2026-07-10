"""
In-process agentic tool loop for bug-hunt (decisions 9a/9d/9h, sub-part B,
step B4).

Ports the PATTERNS proven by ``scripts/mcp_harness.py`` (system-prompt shape,
fallback JSON tool-call parser for models that emit tool-call JSON as plain
content, tool-result reminder, max-rounds guard, token accounting) and
``cli/src/lib/agent.ts`` (added call-dedupe guard) — but WITHOUT spawning the
dokkai MCP server as a stdio subprocess. The tools here call
``services.retriever``/``services.graph_store`` DIRECTLY, the same modules
``src/mcp_server.py`` imports, and mirror that server's small-model-profile
tool semantics/text output. There is no MCP client/subprocess involved.

Ollama-native only (decision 9a): tool-calling in this release is Ollama's
``/api/chat`` ``tools=``/``tool_calls`` mechanism (see
``OllamaProvider.chat_with_tools``). Any other provider raises
:class:`NotImplementedError` up front — the review pipeline stays
provider-agnostic; this loop is bug-hunt-only.

Async, consistent with how ``services.routines.review``'s stages run:
``run_review`` is itself a coroutine executed via ``asyncio.run(core(...))``
inside the job's worker thread (``services.routines.engine._make_runner``),
and its sync-but-asyncio.run()-internally helpers (``documents.
fetch_playbooks_sync``) are bridged with ``asyncio.to_thread`` to avoid
nesting ``asyncio.run()`` inside an already-running loop (see
``review._stage_playbooks``'s docstring) — this module follows the same
rule for ``documents.fetch_skill_sync``/``fetch_skills_catalog_sync``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import weaviate

from services import graph_store
from services.graph_store import resolve_file
from services.llm_provider import LLMProvider
from services.retriever import Retriever
from services.routines.documents import (
    fetch_skill_sync,
    fetch_skills_catalog_sync,
    split_frontmatter,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Budgets (9h) — module constants, not hardcoded inline.
# ---------------------------------------------------------------------------

MAX_ROUNDS = 12
TOOL_PAYLOAD_CHAR_BUDGET = 20_000  # ~20k tool-payload chars (9h's chars/4 token convention)
BUDGET_TRUNCATE_CHARS = 500        # harder per-result cap once the budget is exceeded
MAX_LOADED_SKILLS = 3              # decision 9d — at most 3 skills loaded per run

_MAX_FILE_LINES = 2_000
_MAX_FILE_BYTES = 100_000

EmitFn = Callable[[str, str], None]


# ---------------------------------------------------------------------------
# System prompt (ported from scripts/mcp_harness.py's SYSTEM_PROMPT)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are answering questions about a codebase. You have no knowledge of "
    "this code beyond what the tools return — use them. Start with "
    "search(query) to find relevant entities; run more searches with "
    "different queries if the first one doesn't cover the question. For "
    "detail on a specific entity, call get_entity() or neighbors() on it. "
    "Never call the same tool with the same arguments twice — you already "
    "have that result. As soon as the tool results answer the question, "
    "STOP calling tools and write your final answer as plain English text "
    "(not JSON), citing the file paths you used. Do not answer without "
    "calling at least one tool first. You do not have real results until "
    "the tools return them; never invent or guess a tool's output yourself."
)

_TOOL_RESULT_REMINDER = (
    "\n\n[If the information above answers the question, reply now with your "
    "final answer as plain English text — do not call any more tools.]"
)

_FORCE_FINAL_PROMPT = (
    "You have reached the maximum number of tool-call rounds. Answer now "
    "with your best final answer as plain English text, based on what "
    "you've already learned — do not call any more tools."
)


def build_system_prompt(base: str, skills_catalog: list[dict]) -> str:
    """
    Append the skills catalog (decision 9d) to *base*: a ``{name,
    description}`` list plus an instruction to load at most
    :data:`MAX_LOADED_SKILLS` skills, only when relevant. Returns *base*
    unchanged when the catalog is empty.
    """
    if not skills_catalog:
        return base
    lines = [f"- {s['name']}: {s['description']}" for s in skills_catalog]
    return (
        f"{base}\n\n"
        f"Available skills (call load_skill(name) to load one; load at most "
        f"{MAX_LOADED_SKILLS} skills total, only when clearly relevant to the "
        "question):\n" + "\n".join(lines)
    )


# ---------------------------------------------------------------------------
# Fallback tool-call parser — ported verbatim (shapes) from
# scripts/mcp_harness.py / cli/src/lib/agent.ts.
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_MAX_JSON_CANDIDATES = 20  # bounded scan — see find_json_object_candidates


def find_json_object_candidates(text: str) -> list[str]:
    """
    Every plausible embedded JSON-object substring in *text*, in order of
    appearance: first every ```-fenced block's inner text, then every
    balanced top-level ``{...}`` span found by string-aware brace-scanning
    over the raw text (braces inside JSON string literals are not counted,
    so a stray ``{``/``}`` inside a quoted value doesn't desync the scan).

    Models occasionally narrate around a tool call or verdict ("I'll now
    call...\\n{...}\\nNext I'll...") instead of emitting bare JSON — this
    recovers the embedded object(s) without requiring the whole message to
    be JSON. Bounded to :data:`_MAX_JSON_CANDIDATES` total candidates and,
    per candidate, a single linear scan to its matching close brace (or to
    end-of-text if unbalanced, at which point scanning stops entirely —
    pathological/unbalanced input costs at most one O(len(text)) pass, never
    more). Shared by :func:`extract_fallback_tool_call` (tool-call shape)
    and ``services.routines.bughunt._parse_hunt_answer`` (verdict shape) —
    this function does no shape filtering itself; callers try each
    candidate against their own predicate and take the first match.
    """
    candidates: list[str] = [m.group(1).strip() for m in _FENCE_RE.finditer(text)]

    n = len(text)
    i = 0
    while i < n and len(candidates) < _MAX_JSON_CANDIDATES:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escape = False
        j = i
        closed = False
        while j < n:
            ch = text[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        closed = True
                        j += 1
                        break
            j += 1
        if not closed:
            # No matching close for this open brace before end-of-text —
            # nothing valid can start here or later attempt to close it
            # either; stop scanning rather than re-walking the same tail
            # repeatedly on pathological input.
            break
        candidates.append(text[i:j])
        i = j

    return candidates[:_MAX_JSON_CANDIDATES]


def _match_tool_call_shape(obj: Any) -> dict | None:
    """``{"name": str, "arguments": dict?}`` -> normalized dict (arguments
    defaults to ``{}`` when absent), else ``None``."""
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None
    args = obj.get("arguments", {})
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return {"name": obj["name"], "arguments": args}


def extract_fallback_tool_call(content: str) -> dict | None:
    """
    Best-effort fallback for models that emit a well-formed tool-call JSON as
    plain ``content`` instead of Ollama's structured ``tool_calls`` field.

    Tries, cheapest first: (1) the whole content (after stripping
    ``<tool_call>`` tags / a single wrapping markdown fence) as ONE JSON
    object — the common case; (2) every candidate from
    :func:`find_json_object_candidates` (fenced blocks, then embedded
    ``{...}`` spans), in order, returning the FIRST one that both parses and
    matches the ``{"name": str, "arguments": dict?}`` tool-call shape. Prose
    around/between candidates is ignored. One call per round is the loop's
    model — if the model narrates several tool calls in one message, only
    the first is taken; it gets that result and can continue next round.
    """
    text = content.strip()
    if "<tool_call>" in text:
        start = text.find("<tool_call>") + len("<tool_call>")
        end = text.find("</tool_call>")
        text = text[start : end if end != -1 else None].strip()

    whole = text
    if whole.startswith("```"):
        whole = whole.strip("`").strip()
        if whole.startswith("json"):
            whole = whole[4:].strip()
    try:
        match = _match_tool_call_shape(json.loads(whole))
    except (json.JSONDecodeError, ValueError):
        match = None
    if match is not None:
        return match

    for candidate in find_json_object_candidates(text):
        try:
            match = _match_tool_call_shape(json.loads(candidate))
        except (json.JSONDecodeError, ValueError):
            continue
        if match is not None:
            return match
    return None


def _format_args(args: dict) -> str:
    """First string/number arg, JSON-quoted and truncated — for the compact
    ``→ tool(...)`` event line (mirrors ``cli/src/lib/agent.ts``'s
    ``formatArgs``)."""
    value = next((v for v in args.values() if isinstance(v, (str, int, float))), None)
    if value is None:
        return ""
    text = json.dumps(value)
    return text if len(text) <= 60 else text[:57] + "..."


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

ToolHandler = Callable[[dict], Awaitable[str]]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    handler: ToolHandler


def _tool_to_ollama(tool: ToolSpec) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _require_str(args: dict, key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required and must be a non-empty string")
    return value


def _coerce_int(args: dict, key: str, default: int) -> int:
    if key not in args or args[key] is None:
        return default
    try:
        return int(args[key])
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be an integer")


def _format_hit_lines(chunks, *, with_description: bool) -> list[str]:
    """Shared ranked-hit LINE formatting for search/grep_project (identical
    format to ``src/mcp_server.py``'s ``search``/``grep_project`` tools) —
    empty-result wording is each caller's own (it differs between the two,
    see ``_make_search_handler``/``_make_grep_project_handler``)."""
    lines = []
    for i, c in enumerate(chunks, 1):
        loc = c.absolute_path or c.file_path
        if c.start_line is not None and c.end_line is not None:
            loc += f":{c.start_line}-{c.end_line}"
        score = f"{c.score:.2f}" if c.score is not None else "n/a"
        lines.append(f"{i}. [{c.entity_type}] {c.qualified_name} — {loc} (score {score})")
        if with_description and c.description:
            lines.append(f"   {c.description[:140]}")
    return lines


def _make_search_handler(retriever: Retriever, project: str) -> ToolHandler:
    async def handler(args: dict) -> str:
        query = _require_str(args, "query")
        top_k = _coerce_int(args, "top_k", 5)
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        chunks = await asyncio.to_thread(
            retriever.search_seeds, query, project_name=project, top_k=top_k
        )
        if not chunks:
            return f"(no results for '{query}' in project '{project}')"
        return "\n".join(_format_hit_lines(chunks, with_description=True))

    return handler


def _make_grep_project_handler(retriever: Retriever, project: str) -> ToolHandler:
    async def handler(args: dict) -> str:
        # mcp_server.py's grep_project treats a missing/blank pattern as a
        # (non-error) no-matches result, not a validation error — mirror
        # that exactly rather than using _require_str here.
        pattern = args.get("pattern")
        pattern = pattern if isinstance(pattern, str) else ""
        max_results = _coerce_int(args, "max_results", 20)
        if max_results < 1:
            raise ValueError("max_results must be >= 1")
        if not pattern.strip():
            return f"no matches for '{pattern}' in project '{project}'"
        chunks = await asyncio.to_thread(
            retriever.search_bm25, pattern, project_name=project, top_k=max_results
        )
        if not chunks:
            return f"no matches for '{pattern}' in project '{project}'"
        return "\n".join(_format_hit_lines(chunks, with_description=False))

    return handler


def _make_get_entity_handler(retriever: Retriever, project: str) -> ToolHandler:
    async def handler(args: dict) -> str:
        qualified_name = _require_str(args, "qualified_name")
        chunk = await asyncio.to_thread(
            retriever.get_by_qualified_name, qualified_name, project
        )
        if chunk is None:
            raise ValueError(
                f"entity '{qualified_name}' not found in project '{project}' — try search first"
            )

        lines = [f"[{chunk.entity_type}] {chunk.qualified_name}"]
        loc = chunk.absolute_path or chunk.file_path
        if chunk.start_line is not None and chunk.end_line is not None:
            loc += f" (lines {chunk.start_line}-{chunk.end_line})"
        lines.append(f"File: {loc}")

        if chunk.calls:
            lines.append(f"Calls: {', '.join(chunk.calls)}")
        if chunk.called_by:
            lines.append(f"Called by: {', '.join(chunk.called_by)}")
        if chunk.inherits:
            lines.append(f"Inherits: {', '.join(chunk.inherits)}")
        if chunk.implements:
            lines.append(f"Implements: {', '.join(chunk.implements)}")
        if chunk.overrides:
            lines.append(f"Overrides: {', '.join(chunk.overrides)}")
        if chunk.defined_methods:
            lines.append(f"Methods: {', '.join(chunk.defined_methods)}")
        if chunk.description:
            lines.append(f"Summary: {chunk.description}")

        fence_idx = chunk.chunk_text.find("```")
        source = chunk.chunk_text[fence_idx:] if fence_idx != -1 else chunk.chunk_text
        lines.append("")
        lines.append(source)
        return "\n".join(lines)

    return handler


def _make_neighbors_handler(project: str) -> ToolHandler:
    async def handler(args: dict) -> str:
        qualified_name = _require_str(args, "qualified_name")
        direction = args.get("direction") or "both"
        if direction not in ("in", "out", "both"):
            raise ValueError("direction must be one of: in, out, both")
        depth = _coerce_int(args, "depth", 1)
        if depth < 1:
            raise ValueError("depth must be >= 1")
        limit = _coerce_int(args, "limit", 30)
        if limit < 1:
            raise ValueError("limit must be >= 1")

        result = await asyncio.to_thread(
            graph_store.get_neighborhood,
            project,
            entity=qualified_name,
            depth=depth,
            direction=direction,
            limit=limit,
        )

        index_of: dict[int, int] = {}
        lines = []
        for i, node in enumerate(result["nodes"], 1):
            index_of[node["id"]] = i
            loc = node["absolute_path"] or node["path"] or ""
            if node.get("start_line") is not None:
                loc += f":{node['start_line']}"
            lines.append(f"{i}. hop={node['hop']} [{node['kind']}] {node['qualified_name']} — {loc}")

        for edge in result["edges"]:
            src = index_of.get(edge["source"])
            dst = index_of.get(edge["target"])
            if src is None or dst is None:
                continue
            lines.append(f"{src} -{edge['type']}-> {dst}")

        if result["stats"]["truncated"]:
            lines.append("(truncated — increase limit for more)")

        return "\n".join(lines)

    return handler


def _make_get_file_handler(project: str) -> ToolHandler:
    async def handler(args: dict) -> str:
        path = _require_str(args, "path")
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        start_line = int(start_line) if start_line is not None else None
        end_line = int(end_line) if end_line is not None else None

        node = await asyncio.to_thread(resolve_file, project, path)
        if node is None:
            raise ValueError(
                f"file '{path}' not found in project '{project}' graph — paths must match "
                "the ingested tree (relative like src/a/b.ts or absolute)"
            )

        abs_path = node["absolute_path"] or node["path"]
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            raise ValueError(
                f"file '{path}' is recorded in the graph but missing on disk — re-ingest the project"
            )

        total = len(lines)
        if total == 0:
            if start_line is None and end_line is None:
                return f"File: {abs_path} (0 lines)\n[file is empty]"
            requested = start_line if start_line is not None else 1
            raise ValueError(f"start_line ({requested}) is beyond the file's 0 lines")

        start = start_line if start_line is not None else 1
        end = end_line if end_line is not None else total
        if start < 1 or end < 1:
            raise ValueError("start_line/end_line must be >= 1")
        if start > end:
            raise ValueError(f"invalid range: start_line ({start}) must be <= end_line ({end})")
        if start > total:
            raise ValueError(f"start_line ({start}) is beyond the file's {total} lines")
        end = min(end, total)

        out_lines: list[str] = []
        size = 0
        truncated = False
        actual_end = start - 1
        for offset, line in enumerate(lines[start - 1 : end]):
            line_size = len(line.encode("utf-8"))
            if len(out_lines) >= _MAX_FILE_LINES or size + line_size > _MAX_FILE_BYTES:
                truncated = True
                break
            out_lines.append(line)
            size += line_size
            actual_end = start + offset

        if truncated and not out_lines:
            return (
                f"File: {abs_path}\n"
                f"[line {start} alone exceeds the {_MAX_FILE_BYTES // 1000} KB limit — "
                "file may be minified; no content returned]"
            )

        header = f"File: {abs_path} (lines {start}-{actual_end})"
        body = "".join(out_lines)
        result = f"{header}\n{body}"
        if truncated:
            result += (
                f"\n[truncated — showing lines {start}-{actual_end} of {total}; "
                "use start_line/end_line for more]"
            )
        return result

    return handler


async def _fetch_skill_for_loop(name: str) -> tuple[bool, str]:
    """``(found, text)`` — *text* is the skill body (frontmatter stripped)
    when found, else the "not found" message with the catalog. Uses
    ``asyncio.to_thread`` because ``fetch_skill_sync``/
    ``fetch_skills_catalog_sync`` each do their own internal
    ``asyncio.run()`` — calling them directly from this already-running
    coroutine would raise (see module docstring)."""
    skill = await asyncio.to_thread(fetch_skill_sync, name)
    if skill is None:
        catalog = await asyncio.to_thread(fetch_skills_catalog_sync)
        available = ", ".join(c["name"] for c in catalog) or "(none)"
        return False, f"skill '{name}' not found — available: {available}"
    _frontmatter, body = split_frontmatter(skill["content"])
    return True, body.strip()


def _make_load_skill_handler() -> ToolHandler:
    """Standalone handler (no cap tracking) — used for the tool schema/name
    registration and directly testable in isolation. The cap-aware dispatch
    used by :func:`run_agent_loop` (tracking ``loaded_skills``, emitting
    'loaded skill <name>', enforcing :data:`MAX_LOADED_SKILLS`) lives in
    ``_dispatch_load_skill`` below, since that state belongs to the loop,
    not to a single stateless tool call."""

    async def handler(args: dict) -> str:
        name = _require_str(args, "name")
        _found, text = await _fetch_skill_for_loop(name)
        return text

    return handler


def build_tools(project: str, client: weaviate.WeaviateClient) -> list[ToolSpec]:
    """
    Build the in-process tool registry for *project*, bound to *client* (one
    Weaviate client for the loop's lifetime — closed by the caller/finally,
    NOT by this function or the loop).
    """
    retriever = Retriever(client)
    return [
        ToolSpec(
            name="search",
            description=(
                "Hybrid (semantic + keyword) search for code entities matching a query. "
                "Returns ranked hits with qname, location and score."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query."},
                    "top_k": {"type": "integer", "description": "Max results (default 5)."},
                },
                "required": ["query"],
            },
            handler=_make_search_handler(retriever, project),
        ),
        ToolSpec(
            name="get_entity",
            description=(
                "Fetch a single code entity by exact qualified_name: relations, summary "
                "and source. Call search() first to find the qualified_name."
            ),
            parameters={
                "type": "object",
                "properties": {"qualified_name": {"type": "string"}},
                "required": ["qualified_name"],
            },
            handler=_make_get_entity_handler(retriever, project),
        ),
        ToolSpec(
            name="neighbors",
            description=(
                "Graph neighborhood of an entity (calls/inherits/implements/overrides/"
                "defines), BFS up to depth hops. direction: in|out|both."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "qualified_name": {"type": "string"},
                    "direction": {"type": "string", "enum": ["in", "out", "both"]},
                    "depth": {"type": "integer", "description": "BFS depth (default 1)."},
                    "limit": {"type": "integer", "description": "Max nodes returned (default 30)."},
                },
                "required": ["qualified_name"],
            },
            handler=_make_neighbors_handler(project),
        ),
        ToolSpec(
            name="get_file",
            description=(
                "Read a file from the ingested project tree by path (relative or "
                "absolute), optionally restricted to a 1-indexed inclusive line range."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
            handler=_make_get_file_handler(project),
        ),
        ToolSpec(
            name="grep_project",
            description=(
                "Literal keyword search (BM25 over code text, NOT regex) for a known "
                "identifier. Returns ranked hits with qname, location and BM25 score."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "max_results": {"type": "integer", "description": "Max results (default 20)."},
                },
                "required": ["pattern"],
            },
            handler=_make_grep_project_handler(retriever, project),
        ),
        ToolSpec(
            name="load_skill",
            description=(
                "Load a named skill's full instructions by name (see the skills catalog "
                f"in the system prompt). At most {MAX_LOADED_SKILLS} skills may be loaded per run."
            ),
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=_make_load_skill_handler(),
        ),
    ]


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


async def _dispatch_load_skill(args: dict, loaded_skills: list[str], emit: EmitFn) -> str:
    """Cap-aware ``load_skill`` dispatch (decision 9d) — tracks
    ``loaded_skills`` (cap :data:`MAX_LOADED_SKILLS`), emits ``'loaded skill
    <name>'``. Called by :func:`run_agent_loop` instead of the tool's own
    (stateless) handler."""
    name = args.get("name")
    if not isinstance(name, str) or not name.strip():
        return "ERROR: name is required and must be a non-empty string"
    name = name.strip()

    if name not in loaded_skills and len(loaded_skills) >= MAX_LOADED_SKILLS:
        return (
            f"skill '{name}' not loaded — you already have the maximum of "
            f"{MAX_LOADED_SKILLS} skills loaded ({', '.join(loaded_skills)}); "
            "reuse what you've already loaded."
        )

    found, text = await _fetch_skill_for_loop(name)
    if found and name not in loaded_skills:
        loaded_skills.append(name)
        emit("agent", f"loaded skill {name}")
    return text


def _parse_tool_call_args(raw_args: Any) -> tuple[dict | None, str | None]:
    """``(args, error)`` — *args* is a JSON object dict on success; *error*
    is a malformed-tool-call message otherwise (one of the two is always
    ``None``)."""
    if raw_args is None:
        return {}, None
    if isinstance(raw_args, str):
        if not raw_args.strip():
            return {}, None
        try:
            raw_args = json.loads(raw_args)
        except (json.JSONDecodeError, ValueError):
            return None, f"malformed tool call — arguments is not valid JSON: {raw_args!r}"
    if not isinstance(raw_args, dict):
        return None, f"malformed tool call — arguments must be a JSON object, got {type(raw_args).__name__}"
    return raw_args, None


async def run_agent_loop(
    provider: LLMProvider,
    model: str,
    system_prompt: str,
    user_prompt: str,
    tools: list[ToolSpec],
    emit: EmitFn,
    *,
    max_rounds: int = MAX_ROUNDS,
) -> dict:
    """
    Run the bounded agentic tool loop (decisions 9a/9h): rounds of chat ->
    execute any tool_calls -> append tool results, until the model replies
    with plain content (no tool_calls) or *max_rounds* is reached (the last
    round is a forced, tool-less "answer now" round — the loop always
    terminates with an answer, possibly empty).

    Ollama-native only (9a): raises :class:`NotImplementedError` immediately
    for any provider other than ``"ollama"``.

    The system prompt is extended with the skills catalog (decision 9d)
    before the first round (see :func:`build_system_prompt`).

    Returns ``{"answer", "rounds", "tool_calls_made", "tool_payload_chars",
    "loaded_skills", "tools_called"}`` — ``tools_called`` is the sorted list
    of distinct tool NAMES actually dispatched during the run (regardless of
    whether the handler raised), for callers that need to check whether a
    specific tool (e.g. ``get_entity``/``get_file``) was used without
    threading a bespoke parameter through the loop itself.
    """
    if provider.provider_name != "ollama":
        raise NotImplementedError(
            "bug hunt requires Ollama-native tool calling in this release — "
            f"provider '{provider.provider_name}' is not supported yet"
        )
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")

    catalog = await asyncio.to_thread(fetch_skills_catalog_sync)
    full_system_prompt = build_system_prompt(system_prompt, catalog)

    tools_by_name = {t.name: t for t in tools}
    ollama_tools = [_tool_to_ollama(t) for t in tools]

    messages: list[dict] = [
        {"role": "system", "content": full_system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    called_signatures: set[str] = set()
    loaded_skills: list[str] = []
    tools_called: set[str] = set()
    tool_calls_made = 0
    tool_payload_chars = 0
    budget_warned = False

    for round_no in range(1, max_rounds + 1):
        is_last_round = round_no == max_rounds
        if is_last_round:
            messages.append({"role": "user", "content": _FORCE_FINAL_PROMPT})

        message = await provider.chat_with_tools(
            messages, model=model, tools=None if is_last_round else ollama_tools
        )
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        used_fallback = False
        if not tool_calls and not is_last_round:
            fallback = extract_fallback_tool_call(message.get("content") or "")
            if fallback:
                tool_calls = [{"function": fallback}]
                used_fallback = True

        if not tool_calls or is_last_round:
            return {
                "answer": message.get("content") or "",
                "rounds": round_no,
                "tool_calls_made": tool_calls_made,
                "tool_payload_chars": tool_payload_chars,
                "loaded_skills": loaded_skills,
                "tools_called": sorted(tools_called),
            }

        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name")
            args, error = _parse_tool_call_args(fn.get("arguments"))

            if error is not None:
                emit("agent", f"→ {name or '?'}(...) · {error}")
                messages.append({"role": "tool", "content": f"ERROR: {error}" + _TOOL_RESULT_REMINDER})
                continue

            note = " (fallback parse)" if used_fallback else ""
            emit("agent", f"→ {name}({_format_args(args)}){note}")

            if not name:
                text = "ERROR: malformed tool call — missing tool name"
            elif name not in tools_by_name:
                text = f"ERROR: unknown tool '{name}' — available: {', '.join(tools_by_name)}"
            else:
                signature = f"{name}:{json.dumps(args, sort_keys=True)}"
                if signature in called_signatures:
                    text = (
                        f"ERROR: you already called {name}({_format_args(args)}) with these "
                        "exact arguments earlier in this conversation — reuse that result, "
                        "do not call it again."
                    )
                else:
                    called_signatures.add(signature)
                    tools_called.add(name)
                    if name == "load_skill":
                        text = await _dispatch_load_skill(args, loaded_skills, emit)
                    else:
                        try:
                            text = await tools_by_name[name].handler(args)
                        except Exception as e:  # noqa: BLE001 — model can recover from an error result
                            text = f"ERROR: {e}"

                    tool_calls_made += 1
                    if tool_payload_chars > TOOL_PAYLOAD_CHAR_BUDGET:
                        if not budget_warned:
                            emit(
                                "agent",
                                f"tool payload budget ({TOOL_PAYLOAD_CHAR_BUDGET} chars) "
                                "exceeded — truncating further tool results",
                            )
                            budget_warned = True
                        text = text[:BUDGET_TRUNCATE_CHARS] + "\n[truncated — tool payload budget exceeded]"
                    tool_payload_chars += len(text)

            messages.append({"role": "tool", "content": text + _TOOL_RESULT_REMINDER})

    # Unreachable: the is_last_round branch above always returns by the time
    # round_no == max_rounds.
    raise AssertionError("run_agent_loop: exhausted max_rounds without returning")
