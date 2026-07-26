"""
Dokkai MCP server — stdio server exposing the codebase-retrieval tools
(hybrid search, graph neighborhood, entity/file lookup) to MCP clients.

Launch: uv run --directory <repo root> python src/mcp_server.py

Transport is stdio ONLY: stdout is reserved for the MCP protocol, so this
module and everything it calls must never print() — diagnostics go to
stderr via the module logger below.
"""

from __future__ import annotations

import atexit
import functools
import logging
import os
import sys

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

logger = logging.getLogger("dokkai.mcp")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logger.addHandler(_handler)
logger.propagate = False

from services import graph_store, navigation
from services.graph_store import resolve_file
from services.retriever import Retriever
from services.weaviate_client import count_chunks_by_project, get_client

# DOKKAI_MCP_PROFILE unset (default) keeps the original instructions; "small-model"
# swaps in a more directive, anti-loop variant for small local models (observed
# to loop/re-call tools without it — see scripts/mcp_harness.py's SYSTEM_PROMPT
# for the same phrasing pattern proven there).
_INSTRUCTIONS_DEFAULT = (
    "Explore an ingested codebase. Orient with tree() for the repo's shape, then "
    "narrow with outline(file) to map a file's entities or search(query) for seed "
    "entities by name/concept. Read ONLY the chosen range via get_file(path, "
    "start_line, end_line) — never fetch a whole file when a range will do. Use neighbors()/"
    "get_entity() to explore the call graph and read code, path(source, target) to "
    "answer \"how does X reach Y\", and context() for a one-shot answer bundle."
)
_INSTRUCTIONS_SMALL_MODEL = (
    "Explore an ingested codebase. Call ONE tool, then check if its result already "
    "answers the question. Orient with tree() for the repo's shape, narrow with "
    "outline(file) or search(query)/context(query) for seed entities, "
    "grep_project(pattern) for a known identifier, neighbors()/get_entity() for call "
    "graph and code, path(source, target) for how X reaches Y, get_file(path, "
    "start_line, end_line) for raw ranges — read ONLY the range you need, never a "
    "whole file. Never "
    "call the same tool with the same arguments twice. As soon as you have enough "
    "information, STOP calling tools and answer as plain text."
)
_MCP_PROFILE = os.getenv("DOKKAI_MCP_PROFILE")

mcp = FastMCP(
    "dokkai",
    instructions=(
        _INSTRUCTIONS_SMALL_MODEL if _MCP_PROFILE == "small-model" else _INSTRUCTIONS_DEFAULT
    ),
)


# ---------------------------------------------------------------------------
# Session watchdog — always-on, stderr-only per-tool usage summary
# ---------------------------------------------------------------------------

_watchdog_stats: dict[str, dict[str, int]] = {}


def _watchdog(fn):
    """Wrap a tool function to accumulate its call count and response chars
    into ``_watchdog_stats``, centrally instead of by hand in every tool."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        stats = _watchdog_stats.setdefault(fn.__name__, {"calls": 0, "chars": 0})
        stats["calls"] += 1
        result = fn(*args, **kwargs)
        if isinstance(result, str):
            stats["chars"] += len(result)
        return result

    return wrapper


def _print_watchdog_summary() -> None:
    """Print the per-session tool usage summary to stderr at shutdown."""
    if not _watchdog_stats:
        return
    lines = ["", "=" * 60, "dokkai MCP session summary", "=" * 60]
    total_calls = total_chars = 0
    for name in sorted(_watchdog_stats):
        s = _watchdog_stats[name]
        tokens = s["chars"] // 4
        lines.append(f"{name:<14} calls={s['calls']:<4} chars={s['chars']:<8} ~tokens={tokens}")
        total_calls += s["calls"]
        total_chars += s["chars"]
    lines.append("-" * 60)
    lines.append(f"{'TOTAL':<14} calls={total_calls:<4} chars={total_chars:<8} ~tokens={total_chars // 4}")
    lines.append("=" * 60)
    logger.info("\n".join(lines))


atexit.register(_print_watchdog_summary)


# ---------------------------------------------------------------------------
# Project resolution
# ---------------------------------------------------------------------------


def _resolve_project(project: str | None) -> str:
    """
    Resolve the optional ``project`` param against ingested graphs.

    Explicit project names are validated against list_graphs() when at least
    one graph is ingested; an omitted project auto-resolves if exactly one
    project is ingested.
    """
    known = [g["project"] for g in graph_store.list_graphs()]
    if project:
        if known and project not in known:
            raise ValueError(f"unknown project '{project}' — call list_projects")
        return project
    if len(known) == 1:
        return known[0]
    if not known:
        raise ValueError("no project ingested — run POST /instances/pipeline to ingest one")
    raise ValueError(
        f"multiple projects ingested ({', '.join(known)}) — specify project"
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
@_watchdog
def list_projects() -> str:
    """List ingested projects with chunk counts and graph size.
    Use this first when the project name is unknown."""
    client = get_client()
    try:
        chunk_counts = {c["project"]: c["chunks"] for c in count_chunks_by_project(client)}
    finally:
        client.close()

    graphs = {g["project"]: g for g in graph_store.list_graphs()}
    names = sorted(set(chunk_counts) | set(graphs))
    if not names:
        return "(no projects ingested — run POST /instances/pipeline)"

    lines = []
    for name in names:
        chunks = chunk_counts.get(name, "-")
        g = graphs.get(name)
        nodes = g["nodes"] if g else "-"
        edges = g["edges"] if g else "-"
        generated_at = g["generated_at"] if g else "-"
        lines.append(
            f"{name} — chunks: {chunks}, nodes: {nodes}, edges: {edges}, "
            f"generated_at: {generated_at}"
        )
    return "\n".join(lines)


@mcp.tool()
@_watchdog
def search(query: str, project: str | None = None, k: int = 8) -> str:
    """Hybrid (semantic + keyword) search for code entities matching a query.
    Returns ranked hits with qname, location and score. For a hit's whole
    file, call outline(file) next."""
    if k < 1:
        raise ValueError("k must be >= 1")
    resolved = _resolve_project(project)
    client = get_client()
    try:
        chunks = Retriever(client).search_seeds(query, project_name=resolved, top_k=k)
    finally:
        client.close()

    if not chunks:
        return f"(no results for '{query}' in project '{resolved}')"

    lines = []
    for i, c in enumerate(chunks, 1):
        loc = c.absolute_path or c.file_path
        if c.start_line is not None and c.end_line is not None:
            loc += f":{c.start_line}-{c.end_line}"
        score = f"{c.score:.2f}" if c.score is not None else "n/a"
        lines.append(f"{i}. [{c.entity_type}] {c.qualified_name} — {loc} (score {score})")
        if c.description:
            lines.append(f"   {navigation.first_sentence(c.description)}")
    return "\n".join(lines)


@mcp.tool()
@_watchdog
def grep_project(pattern: str, project: str | None = None, k: int = 10) -> str:
    """Literal keyword search (BM25 over code text, NOT regex) for a known
    identifier. Returns ranked hits with qname, location and BM25 score.
    Test-file hits are down-weighted like in search()."""
    if k < 1:
        raise ValueError("k must be >= 1")
    resolved = _resolve_project(project)
    if not pattern.strip():
        return f"no matches for '{pattern}' in project '{resolved}'"
    client = get_client()
    try:
        chunks = Retriever(client).search_bm25(pattern, project_name=resolved, top_k=k)
    finally:
        client.close()

    if not chunks:
        return f"no matches for '{pattern}' in project '{resolved}'"

    lines = []
    for i, c in enumerate(chunks, 1):
        loc = c.absolute_path or c.file_path
        if c.start_line is not None and c.end_line is not None:
            loc += f":{c.start_line}-{c.end_line}"
        score = f"{c.score:.2f}" if c.score is not None else "n/a"
        lines.append(f"{i}. [{c.entity_type}] {c.qualified_name} — {loc} (score {score})")
    return "\n".join(lines)


@mcp.tool()
@_watchdog
def get_entity(qualified_name: str, project: str | None = None) -> str:
    """Fetch a single code entity by exact qualified_name: relations, summary
    and source. Call search() first to find the qualified_name."""
    resolved = _resolve_project(project)
    client = get_client()
    try:
        chunk = Retriever(client).get_by_qualified_name(qualified_name, resolved)
    finally:
        client.close()

    if chunk is None:
        raise ValueError(
            f"entity '{qualified_name}' not found in project '{resolved}' — try search first"
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


@mcp.tool()
@_watchdog
def neighbors(
    qualified_name: str,
    project: str | None = None,
    direction: str = "both",
    depth: int = 1,
    limit: int = 30,
) -> str:
    """Graph neighborhood of an entity (calls/inherits/implements/overrides/
    defines), BFS up to depth hops, each with a best-effort 1-line summary
    when Weaviate has one. direction: in|out|both."""
    if direction not in ("in", "out", "both"):
        raise ValueError("direction must be one of: in, out, both")
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if limit < 1:
        raise ValueError("limit must be >= 1")
    resolved = _resolve_project(project)
    result = graph_store.get_neighborhood(
        resolved, entity=qualified_name, depth=depth, direction=direction, limit=limit
    )

    # Best-effort, ONE batched round trip — degrades to {} (no summary lines,
    # no error) when Weaviate is unreachable, preserving neighbors()'s
    # offline behavior exactly.
    qnames = [node["qualified_name"] for node in result["nodes"] if node.get("qualified_name")]
    summaries, _ = navigation.fetch_summaries(resolved, qnames)

    index_of: dict[int, int] = {}
    lines = []
    for i, node in enumerate(result["nodes"], 1):
        index_of[node["id"]] = i
        loc = node["absolute_path"] or node["path"] or ""
        if node.get("start_line") is not None:
            loc += f":{node['start_line']}"
        lines.append(f"{i}. hop={node['hop']} [{node['kind']}] {node['qualified_name']} — {loc}")
        summary = summaries.get(node["qualified_name"])
        if summary:
            lines.append(f"   {navigation.first_sentence(summary)}")

    for edge in result["edges"]:
        src = index_of.get(edge["source"])
        dst = index_of.get(edge["target"])
        if src is None or dst is None:
            continue
        lines.append(f"{src} -{edge['type']}-> {dst}")

    if result["stats"]["truncated"]:
        lines.append("(truncated — increase limit for more)")

    return "\n".join(lines)


@mcp.tool()
@_watchdog
def context(query: str, project: str | None = None, k: int = 8) -> str:
    """One-shot graph-expanded context for a query: seeds plus their call-graph
    neighbors, formatted for direct use as LLM context."""
    if k < 1:
        raise ValueError("k must be >= 1")
    resolved = _resolve_project(project)
    client = get_client()
    try:
        retriever = Retriever(client)
        chunks = retriever.search_graph(query, project_name=resolved, top_k_seeds=k)
        return retriever.build_graph_context(chunks, max_chars=10_000)
    finally:
        client.close()


_TREE_BUDGET = 10_000


def _count_label(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def _format_tree_entry(entry: dict) -> str:
    indent = "  " * entry["depth"]
    if entry["type"] == "dir":
        files = _count_label(entry["files"], "file", "files")
        entities = _count_label(entry["entities"], "entity", "entities")
        return f"{indent}{entry['name']}/ ({files}, {entities})"
    kinds = ", ".join(f"{v} {k}" for k, v in entry["entities"].items())
    return f"{indent}{entry['name']}" + (f" ({kinds})" if kinds else "")


def _tree_header(payload: dict, requested_depth: int | None = None) -> str:
    root_label = payload["root"] or "(repo root)"
    depth_label = f"depth {payload['depth']}"
    if requested_depth is not None and requested_depth != payload["depth"]:
        depth_label = (
            f"depth {payload['depth']} of {requested_depth} requested — "
            f"{_TREE_BUDGET:,} char budget"
        )
    files = _count_label(payload["files"], "file", "files")
    entities = _count_label(payload["entities"], "entity", "entities")
    return f"{payload['project']} — {root_label} ({depth_label}): {files}, {entities}"


def _render_tree(payload: dict, requested_depth: int | None = None) -> str:
    lines = [_tree_header(payload, requested_depth)]
    lines.extend(_format_tree_entry(entry) for entry in payload["entries"])
    return "\n".join(lines)


def _truncate_tree_entries(payload: dict, budget: int, requested_depth: int | None = None) -> str:
    """Last-resort fallback when even depth=1 overflows budget: keep as many
    entries as fit (in order), then a single deterministic '(+N more)' line
    — never a mid-line cut."""
    lines = [_tree_header(payload, requested_depth)]
    entries = payload["entries"]
    marker_reserve = len(f"(+{len(entries)} more)")
    kept = 0
    for entry in entries:
        line = _format_tree_entry(entry)
        projected_len = sum(len(l) + 1 for l in lines) + len(line) + marker_reserve + 1
        if projected_len > budget:
            break
        lines.append(line)
        kept += 1
    remaining = len(entries) - kept
    if remaining:
        lines.append(f"(+{remaining} more)")
    return "\n".join(lines)


@mcp.tool()
@_watchdog
def tree(path: str | None = None, depth: int = 2, project: str | None = None) -> str:
    """Directory structure map of the repo: directories with aggregate file/
    entity counts, files with per-kind entity counts (e.g. "2 Class, 5
    Method"). Cheap, deterministic overview to orient before diving in — use
    outline(file) next to see inside a file. `path` is a DIRECTORY path,
    relative (e.g. "src/features") or absolute under the project's repo
    root (stripped automatically); a file path raises a hint to use
    outline() instead."""
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if depth > 10:
        raise ValueError("depth must be <= 10")
    resolved = _resolve_project(project)

    payload = None
    text = ""
    for try_depth in range(depth, 0, -1):
        payload = navigation.get_tree(resolved, path=path, depth=try_depth)
        text = _render_tree(payload, requested_depth=depth)
        if len(text) <= _TREE_BUDGET:
            return text
    return _truncate_tree_entries(payload, _TREE_BUDGET, requested_depth=depth)


_OUTLINE_BUDGET = 10_000


def _render_outline_entity(entry: dict, indent: str = "") -> list[str]:
    if entry["start_line"] and entry["end_line"]:
        loc = f"lines {entry['start_line']}-{entry['end_line']}"
    else:
        loc = "location unknown"
    lines = [f"{indent}[{entry['kind']}] {entry['name']} — {loc}"]
    if entry["signature"]:
        lines.append(f"{indent}  {entry['signature']}")
    if entry["summary"]:
        lines.append(f"{indent}  {entry['summary']}")
    for method in entry["methods"]:
        lines.extend(_render_outline_entity(method, indent + "  "))
    return lines


def _outline_entity_total(payload: dict) -> int:
    """Total entities in the outline, nested methods included — must agree
    with tree()'s per-file counts for the same file."""
    return len(payload["entities"]) + sum(len(e["methods"]) for e in payload["entities"])


def _outline_header(payload: dict) -> str:
    note = "" if payload["summaries_available"] else " (Weaviate unreachable — no summaries)"
    total = _count_label(_outline_entity_total(payload), "entity", "entities")
    return f"{payload['project']} — {payload['file']} ({total}){note}"


@mcp.tool()
@_watchdog
def outline(file: str, project: str | None = None) -> str:
    """Per-file entity map: every Class/Function/Method/etc in a file, with
    line ranges, a disk-derived signature line and a 1-line summary where
    available (methods nested under their class). Pick the exact line range
    you need here, then get_file(file, start_line, end_line) to read it.
    `file` accepts a relative or absolute path (same resolution as
    get_file)."""
    resolved = _resolve_project(project)
    payload = navigation.get_outline(resolved, file)

    lines = [_outline_header(payload)]
    entries = payload["entities"]
    blocks = [_render_outline_entity(entry) for entry in entries]
    marker_reserve = len(f"(+{_outline_entity_total(payload)} more entities)")
    kept = 0
    for block in blocks:
        block_len = sum(len(l) + 1 for l in block)
        projected = sum(len(l) + 1 for l in lines) + block_len + marker_reserve + 1
        if projected > _OUTLINE_BUDGET:
            break
        lines.extend(block)
        kept += 1
    # The marker reports omitted ENTITIES (blocks plus their nested
    # methods), matching the header's unit — not omitted render blocks.
    remaining = sum(1 + len(e["methods"]) for e in entries[kept:])
    if remaining:
        lines.append(f"(+{_count_label(remaining, 'more entity', 'more entities')})")
    return "\n".join(lines)


_PATH_BUDGET = 4_000


_PATH_VIA_EXTERNAL_NOTE = (
    "note: this path only connects through external modules — weak structural coupling"
)


def _format_path_node(index: int, node: dict) -> str:
    loc = node["absolute_path"] or node["path"] or ""
    if loc and node.get("start_line") is not None:
        loc += f":{node['start_line']}"
    # LOW-3: a pathless node (~32 anonymous nested functions in the corpus,
    # no `path`/`absolute_path` property at all) gets no location suffix —
    # never a dangling "— :N"; get_entity(qname) is the fallback.
    suffix = f" — {loc}" if loc else ""
    kind_label = f"{node['kind']}, external" if node.get("is_external") else node["kind"]
    return f"{index + 1}. [{kind_label}] {node['qualified_name']}{suffix}"


def _format_path_hop(hop: dict) -> str:
    arrow = f"-{hop['type']}->" if hop["direction"] == "out" else f"<-{hop['type']}-"
    return f"   {arrow}"


def _path_node_groups(nodes: list[dict], hops: list[dict]) -> list[list[str]]:
    """One render group per node: its own line plus its OUTGOING hop line
    (if any) — the atomic unit truncation keeps or drops whole, never
    splitting a node from its hop line."""
    groups = []
    for i, node in enumerate(nodes):
        group = [_format_path_node(i, node)]
        if i < len(hops):
            group.append(_format_path_hop(hops[i]))
        groups.append(group)
    return groups


def _path_header(payload: dict) -> str:
    return (
        f"{payload['project']} — '{payload['source']}' to '{payload['target']}' "
        f"({_count_label(len(payload['hops']), 'hop', 'hops')})"
    )


def _render_path(payload: dict) -> str:
    if not payload["found"]:
        edge_types = ", ".join(payload["edge_types"])
        return (
            f"{payload['project']} — no path from '{payload['source']}' to "
            f"'{payload['target']}' within {_count_label(payload['max_depth'], 'hop', 'hops')} "
            f"(edge types: {edge_types}). Try neighbors() on each endpoint, "
            "or raise max_depth."
        )

    lines = [_path_header(payload)]
    for group in _path_node_groups(payload["nodes"], payload["hops"]):
        lines.extend(group)
    if payload["via_external"]:
        lines.append(_PATH_VIA_EXTERNAL_NOTE)
    return "\n".join(lines)


def _truncate_path(payload: dict, budget: int) -> str:
    """
    INFO-4 fallback for an over-budget render — keeps BOTH endpoints: a head
    growing from the source and a tail growing from the target, with a
    single exact ``(… N hops omitted …)`` marker in between. Whole node+hop
    groups only (never a mid-line cut), grown greedily one group at a time
    alternating head/tail so a realistic path shows as much of both ends as
    fits. The per-step budget check reserves the WORST-CASE marker width
    (all hops omitted) — a conservative overestimate, same "+1 per line"
    margin as tree's/outline's truncation helpers — so the result is never
    over budget, only occasionally short of the true limit by a few chars.
    """
    header = _path_header(payload)
    note = _PATH_VIA_EXTERNAL_NOTE if payload["via_external"] else None
    reserved_note_len = len(note) + 1 if note else 0
    groups = _path_node_groups(payload["nodes"], payload["hops"])
    total = len(groups)
    worst_marker = len(f"(… {total} hops omitted …)")

    def _fits(h: int, t: int) -> bool:
        kept_lines = [header]
        for g in groups[:h]:
            kept_lines.extend(g)
        kept_lines.append("#" * worst_marker)
        for g in groups[total - t :] if t else []:
            kept_lines.extend(g)
        return sum(len(l) + 1 for l in kept_lines) + reserved_note_len <= budget

    head = tail = 0
    grow_head = True
    while head + tail < total:
        candidate = (head + 1, tail) if grow_head else (head, tail + 1)
        if not _fits(*candidate):
            break
        head, tail = candidate
        grow_head = not grow_head

    lines = [header]
    for g in groups[:head]:
        lines.extend(g)
    if head + tail < total:
        # Drop the last head group's dangling arrow — it points into the
        # elided region. With it gone, rendered hop lines + omitted equals
        # the path's total hops exactly, in every regime (incl. head or
        # tail == 0 under degenerate budgets).
        if head and len(groups[head - 1]) == 2:
            lines.pop()
        rendered_hops = (head - 1 if head else 0) + (tail - 1 if tail else 0)
        omitted_hops = len(payload["hops"]) - rendered_hops
        lines.append(f"(… {omitted_hops} hops omitted …)")
    for g in (groups[total - tail :] if tail else []):
        lines.extend(g)
    if note:
        lines.append(note)
    return "\n".join(lines)


@mcp.tool()
@_watchdog
def path(source: str, target: str, max_depth: int = 10, project: str | None = None) -> str:
    """Shortest structural route between two entities — answers "how does X
    reach Y" for e2e wiring questions. Undirected BFS over calls/inherits/
    implements/overrides/defines/imports edges, excluding third-party
    external modules (e.g. mongoose, express) as pass-through hubs — a path
    that only connects via one is still returned, marked `[Module,
    external]` with a weak-coupling note, never silently preferred over a
    real app-level route. The result is the shortest STRUCTURAL connection,
    not necessarily the runtime call narrative. When a mutual edge exists in
    both directions between two nodes, the rendered arrow reflects whichever
    direction the deterministic tie-break selects — both are equally true.
    Call search() first to get exact qualified_name endpoints (Module
    qualified_names work too, since import edges connect modules)."""
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")
    if max_depth > 20:
        raise ValueError("max_depth must be <= 20")
    resolved = _resolve_project(project)
    payload = navigation.get_path(resolved, source, target, max_depth=max_depth)
    if not payload["found"]:
        return _render_path(payload)
    text = _render_path(payload)
    if len(text) <= _PATH_BUDGET:
        return text
    return _truncate_path(payload, _PATH_BUDGET)


_MAX_FILE_LINES = 2_000
_MAX_FILE_BYTES = 100_000


@mcp.tool()
@_watchdog
def get_file(
    path: str,
    project: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Read a file from the ingested project tree by path (relative or
    absolute), optionally restricted to a 1-indexed inclusive line range.
    Call outline(file) first to pick the exact range — avoid fetching a
    whole file when a range will do."""
    resolved = _resolve_project(project)
    node = resolve_file(resolved, path)
    if node is None:
        raise ValueError(
            f"file '{path}' not found in project '{resolved}' graph — paths must match "
            "the ingested tree (relative like src/a/b.ts or absolute)"
        )

    abs_path = node["absolute_path"] or node["path"]
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        raise ValueError(
            f"file '{path}' is recorded in the graph but missing on disk — "
            "re-ingest the project"
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


if __name__ == "__main__":
    mcp.run()
