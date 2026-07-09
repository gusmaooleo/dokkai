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

from services import graph_store
from services.graph_store import resolve_file
from services.retriever import Retriever
from services.weaviate_client import count_chunks_by_project, get_client

# DOKKAI_MCP_PROFILE unset (default) keeps the original instructions; "small-model"
# swaps in a more directive, anti-loop variant for small local models (observed
# to loop/re-call tools without it — see scripts/mcp_harness.py's SYSTEM_PROMPT
# for the same phrasing pattern proven there).
_INSTRUCTIONS_DEFAULT = (
    "Explore an ingested codebase: start with search(query) to find seed entities, "
    "then use neighbors()/get_entity() to explore call graph and read code, and "
    "get_file() for raw file ranges. Use context() for a one-shot answer bundle."
)
_INSTRUCTIONS_SMALL_MODEL = (
    "Explore an ingested codebase. Call ONE tool, then check if its result already "
    "answers the question. Start with search(query) or context(query) for seed "
    "entities, grep_project(pattern) for a known identifier, neighbors()/get_entity() "
    "for call graph and code, get_file() for raw ranges. Never call the same tool "
    "with the same arguments twice. As soon as you have enough information, STOP "
    "calling tools and answer as plain text."
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
    Returns ranked hits with qname, location and score."""
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
            desc = c.description[:140]
            lines.append(f"   {desc}")
    return "\n".join(lines)


@mcp.tool()
@_watchdog
def grep_project(pattern: str, project: str | None = None, k: int = 10) -> str:
    """Literal keyword search (BM25 over code text, NOT regex) for a known
    identifier. Returns ranked hits with qname, location and BM25 score."""
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
    defines), BFS up to depth hops. direction: in|out|both."""
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
    absolute), optionally restricted to a 1-indexed inclusive line range."""
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
