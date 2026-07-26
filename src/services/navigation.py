"""
Navigation service — deterministic repo-structure maps computed straight
from a project's ingested graph JSON (see :mod:`services.graph_store`), plus
per-file entity outlines that add a light Weaviate assist (1-line summaries)
on top of the same graph-derived structure.

Gives a coding agent the SHAPE of a repo (``get_tree``) or of a single file
(``get_outline``) in one cheap call.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from services.chunker import CODE_ENTITY_LABELS
from services.graph_store import Graph, get_graph, resolve_file
from services.retriever import Retriever
from services.weaviate_client import get_client

logger = logging.getLogger(__name__)

# In-memory directory tree node: "dirs" maps component name -> child node,
# "files" maps filename -> {entity_kind: count} (empty dict = no entities).
# "agg" (added by _compute_aggregates) holds the subtree's (files, entities)
# totals, files included.
_DirNode = dict[str, Any]


def _new_dir() -> _DirNode:
    return {"dirs": {}, "files": {}}


def _insert_file(root: _DirNode, parts: list[str], entity_counts: dict[str, int]) -> None:
    node = root
    for part in parts[:-1]:
        node = node["dirs"].setdefault(part, _new_dir())
    node["files"][parts[-1]] = entity_counts


def _build_tree(graph: Graph) -> _DirNode:
    """
    Build the full-repo directory tree from File-node paths (every file,
    code or not — File nodes cover the whole ingested tree), attaching
    per-file entity kind counts from code-entity nodes (Function/Method/
    Class/Interface/Type/Enum, see :data:`CODE_ENTITY_LABELS`) whose own
    ``path`` matches the file's path.

    Entity nodes with no ``path`` at all (named nested/inner functions that
    cgr emits without a path property — 32 in the reference corpus) can't be
    placed and are skipped — not counted anywhere, so a file's entity count
    here can undercount versus search/get_entity, which do surface those
    entities. Entity nodes whose path doesn't match any File node
    (not observed in the ingested corpus, but not guaranteed absent) are
    defensively added as their own file entry.
    """
    file_paths: set[str] = set()
    for node in graph.nodes:
        if node["labels"][0] == "File":
            path = node.get("properties", {}).get("path")
            if path:
                file_paths.add(path)

    entity_counts_by_path: dict[str, dict[str, int]] = {}
    for node in graph.nodes:
        labels = node["labels"]
        if not labels or labels[0] not in CODE_ENTITY_LABELS:
            continue
        path = node.get("properties", {}).get("path")
        if not path:
            continue
        file_paths.add(path)
        counts = entity_counts_by_path.setdefault(path, {})
        counts[labels[0]] = counts.get(labels[0], 0) + 1

    root = _new_dir()
    for path in file_paths:
        _insert_file(root, path.split("/"), entity_counts_by_path.get(path, {}))
    return root


def _compute_aggregates(node: _DirNode) -> tuple[int, int]:
    """Recursively fill each node's ``agg`` (files, entities) subtree totals."""
    files = len(node["files"])
    entities = sum(sum(counts.values()) for counts in node["files"].values())
    for child in node["dirs"].values():
        child_files, child_entities = _compute_aggregates(child)
        files += child_files
        entities += child_entities
    node["agg"] = (files, entities)
    return files, entities


def _navigate(root: _DirNode, path: str | None) -> _DirNode:
    """Resolve *path* (relative directory path, or ``None``/empty for the
    repo root) to its tree node. Raises :class:`ValueError` with a few
    top-level candidates when *path* doesn't resolve to a directory."""
    if not path:
        return root
    node = root
    for part in path.strip("/").split("/"):
        if part not in node["dirs"]:
            candidates = sorted(root["dirs"])[:8]
            hint = ", ".join(candidates) if candidates else "(no subdirectories)"
            raise ValueError(
                f"path '{path}' not found in project tree — top-level candidates: {hint}"
            )
        node = node["dirs"][part]
    return node


def _render_entries(node: _DirNode, current_depth: int, max_depth: int) -> list[dict[str, Any]]:
    """
    Flatten *node*'s subtree into a pre-order entries list (dirs before
    files, both alphabetical), expanding directories up to *max_depth*
    levels below the starting node. Directories beyond that depth still
    appear (their own line, with aggregate counts) but aren't expanded.

    Single-child directory chains with no files of their own (e.g.
    ``a/b/c/`` where each level holds nothing but the next) are collapsed
    into one entry, so empty nesting doesn't waste lines.
    """
    entries: list[dict[str, Any]] = []
    for name in sorted(node["dirs"]):
        child = node["dirs"][name]
        display_name = name
        while not child["files"] and len(child["dirs"]) == 1:
            ((next_name, next_child),) = child["dirs"].items()
            display_name += "/" + next_name
            child = next_child
        files_count, entities_count = child["agg"]
        entries.append(
            {
                "type": "dir",
                "name": display_name,
                "depth": current_depth,
                "files": files_count,
                "entities": entities_count,
            }
        )
        if current_depth < max_depth:
            entries.extend(_render_entries(child, current_depth + 1, max_depth))
    for name in sorted(node["files"]):
        entries.append(
            {
                "type": "file",
                "name": name,
                "depth": current_depth,
                "entities": dict(sorted(node["files"][name].items())),
            }
        )
    return entries


def get_tree(project: str, path: str | None = None, depth: int = 2) -> dict[str, Any]:
    """
    Assemble the directory-tree payload for *project*.

    ``path`` scopes the tree to a subtree root (relative directory path,
    e.g. ``src/features``); ``None``/empty selects the repo root. ``depth``
    is how many directory levels below the root to expand — deeper
    directories still get a line (aggregate counts) but aren't expanded
    further. Callers are expected to have already validated ``depth``.

    Raises :class:`services.graph_store.GraphNotFoundError` when no graph
    has been ingested for the project (propagated from
    :func:`services.graph_store.get_graph`), and :class:`ValueError` when
    ``path`` doesn't resolve to a directory in the project's tree.
    """
    graph = get_graph(project)
    root = _build_tree(graph)
    _compute_aggregates(root)
    subtree = _navigate(root, path)

    files, entities = subtree["agg"]
    return {
        "project": project,
        "root": path or "",
        "depth": depth,
        "files": files,
        "entities": entities,
        "entries": _render_entries(subtree, 1, depth),
    }


# ---------------------------------------------------------------------------
# outline — per-file entity map
# ---------------------------------------------------------------------------

_SIGNATURE_MAX_CHARS = 120
_SUMMARY_MAX_CHARS = 140
_SENTENCE_END_RE = re.compile(r"[.!?](\s|$)")


def _read_lines(absolute_path: str | None) -> list[str] | None:
    """Read *absolute_path* into lines, or ``None`` if it can't be read —
    signature lines are a best-effort disk read, never an error."""
    if not absolute_path:
        return None
    try:
        with open(absolute_path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.readlines()
    except OSError:
        return None


def _signature_line(lines: list[str] | None, start_line: int | None) -> str | None:
    """
    An entity's first source line (disk-derived), stripped and capped at
    ``_SIGNATURE_MAX_CHARS``. ``None`` (never an error) when *lines* is
    unavailable (unreadable/missing file), *start_line* is missing/out of
    range, or the line is blank after stripping.
    """
    if not lines or not start_line or start_line < 1 or start_line > len(lines):
        return None
    line = lines[start_line - 1].strip()
    if not line:
        return None
    if len(line) > _SIGNATURE_MAX_CHARS:
        # -1 so the ellipsis fits INSIDE the cap (the stated max is the max).
        line = line[: _SIGNATURE_MAX_CHARS - 1].rstrip() + "…"
    return line


def _first_sentence(text: str, max_chars: int = _SUMMARY_MAX_CHARS) -> str:
    """
    First sentence of *text* (up to the first ``.``/``!``/``?`` followed by
    whitespace or end-of-string), capped at *max_chars*. When the sentence
    itself exceeds the cap, cuts at the last whole word inside it — never
    mid-word (unlike ``search()``'s plain ``[:140]`` slice).
    """
    text = text.strip()
    if not text:
        return ""
    match = _SENTENCE_END_RE.search(text)
    sentence = text[: match.end()].strip() if match else text
    if len(sentence) <= max_chars:
        return sentence
    truncated = sentence[: max_chars - 1]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip() + "…"


def _fetch_summaries(project: str, qualified_names: list[str]) -> tuple[dict[str, str], bool]:
    """
    Batch-fetch entity descriptions for *qualified_names* from Weaviate in
    ONE round trip (see :meth:`services.retriever.Retriever._fetch_by_qnames`
    — deterministic UUID lookup, the same pattern
    :func:`services.graph_store._fetch_chunk_fields` uses for a single
    entity). Degrades to ``({}, False)`` — logged, never raised — when
    Weaviate is unreachable or the fetch itself fails; ``get_outline`` must
    always succeed regardless of Weaviate's availability.
    """
    if not qualified_names:
        return {}, True
    try:
        client = get_client()
    except Exception as e:
        logger.warning("outline: Weaviate unreachable, omitting summaries: %s", e)
        return {}, False
    try:
        fetched = Retriever(client)._fetch_by_qnames(qualified_names, project)
    except Exception as e:
        logger.warning("outline: failed to fetch summaries for '%s': %s", project, e)
        return {}, False
    finally:
        client.close()
    return {qn: c.description for qn, c in fetched.items() if c.description}, True


def get_outline(project: str, file: str) -> dict[str, Any]:
    """
    Assemble the per-file entity outline for *file* in *project*.

    Resolves *file* through the same traversal-guarded mechanism as the MCP
    ``get_file`` tool (:func:`services.graph_store.resolve_file` — relative
    or absolute path), then collects the file's code-entity nodes
    (Function/Method/Class/Interface/Type/Enum, matched by their own
    ``path``), sorted by ``start_line``, with Methods nested under their
    Class via ``DEFINES_METHOD`` edges (a Method whose Class isn't in this
    file — not expected, but handled defensively — stays top-level).

    Each entity carries a disk-derived ``signature`` (see
    :func:`_signature_line`) and a Weaviate-derived one-line ``summary``
    (see :func:`_first_sentence`), both ``None`` when unavailable —
    ``summaries_available`` tells the caller whether that's because Weaviate
    was unreachable or because the entity simply had no description.

    A non-code file (e.g. a config or markdown file) is a valid outline with
    an empty entity list — not an error.

    Raises :class:`ValueError` (mirroring ``get_file``'s message) when
    *file* doesn't resolve to a node in the project's graph.
    """
    node = resolve_file(project, file)
    if node is None:
        raise ValueError(
            f"file '{file}' not found in project '{project}' graph — paths must match "
            "the ingested tree (relative like src/a/b.ts or absolute)"
        )

    graph = get_graph(project)
    file_path = node["path"]
    entity_nodes = [
        n
        for n in graph.nodes
        if n["labels"][0] in CODE_ENTITY_LABELS
        and n.get("properties", {}).get("path") == file_path
    ]
    entity_ids = {n["node_id"] for n in entity_nodes}

    parent_of: dict[int, int] = {}
    for n in entity_nodes:
        if n["labels"][0] != "Class":
            continue
        for rtype, target_id in graph.outgoing.get(n["node_id"], []):
            if rtype == "DEFINES_METHOD" and target_id in entity_ids:
                parent_of[target_id] = n["node_id"]

    lines = _read_lines(node["absolute_path"] or node["path"])

    qualified_names = [
        n["properties"]["qualified_name"]
        for n in entity_nodes
        if n["properties"].get("qualified_name")
    ]
    summaries, summaries_available = _fetch_summaries(project, qualified_names)

    def _to_entry(n: dict[str, Any]) -> dict[str, Any]:
        props = n["properties"]
        start_line = props.get("start_line")
        qualified_name = props.get("qualified_name")
        description = summaries.get(qualified_name) if qualified_name else None
        return {
            "kind": n["labels"][0],
            "name": props.get("name"),
            "qualified_name": qualified_name,
            "start_line": start_line,
            "end_line": props.get("end_line"),
            "signature": _signature_line(lines, start_line),
            "summary": _first_sentence(description) if description else None,
            "methods": [],
        }

    entries_by_id = {n["node_id"]: _to_entry(n) for n in entity_nodes}

    top_level: list[dict[str, Any]] = []
    for n in sorted(entity_nodes, key=lambda n: n["properties"].get("start_line") or 0):
        entry = entries_by_id[n["node_id"]]
        parent_id = parent_of.get(n["node_id"])
        if parent_id is not None:
            entries_by_id[parent_id]["methods"].append(entry)
        else:
            top_level.append(entry)

    return {
        "project": project,
        "file": file_path,
        "entities": top_level,
        "summaries_available": summaries_available,
    }
