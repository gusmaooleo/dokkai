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
from services.graph_store import (
    Graph,
    NEIGHBORHOOD_EDGE_TYPES,
    _derive_repo_path,
    get_graph,
    normalize_node,
    resolve_entity,
    resolve_file,
)
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


def _is_file_path(root: _DirNode, normalized: str) -> bool:
    """Whether *normalized* (a repo-relative path, no leading/trailing
    slashes) names a FILE in *root*'s tree — used to give a targeted hint
    instead of the generic "not found" error."""
    parts = normalized.split("/")
    node = root
    for part in parts[:-1]:
        if part not in node["dirs"]:
            return False
        node = node["dirs"][part]
    return parts[-1] in node["files"]


def _resolve_tree_path_input(graph: Graph, path: str | None) -> str | None:
    """
    Normalize *path* for :func:`_navigate`. A repo-relative directory path
    (or ``None``) passes through unchanged. An ABSOLUTE path gets the
    project's repo root stripped — derived the same way
    :func:`services.graph_store.resolve_file` resolution does (see
    :func:`services.graph_store._derive_repo_path`) — so
    ``tree(path="<repo_root>/src")`` behaves exactly like
    ``tree(path="src")``. An absolute path that isn't under the repo root
    raises a targeted :class:`ValueError` instead of falling through to the
    generic "not found" hint.
    """
    if not path or not path.startswith("/"):
        return path
    repo_root = _derive_repo_path(graph)
    if repo_root and (path == repo_root or path.startswith(repo_root + "/")):
        return path[len(repo_root) :].strip("/")
    hint = f" (repo root: {repo_root})" if repo_root else ""
    raise ValueError(
        f"path '{path}' is an absolute path but not under the project's repo root{hint}"
    )


def _navigate(root: _DirNode, path: str | None) -> _DirNode:
    """Resolve *path* (repo-relative directory path, already stripped of any
    repo-root prefix by :func:`_resolve_tree_path_input`, or ``None``/empty
    for the repo root) to its tree node. Raises :class:`ValueError` — with a
    "that's a file" hint when *path* names a file (see :func:`_is_file_path`),
    else a few top-level candidates — when *path* doesn't resolve to a
    directory."""
    if not path:
        return root
    normalized = path.strip("/")
    if _is_file_path(root, normalized):
        raise ValueError(
            f"path '{path}' is a file, not a directory — use outline('{path}') to see inside it"
        )
    node = root
    for part in normalized.split("/"):
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

    ``path`` scopes the tree to a subtree root — a repo-relative directory
    path (e.g. ``src/features``) or an absolute one under the project's repo
    root (see :func:`_resolve_tree_path_input`); ``None``/empty selects the
    repo root. ``depth`` is how many directory levels below the root to
    expand — deeper directories still get a line (aggregate counts) but
    aren't expanded further. Callers are expected to have already validated
    ``depth``.

    Raises :class:`services.graph_store.GraphNotFoundError` when no graph
    has been ingested for the project (propagated from
    :func:`services.graph_store.get_graph`), and :class:`ValueError` when
    ``path`` is an absolute path outside the repo root, names a file instead
    of a directory, or otherwise doesn't resolve in the project's tree.
    """
    graph = get_graph(project)
    root = _build_tree(graph)
    _compute_aggregates(root)
    relative_path = _resolve_tree_path_input(graph, path)
    subtree = _navigate(root, relative_path)

    files, entities = subtree["agg"]
    return {
        "project": project,
        "root": relative_path or "",
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


def first_sentence(text: str, max_chars: int = _SUMMARY_MAX_CHARS) -> str:
    """
    First sentence of *text* (up to the first ``.``/``!``/``?`` followed by
    whitespace or end-of-string), capped at *max_chars*. When the sentence
    itself exceeds the cap, cuts at the last whole word inside it — never
    mid-word. Public: shared by ``get_outline`` and the MCP ``search()``/
    ``neighbors()`` tools' one-line summaries (the uniform replacement for
    ``search()``'s old plain ``[:140]`` slice, which could cut mid-word).
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


def fetch_summaries(project: str, qualified_names: list[str]) -> tuple[dict[str, str], bool]:
    """
    Batch-fetch entity descriptions for *qualified_names* from Weaviate in
    ONE round trip (see :meth:`services.retriever.Retriever._fetch_by_qnames`
    — deterministic UUID lookup, the same pattern
    :func:`services.graph_store._fetch_chunk_fields` uses for a single
    entity). Degrades to ``({}, False)`` — logged, never raised — when
    Weaviate is unreachable or the fetch itself fails.

    Public: shared by ``get_outline`` and the MCP ``neighbors()`` tool's
    best-effort summaries. Callers that must work identically whether or not
    Weaviate is up (``neighbors()``'s offline property) rely on this never
    raising — an empty dict silently produces no summary lines, not an
    error.
    """
    if not qualified_names:
        return {}, True
    try:
        client = get_client()
    except Exception as e:
        logger.warning("navigation: Weaviate unreachable, omitting summaries: %s", e)
        return {}, False
    try:
        fetched = Retriever(client)._fetch_by_qnames(qualified_names, project)
    except Exception as e:
        logger.warning("navigation: failed to fetch summaries for '%s': %s", project, e)
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
    (see :func:`first_sentence`), both ``None`` when unavailable —
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
    summaries, summaries_available = fetch_summaries(project, qualified_names)

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
            "summary": first_sentence(description) if description else None,
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


# ---------------------------------------------------------------------------
# path — shortest structural route between two entities
# ---------------------------------------------------------------------------

# The neighborhood edge set plus IMPORTS — module-level import edges become
# followable here so two entities connected only through their modules'
# import wiring (not a direct call) still resolve to a path. CONTAINS_* stays
# excluded (it would connect virtually everything through the folder tree,
# making "shortest path" meaningless).
_PATH_EDGE_TYPES = NEIGHBORHOOD_EDGE_TYPES | {"IMPORTS"}


def _is_external(node: dict[str, Any]) -> bool:
    return bool(node.get("properties", {}).get("is_external"))


def _node_payload(node: dict[str, Any]) -> dict[str, Any]:
    """``normalize_node``'s shape plus ``is_external`` — needed to mark
    external-hub nodes in a ``via_external`` path's render; ``normalize_node``
    itself omits the property."""
    payload = normalize_node(node)
    payload["is_external"] = _is_external(node)
    return payload


def _path_neighbors(graph: Graph, node_id: int, edge_types: set[str]) -> list[tuple[int, str, str]]:
    """
    ``(neighbor_id, edge_type, direction)`` reachable from *node_id* over
    *edge_types*, in BOTH directions (undirected traversal). ``direction`` is
    ``"out"`` when the stored edge is ``node_id -[edge_type]-> neighbor_id``,
    ``"in"`` when it's ``neighbor_id -[edge_type]-> node_id``.
    """
    neighbors = [
        (nid, rtype, "out") for rtype, nid in graph.outgoing.get(node_id, []) if rtype in edge_types
    ]
    neighbors += [
        (nid, rtype, "in") for rtype, nid in graph.incoming.get(node_id, []) if rtype in edge_types
    ]
    return neighbors


def _reconstruct_path(
    parent: dict[int, tuple[int, str, str]], source_id: int, target_id: int
) -> tuple[list[int], list[tuple[int, int, str, str]]]:
    """Walk ``parent`` pointers from *target_id* back to *source_id*, then
    reverse into source->target order. Each hop is ``(from_id, to_id,
    edge_type, direction)`` — ``direction`` as in :func:`_path_neighbors`,
    relative to ``from_id``."""
    hops: list[tuple[int, int, str, str]] = []
    node_id = target_id
    while node_id != source_id:
        parent_id, rtype, direction = parent[node_id]
        hops.append((parent_id, node_id, rtype, direction))
        node_id = parent_id
    hops.reverse()
    node_ids = [source_id] + [to_id for (_from_id, to_id, _rtype, _direction) in hops]
    return node_ids, hops


def _bfs_path(
    graph: Graph,
    source_id: int,
    target_id: int,
    edge_types: set[str],
    max_depth: int,
    *,
    exclude_external: bool = False,
) -> tuple[list[int] | None, list[tuple[int, int, str, str]] | None]:
    """
    Undirected BFS from *source_id* to *target_id* over *edge_types*, up to
    *max_depth* hops. Returns ``(node_ids, hops)`` for the first shortest
    path found (see :func:`_reconstruct_path`), or ``(None, None)`` when no
    path exists within *max_depth*.

    ``exclude_external=True`` (phase 1 of the two-phase search — decision
    PATH-external) refuses to traverse THROUGH any node whose
    ``properties.is_external`` is truthy: such a candidate is skipped
    entirely UNLESS it's the target itself, so an explicitly-resolved
    external endpoint (e.g. someone asking for a path TO ``express``) still
    resolves — only pass-through external hubs (mongoose, express, vitest —
    the corpus's highest-degree nodes) are excluded as intermediates.

    Deterministic across processes: each BFS layer's frontier, and each
    node's candidate neighbors, are sorted by ``node_id`` (an int assigned at
    ingestion — stable within a graph JSON and immune to PYTHONHASHSEED,
    unlike Python's hash-based set/dict iteration order) before being
    walked, so ties among multiple shortest paths always resolve to the same
    one.
    """
    parent: dict[int, tuple[int, str, str]] = {}
    visited = {source_id}
    frontier = [source_id]
    for _ in range(max_depth):
        next_frontier: list[int] = []
        for node_id in frontier:
            candidates = sorted(_path_neighbors(graph, node_id, edge_types))
            for neighbor_id, rtype, direction in candidates:
                if neighbor_id in visited:
                    continue
                if (
                    exclude_external
                    and neighbor_id != target_id
                    and _is_external(graph.node_by_id[neighbor_id])
                ):
                    continue
                visited.add(neighbor_id)
                parent[neighbor_id] = (node_id, rtype, direction)
                if neighbor_id == target_id:
                    return _reconstruct_path(parent, source_id, target_id)
                next_frontier.append(neighbor_id)
        frontier = sorted(next_frontier)
        if not frontier:
            break
    return None, None


def get_path(project: str, source: str, target: str, max_depth: int = 10) -> dict[str, Any]:
    """
    Assemble the shortest structural-path payload between *source* and
    *target* entities in *project*'s graph.

    Endpoints resolve via :func:`services.graph_store.resolve_entity` — the
    same resolution ``neighbors()``/``get_entity()`` use, so Module
    qualified_names work too (IMPORTS edges connect Modules, not just code
    entities).

    TWO-PHASE search (decision PATH-external — third-party hubs like
    mongoose/express/vitest otherwise tunnel 70% of shortest paths into a
    false "these are close" reading): phase 1 is undirected BFS over
    :data:`_PATH_EDGE_TYPES` excluding external nodes as intermediates (see
    :func:`_bfs_path`'s ``exclude_external``) — this is the real
    app-level structural coupling. Only when phase 1 finds nothing within
    *max_depth* does phase 2 re-run the same BFS unrestricted; a phase-2 hit
    sets ``via_external=True`` in the payload (rendered with a weak-coupling
    note). Both phases share the identical deterministic tie-break.

    ``source == target`` returns a trivial one-node path (``found=True``,
    zero hops) — not an error. No path within *max_depth* hops (in EITHER
    phase) returns ``found=False`` — also not an error, a normal outcome for
    two genuinely unrelated entities.

    The returned dict ALWAYS carries ``found``, ``via_external``,
    ``max_depth`` and ``edge_types`` regardless of outcome (trivial / found /
    not-found), so callers never need to branch on key presence.

    Raises :class:`ValueError` (mirroring ``get_entity``'s message) when
    *source* or *target* doesn't resolve to an entity in the project's
    graph. Callers are expected to have already validated *max_depth*.
    """
    graph = get_graph(project)
    source_node = resolve_entity(graph, source)
    if source_node is None:
        raise ValueError(f"entity '{source}' not found in project '{project}' — try search first")
    target_node = resolve_entity(graph, target)
    if target_node is None:
        raise ValueError(f"entity '{target}' not found in project '{project}' — try search first")

    source_id = source_node["node_id"]
    target_id = target_node["node_id"]
    edge_types = sorted(_PATH_EDGE_TYPES)

    if source_id == target_id:
        return {
            "project": project,
            "source": source,
            "target": target,
            "found": True,
            "via_external": False,
            "max_depth": max_depth,
            "edge_types": edge_types,
            "nodes": [_node_payload(source_node)],
            "hops": [],
        }

    node_ids, hops = _bfs_path(
        graph, source_id, target_id, _PATH_EDGE_TYPES, max_depth, exclude_external=True
    )
    via_external = False
    if node_ids is None:
        node_ids, hops = _bfs_path(
            graph, source_id, target_id, _PATH_EDGE_TYPES, max_depth, exclude_external=False
        )
        via_external = node_ids is not None

    if node_ids is None:
        return {
            "project": project,
            "source": source,
            "target": target,
            "found": False,
            "via_external": False,
            "max_depth": max_depth,
            "edge_types": edge_types,
            "nodes": [],
            "hops": [],
        }

    return {
        "project": project,
        "source": source,
        "target": target,
        "found": True,
        "via_external": via_external,
        "max_depth": max_depth,
        "edge_types": edge_types,
        "nodes": [_node_payload(graph.node_by_id[nid]) for nid in node_ids],
        "hops": [
            {"type": rtype, "direction": direction} for (_from_id, _to_id, rtype, direction) in hops
        ],
    }
