"""
Navigation service — deterministic repo-structure maps computed straight
from a project's ingested graph JSON (see :mod:`services.graph_store`).

Gives a coding agent the SHAPE of a repo in one cheap call: directories with
aggregate (file, entity) counts, files with per-kind entity counts. No
file-level summaries (that's a later feature) — zero LLM, zero Weaviate.
"""

from __future__ import annotations

from typing import Any

from services.chunker import CODE_ENTITY_LABELS
from services.graph_store import Graph, get_graph

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
