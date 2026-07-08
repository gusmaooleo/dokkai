"""
Graph store — loads and caches the ingested code-graph JSON per project.

Serves straight from the raw cgr JSON files in ``ingested/`` (see
:func:`services.vectorize.find_latest_graph_json`) — no new database. The
graph is loaded lazily on first access and kept in memory, indexed once for
fast lookups by later steps (full-graph, neighborhood, file-level views).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.vectorize import find_latest_graph_json


class GraphNotFoundError(Exception):
    """Raised when no ingested graph JSON exists for a project."""


def _not_found_message(project_name: str) -> str:
    return (
        f"no graph found for project '{project_name}' — run POST "
        "/instances/pipeline to ingest it"
    )


@dataclass
class Graph:
    """
    A loaded project graph: raw cgr nodes/relationships/metadata plus
    indexes built once at load time.

    ``by_qualified_name`` resolves first-wins in ``nodes`` array order when
    a qualified name is duplicated (the corpus has at least one such case).
    ``outgoing``/``incoming`` map a node id to its ``(edge_type, other_id)``
    adjacency in each direction.
    """

    project_name: str
    path: Path
    nodes: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    metadata: dict[str, Any]
    node_by_id: dict[int, dict[str, Any]] = field(default_factory=dict, init=False)
    by_qualified_name: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    outgoing: dict[int, list[tuple[str, int]]] = field(default_factory=dict, init=False)
    incoming: dict[int, list[tuple[str, int]]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for node in self.nodes:
            self.node_by_id[node["node_id"]] = node
            qualified_name = node.get("properties", {}).get("qualified_name")
            if qualified_name and qualified_name not in self.by_qualified_name:
                self.by_qualified_name[qualified_name] = node
        for rel in self.relationships:
            from_id, to_id, rtype = rel["from_id"], rel["to_id"], rel["type"]
            self.outgoing.setdefault(from_id, []).append((rtype, to_id))
            self.incoming.setdefault(to_id, []).append((rtype, from_id))


def normalize_node(node: dict[str, Any]) -> dict[str, Any]:
    """Raw cgr node -> the normalized dokkai shape (nodes are single-label)."""
    props = node.get("properties", {})
    return {
        "id": node["node_id"],
        "kind": node["labels"][0],
        "name": props.get("name"),
        "qualified_name": props.get("qualified_name"),
        "path": props.get("path"),
        "absolute_path": props.get("absolute_path"),
        "start_line": props.get("start_line"),
        "end_line": props.get("end_line"),
    }


def normalize_edge(rel: dict[str, Any]) -> dict[str, Any]:
    """Raw cgr relationship -> the normalized dokkai shape."""
    return {"source": rel["from_id"], "target": rel["to_id"], "type": rel["type"]}


# project_name -> ((resolved_path, mtime, size), Graph) — cache validity is
# the (path, mtime, size) tuple; a changed file (e.g. os.replace during
# promotion) simply misses and triggers a reload.
_CacheEntry = tuple[tuple[Path, float, int], Graph]
_cache: dict[str, _CacheEntry] = {}


def _load(project_name: str, path: Path) -> Graph:
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return Graph(
        project_name=project_name,
        path=path,
        nodes=data.get("nodes", []),
        relationships=data.get("relationships", []),
        metadata=data.get("metadata", {}),
    )


def get_graph(project_name: str) -> Graph:
    """
    Return the cached, indexed graph for *project_name*, loading it first
    (or reloading it, if the underlying file changed since the last access).

    Raises :class:`GraphNotFoundError` when no graph has been ingested for
    the project yet.
    """
    return _get_graph(project_name, retried=False)


def _get_graph(project_name: str, *, retried: bool) -> Graph:
    try:
        # find_latest_graph_json itself stats every candidate (to pick the
        # newest), so a candidate vanishing between the glob and that stat —
        # exactly what ingest's promote+cleanup does — raises FileNotFoundError
        # right here, not only from our own path.stat()/open() below. Both
        # must share the same bounded retry.
        path = find_latest_graph_json(project_name)
        if path is None:
            _cache.pop(project_name, None)
            raise GraphNotFoundError(_not_found_message(project_name))

        stat = path.stat()
        cache_key = (path, stat.st_mtime, stat.st_size)
        cached = _cache.get(project_name)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        graph = _load(project_name, path)
    except FileNotFoundError:
        # Promote/delete race: a candidate (or the resolved path) vanished
        # between resolve and stat/open (a job's file lock doesn't cover
        # concurrent HTTP reads). Drop any stale cache entry and re-resolve
        # exactly once — bounded, so a persistently missing graph can't
        # loop — before giving up.
        _cache.pop(project_name, None)
        if retried:
            raise GraphNotFoundError(_not_found_message(project_name))
        return _get_graph(project_name, retried=True)

    _cache[project_name] = (cache_key, graph)
    return graph
