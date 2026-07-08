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
from typing import Any, Literal

from services.chunker import CODE_ENTITY_LABELS
from services.vectorize import find_latest_graph_json


class GraphNotFoundError(Exception):
    """Raised when no ingested graph JSON exists for a project."""


class EntityNotFoundError(Exception):
    """Raised when a qualified name cannot be resolved in a project's graph."""

    def __init__(self, project_name: str, qualified_name: str) -> None:
        super().__init__(
            f"entity '{qualified_name}' not found in project '{project_name}' graph"
        )


def _not_found_message(project_name: str) -> str:
    return (
        f"no graph found for project '{project_name}' — run POST "
        "/instances/pipeline to ingest it"
    )


# The neighborhood endpoint's edge-type superset (decision N3): call/structural
# relations only — never IMPORTS, CONTAINS_*, or DEPENDS_ON_EXTERNAL.
NEIGHBORHOOD_EDGE_TYPES = {
    "CALLS",
    "INHERITS",
    "IMPLEMENTS",
    "OVERRIDES",
    "DEFINES",
    "DEFINES_METHOD",
}


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


def get_full_graph(project_name: str, *, structural: bool, limit: int) -> dict[str, Any]:
    """
    Assemble the full-graph response payload for *project_name*.

    ``structural=False`` selects only code-entity nodes (labels in
    :data:`services.chunker.CODE_ENTITY_LABELS`); ``structural=True`` selects
    every node. ``limit`` caps the *selected* node list (in array order)
    before normalization; edges are always filtered to the actually-returned
    node ids, so the result never has a dangling endpoint. ``total_nodes``/
    ``total_edges`` in ``stats`` are computed pre-limit, over the selected
    set, so callers can tell whether the response was truncated.

    Raises :class:`GraphNotFoundError` when no graph has been ingested for
    the project yet (propagated from :func:`get_graph`).
    """
    graph = get_graph(project_name)

    if structural:
        selected_nodes = graph.nodes
    else:
        selected_nodes = [n for n in graph.nodes if n["labels"][0] in CODE_ENTITY_LABELS]

    selected_ids = {n["node_id"] for n in selected_nodes}
    total_nodes = len(selected_nodes)
    total_edges = sum(
        1
        for rel in graph.relationships
        if rel["from_id"] in selected_ids and rel["to_id"] in selected_ids
    )

    returned_nodes = selected_nodes[:limit]
    returned_ids = {n["node_id"] for n in returned_nodes}
    returned_edges = [
        normalize_edge(rel)
        for rel in graph.relationships
        if rel["from_id"] in returned_ids and rel["to_id"] in returned_ids
    ]

    return {
        "project": project_name,
        "generated_at": graph.metadata.get("exported_at"),
        "nodes": [normalize_node(n) for n in returned_nodes],
        "edges": returned_edges,
        "stats": {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "returned_nodes": len(returned_nodes),
            "returned_edges": len(returned_edges),
            "truncated": len(returned_nodes) < total_nodes,
        },
    }


def resolve_entity(graph: Graph, qualified_name: str) -> dict[str, Any] | None:
    """
    Resolve *qualified_name* to a node, preferring code-entity nodes.

    ``graph.by_qualified_name`` is first-wins across *all* node kinds, so a
    structural node (Package/Module) that shares a qualified name with a code
    entity could shadow it — a defensive concern for future corpora; in the
    current test corpus first-wins already favors the code entity. This
    scans ``graph.nodes`` in array order and returns the first
    node whose kind is a code-entity label (see
    :data:`services.chunker.CODE_ENTITY_LABELS`); if none matches, it falls
    back to the raw first-wins ``by_qualified_name`` lookup.
    """
    for node in graph.nodes:
        if (
            node.get("properties", {}).get("qualified_name") == qualified_name
            and node["labels"][0] in CODE_ENTITY_LABELS
        ):
            return node
    return graph.by_qualified_name.get(qualified_name)


def get_neighborhood(
    project_name: str,
    *,
    entity: str,
    depth: int,
    direction: Literal["in", "out", "both"],
    limit: int,
) -> dict[str, Any]:
    """
    Assemble the neighborhood-query response payload for *entity* in
    *project_name*'s graph.

    BFS from the resolved entity node over :data:`NEIGHBORHOOD_EDGE_TYPES`
    only (CALLS, INHERITS, IMPLEMENTS, OVERRIDES, DEFINES, DEFINES_METHOD —
    never IMPORTS/CONTAINS_*/DEPENDS_ON_EXTERNAL). ``direction`` picks the
    adjacency followed while expanding: ``out`` outgoing edges only, ``in``
    incoming only, ``both`` both.

    Nodes are collected breadth-first, closest hop first, up to ``depth``
    hops. The center is hop 0 and is included in both ``center`` and
    ``nodes`` (UI convenience — ``stats.returned_nodes`` counts it too).
    ``limit`` caps the *total* number of returned nodes, root included;
    because collection order is BFS, truncation always drops the farthest
    nodes first. ``truncated`` is true only when the node cap actually cut
    off nodes BFS would otherwise have reached within ``depth`` hops —
    simply exhausting ``depth`` without hitting the cap does not set it.

    ``edges`` lists every :data:`NEIGHBORHOOD_EDGE_TYPES` relationship whose
    both endpoints are in the returned node set, regardless of
    ``direction`` (mirrors the full-graph endpoint's type+membership-only
    edge filter — a node can be reached via one direction and still have
    its edges in the other direction shown once it's in the result set).

    Raises :class:`GraphNotFoundError` when no graph has been ingested for
    the project, and :class:`EntityNotFoundError` when *entity* cannot be
    resolved in the project's graph.
    """
    graph = get_graph(project_name)
    center = resolve_entity(graph, entity)
    if center is None:
        raise EntityNotFoundError(project_name, entity)

    center_id = center["node_id"]
    hop_of: dict[int, int] = {center_id: 0}
    order: list[int] = [center_id]
    frontier = [center_id]
    for hop in range(1, depth + 1):
        next_frontier: list[int] = []
        for node_id in frontier:
            neighbor_ids: list[int] = []
            if direction in ("out", "both"):
                neighbor_ids += [
                    nid
                    for rtype, nid in graph.outgoing.get(node_id, [])
                    if rtype in NEIGHBORHOOD_EDGE_TYPES
                ]
            if direction in ("in", "both"):
                neighbor_ids += [
                    nid
                    for rtype, nid in graph.incoming.get(node_id, [])
                    if rtype in NEIGHBORHOOD_EDGE_TYPES
                ]
            for nid in neighbor_ids:
                if nid not in hop_of:
                    hop_of[nid] = hop
                    order.append(nid)
                    next_frontier.append(nid)
        frontier = next_frontier
        if not frontier:
            break

    truncated = len(order) > limit
    returned_ids = order[:limit]
    returned_id_set = set(returned_ids)

    returned_edges = [
        normalize_edge(rel)
        for rel in graph.relationships
        if rel["type"] in NEIGHBORHOOD_EDGE_TYPES
        and rel["from_id"] in returned_id_set
        and rel["to_id"] in returned_id_set
    ]

    nodes_payload = []
    for nid in returned_ids:
        node = normalize_node(graph.node_by_id[nid])
        node["hop"] = hop_of[nid]
        nodes_payload.append(node)

    return {
        "project": project_name,
        "entity": entity,
        "center": normalize_node(center),
        "nodes": nodes_payload,
        "edges": returned_edges,
        "stats": {
            "depth": depth,
            "direction": direction,
            "limit": limit,
            "returned_nodes": len(nodes_payload),
            "returned_edges": len(returned_edges),
            "truncated": truncated,
        },
    }
