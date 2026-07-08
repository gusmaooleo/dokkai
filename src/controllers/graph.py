"""
Graph API endpoints.

GET /graph/{project}                — full project graph (code entities by
                                       default, or the full structural graph
                                       with ?include=structural)
GET /graph/{project}/neighborhood   — BFS neighborhood of a single entity
GET /graph/{project}/files          — file-level dependency view
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from models.dtos.graph import FileViewResponse, GraphResponse, NeighborhoodResponse
from services.graph_store import (
    EntityNotFoundError,
    GraphNotFoundError,
    get_file_view,
    get_full_graph,
    get_neighborhood,
)

router = APIRouter(prefix="/graph")


@router.get("/{project}", response_model=GraphResponse)
async def getGraph(
    project: str,
    include: Literal["structural"] | None = Query(
        default=None,
        description="Set to 'structural' to include all nodes (code entities, "
        "folders, files, modules, packages, the project, and external "
        "packages) instead of only code entities.",
    ),
    limit: int = Query(default=5000, ge=1, description="Maximum number of nodes to return."),
):
    """
    Return the ingested graph for *project*.

    By default only code-entity nodes (Class, Function, Method, Interface,
    Enum, Type) are returned. With ``include=structural``, every node is
    returned instead. Edges are always restricted to pairs where both
    endpoints are present in the returned node set — no dangling endpoints.
    """
    try:
        payload = get_full_graph(project, structural=include == "structural", limit=limit)
    except GraphNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return payload


@router.get("/{project}/neighborhood", response_model=NeighborhoodResponse)
async def getNeighborhood(
    project: str,
    entity: str = Query(
        ..., description="Qualified name of the entity to center the neighborhood on."
    ),
    depth: int = Query(default=1, ge=1, description="Number of hops to expand from the entity."),
    direction: Literal["in", "out", "both"] = Query(
        default="both", description="Which edge direction to follow while expanding."
    ),
    limit: int = Query(
        default=200,
        ge=1,
        description="Maximum number of nodes to return, root included, closest-first.",
    ),
):
    """
    Return *entity*'s neighborhood in *project*'s graph.

    BFS from the entity's node over CALLS, INHERITS, IMPLEMENTS, OVERRIDES,
    DEFINES and DEFINES_METHOD edges only — never IMPORTS, CONTAINS_*, or
    DEPENDS_ON_EXTERNAL. ``direction`` picks which adjacency to follow while
    expanding; ``depth`` bounds the number of hops; ``limit`` caps the total
    returned node count (root included, closest-first). See
    :func:`services.graph_store.get_neighborhood` for the full contract.
    """
    try:
        payload = get_neighborhood(
            project, entity=entity, depth=depth, direction=direction, limit=limit
        )
    except GraphNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return payload


@router.get("/{project}/files", response_model=FileViewResponse)
async def getFileView(project: str):
    """
    Return *project*'s file-level dependency view.

    Aggregates every relationship in the graph into file-to-file edges: each
    edge endpoint is mapped to the file it belongs to (code entities,
    modules, and files map via their path; folders, packages, the project,
    and external packages are not files and drop the relationship). Edges
    between the same pair of files are aggregated into one edge with a total
    ``weight`` and a per relationship-type ``types`` breakdown. Self-loops
    and external dependencies are excluded, so only internal file-to-file
    dependencies remain. See :func:`services.graph_store.get_file_view` for
    the full contract.
    """
    try:
        payload = get_file_view(project)
    except GraphNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return payload
