"""
Graph API endpoints.

GET /graph/{project}   — full project graph (code entities by default, or the
                          full structural graph with ?include=structural)
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from models.dtos.graph import GraphResponse
from services.graph_store import GraphNotFoundError, get_full_graph

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
