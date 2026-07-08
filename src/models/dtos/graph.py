"""
DTOs for the graph endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel


class GraphNodeDTO(BaseModel):
    id: int
    kind: str
    name: str | None = None
    qualified_name: str | None = None
    path: str | None = None
    absolute_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None


class GraphEdgeDTO(BaseModel):
    source: int
    target: int
    type: str


class GraphStatsDTO(BaseModel):
    total_nodes: int
    total_edges: int
    returned_nodes: int
    returned_edges: int
    truncated: bool


class GraphResponse(BaseModel):
    project: str
    generated_at: str | None = None
    nodes: list[GraphNodeDTO]
    edges: list[GraphEdgeDTO]
    stats: GraphStatsDTO
