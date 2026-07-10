"""
DTOs for the graph endpoints.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ProjectGraphDTO(BaseModel):
    project: str
    file: str
    nodes: int
    edges: int
    generated_at: str
    chunks: int | None = None
    repo_path: str | None = None


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


class NeighborhoodNodeDTO(GraphNodeDTO):
    hop: int


class EntityDetailResponse(GraphNodeDTO):
    description: str | None = None
    chunk_text: str | None = None


class NeighborhoodStatsDTO(BaseModel):
    depth: int
    direction: Literal["in", "out", "both"]
    limit: int
    returned_nodes: int
    returned_edges: int
    truncated: bool


class NeighborhoodResponse(BaseModel):
    project: str
    entity: str
    center: GraphNodeDTO
    nodes: list[NeighborhoodNodeDTO]
    edges: list[GraphEdgeDTO]
    stats: NeighborhoodStatsDTO


class FileDTO(BaseModel):
    path: str
    name: str
    absolute_path: str | None = None


class FileEdgeDTO(BaseModel):
    source: str
    target: str
    weight: int
    types: dict[str, int]


class FileViewStatsDTO(BaseModel):
    files: int
    edges: int


class FileViewResponse(BaseModel):
    project: str
    files: list[FileDTO]
    edges: list[FileEdgeDTO]
    stats: FileViewStatsDTO
