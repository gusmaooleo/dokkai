"""
Retriever — hybrid search over Weaviate CodeEntity collection.

Performs combined vector (semantic) + BM25 (keyword) search on the
``chunk_text`` field and returns ranked results with full metadata.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, cast

import weaviate
from weaviate.classes.query import MetadataQuery, HybridFusion

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "CodeEntity")


@dataclass
class RetrievedChunk:
    """Single search result returned by the retriever."""

    entity_type: str = ""
    name: str = ""
    qualified_name: str = ""
    file_path: str = ""
    absolute_path: str = ""
    start_line: int | None = None
    end_line: int | None = None
    project_name: str = ""
    module_name: str = ""
    parent_class: str = ""
    chunk_text: str = ""
    calls: list[str] = field(default_factory=list)
    called_by: list[str] = field(default_factory=list)
    inherits: list[str] = field(default_factory=list)
    implements: list[str] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)
    defined_methods: list[str] = field(default_factory=list)
    score: float | None = None


class Retriever:
    """
    Hybrid search retriever for the vectorized codebase.

    Parameters
    ----------
    client : weaviate.WeaviateClient
        An already-connected Weaviate client.
    """

    def __init__(self, client: weaviate.WeaviateClient) -> None:
        self.client = client
        self.collection = client.collections.get(COLLECTION_NAME)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        project_name: str | None = None,
        entity_type: str | None = None,
        top_k: int = 10,
        alpha: float = 0.7,
    ) -> list[RetrievedChunk]:
        """
        Run a hybrid search (vector + keyword) against the collection.

        Parameters
        ----------
        query : str
            Natural language search query.
        project_name : str, optional
            Filter results to a specific project.
        entity_type : str, optional
            Filter results to a specific entity type (Class, Function, etc.).
        top_k : int
            Maximum number of results to return.
        alpha : float
            Balance between vector (1.0) and keyword (0.0) search. Default
            0.7 leans towards semantic similarity.

        Returns
        -------
        list[RetrievedChunk]
        """
        filters = self._build_filters(project_name, entity_type)

        response = self.collection.query.hybrid(
            query=query,
            limit=top_k,
            alpha=alpha,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            filters=filters,
            return_metadata=MetadataQuery(score=True),
        )

        chunks: list[RetrievedChunk] = []
        for obj in response.objects:
            props = obj.properties
            chunk = RetrievedChunk(
                entity_type=str(props.get("entity_type", "")),
                name=str(props.get("name", "")),
                qualified_name=str(props.get("qualified_name", "")),
                file_path=str(props.get("file_path", "")),
                absolute_path=str(props.get("absolute_path", "")),
                start_line=int(cast(int, props["start_line"])) if props.get("start_line") is not None else None,
                end_line=int(cast(int, props["end_line"])) if props.get("end_line") is not None else None,
                project_name=str(props.get("project_name", "")),
                module_name=str(props.get("module_name", "")),
                parent_class=str(props.get("parent_class", "")),
                chunk_text=str(props.get("chunk_text", "")),
                calls=list(props.get("calls", [])),  # type: ignore[arg-type]
                called_by=list(props.get("called_by", [])),  # type: ignore[arg-type]
                inherits=list(props.get("inherits", [])),  # type: ignore[arg-type]
                implements=list(props.get("implements", [])),  # type: ignore[arg-type]
                overrides=list(props.get("overrides", [])),  # type: ignore[arg-type]
                defined_methods=list(props.get("defined_methods", [])),  # type: ignore[arg-type]
                score=obj.metadata.score if obj.metadata else None,
            )
            chunks.append(chunk)

        return chunks

    def build_context(
        self,
        chunks: list[RetrievedChunk],
        *,
        max_chars: int = 12_000,
    ) -> str:
        """
        Format retrieved chunks into a single context string suitable for
        injection into an LLM prompt.

        Parameters
        ----------
        chunks : list[RetrievedChunk]
            Ranked search results from ``search()``.
        max_chars : int
            Truncate the context to at most this many characters.

        Returns
        -------
        str
            A formatted context block.
        """
        if not chunks:
            return "(No relevant code found in the codebase.)"

        sections: list[str] = []
        total = 0

        for i, chunk in enumerate(chunks, 1):
            header = f"--- Source {i}: [{chunk.entity_type}] {chunk.qualified_name} ---"
            location = f"File: {chunk.file_path}"
            if chunk.start_line is not None and chunk.end_line is not None:
                location += f" (lines {chunk.start_line}-{chunk.end_line})"

            relations: list[str] = []
            if chunk.calls:
                relations.append(f"Calls: {', '.join(chunk.calls)}")
            if chunk.called_by:
                relations.append(f"Called by: {', '.join(chunk.called_by)}")
            if chunk.inherits:
                relations.append(f"Inherits: {', '.join(chunk.inherits)}")
            if chunk.implements:
                relations.append(f"Implements: {', '.join(chunk.implements)}")
            if chunk.defined_methods:
                relations.append(f"Methods: {', '.join(chunk.defined_methods)}")

            section_lines = [header, location]
            if relations:
                section_lines.append(" | ".join(relations))
            section_lines.append(chunk.chunk_text)

            section = "\n".join(section_lines)

            if total + len(section) > max_chars:
                remaining = max_chars - total
                if remaining > 200:
                    sections.append(section[:remaining] + "\n[truncated]")
                break

            sections.append(section)
            total += len(section) + 2  # account for double newline separator

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filters(
        project_name: str | None,
        entity_type: str | None,
    ) -> Any | None:
        """Build a Weaviate filter combining project_name and entity_type."""
        from weaviate.classes.query import Filter

        conditions: list[Any] = []

        if project_name:
            conditions.append(
                Filter.by_property("project_name").equal(project_name)
            )
        if entity_type:
            conditions.append(
                Filter.by_property("entity_type").equal(entity_type)
            )

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]

        return Filter.all_of(conditions)
