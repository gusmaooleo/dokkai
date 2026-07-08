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
from weaviate.classes.query import MetadataQuery, HybridFusion, Filter

from services.weaviate_client import uuid_for_entity, VECTOR_CODE, VECTOR_SUMMARY

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "CodeEntity")

# --- graph expansion config ------------------------------------------------
# Chunk fields that hold qualified_names and can be followed as graph edges.
# The weight reflects how much signal each edge type carries about related
# context. ``defined_methods`` is intentionally excluded: it stores plain
# names (not qualified_names), so it cannot be joined by id.
_FORWARD_EDGES = {"calls": 0.60, "inherits": 0.65, "implements": 0.60, "overrides": 0.50}
_BACKWARD_EDGES = {"called_by": 0.45}

_HOP_DECAY = 0.5      # each extra hop multiplies the inherited score by this
_MAX_FANOUT = 8       # max neighbors followed per edge field, per node
_HUB_THRESHOLD = 25   # skip an edge whose neighbor list exceeds this (hub node)

# --- test de-prioritization ------------------------------------------------
# For "how does X work?" queries, test files tend to out-rank the real
# implementation: their descriptions read like specs, so they match the query
# semantically. We softly down-weight test chunks (multiply their score) so the
# implementation surfaces first — without dropping tests entirely, since a test
# can still be the most relevant context. Set RETRIEVAL_TEST_PENALTY=1.0 to
# disable.
_TEST_PENALTY = float(os.getenv("RETRIEVAL_TEST_PENALTY", "0.35"))
_SEARCH_OVERFETCH = 3  # fetch Nx candidates before penalizing, so demoted
                       # tests don't crowd implementation out of the window
_TEST_DIR_SEGMENTS = {"test", "tests", "__tests__", "spec", "specs", "e2e"}
_TEST_FILE_MARKERS = (".test.", ".spec.", ".e2e.")


def _is_test_path(file_path: str) -> bool:
    """
    Heuristically decide whether a file path belongs to a test.

    Uses path *segments* for directories and filename *boundaries* for the
    basename, so a path like ``latest_version.py`` is not misclassified as a
    test just because it contains the substring "test".
    """
    if not file_path:
        return False
    p = file_path.replace("\\", "/").lower()
    segments = p.split("/")
    if any(seg in _TEST_DIR_SEGMENTS for seg in segments):
        return True
    base = segments[-1]
    if any(marker in base for marker in _TEST_FILE_MARKERS):
        return True
    stem = base.rsplit(".", 1)[0]
    return stem.startswith("test_") or stem.endswith(("_test", "_spec"))


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
    description: str = ""  # Tier 2: LLM micro-description (summary-vector source)
    score: float | None = None
    hop: int = 0          # 0 = seed (matched query), 1 = direct neighbor, ...
    via: str = ""         # provenance, e.g. "calls ← registerRoutes"


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
        target_vector: str = VECTOR_SUMMARY,
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
        target_vector : str
            Which named vector to search — ``summary`` (natural-language intent,
            default) or ``code`` (literal source). The collection has both.

        Returns
        -------
        list[RetrievedChunk]
        """
        filters = self._build_filters(project_name, entity_type)

        # Over-fetch so down-weighted test chunks don't crowd real
        # implementation out of the candidate window before we re-rank.
        fetch_limit = top_k * _SEARCH_OVERFETCH if _TEST_PENALTY < 1.0 else top_k
        response = self.collection.query.hybrid(
            query=query,
            limit=fetch_limit,
            alpha=alpha,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            filters=filters,
            target_vector=target_vector,
            return_metadata=MetadataQuery(score=True),
        )

        chunks = [self._to_chunk(obj) for obj in response.objects]
        if _TEST_PENALTY < 1.0:
            for c in chunks:
                if c.score is not None and _is_test_path(c.file_path):
                    c.score *= _TEST_PENALTY
            chunks.sort(key=lambda c: -(c.score or 0.0))
        return chunks[:top_k]

    def search_graph(
        self,
        query: str,
        *,
        project_name: str,
        top_k_seeds: int = 6,
        max_hops: int = 2,
        max_nodes: int = 30,
        direction: str = "both",
        alpha: float = 0.7,
    ) -> list[RetrievedChunk]:
        """
        Hybrid-search for seed chunks, then expand along the dependency graph.

        Seeds are ranked by hybrid score; neighbors inherit a fraction of their
        parent's score (decayed per hop, weighted by edge type), so a helper
        called directly by the top match outranks a weak, loosely-matched seed.

        Parameters
        ----------
        direction:
            ``"forward"`` follows dependencies (calls/inherits/...), good for
            "how does X work"; ``"backward"`` follows callers, good for impact
            / "what uses X"; ``"both"`` (default) combines them.
        """
        edges: dict[str, float] = {}
        if direction in ("forward", "both"):
            edges |= _FORWARD_EDGES
        if direction in ("backward", "both"):
            edges |= _BACKWARD_EDGES

        # 1) seeds via hybrid search on BOTH named vectors, merged: the summary
        #    vector catches conceptual matches, the code vector catches literal
        #    ones (and undescribed entities that have no summary vector).
        summary_seeds = self.search(
            query, project_name=project_name, top_k=top_k_seeds, alpha=alpha,
            target_vector=VECTOR_SUMMARY,
        )
        code_seeds = self.search(
            query, project_name=project_name, top_k=top_k_seeds, alpha=alpha,
            target_vector=VECTOR_CODE,
        )
        seeds = self._merge_seeds(summary_seeds, code_seeds, top_k_seeds)
        selected: dict[str, RetrievedChunk] = {}
        for seed in seeds:
            seed.hop, seed.via = 0, "seed (matched query)"
            selected[seed.qualified_name] = seed

        # 2) breadth-first expansion over the graph edges
        frontier = list(seeds)
        for hop in range(1, max_hops + 1):
            # qname -> (parent, inherited_score, edge_field)
            wanted: dict[str, tuple[RetrievedChunk, float, str]] = {}
            for parent in frontier:
                for field_name, weight in edges.items():
                    neighbors = getattr(parent, field_name, []) or []
                    if len(neighbors) > _HUB_THRESHOLD:
                        continue  # hub node — expanding it would flood the context
                    for qn in neighbors[:_MAX_FANOUT]:
                        if qn in selected or qn.startswith("<unknown"):
                            continue
                        score = (parent.score or 0.0) * weight * (_HOP_DECAY ** (hop - 1))
                        best = wanted.get(qn)
                        if best is None or score > best[1]:
                            wanted[qn] = (parent, score, field_name)
            if not wanted:
                break

            fetched = self._fetch_by_qnames(list(wanted), project_name)
            frontier = []
            for qn, chunk in fetched.items():
                parent, score, field_name = wanted[qn]
                chunk.hop = hop
                chunk.via = f"{field_name} ← {parent.name}"
                # De-prioritize tests reached via expansion too, not just seeds,
                # so they don't re-enter the context through the graph.
                if _TEST_PENALTY < 1.0 and _is_test_path(chunk.file_path):
                    score *= _TEST_PENALTY
                chunk.score = score
                selected[qn] = chunk
                frontier.append(chunk)
            if len(selected) >= max_nodes:
                break

        # 3) rank by score: seeds float to the top, then strongest neighbors
        ranked = sorted(selected.values(), key=lambda c: -(c.score or 0.0))
        return ranked[:max_nodes]

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
            location = f"File: {chunk.absolute_path or chunk.file_path}"
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

    def build_graph_context(
        self,
        chunks: list[RetrievedChunk],
        *,
        max_chars: int = 16_000,
    ) -> str:
        """
        Format graph-expanded chunks for an LLM prompt, in ranked order, each
        prefixed with its provenance (seed vs. how the neighbor was reached) so
        the model understands the structure of the connected context.
        """
        if not chunks:
            return "(No relevant code found in the codebase.)"

        sections: list[str] = []
        total = 0
        for i, chunk in enumerate(chunks, 1):
            tag = "SEED · matched query" if chunk.hop == 0 else f"hop {chunk.hop} · {chunk.via}"
            location = f"File: {chunk.absolute_path or chunk.file_path}"
            if chunk.start_line is not None and chunk.end_line is not None:
                location += f" (lines {chunk.start_line}-{chunk.end_line})"
            summary = f"Summary: {chunk.description}\n" if chunk.description else ""
            section = f"--- Source {i} [{tag}] ---\n{location}\n{summary}{chunk.chunk_text}"
            if sections and total + len(section) > max_chars:
                break
            sections.append(section)
            total += len(section)

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_chunk(obj: Any) -> RetrievedChunk:
        """Map a raw Weaviate object to a RetrievedChunk."""
        props = obj.properties
        return RetrievedChunk(
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
            description=str(props.get("description", "")),
            score=obj.metadata.score if obj.metadata else None,
        )

    @staticmethod
    def _merge_seeds(
        summary_seeds: list[RetrievedChunk],
        code_seeds: list[RetrievedChunk],
        limit: int,
    ) -> list[RetrievedChunk]:
        """Union two seed lists by qualified_name, keeping the higher score."""
        best: dict[str, RetrievedChunk] = {}
        for chunk in (*summary_seeds, *code_seeds):
            existing = best.get(chunk.qualified_name)
            if existing is None or (chunk.score or 0.0) > (existing.score or 0.0):
                best[chunk.qualified_name] = chunk
        return sorted(best.values(), key=lambda c: -(c.score or 0.0))[:limit]

    def _fetch_by_qnames(
        self, qnames: list[str], project_name: str
    ) -> dict[str, RetrievedChunk]:
        """Batch-fetch chunks by deterministic UUID — exact, single round-trip."""
        if not qnames:
            return {}
        ids = [uuid_for_entity(project_name, q) for q in qnames]
        resp = self.collection.query.fetch_objects(
            filters=Filter.by_id().contains_any(ids),
            limit=len(ids),
        )
        return {c.qualified_name: c for c in (self._to_chunk(o) for o in resp.objects)}

    @staticmethod
    def _build_filters(
        project_name: str | None,
        entity_type: str | None,
    ) -> Any | None:
        """Build a Weaviate filter combining project_name and entity_type."""
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
