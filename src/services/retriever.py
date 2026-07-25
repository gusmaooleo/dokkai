"""
Retriever — hybrid search over Weaviate CodeEntity collection.

Performs combined vector (semantic) + BM25 (keyword) search on the
``chunk_text`` field and returns ranked results with full metadata.
"""

from __future__ import annotations

import os
import re
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

# --- identifier fast path ---------------------------------------------------
# "validate_token", "TokenService.authenticate" are literal identifiers, not
# natural language — a deterministic lookup (exact qualified_name) plus a
# keyword scan beats semantic hybrid search for precision and cost. Only
# fires on a STRONG signal so a plain word like "login" (better served by
# semantic search) still takes the hybrid path.
_IDENTIFIER_SHAPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:]*$")
_CAMEL_CASE_HUMP = re.compile(r"[a-z][A-Z]")


def _looks_like_identifier(query: str) -> bool:
    """
    True only when ``query`` is a single whitespace-free token that reads as
    a code identifier: matches the identifier character shape AND carries at
    least one of ``_``, ``.``, ``::``, or an internal camelCase hump — a bare
    lowercase word ("login") returns False on purpose.
    """
    token = query.strip()
    if len(token) < 3 or " " in token or "\t" in token:
        return False
    if not _IDENTIFIER_SHAPE.match(token):
        return False
    return "_" in token or "." in token or "::" in token or bool(_CAMEL_CASE_HUMP.search(token))


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
        target_vector: str | list[str] = VECTOR_SUMMARY,
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
        target_vector : str | list[str]
            Which named vector(s) to search — ``summary`` (natural-language
            intent, default) or ``code`` (literal source), or both at once
            (a single multi-target hybrid call; see ``search_seeds``).

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
            # Scope the BM25 keyword lane to the fields that carry a chunk's
            # OWN identity/content. Without this, `query.hybrid()` matches
            # keywords against every searchable TEXT/TEXT_ARRAY prop — the
            # `calls`/`called_by` arrays (short fields, so BM25 length
            # normalization inflates their hits), `description`, `file_path`,
            # `module_name`, … Residual: `chunk_text` itself still embeds
            # `Calls:`/`Called by:`/`Terms:` lines (chunker.build_text), so
            # hub keyword attraction is reduced here, not closed — capping
            # those embedded lists is a separate planned chunker change.
            # Dropping `description`/`file_path` from the keyword lane is
            # deliberate: both stay reachable through the vector lane. No
            # per-field boosts: tuning weights without an eval harness is
            # guesswork.
            query_properties=["chunk_text", "name", "qualified_name"],
            return_metadata=MetadataQuery(score=True),
        )

        chunks = [self._to_chunk(obj) for obj in response.objects]
        if _TEST_PENALTY < 1.0:
            for c in chunks:
                if c.score is not None and _is_test_path(c.file_path):
                    c.score *= _TEST_PENALTY
            chunks.sort(key=lambda c: -(c.score or 0.0))
        return chunks[:top_k]

    def search_seeds(
        self,
        query: str,
        *,
        project_name: str | None = None,
        top_k: int = 8,
        alpha: float = 0.7,
    ) -> list[RetrievedChunk]:
        """
        Hybrid-search for seed chunks across BOTH named vectors at once: the
        summary vector catches conceptual matches, the code vector catches
        literal ones (and undescribed entities that have no summary vector).

        A single multi-target ``hybrid()`` call, rather than one query per
        vector unioned client-side: two separate calls each get their scores
        RELATIVE_SCORE-normalized *per response*, so the raw numbers aren't
        comparable across them and picking "the higher score" per
        qualified_name is not a valid merge. The guarantee here is one
        response ⇒ one fusion pass ⇒ scores comparable within the result
        set — NOT cross-target normalization: with a plain list the server
        applies its default join (``minimum`` over the raw per-vector
        distances) before fusion, so undescribed entities compete through
        their uncalibrated code-vector distance alone. A
        ``TargetVectors.relative_score`` join changes their standing
        noticeably (measured live) — a knob to evaluate on a real corpus
        before adopting. Confirmed live against Weaviate 1.28.4: entities
        with no ``summary`` vector (empty description) are returned, not
        dropped and not erroring.

        Identifier fast path: when ``query`` reads as a literal code
        identifier (``_looks_like_identifier`` — e.g. "validate_token",
        "TokenService.authenticate"), the hybrid call above is SKIPPED
        entirely in favor of deterministic lookups: an exact qualified_name
        match, then BM25 hits filtered to ones that look like the entity's
        OWN definition rather than a mention (e.g. a caller whose body
        repeats the name). Deterministic-first because for a literal token
        there is a right answer to find, not a similarity to rank — precise
        when it hits; the hybrid path stays as a fallback (at the cost of an
        extra round trip) rather than a first choice. Fallback guarantee: if
        the fast path finds nothing definition-ish, this falls straight
        through to the hybrid call below, so results can never regress
        relative to the pre-fast-path behavior.

        Returns
        -------
        list[RetrievedChunk]
        """
        # Strip once, here, so the gate (_looks_like_identifier) and the
        # executor (_identifier_seeds) see the exact same token — a trailing
        # newline from an MCP client must not silently disable the fast path.
        query = query.strip()
        if _looks_like_identifier(query):
            fast_seeds = self._identifier_seeds(query, project_name=project_name, top_k=top_k)
            if fast_seeds:
                return fast_seeds

        return self.search(
            query, project_name=project_name, top_k=top_k, alpha=alpha,
            target_vector=[VECTOR_SUMMARY, VECTOR_CODE],
        )

    def search_bm25(
        self,
        query: str,
        *,
        project_name: str | None = None,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """
        Literal/keyword search over ``chunk_text`` via Weaviate's BM25 ranking
        (tokenized, NOT full regex) — cheaper and more precise than hybrid
        search when the caller already knows the exact identifier.

        Returns
        -------
        list[RetrievedChunk]
        """
        filters = self._build_filters(project_name, None)

        # Over-fetch so down-weighted test chunks don't crowd real
        # implementation out of the candidate window before we re-rank —
        # same pattern as search().
        fetch_limit = top_k * _SEARCH_OVERFETCH if _TEST_PENALTY < 1.0 else top_k
        response = self.collection.query.bm25(
            query=query,
            query_properties=["chunk_text"],
            filters=filters,
            limit=fetch_limit,
            return_metadata=MetadataQuery(score=True),
        )
        chunks = [self._to_chunk(obj) for obj in response.objects]
        # BM25 never applied the test-file down-weight that search() does,
        # so a test whose name/body happens to match the keyword out-ranked
        # the real implementation. BM25 scores are positive floats, so the
        # same multiply-and-re-sort applies unchanged.
        if _TEST_PENALTY < 1.0:
            for c in chunks:
                if c.score is not None and _is_test_path(c.file_path):
                    c.score *= _TEST_PENALTY
            chunks.sort(key=lambda c: -(c.score or 0.0))
        return chunks[:top_k]

    def get_by_qualified_name(
        self, qualified_name: str, project_name: str
    ) -> RetrievedChunk | None:
        """Fetch a single chunk by its exact qualified_name, or None if absent."""
        return self._fetch_by_qnames([qualified_name], project_name).get(qualified_name)

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

        # 1) seeds via hybrid search, merged across the summary and code vectors.
        seeds = self.search_seeds(
            query, project_name=project_name, top_k=top_k_seeds, alpha=alpha,
        )
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

    def _identifier_seeds(
        self, query: str, *, project_name: str | None, top_k: int
    ) -> list[RetrievedChunk]:
        """
        Deterministic-first seeds for a literal identifier query (called from
        ``search_seeds`` when ``_looks_like_identifier`` is True): an exact
        qualified_name lookup, then BM25 hits that look like the entity's OWN
        definition — ``name`` equals the token's last segment, or
        ``qualified_name`` equals the query or ends with it at a ``.`` / ``::``
        boundary (case-insensitive) — ranked ahead of hits that merely MENTION
        the identifier (e.g. a caller whose body repeats it, which can
        out-score the definition on raw BM25 term frequency). Returns ``[]``
        when nothing definition-ish turns up, so the caller falls back to the
        hybrid path.
        """
        last_segment = re.split(r"[.:]+", query)[-1].lower()
        query_lower = query.lower()

        # project_name is required to derive the deterministic UUID; without
        # it we can only rely on the BM25 pass below.
        exact = self.get_by_qualified_name(query, project_name) if project_name else None

        bm25_hits = self.search_bm25(query, project_name=project_name, top_k=top_k * _SEARCH_OVERFETCH)

        def _is_definition(c: RetrievedChunk) -> bool:
            qn = c.qualified_name.lower()
            # Boundary-checked: a plain `.endswith(query_lower)` would also
            # admit `queue.helpers._send_notification` for query
            # "send_notification" — same trailing characters, different
            # identifier. Requiring a `.`/`::` separator (or full equality)
            # before the match rules that out.
            return (
                c.name.lower() == last_segment
                or qn == query_lower
                or qn.endswith("." + query_lower)
                or qn.endswith("::" + query_lower)
            )

        definition_hits = [c for c in bm25_hits if _is_definition(c)]
        if exact is not None:
            definition_hits = [c for c in definition_hits if c.qualified_name != exact.qualified_name]

        if exact is None and not definition_hits:
            return []

        # Normalize into 0..1: downstream consumers assume that range —
        # the frontend source card renders `score * 100`%, and graph
        # expansion's `parent.score * weight * decay` only stays bounded if
        # the parent score is. Raw BM25 floats (can exceed 1.0) would break
        # both, so scale by the strongest BM25 hit in this response, mirroring
        # Weaviate's own RELATIVE_SCORE convention (top of the CANDIDATE set
        # ≈ 1.0, rest relative to it — a surviving definition can score below
        # 1.0 when a stronger raw keyword candidate was deliberately filtered
        # out, which is honest). Order is preserved (monotonic transform).
        best_bm25 = max((c.score or 0.0) for c in bm25_hits) if bm25_hits else 0.0
        for c in definition_hits:
            c.score = (c.score / best_bm25) if best_bm25 > 0 and c.score is not None else 0.0

        seeds: list[RetrievedChunk] = []
        if exact is not None:
            exact.score = 1.0  # deterministic match: always the top seed
            seeds.append(exact)
        seeds.extend(definition_hits)
        return seeds[:top_k]

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
