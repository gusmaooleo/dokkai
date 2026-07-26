"""
Vectorize orchestrator — ties chunking + descriptions + Weaviate storage.

Call :func:`process_and_store` with the path to the already-generated graph
JSON.  It will:

1. Read and chunk the graph JSON (with source + Tier-1 enrichment).
2. Generate Tier-2 micro-descriptions (selective, cached) — populates the
   ``summary`` vector source.
3. Connect to Weaviate, ensure the named-vector collection exists.
4. Batch-upsert all chunks (replacing previous data for the same project).
5. Return summary statistics.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from services.chunker import CodeChunk, chunk_graph
from services.describe import describe_chunks
from services.weaviate_client import (
    COLLECTION_NAME,
    ensure_collection,
    fetch_project_chunk_index,
    get_client,
    update_chunk_description,
    upsert_chunks,
    uuid_for_entity,
)


# progress(stage, done, total) — stage ∈ {"chunk","describe","upsert","update","done"}
ProgressFn = Callable[[str, int, int], None]

# Timestamp suffix used by ingestByLocalRepository: "-<YYYYMMDDTHHMMSS>".
_TIMESTAMP_SUFFIX_RE = re.compile(r"^-\d{8}T\d{6}$")


async def process_and_store(
    output_json_path: str,
    source_root: str | None = None,
    *,
    recreate: bool = False,
    describe: bool = True,
    progress: ProgressFn | None = None,
) -> dict:
    """
    Chunk the graph JSON, describe entities, and push everything to Weaviate.

    Parameters
    ----------
    output_json_path:
        Path to the code-graph-rag output JSON.
    source_root:
        Repository root used to read the actual source code into each chunk.
        Falls back to the absolute paths recorded in the graph when omitted.
    recreate:
        Drop and rebuild the collection before inserting. Needed once when the
        schema changes (e.g. adopting named vectors). Wipes ALL projects.
    describe:
        Whether to run the Tier-2 description pass. Defaults to ``True``; the
        no-describe mode is an explicit opt-in for internal callers (the
        CLI/API UX for it lands with feature 04) — it never falls back
        silently on a missing/unavailable descriptor model.
    progress:
        Optional callback ``(stage, done, total)`` for job status reporting.

    Returns
    -------
    dict with ``project_name``, ``chunks_created``, ``ingestion_id`` and the
    ``descriptions`` stats block.
    """
    path = Path(output_json_path)
    # Unique per run — even with a constant canonical stem (post-promotion,
    # ingested/<project>.json) — so upsert_chunks' stale-entity purge (see
    # weaviate_client.upsert_chunks) still distinguishes "this run" from "an
    # earlier run" of the same project.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    ingestion_id = f"{path.stem}-{timestamp}"

    def report(stage: str, done: int, total: int) -> None:
        if progress:
            progress(stage, done, total)

    report("chunk", 0, 1)
    chunks = chunk_graph(path, source_root=source_root)
    report("chunk", len(chunks), len(chunks))

    # Tier 2 — micro-descriptions. Raises DescriptorError (via describe_chunks)
    # when no descriptor model is configured or it's unavailable — describe=False
    # is the only way to skip this pass.
    report("describe", 0, len(chunks))
    if describe:
        desc_stats = await describe_chunks(
            chunks,
            progress=lambda done, total: report("describe", done, total),
        )
    else:
        desc_stats = {"enabled": False, "reason": "descriptions disabled for this ingestion"}
        report("describe", len(chunks), len(chunks))

    client = get_client()
    purge_stats = {"count": 0}
    try:
        ensure_collection(client, recreate=recreate)
        report("upsert", 0, len(chunks))
        inserted = upsert_chunks(
            client,
            chunks,
            ingestion_id,
            on_purge=lambda n: purge_stats.__setitem__("count", n),
        )
        report("upsert", inserted, len(chunks))
    finally:
        client.close()

    project_name = chunks[0].project_name if chunks else ""
    report("done", inserted, len(chunks))

    return {
        "project_name": project_name,
        "chunks_created": inserted,
        "ingestion_id": ingestion_id,
        "descriptions": desc_stats,
        "stale_chunks_purged": purge_stats["count"],
    }


def _ingested_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "ingested"


def graph_json_candidates(project_name: str) -> list[Path]:
    """
    Find every graph JSON for *project_name* in ``ingested/``.

    Matches only ``<project_name>.json`` or
    ``<project_name>-<YYYYMMDDTHHMMSS>.json`` — anchored so a project name
    that prefixes another (e.g. ``foo`` vs ``foo-bar``) never matches the
    wrong file.
    """
    ingested_dir = _ingested_dir()
    if not ingested_dir.is_dir():
        return []

    candidates: list[Path] = []
    for path in ingested_dir.glob("*.json"):
        stem = path.stem
        if stem == project_name:
            candidates.append(path)
        elif stem.startswith(project_name) and _TIMESTAMP_SUFFIX_RE.match(stem[len(project_name):]):
            candidates.append(path)

    return candidates


def find_latest_graph_json(project_name: str) -> Path | None:
    """
    Find the newest graph JSON for *project_name* in ``ingested/``.

    Newest mtime wins when several candidates exist (see
    :func:`graph_json_candidates` for the matching rules).
    """
    candidates = graph_json_candidates(project_name)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def no_graph_json_message(project_name: str) -> str:
    """English message for a missing graph JSON — shared by the refresh
    pre-flight (409) and the fail-loud check inside the refresh job itself."""
    return (
        f"no graph JSON found for project '{project_name}' in ingested/ — "
        "re-run POST /instances/pipeline to regenerate it"
    )


async def refresh_descriptions(
    project_name: str,
    *,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> dict:
    """
    Re-run ONLY the describe pass over a project's already-ingested chunks.

    Re-chunks the latest graph JSON in ``ingested/`` and, for every rebuilt
    chunk whose ``chunk_text`` is byte-identical to what's stored in
    Weaviate (the anti-drift guard), re-describes it and writes back the
    ``description`` property when it changed. Chunks whose rebuilt text
    differs from the stored one (source edited since ingestion) are counted
    as ``stale_source`` and never described; rebuilt chunks with no stored
    counterpart are counted as ``not_indexed`` and skipped.

    Never re-chunks the graph relationships/upsert path and never touches
    embeddings of unchanged ``chunk_text`` — only ``description`` (and the
    ``summary`` vector it feeds) can change. A transient generation failure
    (``describe_chunks`` swallows per-entity errors, leaving
    ``chunk.description`` empty) never overwrites a non-empty stored
    description with an empty one — such chunks are counted as
    ``preserved`` instead of ``updated``.
    """

    def report(stage: str, done: int, total: int) -> None:
        if progress:
            progress(stage, done, total)

    graph_path = find_latest_graph_json(project_name)
    if graph_path is None:
        raise RuntimeError(no_graph_json_message(project_name))

    report("chunk", 0, 1)
    chunks = chunk_graph(graph_path)
    report("chunk", len(chunks), len(chunks))

    if chunks and chunks[0].project_name != project_name:
        raise RuntimeError(
            f"graph JSON '{graph_path.name}' resolves to project "
            f"'{chunks[0].project_name}', expected '{project_name}'"
        )

    client = get_client()
    try:
        index = fetch_project_chunk_index(client, project_name)

        eligible: list[tuple[str, CodeChunk, dict]] = []
        stale_source = 0
        not_indexed = 0
        for chunk in chunks:
            uuid = uuid_for_entity(chunk.project_name, chunk.qualified_name)
            stored = index.get(uuid)
            if stored is None:
                not_indexed += 1
                continue
            if stored.get("chunk_text") != chunk.chunk_text:
                stale_source += 1
                continue
            eligible.append((uuid, chunk, stored))

        report("describe", 0, len(eligible))
        desc_stats = await describe_chunks(
            [chunk for _, chunk, _ in eligible],
            force=force,
            progress=lambda done, total: report("describe", done, total),
        )

        collection = client.collections.get(COLLECTION_NAME)
        total_eligible = len(eligible)
        report("update", 0, total_eligible)
        updated = 0
        unchanged = 0
        preserved = 0
        for i, (uuid, chunk, stored) in enumerate(eligible, start=1):
            new_description = chunk.description or chunk.doc
            stored_description = stored.get("description")
            if not new_description and stored_description:
                # An eligible chunk's doc is byte-identical to ingestion time
                # (chunk_text matched), so an empty new_description here can
                # only mean a transient generation failure — never overwrite
                # a good stored description with an empty one.
                preserved += 1
            elif new_description != stored_description:
                update_chunk_description(collection, uuid, new_description)
                updated += 1
            else:
                unchanged += 1
            report("update", i, total_eligible)
    finally:
        client.close()

    stats = {
        "project_name": project_name,
        "chunks_indexed": len(index),
        "eligible": total_eligible,
        "stale_source": stale_source,
        "not_indexed": not_indexed,
        "updated": updated,
        "unchanged": unchanged,
        "preserved": preserved,
        "descriptions": desc_stats,
    }
    if stale_source > 0:
        stats["note"] = (
            f"{stale_source} entities changed on disk since ingestion — "
            "re-ingest to refresh them"
        )
    return stats
