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

from collections.abc import Callable
from pathlib import Path

from services.chunker import chunk_graph
from services.describe import describe_chunks
from services.weaviate_client import get_client, ensure_collection, upsert_chunks


# progress(stage, done, total) — stage ∈ {"chunking","describing","embedding","done"}
ProgressFn = Callable[[str, int, int], None]


async def process_and_store(
    output_json_path: str,
    source_root: str | None = None,
    *,
    recreate: bool = False,
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
    progress:
        Optional callback ``(stage, done, total)`` for job status reporting.

    Returns
    -------
    dict with ``project_name``, ``chunks_created``, ``ingestion_id`` and the
    ``descriptions`` stats block.
    """
    path = Path(output_json_path)
    ingestion_id = path.stem

    def report(stage: str, done: int, total: int) -> None:
        if progress:
            progress(stage, done, total)

    report("chunking", 0, 1)
    chunks = chunk_graph(path, source_root=source_root)
    report("chunking", len(chunks), len(chunks))

    # Tier 2 — micro-descriptions (skipped gracefully if DESC_MODEL is unset).
    report("describing", 0, len(chunks))
    desc_stats = await describe_chunks(
        chunks,
        progress=lambda done, total: report("describing", done, total),
    )

    client = get_client()
    try:
        ensure_collection(client, recreate=recreate)
        report("embedding", 0, len(chunks))
        inserted = upsert_chunks(client, chunks, ingestion_id)
        report("embedding", inserted, len(chunks))
    finally:
        client.close()

    project_name = chunks[0].project_name if chunks else ""
    report("done", inserted, len(chunks))

    return {
        "project_name": project_name,
        "chunks_created": inserted,
        "ingestion_id": ingestion_id,
        "descriptions": desc_stats,
    }
