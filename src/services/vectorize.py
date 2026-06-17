"""
Vectorize orchestrator — ties chunking + Weaviate storage together.

Call :func:`process_and_store` with a ``repo_path`` and the path to the
already-generated graph JSON.  It will:

1. Read and chunk the graph JSON.
2. Connect to Weaviate, ensure the collection exists.
3. Batch-insert all chunks (replacing previous data for the same project).
4. Return summary statistics.
"""

from __future__ import annotations

from pathlib import Path

from services.chunker import chunk_graph
from services.weaviate_client import get_client, ensure_collection, upsert_chunks


async def process_and_store(
    output_json_path: str,
    source_root: str | None = None,
) -> dict:
    """
    Main entry-point: chunk the graph JSON and push to Weaviate.

    Parameters
    ----------
    output_json_path:
        Absolute or relative path to the code-graph-rag output JSON.
    source_root:
        Repository root used to read the actual source code into each chunk.
        Falls back to the absolute paths recorded in the graph when omitted.

    Returns
    -------
    dict with ``project_name``, ``chunks_created``, ``ingestion_id``.
    """
    path = Path(output_json_path)
    ingestion_id = path.stem

    chunks = chunk_graph(path, source_root=source_root)
    client = get_client()
    
    try:
        ensure_collection(client)
        inserted = upsert_chunks(client, chunks, ingestion_id)
    finally:
        client.close()

    project_name = chunks[0].project_name if chunks else ""

    return {
        "project_name": project_name,
        "chunks_created": inserted,
        "ingestion_id": ingestion_id,
    }
