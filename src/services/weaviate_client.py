"""
Weaviate client wrapper for Dokkai.

Manages the connection to Weaviate and the ``CodeEntity`` collection schema.
The collection uses two named vectors — ``code`` and ``summary`` — produced by
the configured provider (Ollama by default; OpenAI/Cohere/transformers also
supported) via environment variables.

Environment variables
---------------------
WEAVIATE_HOST          - default ``localhost``
WEAVIATE_HTTP_PORT     - default ``8080``
WEAVIATE_GRPC_PORT     - default ``50051``
VECTORIZER_PROVIDER    - ``ollama`` (default) | ``openai`` | ``cohere`` | ``local``
OLLAMA_EMBED_ENDPOINT  - Ollama URL as seen by the Weaviate server
                         (default ``http://host.docker.internal:11434``)
EMBED_MODEL            - Ollama embedding model (default ``nomic-embed-text``)
OPENAI_API_KEY         - required when VECTORIZER_PROVIDER=openai
COHERE_API_KEY         - required when VECTORIZER_PROVIDER=cohere
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
from weaviate.util import generate_uuid5

from services.chunker import CodeChunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "CodeEntity")

# Named vectors (Tier 2). Each object carries two embeddings:
#   - VECTOR_CODE    over ``chunk_text``  (header + terms + doc + real source)
#   - VECTOR_SUMMARY over ``description`` (LLM micro-description, when present)
# so retrieval can target literal code or natural-language intent.
VECTOR_CODE = "code"
VECTOR_SUMMARY = "summary"


def uuid_for_entity(project_name: str, qualified_name: str) -> str:
    """
    Deterministic object id for a code entity.

    The SAME formula is used at insert time (so it doubles as the upsert /
    dedup key) and by the retriever to follow graph edges by id. Stable across
    re-ingestions because it is derived only from the entity's identity.
    """
    return generate_uuid5(f"{project_name}:{qualified_name}")

def _named_vectors():
    """
    Two named vectors — ``code`` (over ``chunk_text``) and ``summary`` (over the
    LLM ``description``) — both produced by the configured embedding provider.
    Objects with an empty ``description`` simply get no ``summary`` vector, so
    only described entities are reachable through summary-targeted search.
    """
    provider = os.getenv("VECTORIZER_PROVIDER", "ollama").lower()

    def pair(factory, **kw):
        return [
            factory(name=VECTOR_CODE, source_properties=["chunk_text"], **kw),
            factory(name=VECTOR_SUMMARY, source_properties=["description"], **kw),
        ]

    if provider == "ollama":
        # ``api_endpoint`` is resolved from the Weaviate *server's* perspective,
        # so under Docker it must reach the host (host.docker.internal).
        return pair(
            Configure.NamedVectors.text2vec_ollama,
            api_endpoint=os.getenv("OLLAMA_EMBED_ENDPOINT", "http://host.docker.internal:11434"),
            model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            vectorize_collection_name=False,
        )
    elif provider == "openai":
        return pair(Configure.NamedVectors.text2vec_openai, model="text-embedding-3-small")
    elif provider == "cohere":
        return pair(Configure.NamedVectors.text2vec_cohere, model="embed-multilingual-v3.0")
    else:
        return pair(Configure.NamedVectors.text2vec_transformers)

def get_client() -> weaviate.WeaviateClient:
    '''
    Stablish connection with Weaviate.
    '''
    host = os.getenv("WEAVIATE_HOST", "localhost")
    http_port = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
    grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

    headers: dict[str, str] = {}
    provider = os.getenv("VECTORIZER_PROVIDER", "ollama").lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            headers["X-OpenAI-Api-Key"] = api_key
    elif provider == "cohere":
        api_key = os.getenv("COHERE_API_KEY", "")
        if api_key:
            headers["X-Cohere-Api-Key"] = api_key

    client = weaviate.connect_to_local(
        host=host,
        port=http_port,
        grpc_port=grpc_port,
        headers=headers,
    )
    return client

_PROPERTIES = [
    Property(name="chunk_text",      data_type=DataType.TEXT),
    Property(name="description",     data_type=DataType.TEXT),
    Property(name="entity_type",     data_type=DataType.TEXT,       skip_vectorization=True),
    Property(name="name",            data_type=DataType.TEXT,       skip_vectorization=True),
    Property(name="qualified_name",  data_type=DataType.TEXT,       skip_vectorization=True),
    Property(name="file_path",       data_type=DataType.TEXT,       skip_vectorization=True),
    Property(name="absolute_path",   data_type=DataType.TEXT,       skip_vectorization=True),
    Property(name="start_line",      data_type=DataType.INT,        skip_vectorization=True),
    Property(name="end_line",        data_type=DataType.INT,        skip_vectorization=True),
    Property(name="project_name",    data_type=DataType.TEXT,       skip_vectorization=True),
    Property(name="module_name",     data_type=DataType.TEXT,       skip_vectorization=True),
    Property(name="parent_class",    data_type=DataType.TEXT,       skip_vectorization=True),
    Property(name="decorators",      data_type=DataType.TEXT_ARRAY, skip_vectorization=True),
    Property(name="is_exported",     data_type=DataType.BOOL,       skip_vectorization=True),
    Property(name="calls",           data_type=DataType.TEXT_ARRAY, skip_vectorization=True),
    Property(name="called_by",       data_type=DataType.TEXT_ARRAY, skip_vectorization=True),
    Property(name="inherits",        data_type=DataType.TEXT_ARRAY, skip_vectorization=True),
    Property(name="implements",      data_type=DataType.TEXT_ARRAY, skip_vectorization=True),
    Property(name="overrides",       data_type=DataType.TEXT_ARRAY, skip_vectorization=True),
    Property(name="defined_methods", data_type=DataType.TEXT_ARRAY, skip_vectorization=True),
    Property(name="imports",         data_type=DataType.TEXT_ARRAY, skip_vectorization=True),
    Property(name="ingestion_id",    data_type=DataType.TEXT,       skip_vectorization=True),
]


def _assert_named_vectors(client: weaviate.WeaviateClient) -> None:
    """Fail loudly if an existing collection predates the named-vector schema."""
    cfg = client.collections.get(COLLECTION_NAME).config.get()
    names = set((cfg.vector_config or {}).keys())
    if not {VECTOR_CODE, VECTOR_SUMMARY} <= names:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' has an outdated schema "
            f"(vectors: {sorted(names) or 'single default vector'}). Tier 2 needs "
            f"named vectors '{VECTOR_CODE}' + '{VECTOR_SUMMARY}'. Recreate it once: "
            "pass recreate=true to the pipeline (or set DOKKAI_RECREATE_COLLECTION=1), "
            "or run `docker compose down -v && docker compose up -d`, then re-ingest."
        )


def ensure_collection(client: weaviate.WeaviateClient, *, recreate: bool = False) -> None:
    # A one-time env escape hatch for the schema migration, mirrored in the
    # _assert_named_vectors error message. Leaving it set drops the collection
    # on every ingest, so prefer the per-request ``recreate`` flag.
    recreate = recreate or os.getenv("DOKKAI_RECREATE_COLLECTION", "").lower() in ("1", "true", "yes")

    exists = client.collections.exists(COLLECTION_NAME)
    if exists and recreate:
        client.collections.delete(COLLECTION_NAME)  # drops ALL projects in the collection
        exists = False

    if exists:
        _assert_named_vectors(client)
        _ensure_imports_property(client)
        return

    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=_named_vectors(),
        properties=_PROPERTIES,
    )


def _ensure_imports_property(client: weaviate.WeaviateClient) -> None:
    """
    Additive migration for pre-existing collections created before the
    ``imports`` property existed: an early-return on ``exists`` above would
    otherwise leave a deployed collection without it forever. No recreation,
    no data loss — just adds the missing property (existing objects simply
    read back an empty ``imports`` list until re-ingested).
    """
    collection = client.collections.get(COLLECTION_NAME)
    existing_names = {p.name for p in collection.config.get().properties}
    if "imports" in existing_names:
        return
    try:
        collection.config.add_property(
            Property(name="imports", data_type=DataType.TEXT_ARRAY, skip_vectorization=True)
        )
    except weaviate.exceptions.WeaviateInvalidInputError:
        # Check-then-add races with a concurrent ingest of ANOTHER project
        # (jobs are serialized per project only): the loser's add_property
        # raises "already exists". The operation is idempotent — verify and
        # move on rather than killing that ingest job.
        current = {p.name for p in collection.config.get().properties}
        if "imports" not in current:
            raise
        return
    logger.info("Added missing 'imports' property to collection '%s'", COLLECTION_NAME)

# Per-call cap on Filter.by_id().contains_any(...) — keeps each delete_many
# gRPC call bounded regardless of how many objects need removing.
_DELETE_CHUNK_SIZE = 500


def _uuids_matching(
    collection,
    *,
    return_properties: list[str],
    predicate: Callable[[dict], bool],
) -> list[str]:
    """
    Cursor-scan EVERY object in the collection (``collection.iterator()`` —
    no offset/``QUERY_MAXIMUM_RESULTS`` ceiling, unlike ``fetch_objects``'
    offset paging — live-confirmed to raise at exactly 10,000 matched
    objects), collecting the uuid of each whose *predicate* (evaluated on its
    ``properties``) returns True.

    Deliberately does NOT pass a server-side ``project_name`` filter to
    narrow the scan: ``iterator()`` doesn't accept a ``filters`` argument on
    weaviate-client 4.20.4 (confirmed via signature introspection) — but even
    if it did, this schema's TEXT properties use the default WORD
    tokenization, under which ``equal`` matches by "document's tokens are a
    superset of the query's tokens", NOT literal string equality. Live
    -confirmed cross-project contamination: ``Filter.by_property
    ("project_name").equal("back-end")`` matches an object whose
    ``project_name`` is ``"sample_back-end"``. *predicate* MUST do its own
    exact (``==``) comparison — never trust a server-side text filter alone
    for tenant isolation on this schema.
    """
    return [
        str(obj.uuid)
        for obj in collection.iterator(return_properties=return_properties)
        if predicate(obj.properties)
    ]


def _delete_by_uuids(collection, uuids: list[str], *, context: str) -> int:
    """
    Delete objects by id, chunked at :data:`_DELETE_CHUNK_SIZE` per
    ``delete_many`` call. Raises loudly on ANY failure — mirroring
    ``upsert_chunks``' own ``batch.failed_objects`` convention ("fail loudly
    instead of silently under-reporting") — rather than returning a
    successful count that quietly omits failures. Returns the total
    successful-delete count.
    """
    if not uuids:
        return 0
    successful = 0
    failed = 0
    for i in range(0, len(uuids), _DELETE_CHUNK_SIZE):
        batch_uuids = uuids[i : i + _DELETE_CHUNK_SIZE]
        result = collection.data.delete_many(where=Filter.by_id().contains_any(batch_uuids))
        successful += result.successful
        failed += result.failed
    if failed:
        raise RuntimeError(f"{context}: {failed}/{len(uuids)} objects failed to delete")
    return successful


def delete_project_chunks(client: weaviate.WeaviateClient, project_name: str) -> int:
    collection = client.collections.get(COLLECTION_NAME)
    uuids = _uuids_matching(
        collection,
        return_properties=["project_name"],
        predicate=lambda props: props.get("project_name") == project_name,
    )
    return _delete_by_uuids(
        collection, uuids, context=f"delete_project_chunks(project_name={project_name!r})"
    )

def fetch_project_chunk_index(client: weaviate.WeaviateClient, project_name: str) -> dict[str, dict]:
    """
    Fetch every chunk of a project, keyed by object uuid, carrying only the
    ``qualified_name``, ``chunk_text`` and ``description`` properties.

    Paged via offset/limit. Weaviate's default ``QUERY_MAXIMUM_RESULTS`` caps
    offset-based pagination at 10,000 objects — the practical ceiling of this
    approach, well above any single project's expected chunk count.
    """
    collection = client.collections.get(COLLECTION_NAME)
    page_size = 500
    index: dict[str, dict] = {}
    offset = 0
    while True:
        response = collection.query.fetch_objects(
            filters=Filter.by_property("project_name").equal(project_name),
            limit=page_size,
            offset=offset,
            return_properties=["qualified_name", "chunk_text", "description"],
        )
        if not response.objects:
            break
        for obj in response.objects:
            index[str(obj.uuid)] = {
                "qualified_name": obj.properties.get("qualified_name"),
                "chunk_text":     obj.properties.get("chunk_text"),
                "description":    obj.properties.get("description"),
            }
        if len(response.objects) < page_size:
            break
        offset += page_size
    return index

def update_chunk_description(collection, uuid: str, description: str) -> None:
    """
    Merge-update only the ``description`` property of an existing object.

    Updating a single property re-vectorizes only the named vector(s) sourced
    from it — here the ``summary`` vector (sourced from ``description``) is
    recomputed while ``code`` (sourced from ``chunk_text``) is left untouched.
    """
    collection.data.update(uuid=uuid, properties={"description": description})

def project_has_chunks(client: weaviate.WeaviateClient, project_name: str) -> bool:
    """Cheap existence check: does this project have any chunks at all?"""
    collection = client.collections.get(COLLECTION_NAME)
    response = collection.query.fetch_objects(
        filters=Filter.by_property("project_name").equal(project_name),
        limit=1,
        return_properties=["qualified_name"],
    )
    return len(response.objects) > 0

def count_chunks_by_project(client: weaviate.WeaviateClient) -> list[dict]:
    """
    Chunk count per ingested project, via a group-by aggregate over
    ``project_name`` (no vector search, no per-object fetch).

    Returns ``[{"project": <name>, "chunks": <count>}, ...]`` sorted by
    project name.
    """
    collection = client.collections.get(COLLECTION_NAME)
    result = collection.aggregate.over_all(group_by="project_name")
    counts = [
        {"project": group.grouped_by.value, "chunks": group.total_count}
        for group in result.groups
    ]
    counts.sort(key=lambda c: c["project"])
    return counts

def upsert_chunks(
    client: weaviate.WeaviateClient,
    chunks: list[CodeChunk],
    ingestion_id: str,
    *,
    on_purge: Callable[[int], None] | None = None,
) -> int:
    # NOTE: chunker.CodeChunk.build_text now caps the relation lists it
    # embeds into chunk_text (_MAX_RELATION_ITEMS) — an already-ingested
    # corpus keeps its OLD (uncapped) chunk_text/vector until it goes
    # through this upsert again: the deterministic UUID (uuid_for_entity)
    # means a re-ingest overwrites the SAME object with the new chunk_text,
    # which re-triggers vectorization naturally — no separate migration path
    # needed here, just a re-ingestion of the project.
    collection = client.collections.get(COLLECTION_NAME)
    if not chunks:
        return 0

    project_name = chunks[0].project_name
    seen_ids: set[str] = set()
    with collection.batch.dynamic() as batch:
        for chunk in chunks:
            properties = {
                "chunk_text":      chunk.chunk_text,
                # summary vector source: LLM description, falling back to any
                # doc/JSDoc we extracted for free (Tier 1) so more entities are
                # reachable via summary-targeted search.
                "description":     chunk.description or chunk.doc,
                "entity_type":     chunk.entity_type,
                "name":            chunk.name,
                "qualified_name":  chunk.qualified_name,
                "file_path":       chunk.file_path,
                "absolute_path":   chunk.absolute_path,
                "start_line":      chunk.start_line or 0,
                "end_line":        chunk.end_line or 0,
                "project_name":    chunk.project_name,
                "module_name":     chunk.module_name,
                "parent_class":    chunk.parent_class or "",
                "decorators":      chunk.decorators,
                "is_exported":     chunk.is_exported,
                "calls":           chunk.calls,
                "called_by":       chunk.called_by,
                "inherits":        chunk.inherits,
                "implements":      chunk.implements,
                "overrides":       chunk.overrides,
                "defined_methods": chunk.defined_methods,
                "imports":         chunk.imports,
                "ingestion_id":    ingestion_id,
            }
            object_id = uuid_for_entity(chunk.project_name, chunk.qualified_name)
            batch.add_object(properties=properties, uuid=object_id)
            seen_ids.add(object_id)

    # The batch context does NOT raise on per-object embedding/insert errors —
    # it only collects them. Fail loudly instead of silently under-reporting.
    failed = collection.batch.failed_objects
    if failed:
        first = getattr(failed[0], "message", failed[0])
        raise RuntimeError(
            f"{len(failed)}/{len(seen_ids)} objects failed to insert; first error: {first}"
        )

    # Remove entities left over from earlier ingests of this project (deleted at
    # the source, or filtered out by ignore_rules on a re-ingest — feature 21).
    # Done AFTER a successful insert — deterministic UUIDs upsert current
    # entities in place, so there is no wipe-then-fail data-loss window.
    #
    # Deliberately NOT Filter.by_property("ingestion_id").not_equal(ingestion_id)
    # (nor an `equal("project_name")` server-side filter to narrow the scan):
    # confirmed live (Weaviate 1.28.4 / weaviate-client 4.20.4) that `not_equal`
    # on a WORD-tokenized TEXT property never matches anything — even
    # single-token values like "alpha" vs "beta" — and `equal` matches by
    # token-superset, not exact string (`equal("back-end")` matches
    # "sample_back-end"). See `_uuids_matching`'s docstring — this is the
    # exact-client-side-match, no-10k-ceiling helper shared with
    # `delete_project_chunks`.
    stale_uuids = _uuids_matching(
        collection,
        return_properties=["project_name", "ingestion_id"],
        predicate=lambda props: (
            props.get("project_name") == project_name
            and props.get("ingestion_id") != ingestion_id
        ),
    )
    purged = _delete_by_uuids(
        collection,
        stale_uuids,
        context=f"upsert_chunks purge (project={project_name!r}, ingestion_id={ingestion_id!r})",
    )
    logger.info(
        "upsert_chunks: purged %d stale chunk(s) for project '%s' (ingestion_id != '%s')",
        purged, project_name, ingestion_id,
    )
    if on_purge is not None:
        on_purge(purged)

    return len(seen_ids)
