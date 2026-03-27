"""
Weaviate client wrapper for Dokkai.

Manages the connection to Weaviate and the ``CodeEntity`` collection schema.
Supports both local ``text2vec-transformers`` (default) and API-based
vectorizers (OpenAI, Cohere) via environment variables.

Environment variables
---------------------
WEAVIATE_HOST          - default ``localhost``
WEAVIATE_HTTP_PORT     - default ``8080``
WEAVIATE_GRPC_PORT     - default ``50051``
VECTORIZER_PROVIDER    - ``local`` (default) | ``openai`` | ``cohere``
OPENAI_API_KEY         - required when VECTORIZER_PROVIDER=openai
COHERE_API_KEY         - required when VECTORIZER_PROVIDER=cohere
"""

from __future__ import annotations

import os
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter

from services.chunker import CodeChunk

COLLECTION_NAME = "CodeEntity"

def _get_vectorizer_config():
    provider = os.getenv("VECTORIZER_PROVIDER", "local").lower()

    if provider == "openai":
        return Configure.Vectorizer.text2vec_openai(
            model="text-embedding-3-small",
        )
    elif provider == "cohere":
        return Configure.Vectorizer.text2vec_cohere(
            model="embed-multilingual-v3.0",
        )
    else:
        return Configure.Vectorizer.text2vec_transformers()

def get_client() -> weaviate.WeaviateClient:
    '''
    Stablish connection with Weaviate.
    '''
    host = os.getenv("WEAVIATE_HOST", "localhost")
    http_port = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
    grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

    headers: dict[str, str] = {}
    provider = os.getenv("VECTORIZER_PROVIDER", "local").lower()
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
    Property(name="chunk_text",      data_type=DataType.TEXT,       skip_vectorization=False),
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
    Property(name="ingestion_id",    data_type=DataType.TEXT,       skip_vectorization=True),
]


def ensure_collection(client: weaviate.WeaviateClient) -> None:
    if client.collections.exists(COLLECTION_NAME):
        return

    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=_get_vectorizer_config(),
        properties=_PROPERTIES,
    )

def delete_project_chunks(client: weaviate.WeaviateClient, project_name: str) -> int:
    collection = client.collections.get(COLLECTION_NAME)
    result = collection.data.delete_many(
        where=Filter.by_property("project_name").equal(project_name),
    )
    return result.successful

def upsert_chunks(
    client: weaviate.WeaviateClient,
    chunks: list[CodeChunk],
    ingestion_id: str,
) -> int:
    collection = client.collections.get(COLLECTION_NAME)

    if chunks:
        delete_project_chunks(client, chunks[0].project_name)

    inserted = 0
    with collection.batch.dynamic() as batch:
        for chunk in chunks:
            properties = {
                "chunk_text":      chunk.chunk_text,
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
                "ingestion_id":    ingestion_id,
            }
            batch.add_object(properties=properties)
            inserted += 1

    return inserted
