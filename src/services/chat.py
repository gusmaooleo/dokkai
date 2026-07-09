"""
Chat / RAG service — orchestrates retrieval + LLM generation.

1. Retrieve relevant code chunks from Weaviate via the Retriever
2. Build an augmented prompt with audience-specific system instructions
3. Stream the LLM response
4. Persist messages to the chat store
"""

from __future__ import annotations

import asyncio
import uuid
from typing import AsyncIterator

from services.retriever import Retriever, RetrievedChunk
from services.llm_provider import LLMMessage
from services.llm_config import get_active_provider
from services.chat_store import get_chat_store, ChatMessage
from services.db import DatabaseUnavailableError
from services.weaviate_client import get_client


# -----------------------------------------------------------------------
# Audience-specific system prompts
# -----------------------------------------------------------------------

_SYSTEM_PROMPTS = {
    "developer": (
        "You are Dokkai, a senior software engineer assistant that helps developers "
        "understand codebases. You have access to the project's source code context below.\n\n"
        "Guidelines:\n"
        "- Reference specific files, classes, functions, and line numbers when answering.\n"
        "- Explain architecture, design patterns, and data flow clearly.\n"
        "- Include code snippets when helpful.\n"
        "- Note relationships between components (calls, inheritance, interfaces).\n"
        "- If the context doesn't contain enough information, say so honestly.\n"
        "- Answer in the same language as the user's question."
    ),
    "manager": (
        "You are Dokkai, a technical project advisor that explains software systems "
        "to managers and executives in clear, non-technical language.\n\n"
        "Guidelines:\n"
        "- Avoid jargon; use business-friendly language.\n"
        "- Focus on what the system DOES, not how it's coded.\n"
        "- Explain impact: what features does this enable? What risks exist?\n"
        "- Use analogies when they help clarify concepts.\n"
        "- Summarize complexity in terms of effort, scope, and dependencies.\n"
        "- If the context doesn't contain enough information, say so honestly.\n"
        "- Answer in the same language as the user's question."
    ),
    "customer": (
        "You are Dokkai, a friendly product documentation assistant that explains "
        "software features from an end-user perspective.\n\n"
        "Guidelines:\n"
        "- Focus on what the user CAN DO with the software.\n"
        "- Explain features, workflows, and capabilities — not implementation details.\n"
        "- Use simple, clear language suitable for non-technical users.\n"
        "- Describe UI flows and user-visible behavior when possible.\n"
        "- If the system has an API, explain what each endpoint lets the user accomplish.\n"
        "- If the context doesn't contain enough information, say so honestly.\n"
        "- Answer in the same language as the user's question."
    ),
}


def _get_system_prompt(audience: str) -> str:
    return _SYSTEM_PROMPTS.get(audience, _SYSTEM_PROMPTS["developer"])


# -----------------------------------------------------------------------
# Core chat function
# -----------------------------------------------------------------------

async def chat_with_codebase(
    message: str,
    project_name: str,
    *,
    audience: str = "developer",
    conversation_id: str | None = None,
    top_k: int = 8,
    max_hops: int = 2,
    max_nodes: int = 30,
    direction: str = "both",
) -> AsyncIterator[dict]:
    """
    RAG chat pipeline — yields SSE-ready dicts.

    Yields
    ------
    dict with one of these shapes:
        {"type": "sources", "data": [...]}       — retrieved source chunks
        {"type": "token", "data": "..."}         — streaming token
        {"type": "done", "data": {"conversation_id": "...", "answer": "..."}}
        {"type": "error", "data": "..."}
    """
    store = get_chat_store()

    # Resolve or create conversation
    try:
        if conversation_id:
            conv = await store.get_conversation(conversation_id)
            if conv is None:
                conversation_id = str(uuid.uuid4())
        else:
            conversation_id = str(uuid.uuid4())
    except DatabaseUnavailableError as e:
        yield {"type": "error", "data": str(e)}
        return

    # ---- Step 1: Retrieve relevant code (graph-expanded) ----
    # The Weaviate client is synchronous (blocking gRPC), so run the whole
    # retrieval off the event loop — otherwise it freezes every other request
    # (concurrent chats, job-status polls) for the full retrieval window.
    def _retrieve() -> tuple[list[RetrievedChunk], str]:
        client = get_client()
        try:
            retriever = Retriever(client)
            chunks = retriever.search_graph(
                query=message,
                project_name=project_name,
                top_k_seeds=top_k,
                max_hops=max_hops,
                max_nodes=max_nodes,
                direction=direction,
            )
            return chunks, retriever.build_graph_context(chunks)
        finally:
            client.close()

    chunks, context = await asyncio.to_thread(_retrieve)

    # Yield sources to the client
    source_dicts = [
        {
            "entity_type": c.entity_type,
            "name": c.name,
            "qualified_name": c.qualified_name,
            "file_path": c.file_path,
            "absolute_path": c.absolute_path,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "chunk_text": c.chunk_text,
            "description": c.description,
            "score": c.score,
            "hop": c.hop,
            "via": c.via,
        }
        for c in chunks
    ]
    yield {"type": "sources", "data": source_dicts}

    # ---- Step 2: Build augmented prompt ----
    system_prompt = _get_system_prompt(audience)
    system_with_context = (
        f"{system_prompt}\n\n"
        f"=== CODEBASE CONTEXT ===\n{context}\n=== END CONTEXT ==="
    )

    # Build message list from conversation history
    llm_messages: list[LLMMessage] = [
        LLMMessage(role="system", content=system_with_context),
    ]

    # Add previous conversation history (if any)
    try:
        conv = await store.get_conversation(conversation_id)
        if conv and conv.messages:
            for msg in conv.messages:
                if msg.role in ("user", "assistant"):
                    llm_messages.append(LLMMessage(role=msg.role, content=msg.content))

        # Add current user message
        llm_messages.append(LLMMessage(role="user", content=message))

        # Save user message
        await store.save_message(
            conversation_id,
            ChatMessage(role="user", content=message),
            project_name=project_name,
            audience=audience,
        )
    except DatabaseUnavailableError as e:
        yield {"type": "error", "data": str(e)}
        return

    # ---- Step 3: Stream LLM response ----
    try:
        provider, model = get_active_provider()
    except ValueError as e:
        yield {"type": "error", "data": str(e)}
        return

    full_answer = ""
    try:
        async for token in provider.stream(
            llm_messages,
            model=model,
        ):
            full_answer += token
            yield {"type": "token", "data": token}
    except Exception as e:
        detail = str(e) or repr(e)
        yield {"type": "error", "data": f"LLM error ({type(e).__name__}): {detail}"}
        return

    # Save assistant response
    try:
        await store.save_message(
            conversation_id,
            ChatMessage(
                role="assistant",
                content=full_answer,
                sources=source_dicts,
            ),
            project_name=project_name,
            audience=audience,
        )
    except DatabaseUnavailableError as e:
        yield {"type": "error", "data": str(e)}
        return

    yield {
        "type": "done",
        "data": {
            "conversation_id": conversation_id,
            "answer": full_answer,
        },
    }
