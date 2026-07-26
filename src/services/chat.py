"""
Chat / RAG service — orchestrates retrieval + LLM generation.

1. Retrieve relevant code chunks from Weaviate via the Retriever
2. Build an augmented prompt with audience-specific system instructions
3. Stream the LLM response
4. Persist messages to the chat store
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import AsyncIterator

from services.retriever import Retriever, RetrievedChunk, _looks_like_identifier
from services.llm_provider import LLMMessage
from services.llm_config import get_active_provider
from services.chat_store import get_chat_store, ChatMessage
from services.db import DatabaseUnavailableError
from services.weaviate_client import get_client

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Prompt budget config
# -----------------------------------------------------------------------
# Ollama's context window (num_ctx) bounds the FULL prompt (system + history
# + current message) in tokens; overflow is truncated from the FRONT, which
# eats the retrieved context first. OpenAI/Anthropic don't expose a window
# here, so a conservative fixed budget stands in for them. Token counts are
# ESTIMATED as chars/4 — the same rough heuristic the MCP watchdog uses, not
# an exact tokenizer count, but cheap enough to budget prompts with.
_CHARS_PER_TOKEN = 4
_REMOTE_CTX_TOKENS = 100_000  # conservative window for providers that don't report one
_RESERVED_OUTPUT_TOKENS = 4096 + 300  # provider's default max_tokens + a small safety margin
_MIN_CONTEXT_CHARS = 2_000  # floor for the retrieved-context budget, even under extreme pressure
_DEFAULT_CONTEXT_MAX_CHARS = 16_000  # Retriever.build_graph_context's own default; never exceed it
_HISTORY_RESERVE_TOKENS_CAP = 1_200  # ceiling on the slice guaranteed to history, before context is sized

# History-aware retrieval query (see ``_retrieval_query``).
_MAX_RETRIEVAL_HISTORY_TURNS = 2
_MAX_RETRIEVAL_QUERY_CHARS = 600


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (chars/4) — see the prompt-budget config comment."""
    return len(text) // _CHARS_PER_TOKEN


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
# Prompt assembly (pure, testable — see scripts/test_chat_budget.py)
# -----------------------------------------------------------------------

# Fixed chars the system message adds around the retrieved context.
_CONTEXT_WRAPPER_OVERHEAD = "\n\n=== CODEBASE CONTEXT ===\n\n=== END CONTEXT ==="


def _fitted_context_chars(
    budget_tokens: int, system_prompt: str, message: str, *, has_history: bool
) -> int:
    """
    Size the retrieved-context budget (the ``max_chars`` handed to
    ``build_graph_context``) within ``budget_tokens``.

    A slice is reserved for history BEFORE context is sized — sizing context
    against the full budget left nothing for history at ordinary
    OLLAMA_NUM_CTX values (the history window was dead code). The honest
    priority: retrieved context gets whatever remains AFTER a guaranteed
    history slice — context > recent history > old history. A first turn has
    no history, so no slice is carved out for it (~26% more context on
    single-turn questions at the 8192 default). Floors at
    ``_MIN_CONTEXT_CHARS`` with a warning; the prompt may then still exceed
    the budget, and ``_fit_messages`` fails open downstream.
    """
    history_reserve_tokens = (
        min(budget_tokens // 4, _HISTORY_RESERVE_TOKENS_CAP) if has_history else 0
    )
    current_tokens = _estimate_tokens(message)
    fixed_overhead_tokens = _estimate_tokens(system_prompt + _CONTEXT_WRAPPER_OVERHEAD)
    fitted = min(
        _DEFAULT_CONTEXT_MAX_CHARS,
        (budget_tokens - history_reserve_tokens - current_tokens - fixed_overhead_tokens)
        * _CHARS_PER_TOKEN,
    )
    if fitted < _MIN_CONTEXT_CHARS:
        logger.warning(
            "Retrieved-context budget too small for a %d-token prompt window "
            "(fitted %d chars) — flooring to %d chars; the system prompt may "
            "still exceed the budget.",
            budget_tokens, fitted, _MIN_CONTEXT_CHARS,
        )
        fitted = _MIN_CONTEXT_CHARS
    return fitted


def _fit_messages(
    system_text: str,
    history: list[LLMMessage],
    current_text: str,
    budget_tokens: int,
) -> tuple[list[LLMMessage], int]:
    """
    Assemble the final provider message list within ``budget_tokens``.

    Priority:
    - The system message and the current user message are PROTECTED — never
      dropped, even if they alone exceed the budget. That's a degenerate,
      fail-open case: send them anyway (with all history dropped) and log a
      warning; the provider does its own truncation rather than dokkai
      crashing or shipping an empty request.
    - History fills whatever budget remains via a sliding window: walk
      newest-first, keeping whole messages while they still fit, then STOP
      at the first one that doesn't (older turns fall off; a message is
      never split in half). Output preserves chronological order. This
      function does NOT itself guarantee that "whatever remains" is
      non-trivial — the system message can consume nearly all of
      ``budget_tokens`` if the caller sizes it that way. ``chat_with_
      codebase`` carves out a history slice (``history_reserve_tokens``)
      BEFORE sizing the retrieved context specifically so this window has
      something real to work with in practice; see its comment for the
      resulting priority order (context > recent history > old history).
    - A window that would START with an assistant turn (possible after the
      cut above lands mid-conversation) is invalid for providers that
      require the first non-system message to be from the user (e.g.
      Anthropic's Messages API 400s otherwise) — that leading assistant
      message is dropped too.

    Returns the assembled messages plus how many history messages were
    dropped (for logging).
    """
    system_msg = LLMMessage(role="system", content=system_text)
    current_msg = LLMMessage(role="user", content=current_text)

    protected_tokens = _estimate_tokens(system_text) + _estimate_tokens(current_text)
    remaining = budget_tokens - protected_tokens

    if remaining <= 0:
        logger.warning(
            "Prompt budget exhausted by the system+current message alone "
            "(~%d tokens estimated, budget %d) — dropping all %d history "
            "message(s); the provider may still truncate this prompt.",
            protected_tokens, budget_tokens, len(history),
        )
        return [system_msg, current_msg], len(history)

    kept_reversed: list[LLMMessage] = []
    used = 0
    for msg in reversed(history):
        cost = _estimate_tokens(msg.content)
        if used + cost > remaining:
            break
        kept_reversed.append(msg)
        used += cost

    kept = list(reversed(kept_reversed))
    if kept and kept[0].role == "assistant":
        # The oldest kept turn is an assistant reply with no preceding user
        # turn in the window — drop it so the conversation still opens on a
        # user message.
        kept = kept[1:]
    dropped = len(history) - len(kept)
    messages = [system_msg, *kept, current_msg]

    if dropped > 0:
        # NOTE: uvicorn's default logging config has no root handler for app
        # loggers (lastResort floor is WARNING), so this INFO line is
        # currently invisible in that setup — pre-existing repo-wide gap,
        # not escalated to WARNING here since it would fire on every long
        # conversation (noise, not a real problem).
        logger.info(
            "Dropped %d oldest history message(s) to fit the %d-token prompt "
            "budget (kept %d).",
            dropped, budget_tokens, len(kept),
        )

    return messages, dropped


def _retrieval_query(history: list[ChatMessage], current: str) -> str:
    """
    Build the query text used for RETRIEVAL only (the LLM prompt still uses
    the real ``current`` message unchanged) — deterministic, no LLM call.

    Anaphoric follow-ups ("and where is it validated?") are underspecified
    on their own; prepending the last couple of user turns gives retrieval a
    referent. Assistant turns are excluded — they're long and already echo
    the retrieved context back, so they'd dilute the query rather than add
    signal.
    """
    if _looks_like_identifier(current):
        # Preserves the retriever's own identifier fast path (C2) — an exact
        # qualified_name lookup should never be diluted with history.
        return current

    prev_user = [
        m.content for m in history if m.role == "user" and m.content.strip()
    ][-_MAX_RETRIEVAL_HISTORY_TURNS:]

    # Drop older extras first (oldest of the kept turns), then — if even the
    # bare current message doesn't fit — truncate the FRONT of the whole
    # concatenation, since the tail (the current message) matters most.
    while prev_user:
        candidate = "\n".join([*prev_user, current])
        if len(candidate) <= _MAX_RETRIEVAL_QUERY_CHARS:
            return candidate
        prev_user.pop(0)

    if len(current) > _MAX_RETRIEVAL_QUERY_CHARS:
        return current[-_MAX_RETRIEVAL_QUERY_CHARS:]
    return current


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
            conv = None
            conversation_id = str(uuid.uuid4())
    except DatabaseUnavailableError as e:
        yield {"type": "error", "data": str(e)}
        return

    history: list[ChatMessage] = conv.messages if conv else []

    # Resolve the provider up front: its context window drives both the
    # retrieved-context size (Step 1) and the history window (Step 2) below.
    try:
        provider, model = get_active_provider()
    except ValueError as e:
        yield {"type": "error", "data": str(e)}
        return

    # ``is None`` (never ``or``): a provider may legitimately report a small
    # window, and 0 must not be treated as "unknown". Ollama reports a
    # conservative default when OLLAMA_NUM_CTX <= 0 (see its
    # ``context_window``); ``None`` here means a remote provider that
    # doesn't report one — those get the large conservative constant.
    context_window = provider.context_window
    budget_tokens = (
        context_window if context_window is not None else _REMOTE_CTX_TOKENS
    ) - _RESERVED_OUTPUT_TOKENS

    # History-aware retrieval query (defect B) — a follow-up like "and where
    # is it validated?" is underspecified on its own.
    retrieval_query = _retrieval_query(history, message)

    # Fit the retrieved-context size to the budget BEFORE building it
    # (defect A): if the system message (audience prompt + context) alone
    # would already exceed the budget, Ollama truncates the prompt from the
    # FRONT — eating the retrieved context itself. Sizing max_chars here is
    # the actual overflow kill; ``_fit_messages`` below only handles the
    # history window and the (now rare) degenerate case.
    system_prompt = _get_system_prompt(audience)
    fitted_max_chars = _fitted_context_chars(
        budget_tokens, system_prompt, message, has_history=bool(history)
    )

    # ---- Step 1: Retrieve relevant code (graph-expanded) ----
    # The Weaviate client is synchronous (blocking gRPC), so run the whole
    # retrieval off the event loop — otherwise it freezes every other request
    # (concurrent chats, job-status polls) for the full retrieval window.
    def _retrieve() -> tuple[list[RetrievedChunk], str]:
        client = get_client()
        try:
            retriever = Retriever(client)
            chunks = retriever.search_graph(
                query=retrieval_query,
                project_name=project_name,
                top_k_seeds=top_k,
                max_hops=max_hops,
                max_nodes=max_nodes,
                direction=direction,
            )
            return chunks, retriever.build_graph_context(chunks, max_chars=fitted_max_chars)
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
    system_with_context = (
        f"{system_prompt}\n\n"
        f"=== CODEBASE CONTEXT ===\n{context}\n=== END CONTEXT ==="
    )

    # Fit history + system + current message into the prompt budget
    # (defect A) — a sliding window over history, oldest turns dropped first.
    history_messages = [
        LLMMessage(role=msg.role, content=msg.content)
        for msg in history
        if msg.role in ("user", "assistant")
    ]
    llm_messages, _dropped = _fit_messages(
        system_with_context, history_messages, message, budget_tokens,
    )

    # Save user message
    try:
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
