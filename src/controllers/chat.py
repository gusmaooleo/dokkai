"""
Chat API endpoints.

POST   /chat                        — send a message, get SSE streaming response
GET    /chat/conversations           — list all conversations
GET    /chat/conversations/{id}      — get a conversation's history
DELETE /chat/conversations/{id}      — delete a conversation
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from models.dtos.chat import (
    ChatRequest,
    ChatResponse,
    SourceChunk,
    ConversationSummary,
)
from services.auth import require_auth, require_role
from services.chat import chat_with_codebase
from services.chat_store import get_chat_store
from services.db import DatabaseUnavailableError

router = APIRouter(prefix="/chat", dependencies=[Depends(require_auth)])

# Sending a message / deleting a conversation mutate state — role 'viewer'
# is read-only (6k).
require_write = require_role("admin", "user")


@router.post("", dependencies=[Depends(require_write)])
async def chat(data: ChatRequest):
    """
    Send a message and receive a Server-Sent Events (SSE) streaming response.

    Event types:
    - ``sources``  — retrieved code chunks used as context
    - ``token``    — a single streaming token from the LLM
    - ``done``     — final event with the full answer and conversation_id
    - ``error``    — an error occurred during processing
    """

    async def event_stream():
        async for event in chat_with_codebase(
            message=data.message,
            project_name=data.project_name,
            audience=data.audience.value,
            conversation_id=data.conversation_id,
            top_k=data.top_k,
        ):
            event_type = event["type"]
            event_data = json.dumps(event["data"], ensure_ascii=False)
            yield f"event: {event_type}\ndata: {event_data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations():
    """List all stored conversations."""
    store = get_chat_store()
    try:
        conversations = await store.list_conversations()
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return [
        ConversationSummary(
            conversation_id=conv.conversation_id,
            project_name=conv.project_name,
            audience=conv.audience,
            title=conv.title,
            message_count=len(conv.messages),
            last_message_preview=(
                conv.messages[-1].content[:120] + "..."
                if conv.messages and len(conv.messages[-1].content) > 120
                else conv.messages[-1].content if conv.messages else ""
            ),
            created_at=conv.created_at,
            updated_at=conv.updated_at,
        )
        for conv in conversations
    ]


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get the full message history for a conversation."""
    store = get_chat_store()
    try:
        conv = await store.get_conversation(conversation_id)
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "conversation_id": conv.conversation_id,
        "project_name": conv.project_name,
        "audience": conv.audience,
        "title": conv.title,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "sources": m.sources,
            }
            for m in conv.messages
        ],
    }


@router.delete("/conversations/{conversation_id}", dependencies=[Depends(require_write)])
async def delete_conversation(conversation_id: str):
    """Delete a conversation and its history."""
    store = get_chat_store()
    try:
        deleted = await store.delete_conversation(conversation_id)
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"message": "Conversation deleted", "conversation_id": conversation_id}
