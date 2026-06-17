"""
Chat API endpoints.

POST   /chat                        — send a message, get SSE streaming response
GET    /chat/conversations           — list all conversations
GET    /chat/conversations/{id}      — get a conversation's history
DELETE /chat/conversations/{id}      — delete a conversation
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from models.dtos.chat import (
    ChatRequest,
    ChatResponse,
    SourceChunk,
    ConversationSummary,
)
from services.chat import chat_with_codebase
from services.chat_store import get_chat_store

router = APIRouter(prefix="/chat")


@router.post("")
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
    conversations = store.list_conversations()

    return [
        ConversationSummary(
            conversation_id=conv.conversation_id,
            project_name=conv.project_name,
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
    conv = store.get_conversation(conversation_id)

    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "conversation_id": conv.conversation_id,
        "project_name": conv.project_name,
        "audience": conv.audience,
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


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and its history."""
    store = get_chat_store()
    deleted = store.delete_conversation(conversation_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"message": "Conversation deleted", "conversation_id": conversation_id}
