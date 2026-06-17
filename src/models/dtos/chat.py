"""
DTOs for the chat endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from enum import Enum


class AudienceRole(str, Enum):
    DEVELOPER = "developer"
    MANAGER = "manager"
    CUSTOMER = "customer"


class ChatMessageDTO(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    message: str
    project_name: str
    audience: AudienceRole = AudienceRole.DEVELOPER
    conversation_id: str | None = None
    top_k: int = Field(default=8, ge=1, le=50)


class SourceChunk(BaseModel):
    entity_type: str
    name: str
    qualified_name: str
    file_path: str
    start_line: int | None = None
    end_line: int | None = None
    chunk_text: str
    score: float | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[SourceChunk]


class ConversationSummary(BaseModel):
    conversation_id: str
    project_name: str
    message_count: int
    last_message_preview: str
    created_at: str
    updated_at: str
