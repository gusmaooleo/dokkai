"""
Chat persistence layer with storage abstraction.

Stores conversation history in Postgres. The ``ChatStore`` interface keeps
the chat service and controllers decoupled from the storage backend.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any

import asyncpg

from services.db import DatabaseUnavailableError, get_pool


@dataclass
class ChatMessage:
    """A single message in a conversation."""

    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class Conversation:
    """A full conversation with metadata."""

    conversation_id: str
    project_name: str
    audience: str = "developer"
    title: str | None = None
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


# -----------------------------------------------------------------------
# Abstract store interface
# -----------------------------------------------------------------------

class ChatStore(ABC):
    """
    Interface for chat persistence.

    Implementations:
        - PostgresChatStore (current)
    """

    @abstractmethod
    async def save_message(
        self,
        conversation_id: str,
        message: ChatMessage,
        *,
        project_name: str = "",
        audience: str = "developer",
    ) -> Conversation:
        """Append a message to a conversation. Create the conversation if new."""
        ...

    @abstractmethod
    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Return a full conversation, or None."""
        ...

    @abstractmethod
    async def list_conversations(self) -> list[Conversation]:
        """Return all conversations (without full message bodies for speed)."""
        ...

    @abstractmethod
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation. Returns True if it existed."""
        ...

    @abstractmethod
    async def create_conversation(
        self,
        project_name: str,
        audience: str = "developer",
    ) -> Conversation:
        """Create a new empty conversation and return it."""
        ...


# -----------------------------------------------------------------------
# Postgres-based implementation
# -----------------------------------------------------------------------

_TITLE_MAX_LEN = 80


def derive_title(text: str, max_len: int = _TITLE_MAX_LEN) -> str:
    """
    Derive a conversation title from a message.

    Collapses internal whitespace/newlines to single spaces, then truncates
    to ``max_len`` characters on a whole-word boundary where possible,
    appending a single-character ellipsis ("…") when truncated.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_len:
        return collapsed

    truncated = collapsed[:max_len]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + "…"


def _row_to_message(row: Any) -> ChatMessage:
    raw_sources = row["sources"]
    sources = json.loads(raw_sources) if isinstance(raw_sources, str) else (raw_sources or [])
    return ChatMessage(
        role=row["role"],
        content=row["content"],
        timestamp=row["created_at"].isoformat(),
        sources=sources,
    )


def _translate_connection_errors(fn):
    """
    ``get_pool()`` only raises ``DatabaseUnavailableError`` when no pool has
    ever been created. Once a pool exists, a query against a since-downed
    Postgres raises a raw connection error instead — normalize that to the
    same ``DatabaseUnavailableError`` so callers only handle one error type.
    """

    @wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except DatabaseUnavailableError:
            raise
        except (OSError, asyncpg.PostgresError) as e:
            raise DatabaseUnavailableError(
                f"cannot reach Postgres — conversation history is unavailable ({e})"
            ) from e

    return wrapper


class PostgresChatStore(ChatStore):
    """Stores conversations and messages in Postgres via ``services.db.get_pool()``."""

    def _parse_id(self, conversation_id: str) -> uuid.UUID | None:
        try:
            return uuid.UUID(conversation_id)
        except (ValueError, AttributeError, TypeError):
            return None

    @_translate_connection_errors
    async def create_conversation(
        self,
        project_name: str,
        audience: str = "developer",
    ) -> Conversation:
        pool = await get_pool()
        conv_id = uuid.uuid4()
        row = await pool.fetchrow(
            """
            INSERT INTO conversations (id, project_name, audience)
            VALUES ($1, $2, $3)
            RETURNING id, project_name, audience, title, created_at, updated_at
            """,
            conv_id,
            project_name,
            audience,
        )
        return Conversation(
            conversation_id=str(row["id"]),
            project_name=row["project_name"],
            audience=row["audience"],
            title=row["title"],
            messages=[],
            created_at=row["created_at"].isoformat(),
            updated_at=row["updated_at"].isoformat(),
        )

    @_translate_connection_errors
    async def save_message(
        self,
        conversation_id: str,
        message: ChatMessage,
        *,
        project_name: str = "",
        audience: str = "developer",
    ) -> Conversation:
        pool = await get_pool()
        conv_uuid = uuid.UUID(conversation_id)

        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO conversations (id, project_name, audience, title)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    conv_uuid,
                    project_name,
                    audience,
                    derive_title(message.content) if message.role == "user" else None,
                )
                await conn.execute(
                    """
                    INSERT INTO messages (conversation_id, role, content, sources)
                    VALUES ($1, $2, $3, $4::jsonb)
                    """,
                    conv_uuid,
                    message.role,
                    message.content,
                    _dump_sources(message.sources),
                )
                row = await conn.fetchrow(
                    """
                    UPDATE conversations SET updated_at = now()
                    WHERE id = $1
                    RETURNING id, project_name, audience, created_at, updated_at
                    """,
                    conv_uuid,
                )

        conv = await self.get_conversation(str(row["id"]))
        assert conv is not None
        return conv

    @_translate_connection_errors
    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        conv_uuid = self._parse_id(conversation_id)
        if conv_uuid is None:
            return None

        pool = await get_pool()
        conv_row = await pool.fetchrow(
            """
            SELECT id, project_name, audience, title, created_at, updated_at
            FROM conversations WHERE id = $1
            """,
            conv_uuid,
        )
        if conv_row is None:
            return None

        message_rows = await pool.fetch(
            """
            SELECT role, content, sources, created_at
            FROM messages WHERE conversation_id = $1
            ORDER BY id
            """,
            conv_uuid,
        )

        return Conversation(
            conversation_id=str(conv_row["id"]),
            project_name=conv_row["project_name"],
            audience=conv_row["audience"],
            title=conv_row["title"],
            messages=[_row_to_message(r) for r in message_rows],
            created_at=conv_row["created_at"].isoformat(),
            updated_at=conv_row["updated_at"].isoformat(),
        )

    @_translate_connection_errors
    async def list_conversations(self) -> list[Conversation]:
        pool = await get_pool()
        conv_rows = await pool.fetch(
            """
            SELECT id, project_name, audience, title, created_at, updated_at
            FROM conversations ORDER BY updated_at DESC
            """
        )
        message_rows = await pool.fetch(
            """
            SELECT conversation_id, role, content, sources, created_at
            FROM messages ORDER BY conversation_id, id
            """
        )

        messages_by_conv: dict[uuid.UUID, list[ChatMessage]] = defaultdict(list)
        for row in message_rows:
            messages_by_conv[row["conversation_id"]].append(_row_to_message(row))

        return [
            Conversation(
                conversation_id=str(row["id"]),
                project_name=row["project_name"],
                audience=row["audience"],
                title=row["title"],
                messages=messages_by_conv.get(row["id"], []),
                created_at=row["created_at"].isoformat(),
                updated_at=row["updated_at"].isoformat(),
            )
            for row in conv_rows
        ]

    @_translate_connection_errors
    async def delete_conversation(self, conversation_id: str) -> bool:
        conv_uuid = self._parse_id(conversation_id)
        if conv_uuid is None:
            return False

        pool = await get_pool()
        result = await pool.execute(
            "DELETE FROM conversations WHERE id = $1", conv_uuid
        )
        return result == "DELETE 1"


def _dump_sources(sources: list[dict[str, Any]]) -> str:
    return json.dumps(sources, ensure_ascii=False)


# -----------------------------------------------------------------------
# Singleton chat store instance
# -----------------------------------------------------------------------

_chat_store: ChatStore = PostgresChatStore()


def get_chat_store() -> ChatStore:
    """Return the global chat store instance."""
    return _chat_store


def set_chat_store(store: ChatStore) -> None:
    """Replace the global chat store (e.g., switch to DB-backed)."""
    global _chat_store
    _chat_store = store
