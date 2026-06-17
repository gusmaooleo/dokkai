"""
Chat persistence layer with storage abstraction.

Stores conversation history as JSON files for now. The ``ChatStore``
interface guarantees a seamless migration to a database later.
"""

from __future__ import annotations

import json
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
        - JsonChatStore (current)
        - DatabaseChatStore (future — same interface, zero code changes)
    """

    @abstractmethod
    def save_message(
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
    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Return a full conversation, or None."""
        ...

    @abstractmethod
    def list_conversations(self) -> list[Conversation]:
        """Return all conversations (without full message bodies for speed)."""
        ...

    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation. Returns True if it existed."""
        ...

    @abstractmethod
    def create_conversation(
        self,
        project_name: str,
        audience: str = "developer",
    ) -> Conversation:
        """Create a new empty conversation and return it."""
        ...


# -----------------------------------------------------------------------
# JSON file-based implementation
# -----------------------------------------------------------------------

class JsonChatStore(ChatStore):
    """
    Stores each conversation as a separate JSON file under ``data_dir``.

    File layout::

        data/conversations/
            <conversation_id>.json
    """

    def __init__(self, data_dir: str | None = None) -> None:
        if data_dir is None:
            base = Path(__file__).resolve().parent.parent.parent
            data_dir = str(base / "data" / "conversations")
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, conversation_id: str) -> Path:
        return self._dir / f"{conversation_id}.json"

    def _load(self, conversation_id: str) -> Conversation | None:
        path = self._path(conversation_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        messages = [ChatMessage(**m) for m in data.get("messages", [])]
        return Conversation(
            conversation_id=data["conversation_id"],
            project_name=data.get("project_name", ""),
            audience=data.get("audience", "developer"),
            messages=messages,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    def _save(self, conv: Conversation) -> None:
        conv.updated_at = datetime.now(timezone.utc).isoformat()
        path = self._path(conv.conversation_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(conv), f, indent=2, ensure_ascii=False)

    def create_conversation(
        self,
        project_name: str,
        audience: str = "developer",
    ) -> Conversation:
        conv = Conversation(
            conversation_id=str(uuid.uuid4()),
            project_name=project_name,
            audience=audience,
        )
        self._save(conv)
        return conv

    def save_message(
        self,
        conversation_id: str,
        message: ChatMessage,
        *,
        project_name: str = "",
        audience: str = "developer",
    ) -> Conversation:
        conv = self._load(conversation_id)
        if conv is None:
            conv = Conversation(
                conversation_id=conversation_id,
                project_name=project_name,
                audience=audience,
            )
        conv.messages.append(message)
        self._save(conv)
        return conv

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        return self._load(conversation_id)

    def list_conversations(self) -> list[Conversation]:
        conversations: list[Conversation] = []
        for path in sorted(self._dir.glob("*.json"), key=os.path.getmtime, reverse=True):
            conv = self._load(path.stem)
            if conv:
                conversations.append(conv)
        return conversations

    def delete_conversation(self, conversation_id: str) -> bool:
        path = self._path(conversation_id)
        if path.exists():
            path.unlink()
            return True
        return False


# -----------------------------------------------------------------------
# Singleton chat store instance
# -----------------------------------------------------------------------

_chat_store: ChatStore = JsonChatStore()


def get_chat_store() -> ChatStore:
    """Return the global chat store instance."""
    return _chat_store


def set_chat_store(store: ChatStore) -> None:
    """Replace the global chat store (e.g., switch to DB-backed)."""
    global _chat_store
    _chat_store = store
