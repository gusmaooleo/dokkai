"""
Multi-provider LLM abstraction layer.

Supports OpenAI, Anthropic, and Ollama (local) through a unified interface.
Each provider implements ``chat()`` (blocking) and ``stream()`` (async
generator) methods.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import AsyncIterator, cast

from openai.types.chat import ChatCompletionMessageParam

import httpx
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

logger = logging.getLogger(__name__)


class LLMMessage:
    """Simple message container for the LLM conversation."""

    __slots__ = ("role", "content")

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    provider_name: str

    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Send messages and return the full response as a string."""
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Send messages and yield response tokens as they arrive."""
        ...

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Return a list of available model names."""
        ...

    @abstractmethod
    async def health_check(self) -> tuple[bool, str]:
        """
        Check connectivity to the provider.
        Returns (is_healthy, message).
        """
        ...


# -----------------------------------------------------------------------
# OpenAI
# -----------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=model,
            messages=[cast(ChatCompletionMessageParam, m.to_dict()) for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
        
    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        response = await self._client.chat.completions.create(
            model=model,
            messages=[cast(ChatCompletionMessageParam, m.to_dict()) for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    _FALLBACK_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "o1",
        "o1-mini",
        "o3-mini",
    ]

    async def list_models(self) -> list[str]:
        try:
            models = await self._client.models.list()
            return sorted(
                m.id for m in models.data
                if "gpt" in m.id or "o1" in m.id or "o3" in m.id
            )
        except Exception as e:
            logger.warning(
                "Failed to fetch live OpenAI model catalog, using static fallback: %s", e
            )
            return sorted(self._FALLBACK_MODELS)

    async def health_check(self) -> tuple[bool, str]:
        try:
            await self._client.models.list()
            return True, "OpenAI API is reachable"
        except Exception as e:
            return False, f"OpenAI API error: {e}"


# -----------------------------------------------------------------------
# Anthropic
# -----------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    def __init__(self, api_key: str) -> None:
        self._client = AsyncAnthropic(api_key=api_key)

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        # Anthropic separates the system message from user/assistant messages
        system_msg, conv_messages = self._split_system(messages)

        response = await self._client.messages.create(
            model=model,
            system=system_msg,
            messages=[cast(MessageParam, m.to_dict()) for m in conv_messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return "".join(block.text for block in response.content if block.type == "text")

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        system_msg, conv_messages = self._split_system(messages)

        async with self._client.messages.stream(
            model=model,
            system=system_msg,
            messages=[cast(MessageParam, m.to_dict()) for m in conv_messages],
            temperature=temperature,
            max_tokens=max_tokens,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    _FALLBACK_MODELS = [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-3-5-haiku-20241022",
    ]

    async def list_models(self) -> list[str]:
        try:
            models = await self._client.models.list(timeout=5.0)
            return sorted([m.id async for m in models])
        except Exception as e:
            logger.warning(
                "Failed to fetch live Anthropic model catalog, using static fallback: %s", e
            )
            return list(self._FALLBACK_MODELS)

    async def health_check(self) -> tuple[bool, str]:
        try:
            # A lightweight message to verify the key works
            await self._client.messages.create(
                model="claude-3-5-haiku-20241022",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
            return True, "Anthropic API is reachable"
        except Exception as e:
            return False, f"Anthropic API error: {e}"

    @staticmethod
    def _split_system(
        messages: list[LLMMessage],
    ) -> tuple[str, list[LLMMessage]]:
        """Extract the system message (if any) from the message list."""
        system = ""
        conv: list[LLMMessage] = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                conv.append(m)
        return system, conv


# -----------------------------------------------------------------------
# Ollama (local)
# -----------------------------------------------------------------------

def _parse_keep_alive(value: str) -> int | str:
    """
    Ollama accepts keep_alive as an int (seconds; -1 = stay loaded forever) or
    a duration string like "30m". A numeric string such as "-1" must be sent as
    an int — sending it as a string makes Ollama reject the request with 400.
    """
    try:
        return int(value)
    except ValueError:
        return value


class OllamaProvider(LLMProvider):
    provider_name = "ollama"

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        # Ollama defaults num_ctx to 2048, which silently truncates the RAG
        # context. Size it for the graph context + system prompt + answer.
        self._num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
        # The first request cold-loads the model into memory, which can take far
        # longer than a normal token read. Short connect, generous read — so a
        # cold load doesn't surface as an opaque ReadTimeout.
        self._timeout = httpx.Timeout(float(os.getenv("OLLAMA_TIMEOUT", "600")), connect=10.0)
        # Keep the model resident in memory between requests. "-1" = stay loaded
        # indefinitely; a duration like "30m" lets Ollama unload it when idle.
        self._keep_alive = _parse_keep_alive(os.getenv("OLLAMA_KEEP_ALIVE", "-1"))

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [m.to_dict() for m in messages],
                    "stream": False,
                    "keep_alive": self._keep_alive,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "num_ctx": self._num_ctx,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        import json as _json

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [m.to_dict() for m in messages],
                    "stream": True,
                    "keep_alive": self._keep_alive,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "num_ctx": self._num_ctx,
                    },
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = _json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except _json.JSONDecodeError:
                        continue

    async def warmup(self, model: str) -> None:
        """
        Preload the model into memory (respecting keep_alive) so the first real
        request isn't a cold start. Best-effort: a 1-token no-op generation.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "ok"}],
                    "stream": False,
                    "keep_alive": self._keep_alive,
                    "options": {"num_predict": 1},
                },
            )
            resp.raise_for_status()

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self._base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]

    async def health_check(self) -> tuple[bool, str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                return True, "Ollama is reachable"
        except Exception as e:
            return False, f"Ollama error: {e}"


# -----------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------

def get_provider(
    provider_name: str,
    *,
    api_key: str = "",
    base_url: str | None = None,
) -> LLMProvider:
    """
    Instantiate an LLM provider by name.

    Parameters
    ----------
    provider_name : str
        One of 'openai', 'anthropic', 'ollama'.
    api_key : str
        API key for remote providers.
    base_url : str, optional
        Custom base URL override (useful for Ollama or OpenAI-compatible APIs).
    """
    name = provider_name.lower().strip()

    if name == "openai":
        if not api_key:
            raise ValueError("OpenAI requires an API key")
        return OpenAIProvider(api_key=api_key, base_url=base_url)

    elif name == "anthropic":
        if not api_key:
            raise ValueError("Anthropic requires an API key")
        return AnthropicProvider(api_key=api_key)

    elif name == "ollama":
        return OllamaProvider(base_url=base_url)

    else:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Supported: openai, anthropic, ollama"
        )
