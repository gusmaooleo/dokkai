"""
Multi-provider LLM abstraction layer.

Providers are config entries, not bespoke classes (feature 22): a handful
of API "shapes" are implemented once, and any provider — built-in or
custom — is served by pointing a shape at a base URL and a key. This
mirrors OpenClaw's provider-registry pattern and needs zero new
dependencies: both shapes ride SDKs already in ``pyproject.toml``.

- ``openai-completions`` — the ``openai`` SDK with a ``base_url``
  override. Serves OpenAI, Gemini's OpenAI-compatible surface, and any
  other OpenAI-compatible endpoint (Groq, Together, OpenRouter, LM
  Studio, vLLM, ...).
- ``anthropic-messages`` — the ``anthropic`` SDK's Messages API.
- Ollama (local) keeps its own native path — it isn't one of the two
  shapes above, so it's handled directly rather than routed through one.

Each provider implements ``chat()`` (blocking) and ``stream()`` (async
generator) methods.

Embeddings are out of scope here — dokkai's vector search stays
Weaviate-native (``text2vec-ollama``); this module only covers chat-style
completions (the describe pass and the chat/routines answer step).
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, cast
from urllib.parse import urlsplit, urlunsplit

from openai.types.chat import ChatCompletionMessageParam

import httpx
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

logger = logging.getLogger(__name__)


def effective_base_url(url: str) -> str:
    """
    Rewrite a ``localhost``/``127.0.0.1`` host in *url* to
    ``host.docker.internal`` when running inside the dokkai API container
    (``DOKKAI_IN_CONTAINER=1``).

    A persisted LLM config's ``base_url`` (saved e.g. via the UI while
    running ``./dev.sh`` on the host) points at "localhost", which inside a
    container means the container itself — not the host machine running
    Ollama. This rewrite is applied at USE time, where the base_url is about
    to feed an HTTP call (see ``OllamaProvider.__init__``); the persisted
    value itself is never rewritten. URLs that already resolve correctly
    (a non-localhost host, e.g. a remote provider or an env default of
    ``host.docker.internal``) pass through unchanged.
    """
    if os.getenv("DOKKAI_IN_CONTAINER", "").lower() not in ("1", "true", "yes"):
        return url

    parsed = urlsplit(url)
    if parsed.hostname not in ("localhost", "127.0.0.1"):
        return url

    netloc = "host.docker.internal"
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username:
        userinfo = parsed.username + (f":{parsed.password}" if parsed.password else "")
        netloc = f"{userinfo}@{netloc}"

    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


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

    @property
    def context_window(self) -> int | None:
        """
        Input context window in tokens, when the provider reports/configures
        one. ``None`` for remote providers (OpenAI/Anthropic) — dokkai
        doesn't track their per-model limits here, so callers fall back to a
        conservative constant (see ``services.chat._REMOTE_CTX_TOKENS``).
        """
        return None

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
    async def health_check(self, model: str | None = None) -> tuple[bool, str]:
        """
        Check connectivity to the provider. When ``model`` is given, also
        verify that the model exists. Returns (is_healthy, message).
        """
        ...


# -----------------------------------------------------------------------
# openai-completions shape (OpenAI, Gemini's OpenAI-compat surface, and
# any other OpenAI-compatible endpoint)
# -----------------------------------------------------------------------

class OpenAICompletionsProvider(LLMProvider):
    """
    The ``openai-completions`` shape: the ``openai`` SDK pointed at a
    ``base_url``. One class, many providers — ``provider_name``/
    ``display_name`` only affect model-catalog filtering and
    human-readable messages, not the request wire format.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        provider_name: str,
        display_name: str,
    ) -> None:
        self.provider_name = provider_name
        self._display_name = display_name
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

    async def list_models(self) -> list[str]:
        try:
            models = await self._client.models.list()
            ids = [m.id for m in models.data]
            if self.provider_name == "openai":
                # The live OpenAI catalog includes non-chat models
                # (embeddings, TTS, etc.) that need filtering out.
                ids = [i for i in ids if "gpt" in i or "o1" in i or "o3" in i]
            elif self.provider_name == "gemini":
                # Gemini's catalog ids are prefixed "models/" (confirmed
                # live: e.g. "models/gemini-flash-latest"), which the
                # chat/completions body and the persisted config accept,
                # but which makes models.retrieve() build a double
                # "models/models/..." URL (it appends its own "models/"
                # segment) and disagrees with the unprefixed static
                # fallback catalog below. Strip it so both agree. Gemini's
                # /models also mirrors its FULL catalog (embeddings,
                # imagen/veo/lyria generation, TTS, ...), not just
                # chat-capable models — denylist the known non-chat
                # modalities; this is a minimal filter, not a verified
                # allowlist, so an unrecognized non-chat id could still
                # slip through.
                ids = [i.removeprefix("models/") for i in ids]
                ids = [
                    i for i in ids
                    if not any(marker in i for marker in _GEMINI_NON_CHAT_MARKERS)
                ]
            return sorted(ids)
        except Exception as e:
            logger.warning(
                "Failed to fetch live %s model catalog, using static fallback: %s",
                self._display_name, e,
            )
            return sorted(_FALLBACK_MODELS.get(self.provider_name, []))

    async def health_check(self, model: str | None = None) -> tuple[bool, str]:
        try:
            if model:
                await self._client.models.retrieve(model)
                return True, f"Model '{model}' is available"
            await self._client.models.list()
            return True, f"{self._display_name} API is reachable"
        except Exception as e:
            if model:
                return False, f"{self._display_name} model '{model}' error: {e}"
            return False, f"{self._display_name} API error: {e}"


# -----------------------------------------------------------------------
# anthropic-messages shape
# -----------------------------------------------------------------------

class AnthropicMessagesProvider(LLMProvider):
    """The ``anthropic-messages`` shape: the ``anthropic`` SDK's Messages API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        provider_name: str,
        display_name: str,
    ) -> None:
        self.provider_name = provider_name
        self._display_name = display_name
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url)

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

    async def list_models(self) -> list[str]:
        try:
            models = await self._client.models.list(timeout=5.0)
            return sorted([m.id async for m in models])
        except Exception as e:
            logger.warning(
                "Failed to fetch live %s model catalog, using static fallback: %s",
                self._display_name, e,
            )
            return list(_FALLBACK_MODELS.get(self.provider_name, []))

    async def health_check(self, model: str | None = None) -> tuple[bool, str]:
        try:
            if model:
                await self._client.models.retrieve(model)
                return True, f"Model '{model}' is available"
            await self._client.models.list()
            return True, f"{self._display_name} API is reachable"
        except Exception as e:
            if model:
                return False, f"{self._display_name} model '{model}' error: {e}"
            return False, f"{self._display_name} API error: {e}"

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
        self._base_url = effective_base_url(
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

    @property
    def context_window(self) -> int | None:
        # 0 (or negative) is a plausible OLLAMA_NUM_CTX value meaning "let
        # Ollama pick its own default". Returning None here would send the
        # caller to the large remote-provider fallback — a phantom window
        # that re-enables the exact overflow the budget exists to kill. The
        # server's own default is SMALL (2k–4k depending on version/model),
        # so budget conservatively against 4096; set OLLAMA_NUM_CTX
        # explicitly for a bigger window. (Note: the request still sends the
        # user's literal num_ctx value — 0 delegates to the server.)
        if self._num_ctx <= 0:
            return 4096
        return self._num_ctx

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

    async def chat_with_tools(
        self,
        messages: list[dict],
        *,
        model: str,
        tools: list[dict] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict:
        """
        Ollama-native tool-calling chat call (decision 9a — v1 is Ollama-only;
        NOT part of the :class:`LLMProvider` ABC, and not implemented by the
        other providers).

        Unlike :meth:`chat`, this takes and returns raw Ollama message dicts
        (``{"role", "content", "tool_calls"?}``) rather than
        :class:`LLMMessage` — tool-role messages and ``tool_calls`` have no
        equivalent in the ``LLMMessage``/single-string-reply abstraction the
        other providers share, and the agent loop (``services.routines.
        agent_loop``) needs the raw shape to append tool results and read
        ``message.tool_calls`` back. ``tools`` follows Ollama's ``/api/chat``
        tool schema (list of ``{"type": "function", "function": {...}}``);
        omitted/empty disables tool-calling for that call (used by the agent
        loop's forced final-answer round).
        """
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self._num_ctx,
            },
        }
        if tools:
            payload["tools"] = tools
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message") or {"role": "assistant", "content": ""}

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

    async def health_check(self, model: str | None = None) -> tuple[bool, str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                if model:
                    available = [m["name"] for m in resp.json().get("models", [])]
                    matched = model in available or any(
                        a.split(":")[0] == model for a in available
                    )
                    if not matched:
                        return False, f"model '{model}' unavailable"
                    return True, f"Model '{model}' is available"
                return True, "Ollama is reachable"
        except Exception as e:
            return False, f"Ollama error: {e}"


# -----------------------------------------------------------------------
# Built-in registry — provider id -> {shape, default base URL, key env,
# display name}. This is the "config entry, not a class" from the module
# docstring: registering a new built-in is adding a row here, not writing
# a new provider class.
# -----------------------------------------------------------------------

_SHAPE_OPENAI_COMPLETIONS = "openai-completions"
_SHAPE_ANTHROPIC_MESSAGES = "anthropic-messages"


@dataclass(frozen=True)
class _ProviderSpec:
    shape: str
    # None => don't override; let the SDK use its own default (the real
    # OpenAI API for the openai-completions shape).
    default_base_url: str | None
    # Which env var the key normally lives in. get_provider itself never
    # reads env vars — callers resolve and pass ``api_key`` explicitly —
    # so this is used only to (a) decide whether a key is mandatory for
    # this provider and (b) name the var in the "requires an API key"
    # message below. None for custom (non-built-in) ids: many
    # OpenAI-compatible local servers (LM Studio, vLLM, ...) don't
    # require a key at all.
    key_env: str | None
    display_name: str


# Ollama is deliberately not in this table: it's not one of the two
# shapes above (its own native path, see OllamaProvider) and is handled
# directly in get_provider().
_REGISTRY: dict[str, _ProviderSpec] = {
    "openai": _ProviderSpec(
        shape=_SHAPE_OPENAI_COMPLETIONS,
        default_base_url=None,
        key_env="OPENAI_API_KEY",
        display_name="OpenAI",
    ),
    "gemini": _ProviderSpec(
        shape=_SHAPE_OPENAI_COMPLETIONS,
        # Gemini's OpenAI-compatible surface. Trailing slash required —
        # not an openai-SDK requirement (it normalizes base URLs with or
        # without one); Gemini's own compat-endpoint routing needs it.
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_env="GEMINI_API_KEY",
        display_name="Gemini",
    ),
    "anthropic": _ProviderSpec(
        shape=_SHAPE_ANTHROPIC_MESSAGES,
        default_base_url=None,
        key_env="ANTHROPIC_API_KEY",
        display_name="Anthropic",
    ),
}

# Minimal denylist of substrings marking a Gemini catalog id as a known
# non-chat modality (embeddings, image/video/music generation, TTS, the
# QA-only "aqa" model) — see the comment in
# OpenAICompletionsProvider.list_models(). Not a verified allowlist: an
# id that matches none of these still isn't guaranteed to support
# chat/completions, only known *not* to be one of these.
_GEMINI_NON_CHAT_MARKERS = ("embedding", "imagen", "veo-", "lyria", "tts", "aqa")

# Static model-catalog fallbacks (feature 10), keyed by provider id — used
# by list_models() when the live catalog call fails.
_FALLBACK_MODELS: dict[str, list[str]] = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "o1",
        "o1-mini",
        "o3-mini",
    ],
    "gemini": [
        "gemini-pro-latest",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
    ],
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-3-5-haiku-20241022",
    ],
}


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
    Instantiate an LLM provider by id.

    ``provider_name`` resolves through the built-in registry (openai,
    gemini, anthropic, ollama). An id that isn't built in is still
    servable — through the openai-completions shape — as long as
    ``base_url`` is given, which covers any OpenAI-compatible endpoint
    (Groq, Together, OpenRouter, LM Studio, vLLM, ...).

    Parameters
    ----------
    provider_name : str
        A built-in id ('openai', 'gemini', 'anthropic', 'ollama') or a
        custom id paired with ``base_url``.
    api_key : str
        API key for remote providers. Required for built-in remotes
        (openai/gemini/anthropic); optional for a custom endpoint.
    base_url : str, optional
        Base URL override. Required for a custom (non-built-in) id.
    """
    name = provider_name.lower().strip()

    if name == "ollama":
        return OllamaProvider(base_url=base_url)

    spec = _REGISTRY.get(name)
    if spec is None:
        if not base_url:
            builtins = ", ".join(sorted(list(_REGISTRY) + ["ollama"]))
            raise ValueError(
                f"Unknown provider '{provider_name}'. Built-in providers: "
                f"{builtins}. A custom provider needs a base_url pointing "
                f"at an OpenAI-compatible endpoint."
            )
        spec = _ProviderSpec(
            shape=_SHAPE_OPENAI_COMPLETIONS,
            default_base_url=None,
            key_env=None,
            display_name=provider_name.strip(),
        )

    if spec.key_env and not api_key:
        raise ValueError(f"{spec.display_name} requires an API key")

    resolved_base_url = base_url or spec.default_base_url

    if spec.shape == _SHAPE_ANTHROPIC_MESSAGES:
        return AnthropicMessagesProvider(
            api_key=api_key,
            base_url=resolved_base_url,
            provider_name=name,
            display_name=spec.display_name,
        )

    return OpenAICompletionsProvider(
        api_key=api_key,
        base_url=resolved_base_url,
        provider_name=name,
        display_name=spec.display_name,
    )
