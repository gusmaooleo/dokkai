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

Beyond the three built-ins below, ``config/providers.json`` (feature 22,
C3) registers additional providers — id, shape, base URL, key — without
code, loaded once into the same registry at import time. See the "File-
registered providers" section near the bottom for the loader and its
rules (unknown shape, built-in shadowing, malformed JSON, ``${ENV}`` key
interpolation).
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
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


# Matches the userinfo segment of a URL's netloc (``scheme://user:pass@``)
# — greedy up to the LAST ``@`` before the next ``/``/whitespace/end of
# string, mirroring how a real URL parser resolves the userinfo boundary
# (an unencoded ``@`` inside a password is technically invalid URL syntax,
# but this still redacts up to the rightmost one rather than stopping at
# the first, leaking the remainder). Never matches when there's no ``@``
# before the first path segment — an ordinary host-only URL is untouched.
_USERINFO_RE = re.compile(r"://[^/\s]*@")


def _redact_userinfo(text: str) -> str:
    """
    Redact an embedded userinfo credential (``scheme://user:pass@host``)
    from *text* — used to scrub ``validate_base_url``'s error messages
    before they're returned, never the successfully-validated URL itself
    (which the caller still needs, intact, to actually connect/persist).

    ``validate_base_url`` is shared by three entry points that each put a
    raw, caller-supplied ``base_url`` straight into a returned error
    string — the ``/config/llm`` POST body, the ``config/providers.json``
    loader (feature 22, C3), and ``DESC_BASE_URL`` (C4) — any of which can
    read back to the caller (an HTTP 400 body, a boot-time traceback, a
    job log). A malformed value with embedded Basic-auth credentials
    (``https://bob:secret@`` — hostless, so it fails validation) must not
    echo the password back in the rejection message.
    """
    return _USERINFO_RE.sub("://***@", text)


def validate_base_url(raw: str) -> tuple[str | None, str]:
    """
    Normalize and validate a remote-provider ``base_url``.

    Shared by both entry points that accept a caller-supplied base URL:
    ``services.llm_config.validate_and_save_config`` (the ``/config/llm``
    POST body) and this module's own ``config/providers.json`` loader
    (feature 22, C3) — one validator, so a scheme-less URL or an embedded
    control character is rejected the same way whether it arrived via the
    API or the file, instead of the file's ``baseUrl`` bypassing this
    entirely and failing opaquely later (an unhandled error out of
    ``health_check`` or an SDK call, instead of a clean config-time
    rejection).

    Strips surrounding whitespace — ``urlsplit`` does not, so a pasted
    URL with a trailing newline would otherwise validate cleanly here and
    then die inside the SDK with ``InvalidURL: Invalid non-printable
    ASCII character`` — and requires a full ``http``/``https`` URL with a
    non-empty host. Rejects, before any network call:

    - a scheme-less URL (``localhost:1234/v1``);
    - a hostless one, including the single-slash typo
      (``http:/x``, ``http://``, ``http:///v1``, ``http:/localhost:1234/v1``);
    - anything ``urlsplit`` itself refuses to parse, e.g. an unbalanced
      IPv6 bracket (``http://[::1:8000/v1``), which raises ``ValueError``
      rather than returning a result;
    - a tab/CR/LF anywhere INSIDE the string. ``urlsplit`` deletes those
      characters before parsing while ``.strip()`` only touches the ends,
      so ``ht\\ntp://host/v1`` would otherwise be validated as the cleaned
      URL but stored and dialled as the raw one. httpx refuses it either
      way, so nothing was ever silently mis-dialled — this exists so the
      caller gets THIS function's actionable message instead of an SDK
      one, and so the guarantee below is literally true.

    Every error message below is passed through ``_redact_userinfo``
    before being returned — a rejected URL with embedded Basic-auth
    credentials (``https://bob:secret@`` — hostless, so it always fails
    validation) never echoes the password back; only the userinfo segment
    is scrubbed, the rest of the URL (host, port, path — the actionable
    part) stays visible.

    Returns ``(normalized_url, error_message)``. On success,
    ``error_message`` is ``""`` and ``normalized_url`` is the trimmed
    string, WITH any userinfo it legitimately contains left intact — the
    exact value the caller must go on to validate further, persist, and
    pass to ``get_provider``, never the raw input. On failure,
    ``normalized_url`` is ``None`` and ``error_message`` explains why.
    """
    normalized = raw.strip()
    if any(c in normalized for c in "\t\r\n"):
        return None, _redact_userinfo(
            f"Invalid base_url '{raw}': contains a tab or line break"
        )
    try:
        parsed = urlsplit(normalized)
    except ValueError as e:
        return None, _redact_userinfo(f"Invalid base_url '{raw}': {e}")
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None, _redact_userinfo(
            f"Invalid base_url '{raw}': must be a full http:// or https:// "
            "URL with a host, e.g. 'http://localhost:1234/v1'"
        )
    return normalized, ""


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
    # The next two fields are populated ONLY for file-registered providers
    # (config/providers.json, see "File-registered providers" below) —
    # always None for the three built-ins above, and mutually exclusive
    # with each other (a literal apiKey sets default_api_key; a
    # ``${ENV_VAR}`` one sets api_key_env; neither is set for a
    # deliberately keyless entry).
    #
    # repr=False: never let an incidental repr()/log of a _ProviderSpec
    # (or a dict/list containing one) print the resolved literal key.
    default_api_key: str | None = field(default=None, repr=False)
    # The env var named by a ``"${VAR}"`` apiKey — resolved LAZILY, via
    # os.getenv(), in get_provider()/key_env_for() at USE time, never at
    # load time. This matters for two reasons: (1) this module is
    # imported (triggering the file load) before main.py's load_dotenv()
    # call runs, so a var that only lives in .env would read back empty
    # if resolved eagerly here; (2) resolving lazily means exporting the
    # var after boot works on the next call, no restart needed. A var
    # that isn't set yet is not a value to log — this field only ever
    # holds the (non-secret) variable NAME.
    api_key_env: str | None = None
    # providers.json's models[] — feeds TWO consumers (feature 22): (1)
    # list_models()'s fallback (_FALLBACK_MODELS, wired in below) when
    # this id is the ACTIVE provider and its live catalog call fails, and
    # (2) GET /config/llm/providers' `models` field (via models_for(),
    # "Public registry accessors" below, C5), which lets the Settings
    # card's model dropdown work for a not-yet-active file-registered
    # provider too, not only the active one. Empty = free-text model
    # field for both.
    models: tuple[str, ...] = field(default_factory=tuple)


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
# File-registered providers (config/providers.json, feature 22 C3) — lets a
# user register an arbitrary provider (id + shape + base URL + key) without
# touching this file. Loaded once, at import time, straight into _REGISTRY:
# from every consumer's point of view (get_provider, get_registered_provider_ids,
# key_env_for) a file-registered id becomes a first-class registered
# provider — carrying its own base URL and its own key, just like a
# built-in — not an anonymous "custom" id that needs a base_url handed to
# it on every call.
#
# The four ids below are frozen from _REGISTRY BEFORE this section runs, so
# a file entry can never shadow openai/gemini/anthropic/ollama regardless of
# what _REGISTRY grows to contain afterwards.
# -----------------------------------------------------------------------

_RESERVED_PROVIDER_IDS = frozenset(_REGISTRY) | {"ollama"}

# Charset for a normalized (lowercased/stripped) provider id — lowercase
# ASCII letters, digits, '.', '_', '-'. Underscore is INCLUDED
# deliberately (feature 22, C5 review, round 2): 'lm_studio' is a
# plausible real id (LM Studio being exactly the kind of OpenAI-compatible
# local server this feature exists to support) and underscore is a common
# separator — rejecting it to defend against a collision with a frontend
# sentinel string would be a real, permanent usability cost paid to guard
# against a string the FRONTEND chose, not one inherent to what a
# provider id can legitimately be. This still rejects a space
# (``"corp gateway"``, which loaded cleanly before this rule existed) and
# anything else outside this set.
#
# The Settings card's "Other" tab sentinel (``frontend/components/
# settings/llm-card.tsx``'s ``OTHER_TAB``) is kept OUT of this charset
# instead: it's ``"@other"`` — '@' is not in ``[a-z0-9._-]``, so no
# provider id can EVER equal it; the loader rejects any file entry
# keyed ``"@other"`` at boot (see the id-charset check below), the same
# way it rejects ``"corp gateway"``. This makes the two files structurally
# unable to collide (a file entry can't accidentally become unreachable
# behind the sentinel tab) rather than relying on two independently
# maintained regexes happening to agree — see
# ``scripts/test_providers_file.py``'s dedicated check tying the two
# together, so a future edit to either side that reopens the collision
# fails a test instead of shipping quietly.
_PROVIDER_ID_RE = re.compile(r"^[a-z0-9._-]+$")

_ENTRY_ALLOWED_KEYS = frozenset({"api", "baseUrl", "apiKey", "models"})

# The actual path _load_providers_file() read from, remembered so a
# use-time diagnostic (get_provider(), below) can name the REAL file —
# which may be DOKKAI_PROVIDERS_FILE, not the "config/providers.json"
# default — instead of pointing the user at a file that isn't the
# problem. Stays None when no file was loaded (no diagnostic ever needs
# it in that case: api_key_env is only ever set on a spec that came from
# an actually-loaded file).
_PROVIDERS_FILE_PATH: Path | None = None

# The exact supported form, matched anchored end-to-end. UPPERCASE-only
# var name — matching OpenClaw's own convention exactly (env var names
# match ``^[A-Z_][A-Z0-9_]*$``) rather than an invented one. This is the
# load-bearing choice that makes
# the discrimination below actually work: env vars are conventionally
# UPPERCASE, so a value starting with ``$`` + a LOWERCASE letter, a digit,
# or punctuation is confidently a literal, never a reference — there is no
# real-world ambiguity to heuristically guess at.
_ENV_VAR_RE = re.compile(r"^\$\{([A-Z_][A-Z0-9_]*)\}$")
# Anything containing "${" that doesn't match the exact (uppercase) form
# above — including a lowercase name, e.g. "${my_key}" — is a typo'd
# attempt at the braces form (``Bearer ${VAR}`` wrapping one in other
# text, "${ VAR }" with spaces, "${VAR:-default}", a lowercase name, ...).
_ENV_VAR_BRACE_ATTEMPT_RE = re.compile(r"\$\{")
# A value that IS, in its entirety, "$" + an uppercase-leading identifier
# (OpenClaw's own bare ``$VAR`` shorthand form) — dokkai deliberately does
# NOT implement that shorthand (only "${VAR}" resolves), so this is
# rejected as the "missing braces" typo it almost certainly is
# (``$GROQ_API_KEY``), pointing the user at "${VAR}". Anchored at both
# ends: a longer/punctuated string that merely STARTS this way
# (``$argon2id$v=19$m=1``, mixed case, ...) does not match and is a
# literal — see the module-level tests for the exact accept/reject matrix.
_ENV_VAR_BARE_SHORTHAND_RE = re.compile(r"^\$[A-Z][A-Z0-9_]*$")


def _classify_provider_api_key(
    path: Path, provider_id: str, raw: object
) -> tuple[str | None, str | None]:
    """
    Classify (never resolve) a providers.json entry's ``apiKey`` field at
    LOAD time.

    Returns ``(literal_value, env_var_name)`` — at most one is non-
    ``None``. Both are ``None`` for an absent/blank ``apiKey``: a
    deliberately keyless entry (local OpenAI-compatible servers like LM
    Studio/vLLM commonly need none).

    ``"${ENV_VAR}"`` (the exact, UPPERCASE-name form — matching OpenClaw's
    own convention, see ``_ENV_VAR_RE``) is recognized here but
    deliberately NOT interpolated here — see ``_ProviderSpec.api_key_env``'s
    docstring for why resolution is deferred to
    ``get_provider()``/``key_env_for()`` at use time. Anything else is
    accepted as a literal value (the file is gitignored) UNLESS it looks
    like a botched attempt at that syntax — a lowercase/malformed
    ``${...}`` or the bare ``$UPPERCASE_NAME`` shorthand dokkai doesn't
    implement — in which case it's a loud, load-time error naming the
    supported syntax, not a silently-wrong literal key that fails
    opaquely with a 401 later. A ``$``-leading value that ISN'T one of
    those two shapes (lowercase, a digit, punctuation right after the
    ``$`` — e.g. a gateway token, a bcrypt/argon2 hash) is unambiguously a
    literal and is accepted with no error: env vars are conventionally
    UPPERCASE, so there's no real ambiguity left to guess at once that
    convention is the discriminator.

    The error NEVER echoes the value itself (only its length) — this is a
    boot-time exception, and uvicorn prints exceptions to
    ``docker compose logs``/CI output/wherever the user pastes them next.

    Raises ``ValueError`` for a non-string ``apiKey`` (``12345``, ``null``
    written as an object/array, ...) — the wrong type must be as loud as
    every other malformed field in this loop, not a quietly keyless
    provider that ends up sending no ``Authorization`` header at all. This
    message also never echoes the value (only its Python type name).
    """
    if raw is None:
        return None, None
    if not isinstance(raw, str):
        raise ValueError(
            f"{path}: provider '{provider_id}' 'apiKey' must be a string "
            f"(got {type(raw).__name__})"
        )
    if not raw.strip():
        return None, None
    stripped = raw.strip()
    match = _ENV_VAR_RE.match(stripped)
    if match:
        return None, match.group(1)
    if _ENV_VAR_BRACE_ATTEMPT_RE.search(stripped) or _ENV_VAR_BARE_SHORTHAND_RE.match(stripped):
        raise ValueError(
            f"{path}: provider '{provider_id}' 'apiKey' (a {len(stripped)}-"
            "character string) looks like it references an environment "
            "variable but isn't the supported syntax — use exactly "
            "'${VAR_NAME}' (uppercase, e.g. '${GROQ_API_KEY}'). If the "
            "actual key legitimately starts with '$' (a token/hash you "
            "don't control), put IT in an environment variable and "
            "reference that variable the same way — env var values are "
            "used verbatim, with no further '$' interpretation."
        )
    return stripped, None


def _load_providers_file() -> None:
    """
    Load ``config/providers.json`` (path overridable by
    ``DOKKAI_PROVIDERS_FILE``) into ``_REGISTRY``.

    Runs once, as a module-level side effect at import time — this module
    is imported from ``services.llm_config``, which ``controllers.config``
    imports at module scope, itself imported by ``main`` before the
    FastAPI app is constructed — so a malformed file fails the whole
    process at boot rather than degrading into "file present, nothing
    loaded", which would look like a working system quietly serving a
    different provider than intended.

    - File absent (default path or an overridden one pointing at
      nothing): clean no-op. A path that EXISTS but isn't a regular file
      (e.g. a directory — the exact shape a bind-mounted-but-missing-FILE
      turns into, see the compose comment) is loud, not a silent no-op:
      the whole point of mounting the ``config/`` directory instead of the
      file is to avoid that trap, and staying silent here for anyone who
      overrides ``DOKKAI_PROVIDERS_FILE`` to a directory would reopen it.
    - Malformed JSON, or valid JSON missing the required ``providers``
      key: raises naming the path (and, for malformed JSON, the exact
      parse position). A top-level typo (``"Providers"``, ``"provider"``)
      must not silently register nothing — the same "must not degrade
      into quietly serving something other than intended" reasoning as
      malformed JSON. Unrecognized OTHER top-level keys (e.g. a
      ``"$schema"`` hint) are tolerated.
    - An entry whose id is empty/contains a character outside
      ``[a-z0-9._-]`` (after lowercasing — rules out a space, e.g.
      ``"corp gateway"``, and the frontend's "Other" tab sentinel,
      ``"@other"``, whose ``'@'`` can never appear in a valid id)/shadows
      a built-in
      (openai/gemini/anthropic/ollama)/duplicates another entry's id
      after case/whitespace normalization, an unknown ``api`` shape, a
      missing/invalid ``baseUrl``, an unknown field (catches the
      ``"apikey"`` mis-casing typo), or a wrong-typed/null/blank
      ``apiKey``: raises naming the offending id/value (never the value
      of a secret field).
    """
    global _PROVIDERS_FILE_PATH
    path = Path(os.getenv("DOKKAI_PROVIDERS_FILE", "config/providers.json"))
    if not path.exists():
        return
    if not path.is_file():
        raise ValueError(
            f"{path}: expected a JSON file, found something else (a "
            "directory?) — DOKKAI_PROVIDERS_FILE must point at the JSON "
            "file itself, e.g. 'config/providers.json', not a directory."
        )
    _PROVIDERS_FILE_PATH = path

    try:
        raw_text = path.read_text()
    except OSError as e:
        raise ValueError(f"{path}: cannot read providers file: {e}") from e

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{path}: malformed JSON at line {e.lineno}, column {e.colno} "
            f"(char {e.pos}): {e.msg}"
        ) from e

    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON must be an object with a 'providers' key")

    if "providers" not in data:
        raise ValueError(f"{path}: top-level JSON is missing the required 'providers' key")

    providers = data["providers"]
    if not isinstance(providers, dict):
        raise ValueError(f"{path}: 'providers' must be a JSON object of {{id: entry}}")

    known_shapes = (_SHAPE_OPENAI_COMPLETIONS, _SHAPE_ANTHROPIC_MESSAGES)
    seen_pids: dict[str, str] = {}

    # Every id-rejection message below echoes the raw JSON key
    # (provider_id) verbatim for its actionable value — but that key is
    # admin-authored, unvalidated text, and the realistic way it goes
    # wrong is precisely the mistake the id-charset rule below exists to
    # catch: pasting a gateway URL into the id SLOT instead of baseUrl
    # (e.g. "https://svc-user:svc-pw@gw.example/v1": {...}). Without
    # redaction that credential would be echoed in a boot-time exception
    # — which typically ships to uvicorn's stderr / `docker compose logs`
    # / a log aggregator, readable by more people than the gitignored
    # file itself. _redact_userinfo is applied uniformly to every raise
    # in this loop for that reason, matching validate_base_url's own
    # userinfo-redaction discipline rather than leaving some exempt.
    for provider_id, entry in providers.items():
        pid = str(provider_id).strip().lower()

        if not pid:
            raise ValueError(
                _redact_userinfo(f"{path}: provider id {provider_id!r} must not be empty/whitespace")
            )

        if not _PROVIDER_ID_RE.match(pid):
            raise ValueError(
                _redact_userinfo(
                    f"{path}: provider id {provider_id!r} is invalid — ids may "
                    "only contain lowercase letters, digits, '.', '_' and '-' "
                    "(e.g. 'groq', 'my-gateway', 'lm_studio')."
                )
            )

        if pid in _RESERVED_PROVIDER_IDS:
            raise ValueError(
                _redact_userinfo(
                    f"{path}: provider id '{provider_id}' shadows a built-in "
                    f"provider ({', '.join(sorted(_RESERVED_PROVIDER_IDS))}) "
                    "— choose a different id."
                )
            )

        if pid in seen_pids:
            raise ValueError(
                _redact_userinfo(
                    f"{path}: provider ids '{seen_pids[pid]}' and '{provider_id}' "
                    f"both normalize to '{pid}' — duplicate ids "
                    "(case/whitespace-insensitive) are not allowed; rename one. "
                    "(Whichever JSON key sorts last would otherwise silently "
                    "win, discarding the other's shape/base_url/key.)"
                )
            )
        seen_pids[pid] = str(provider_id)

        if not isinstance(entry, dict):
            raise ValueError(f"{path}: provider '{provider_id}' must be a JSON object")

        unknown_keys = set(entry) - _ENTRY_ALLOWED_KEYS
        if unknown_keys:
            raise ValueError(
                f"{path}: provider '{provider_id}' has unknown field(s): "
                f"{', '.join(sorted(unknown_keys))}. Allowed: "
                f"{', '.join(sorted(_ENTRY_ALLOWED_KEYS))}."
            )

        shape = entry.get("api")
        if shape not in known_shapes:
            raise ValueError(
                f"{path}: provider '{provider_id}' has unknown api shape "
                f"{shape!r}. Valid shapes: {', '.join(known_shapes)}."
            )

        base_url_raw = entry.get("baseUrl")
        if not isinstance(base_url_raw, str) or not base_url_raw.strip():
            raise ValueError(
                f"{path}: provider '{provider_id}' is missing required 'baseUrl'"
            )
        base_url, url_error = validate_base_url(base_url_raw)
        if url_error:
            raise ValueError(f"{path}: provider '{provider_id}': {url_error}")

        models = entry.get("models") or []
        if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
            raise ValueError(
                f"{path}: provider '{provider_id}' 'models' must be a list of strings"
            )

        # An explicitly-written "apiKey": null or "" is a JSON author's
        # choice, not the same as omitting the field — dict.get() can't
        # tell "absent" and "present but null" apart (both read back
        # None), so check membership first. Both are treated like any
        # other wrong/malformed value: loud, not a silent keyless
        # fallback (a plain OMISSION still is one — see
        # _classify_provider_api_key). Blank/whitespace-only is the
        # realistic trigger for a templated file (envsubst/sed) whose
        # substituted var was unset, producing exactly "".
        if "apiKey" in entry:
            raw_api_key = entry["apiKey"]
            if raw_api_key is None:
                raise ValueError(
                    f"{path}: provider '{provider_id}' 'apiKey' must be a "
                    "string (got null) — omit the field entirely for a "
                    "keyless provider."
                )
            if isinstance(raw_api_key, str) and not raw_api_key.strip():
                raise ValueError(
                    f"{path}: provider '{provider_id}' 'apiKey' must not be "
                    "blank/whitespace-only — omit the field entirely for a "
                    "keyless provider."
                )
        literal_key, api_key_env = _classify_provider_api_key(path, provider_id, entry.get("apiKey"))

        _REGISTRY[pid] = _ProviderSpec(
            shape=shape,
            default_base_url=base_url,
            key_env=None,
            display_name=str(provider_id).strip(),
            default_api_key=literal_key,
            api_key_env=api_key_env,
            models=tuple(models),
        )

        # Wire models[] as this id's list_models() fallback (used when the
        # live catalog call fails) — the only consumer that exists yet;
        # C4/C5 land more later. Skipped when empty (free-text model
        # field), matching the schema's stated meaning for [].
        if models:
            _FALLBACK_MODELS[pid] = list(models)


_load_providers_file()


# -----------------------------------------------------------------------
# Public registry accessors — services.llm_config and controllers.config
# need to know which ids are registered/built-in, whether a key is
# required, a display name, and a built-in's default base URL, without
# reaching into the private _REGISTRY dict (keeps _ProviderSpec internal).
# -----------------------------------------------------------------------

def get_registered_provider_ids() -> frozenset[str]:
    """
    Ids servable directly through the registry — the three built-ins
    (``openai``, ``gemini``, ``anthropic``) plus any id registered from
    ``config/providers.json`` (feature 22, C3). A file-registered id
    carries its own base URL (and, usually, its own key) exactly like a
    built-in, so from this predicate's caller's point of view
    (``services.llm_config``, deciding whether ``base_url`` must be
    supplied on a config write) it belongs in this set, not treated as an
    anonymous "custom" id. "Registered", not "built-in", is deliberate:
    this set is no longer only the three hardcoded ids. Excludes
    ``ollama``, which isn't part of this registry at all — it's handled
    through its own native path (see module docstring), not one of the
    two shapes above.
    """
    return frozenset(_REGISTRY)


def key_env_for(provider_name: str) -> str | None:
    """
    The env var a provider's API key normally lives in, or ``None`` if
    *provider_name* isn't a registered id, or is registered but doesn't
    require a key. Callers (``services.llm_config``) use it as the "is a
    key mandatory here?" predicate — ``get_provider`` itself never reads
    env vars, so the var is named for the caller's benefit, not consulted.

    For the three built-ins this is always their fixed env var name (key
    always mandatory). For a file-registered provider (feature 22, C3)
    it's ``None`` — key not mandatory on a config write — UNLESS its
    ``apiKey`` names an ``${ENV_VAR}`` that ISN'T CURRENTLY set (checked
    live, via ``os.getenv``, on every call — never cached from load time),
    in which case it's that variable's name, making a key mandatory here
    too until it's exported. Because this re-checks live, a var exported
    after boot makes the key optional again on the very next call, no
    restart needed.
    """
    spec = _REGISTRY.get(provider_name.lower().strip())
    if spec is None:
        return None
    if spec.key_env:
        return spec.key_env
    if spec.api_key_env and not os.getenv(spec.api_key_env):
        return spec.api_key_env
    return None


def get_builtin_provider_ids() -> frozenset[str]:
    """
    The ids ``dokkai`` ships built-in support for (``openai``, ``gemini``,
    ``anthropic``) — ``ollama`` excluded, same reasoning as
    ``get_registered_provider_ids``. Derived from ``_RESERVED_PROVIDER_IDS``,
    which is frozen from ``_REGISTRY`` BEFORE ``config/providers.json`` is
    loaded (see that constant's comment), so this is never affected by
    what the file registers — it answers "is this id code, or config?"
    for a caller (``controllers.config``'s provider-listing endpoint) that
    needs to tell the two apart, without that caller keeping its own
    separate copy of the three ids (a copy that silently goes stale the
    day a fourth built-in is added here).
    """
    return _RESERVED_PROVIDER_IDS - {"ollama"}


def display_name_for(provider_name: str) -> str | None:
    """
    The human-readable display name for a registered id (built-in or
    file-registered), or ``None`` if *provider_name* isn't registered at
    all. For the three built-ins this is their fixed name (``'OpenAI'``,
    ``'Gemini'``, ``'Anthropic'``). For a file-registered provider it's
    the id exactly as written in ``config/providers.json`` (original
    case preserved — the admin's own choice, e.g. ``'OpenRouter'``), NOT
    a caller-side ``str.title()`` of the normalized lowercase id, which
    mangles anything with internal capitals (``'gpt4all'`` ->
    ``'Gpt4All'``). Not secret — the id is already public (it's what a
    caller sends back as ``provider_name`` on every config write).
    """
    spec = _REGISTRY.get(provider_name.lower().strip())
    return spec.display_name if spec is not None else None


def default_base_url_for(provider_name: str) -> str | None:
    """
    The DEFAULT base URL for a BUILT-IN id ONLY (``openai``, ``gemini``,
    ``anthropic``) — ``None`` for anything else, including a
    file-registered provider, DELIBERATELY: unlike a built-in's default
    (a fixed literal in this module, e.g. Gemini's OpenAI-compat URL,
    never secret), a file-registered ``baseUrl`` is admin-authored and
    ``validate_base_url`` legitimately accepts embedded Basic-auth
    userinfo in a URL that has a host (``https://user:pass@host/v1``) —
    and this function feeds a read endpoint any authenticated viewer, not
    just an admin, can call. Returning ``None`` for anything outside the
    three built-ins is what keeps that possibility structurally out of
    reach here, rather than relying on every caller to remember not to
    ask for it. ``None`` too when the built-in itself has no override
    (``openai``/``anthropic`` — the SDK's own official default applies).

    The returned value is still passed through ``_redact_userinfo`` as a
    second line of defense — the three built-in defaults never contain
    userinfo today, so this is normally a no-op, but it makes "never
    leaks a credential" a checked invariant of this function rather than
    an assumption about what the three literals happen to contain.
    """
    name = provider_name.lower().strip()
    if name not in get_builtin_provider_ids():
        return None
    spec = _REGISTRY.get(name)
    if spec is None or spec.default_base_url is None:
        return None
    return _redact_userinfo(spec.default_base_url)


def models_for(provider_name: str) -> tuple[str, ...]:
    """
    A registered provider's declared ``models[]`` (``config/providers.json``,
    feature 22, C3) — empty for an unregistered id, or a registered one
    that declared none (``models[]`` absent/empty in the file means
    "free-text model field", the same meaning ``_load_providers_file``
    already gives it for the ``list_models()`` fallback below).

    Unlike ``default_base_url_for``, this is NOT restricted to the three
    built-ins: a model NAME carries no secret and isn't admin-authored
    connection info the way a file-registered ``baseUrl`` is, so there's
    no equivalent leak surface to guard against — safe to expose for any
    registered id, built-in or file-registered. (The built-ins never
    populate this field today; only ``config/providers.json`` entries do.)
    """
    spec = _REGISTRY.get(provider_name.lower().strip())
    return spec.models if spec is not None else ()


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

    ``provider_name`` resolves through the registry: the built-ins
    (openai, gemini, anthropic, ollama) plus any id registered from
    ``config/providers.json`` (feature 22, C3), which carries its own
    default base URL and, usually, its own key — neither needs to be
    passed by the caller. An id that resolves to neither is still
    servable — through the openai-completions shape — as long as
    ``base_url`` is given, which covers any OpenAI-compatible endpoint
    (Groq, Together, OpenRouter, LM Studio, vLLM, ...).

    Parameters
    ----------
    provider_name : str
        A built-in id ('openai', 'gemini', 'anthropic', 'ollama'), a
        file-registered id, or a custom id paired with ``base_url``.
    api_key : str
        API key for remote providers. Required for built-in remotes
        (openai/gemini/anthropic); for a file-registered id, overrides
        (or supplies, if the file left it unset) its own key; optional
        for a fully custom endpoint.
    base_url : str, optional
        Base URL override. Required for a custom (unregistered) id;
        overrides a registered id's own default when given.
    """
    name = provider_name.lower().strip()

    if name == "ollama":
        return OllamaProvider(base_url=base_url)

    spec = _REGISTRY.get(name)
    if spec is None:
        if not base_url:
            registered = ", ".join(sorted(list(_REGISTRY) + ["ollama"]))
            raise ValueError(
                f"Unknown provider '{provider_name}'. Registered providers: "
                f"{registered}. A custom provider needs a base_url pointing "
                f"at an OpenAI-compatible endpoint."
            )
        spec = _ProviderSpec(
            shape=_SHAPE_OPENAI_COMPLETIONS,
            default_base_url=None,
            key_env=None,
            display_name=provider_name.strip(),
        )

    # api_key_env is resolved LAZILY here, via os.getenv(), never cached
    # from load time — see _ProviderSpec.api_key_env's docstring: this
    # module is imported (triggering the providers.json load) before
    # main.py's load_dotenv() runs, so an eager read at load time would
    # provably miss a .env-only var; resolving here also means exporting
    # the var after boot works on the very next call, no restart needed.
    env_key = os.getenv(spec.api_key_env) if spec.api_key_env else None
    resolved_api_key = api_key or spec.default_api_key or env_key or ""

    if not resolved_api_key and spec.api_key_env:
        # Name the file that ACTUALLY produced this spec — which may be
        # DOKKAI_PROVIDERS_FILE, not the default path — so the diagnostic
        # points at the real file to edit, not necessarily
        # "config/providers.json".
        source = _PROVIDERS_FILE_PATH if _PROVIDERS_FILE_PATH is not None else "config/providers.json"
        raise ValueError(
            f"{spec.display_name}: {source} references "
            f"${{{spec.api_key_env}}} for its API key, but the "
            f"{spec.api_key_env} environment variable is not set. Export "
            f"it, or pass an API key explicitly."
        )
    if spec.key_env and not resolved_api_key:
        raise ValueError(f"{spec.display_name} requires an API key")

    resolved_base_url = base_url or spec.default_base_url

    if spec.shape == _SHAPE_ANTHROPIC_MESSAGES:
        return AnthropicMessagesProvider(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            provider_name=name,
            display_name=spec.display_name,
        )

    return OpenAICompletionsProvider(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        provider_name=name,
        display_name=spec.display_name,
    )
