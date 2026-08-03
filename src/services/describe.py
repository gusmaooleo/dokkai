"""
Tier 2 — per-entity micro-descriptions.

Generates a one-sentence natural-language summary for each *describable* code
entity with a small LLM (local Ollama by default, or a remote provider), so
conceptual queries ("how does the alarm flow work?") reach implementation
code through the ``summary`` vector. Description calls go through the shared
provider abstraction in ``services.llm_provider``.

Configured entirely through env vars (never the ``/config/llm`` UI/DB path —
decision 22-lote19 Q4a): ``DESC_PROVIDER`` (default ``ollama``) accepts any
id the registry can serve — a built-in (``openai``, ``gemini``,
``anthropic``), an id registered via ``config/providers.json``, or an
arbitrary OpenAI-compatible id paired with ``DESC_BASE_URL``. ``DESC_MODEL``
and ``DESC_CONCURRENCY`` name the model and the parallel-request cap. Ollama
alone keeps using ``OLLAMA_BASE_URL``, unchanged.

``DESC_BASE_URL`` is validated the same way as every other caller-supplied
base URL in this codebase (``services.llm_provider.validate_base_url`` —
scheme, host, no embedded control characters) before it ever reaches an
SDK client; a rejected value is a graceful skip, like every other
descriptor misconfiguration, never a raised exception or an opaque SDK
error. ``DESC_BASE_URL`` overriding a REGISTERED provider's own base URL —
a built-in's default, or a file-registered id's own ``baseUrl`` alike —
redirects that provider's key to the override host too — allowed, since
``get_provider`` documents that override, but logged as a warning naming
the provider and the target host (never the key). Both cases are
"leftover-prone" the same way: ``DESC_BASE_URL`` is a single global env
var, so switching ``DESC_PROVIDER`` to a different already-registered id
without unsetting a ``DESC_BASE_URL`` left over from an earlier
custom-endpoint experiment silently redirects the NEW id's key too. Only a
genuinely UNREGISTERED id — where ``DESC_BASE_URL`` is the sole source of
a base URL, not an override of anything — stays silent.

Key precedence: an explicit ``DESC_API_KEY`` (blank/whitespace-only counts
as unset) always wins (this is what makes an UNREGISTERED remote id, e.g.
``DESC_PROVIDER=groq`` pointed at Groq's endpoint via ``DESC_BASE_URL``
with no ``config/providers.json`` entry, actually authenticate — without
it, a provider id the registry doesn't recognize has no other way to
receive a key and would silently send an unauthenticated request).
Otherwise a built-in reads its own standard env var (``OPENAI_API_KEY``
etc, via ``key_env_for()`` — and a missing one is reported by name,
alongside ``DESC_API_KEY``, in the skip reason). Otherwise a
file-registered provider's own key (resolved inside ``get_provider``
itself, from ``config/providers.json``'s ``${ENV}`` or literal
``apiKey``, with its own already-specific missing-key diagnostic) applies.
A key is optional throughout — keyless is legitimate for a local
OpenAI-compatible server (LM Studio, vLLM, ...).

Cost control (a description is one LLM call — the pipeline's dominant cost):
  - **Selective**: skip tests, entities without source, and trivial one-liners.
  - **Cached by source hash ALONE**: unchanged source reuses its description
    across re-ingestions, regardless of ``DESC_PROVIDER``/``DESC_MODEL`` —
    pairs with the deterministic UUID for incremental ingest. This is
    deliberate, not an oversight: keying on model too would force a full
    paid re-describe of every existing corpus the first time anyone
    upgrades a model. The trade-off is that switching provider/model is a
    no-op on chunks already described — it only affects new/changed
    source. To force EXISTING chunks to regenerate under the currently
    configured provider/model, use ``POST /instances/{project}/describe``
    with ``{"force": true}``. Note :func:`describe_chunks`'s returned
    ``stats["model"]`` names the CONFIGURED model, not necessarily the one
    that produced every description in the batch — some may be cache hits
    generated earlier, by a different provider or model.
  - **Concurrent**: bounded parallel requests to the configured provider.

If no model is configured (``DESC_MODEL`` unset) or the configured provider
is unavailable (missing API key, unreachable, or model not found),
description generation **fails loudly** — see :class:`DescriptorError` and
:func:`ensure_descriptor_available`. Ingesting without descriptions is an
explicit opt-in (``process_and_store(..., describe=False)``), not a silent
fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from services.chunker import CodeChunk
from services.llm_provider import (
    LLMMessage,
    LLMProvider,
    get_provider,
    get_registered_provider_ids,
    key_env_for,
    validate_base_url,
)
from services.retriever import _is_test_path

logger = logging.getLogger(__name__)


DESC_SYSTEM_PROMPT = (
    "You summarize code for a search index. Reply with ONE concise sentence "
    "(max 25 words) describing what the code does. No preamble, no code, no "
    "bullet points — just the sentence."
)

_MIN_SOURCE_LINES = 3  # fewer non-blank lines ⇒ too trivial to be worth an LLM call
_GEN_TEMPERATURE = 0.1
_GEN_MAX_TOKENS = 64
_MAX_ENUM_MEMBERS = 10
_LEADING_IDENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)")
_ENUM_LINE_SKIP = ("//", "/*", "*", "#", "@", "{", "}")


# ---------------------------------------------------------------------------
# Template (no-LLM) descriptions for trivial entity types — decision 1d
# ---------------------------------------------------------------------------

def _enum_members(source_code: str) -> list[str]:
    """Cheap leading-identifier scan of an enum body, skipping the header line."""
    members: list[str] = []
    for line in source_code.splitlines()[1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith(_ENUM_LINE_SKIP):
            continue
        match = _LEADING_IDENT_RE.match(stripped)
        if match and match.group(1) not in members:
            members.append(match.group(1))
    return members


def _template_description(chunk: CodeChunk) -> str | None:
    """
    Model-independent description for entity types too trivial (or too
    logic-free) to warrant an LLM call. Returns ``None`` when the chunk
    doesn't qualify — a human docstring always wins over a template.
    """
    if chunk.doc or _is_test_path(chunk.file_path):
        return None
    if chunk.entity_type == "Type":
        return f"Type alias {chunk.name} defined in {chunk.file_path}."
    if chunk.entity_type == "Enum" and not chunk.defined_methods:
        members = _enum_members(chunk.source_code)
        if not members:
            return f"Enumeration {chunk.name}."
        shown = members[:_MAX_ENUM_MEMBERS]
        member_list = ", ".join(shown)
        if len(members) > _MAX_ENUM_MEMBERS:
            member_list += ", …"
        return f"Enumeration {chunk.name} with members {member_list}."
    return None


# ---------------------------------------------------------------------------
# Selectivity + hashing
# ---------------------------------------------------------------------------

def _should_describe(chunk: CodeChunk) -> bool:
    """Whether an entity is worth an LLM description."""
    if not chunk.source_code or _is_test_path(chunk.file_path):
        return False
    if chunk.entity_type in ("Class", "Interface", "Enum"):
        return True  # container types carry the most conceptual signal
    non_blank = [ln for ln in chunk.source_code.splitlines() if ln.strip()]
    return len(non_blank) >= _MIN_SOURCE_LINES


def _source_hash(chunk: CodeChunk) -> str:
    """Content key for the cache — description depends only on the source."""
    return hashlib.sha256(chunk.source_code.encode("utf-8", "replace")).hexdigest()


# ---------------------------------------------------------------------------
# Persistent content-addressed cache
# ---------------------------------------------------------------------------

def _default_cache_path() -> Path:
    base = Path(__file__).resolve().parent.parent.parent
    return base / "data" / "description_cache.json"


class DescriptionCache:
    """``source_hash -> description``, persisted as a single JSON file."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _default_cache_path()
        self._data: dict[str, str] = {}
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, source_hash: str) -> str | None:
        return self._data.get(source_hash)

    def set(self, source_hash: str, description: str) -> None:
        self._data[source_hash] = description

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Merge with what's on disk (a concurrent job may have written entries we
        # don't have) and write atomically, so a full-file overwrite can't clobber
        # a parallel writer or leave a truncated file behind.
        merged = dict(self._data)
        if self._path.exists():
            try:
                on_disk = json.loads(self._path.read_text(encoding="utf-8"))
                merged = {**on_disk, **self._data}
            except (json.JSONDecodeError, OSError):
                pass
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)
        self._data = merged


# ---------------------------------------------------------------------------
# Provider-backed generation (services.llm_provider)
# ---------------------------------------------------------------------------

def _descriptor_base_url(provider_name: str) -> str:
    """
    Base URL env var for *provider_name*.

    ``OLLAMA_BASE_URL`` for Ollama — unchanged, byte-identical to before
    this function existed. ``DESC_BASE_URL`` (new, feature 22 C4) for any
    other provider — a built-in (openai/gemini/anthropic), a
    ``config/providers.json`` id, or a fully custom one. Named to match the
    existing ``DESC_PROVIDER``/``DESC_MODEL``/``DESC_CONCURRENCY``
    convention rather than reusing ``OLLAMA_BASE_URL``, whose name is
    Ollama-specific.

    Empty (unset) is not an error here: a registered id (built-in or
    file-registered) already carries its own default/registered base URL
    inside the registry, so ``DESC_BASE_URL`` is only REQUIRED for a
    genuinely unregistered custom id — and that requirement is enforced by
    ``get_provider`` itself (via ``_resolve_provider`` below), not here.
    """
    if provider_name == "ollama":
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return os.getenv("DESC_BASE_URL", "").strip()


# The three ids that ship inside services.llm_provider's own _REGISTRY at
# import time (C1). Used ONLY to enrich the missing-key skip reason below
# (point 4, review round 2) — a file-registered provider missing its
# ${ENV} key already gets a MORE specific diagnostic straight from
# get_provider (it names the exact var AND the config/providers.json path
# that references it), which this set deliberately does not override; only
# a built-in's generic "requires an API key" (which names no variable at
# all) needs the enrichment. NOT used for the DESC_BASE_URL-override
# warning below anymore (review round 2, point 3) — see
# get_registered_provider_ids() at that call site instead.
_BUILTIN_IDS = frozenset({"openai", "gemini", "anthropic"})


def _resolve_provider(provider_name: str, base_url: str) -> tuple[LLMProvider | None, str]:
    """
    Instantiate the configured provider, or return a graceful-skip reason.

    Resolves *provider_name* through the shared registry in
    ``services.llm_provider`` (feature 22): ``get_provider`` itself already
    knows every built-in id (openai, gemini, anthropic), every id
    registered via ``config/providers.json`` (C3, carrying its own base URL
    and, usually, its own key), and any other id paired with *base_url* (an
    OpenAI-compatible endpoint). This function adds:

    - ``DESC_BASE_URL`` validation (``validate_base_url``, the SAME
      validator every other caller-supplied base URL in this codebase goes
      through — ``config/providers.json``'s loader and ``/config/llm``'s
      POST body) BEFORE it ever reaches ``get_provider``/the SDK. Skipping
      this would let a malformed value either raise ``httpx.InvalidURL``
      (not a ``ValueError`` — would escape this function's except clause
      entirely, surfacing as an unattributed HTTP 500 through
      ``controllers.instances``, which only catches ``DescriptorError``)
      or, for a scheme-less value, get silently accepted and fail later
      with a message that never mentions the URL.
    - the descriptor's own key-lookup convention: an explicit
      ``DESC_API_KEY`` (blank/whitespace-only treated as unset — a
      whitespace value must fall through to the next level, never be sent
      verbatim as a garbage credential) always wins; otherwise a built-in
      reads its standard env var via ``key_env_for()``; a
      file-registered/custom provider otherwise needs no env lookup here,
      since ``get_provider`` resolves its own key internally.
    - a warning (never the key) when ``DESC_BASE_URL`` overrides a
      provider that has its own configured base URL — a built-in's
      default, OR a file-registered id's own ``baseUrl``. Both are
      "leftover-prone" the same way: ``DESC_BASE_URL`` is one global env
      var, so switching ``DESC_PROVIDER`` to a DIFFERENT already-registered
      id without unsetting a ``DESC_BASE_URL`` left over from an earlier
      custom-endpoint experiment silently redirects that id's key too —
      whether the key came from a built-in's fixed env var or from a file
      entry's own ``${ENV}``/literal ``apiKey``. Only a genuinely
      UNREGISTERED id (``DESC_BASE_URL`` is its only source of a base URL
      at all — there is nothing to "override") stays silent.
    - when a built-in's key is missing (empty after the above), a skip
      reason naming BOTH ``DESC_API_KEY`` and the built-in's own standard
      var — ``get_provider``'s own message here names neither. A
      file-registered id's own missing-``${ENV}`` message is untouched:
      it already names the exact variable and the ``config/providers.json``
      path that references it, which is more specific than anything this
      function could add.

    Every one of the above turns into the graceful ``(None, reason)`` skip
    this module's callers expect — this function NEVER raises.

    Ollama keeps its own native path here, unchanged: it isn't part of the
    registry (see ``services.llm_provider``'s module docstring — it's
    handled directly, not through either of the two shapes), *base_url*
    for it is always ``OLLAMA_BASE_URL`` (never validated here — untouched,
    byte-identical to before this function was generalized), and it is
    NEVER subject to the DESC_BASE_URL-override warning above.
    """
    if provider_name == "ollama":
        return get_provider("ollama", base_url=base_url), ""

    if base_url:
        normalized_base_url, url_error = validate_base_url(base_url)
        if url_error:
            return None, f"invalid DESC_BASE_URL: {url_error}"
        base_url = normalized_base_url

    if provider_name in get_registered_provider_ids() and base_url:
        host = urlsplit(base_url).hostname or base_url
        logger.warning(
            "descriptor: DESC_PROVIDER=%s is resolving with a DESC_BASE_URL "
            "override (host: %s) — its API key is being sent to that host "
            "instead of %s's own configured base URL. Unset DESC_BASE_URL "
            "if this wasn't intentional.",
            provider_name, host, provider_name,
        )

    key_env = key_env_for(provider_name)
    override_key = os.getenv("DESC_API_KEY", "").strip()
    api_key = override_key if override_key else (os.getenv(key_env, "") if key_env else "")

    try:
        provider = get_provider(provider_name, api_key=api_key, base_url=base_url or None)
    except ValueError as e:
        if not api_key and key_env and provider_name in _BUILTIN_IDS:
            return None, (
                f"{provider_name} requires an API key — set DESC_API_KEY, "
                f"or {key_env}"
            )
        return None, str(e)
    return provider, ""


_TRIMMED_DOC_CAP = 600
_TRIMMED_BODY_CAP = 1200


def _cut_at_line_boundary(text: str, cap: int) -> str:
    """Truncate ``text`` to at most ``cap`` chars, cutting at a line boundary."""
    if len(text) <= cap:
        return text
    truncated = text[:cap]
    last_newline = truncated.rfind("\n")
    if last_newline > 0:
        truncated = truncated[:last_newline]
    return truncated


def build_describe_prompt(chunk: CodeChunk, mode: str = "full") -> str:
    """
    Build the per-entity user prompt sent to the descriptor LLM.

    ``mode="full"`` is the original prompt (entity type + qualified name +
    the complete source). ``mode="trimmed"`` — the shipped default since the
    1c measurement gate — adds the extracted doc/comment (capped) and caps
    the source excerpt, cutting at a line boundary. Both modes are measurable
    via ``scripts/measure_descriptions.py``.
    """
    if mode == "full":
        return f"{chunk.entity_type} {chunk.qualified_name}\n\n{chunk.source_code}"
    if mode == "trimmed":
        parts = [f"{chunk.entity_type} {chunk.qualified_name}"]
        if chunk.doc:
            parts.append(f"Doc: {_cut_at_line_boundary(chunk.doc, _TRIMMED_DOC_CAP)}")
        parts.append(_cut_at_line_boundary(chunk.source_code, _TRIMMED_BODY_CAP))
        return "\n\n".join(parts)
    raise ValueError(f"unknown prompt mode '{mode}'")


async def _generate_one(provider: LLMProvider, model: str, chunk: CodeChunk) -> str:
    messages = [
        LLMMessage("system", DESC_SYSTEM_PROMPT),
        LLMMessage("user", build_describe_prompt(chunk, "trimmed")),
    ]
    response = await provider.chat(
        messages, model=model, temperature=_GEN_TEMPERATURE, max_tokens=_GEN_MAX_TOKENS
    )
    return response.strip()


class DescriptorError(RuntimeError):
    """Raised when no descriptor model is configured or it is unavailable."""


_NO_MODEL_MESSAGE = (
    "no descriptor model specified — set DESC_MODEL (e.g. qwen2.5-coder:3b) "
    "or configure a remote provider via DESC_PROVIDER"
)


async def _require_provider(provider_name: str, base_url: str, model: str) -> LLMProvider:
    """Resolve + probe the descriptor provider, raising DescriptorError on failure."""
    provider, reason = _resolve_provider(provider_name, base_url)
    if provider is None:
        raise DescriptorError(f"descriptor provider '{provider_name}' unavailable: {reason}")

    available, reason = await provider.health_check(model)
    if not available:
        if provider_name == "ollama":
            raise DescriptorError(
                f"descriptor model '{model}' is not available in Ollama at "
                f"{base_url} — pull it with 'ollama pull {model}'"
            )
        raise DescriptorError(f"descriptor model '{model}' unavailable: {reason}")
    return provider


async def ensure_descriptor_available() -> tuple[str, str]:
    """
    Resolve and probe the configured descriptor provider/model, raising
    :class:`DescriptorError` with a clear English message when description
    generation cannot proceed. Returns ``(provider_name, model)`` on success.
    """
    model = os.getenv("DESC_MODEL", "").strip()
    if not model:
        raise DescriptorError(_NO_MODEL_MESSAGE)

    provider_name = os.getenv("DESC_PROVIDER", "ollama").lower().strip()
    base_url = _descriptor_base_url(provider_name)

    await _require_provider(provider_name, base_url, model)
    return provider_name, model


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

async def describe_chunks(
    chunks: list[CodeChunk],
    *,
    model: str | None = None,
    base_url: str | None = None,
    concurrency: int | None = None,
    cache: DescriptionCache | None = None,
    progress: Callable[[int, int], None] | None = None,
    force: bool = False,
) -> dict:
    """
    Populate ``chunk.description`` in place for describable entities.

    ``force=True`` skips cache reads — every describable target is
    regenerated — while cache writes still persist the fresh results (a
    later re-run without ``force`` sees them as cache hits).

    Raises :class:`DescriptorError` when no descriptor model is configured, or
    when the configured provider/model is unavailable and there is at least
    one entity left to describe (a fully-cached or fully-templated pass with
    an unavailable provider still succeeds offline).

    Returns a stats dict (enabled / describable / cached / generated / failed /
    skipped / templated) suitable for surfacing in a job status.
    """
    # Template pass first — model-independent, so it runs even without an LLM
    # configured. Templated chunks never reach the LLM target list.
    templated = 0
    remaining: list[CodeChunk] = []
    for chunk in chunks:
        template = _template_description(chunk)
        if template is not None:
            chunk.description = template
            templated += 1
        else:
            remaining.append(chunk)

    model = model if model is not None else os.getenv("DESC_MODEL", "").strip()
    if not model:
        raise DescriptorError(_NO_MODEL_MESSAGE)

    provider_name = os.getenv("DESC_PROVIDER", "ollama").lower().strip()
    base_url = base_url or _descriptor_base_url(provider_name)
    concurrency = concurrency or int(os.getenv("DESC_CONCURRENCY", "4"))
    cache = cache if cache is not None else DescriptionCache()

    targets = [c for c in remaining if _should_describe(c)]
    skipped = len(remaining) - len(targets)

    # Cache pass first — reuse descriptions for unchanged source (no LLM call).
    # force=True skips reads only; every target below still gets its fresh
    # result written back to the cache.
    to_generate: list[tuple[CodeChunk, str]] = []
    cached = 0
    for chunk in targets:
        source_hash = _source_hash(chunk)
        hit = None if force else cache.get(source_hash)
        if hit is not None:
            chunk.description = hit
            cached += 1
        else:
            to_generate.append((chunk, source_hash))

    generated = failed = 0
    if to_generate:
        # Probe availability only when there's actually something to generate —
        # a fully-cached (or fully-templated) re-run must still work offline.
        provider = await _require_provider(provider_name, base_url, model)

        sem = asyncio.Semaphore(concurrency)
        done = 0

        async def worker(chunk: CodeChunk, source_hash: str) -> None:
            nonlocal generated, failed, done
            async with sem:
                try:
                    desc = await _generate_one(provider, model, chunk)
                except Exception:
                    desc = ""
                if desc:
                    chunk.description = desc
                    cache.set(source_hash, desc)
                    generated += 1
                else:
                    failed += 1
                done += 1
                if progress and (done % 25 == 0 or done == len(to_generate)):
                    progress(done, len(to_generate))

        await asyncio.gather(*(worker(c, h) for c, h in to_generate))

    if generated:
        cache.save()

    return {"enabled": True, "model": model, "describable": len(targets),
            "cached": cached, "generated": generated, "failed": failed,
            "pending": 0, "skipped": skipped, "templated": templated}
