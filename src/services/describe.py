"""
Tier 2 — per-entity micro-descriptions.

Generates a one-sentence natural-language summary for each *describable* code
entity with a small LLM (local Ollama by default, or a remote provider), so
conceptual queries ("how does the alarm flow work?") reach implementation
code through the ``summary`` vector. Description calls go through the shared
provider abstraction in ``services.llm_provider``.

Cost control (a description is one LLM call — the pipeline's dominant cost):
  - **Selective**: skip tests, entities without source, and trivial one-liners.
  - **Cached by source hash**: unchanged source reuses its description across
    re-ingestions — pairs with the deterministic UUID for incremental ingest.
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
import os
import re
from collections.abc import Callable
from pathlib import Path

from services.chunker import CodeChunk
from services.llm_provider import LLMMessage, LLMProvider, get_provider
from services.retriever import _is_test_path


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

def _resolve_provider(provider_name: str, base_url: str) -> tuple[LLMProvider | None, str]:
    """Instantiate the configured provider, or return a graceful-skip reason."""
    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return None, "OPENAI_API_KEY not set"
        return get_provider("openai", api_key=api_key), ""
    if provider_name == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None, "ANTHROPIC_API_KEY not set"
        return get_provider("anthropic", api_key=api_key), ""
    if provider_name == "ollama":
        return get_provider("ollama", base_url=base_url), ""
    return None, f"unknown DESC_PROVIDER '{provider_name}'"


async def _provider_available(
    provider: LLMProvider, provider_name: str, model: str
) -> tuple[bool, str]:
    """Whether the configured provider/model is ready to generate."""
    if provider_name == "ollama":
        try:
            available = await provider.list_models()
        except Exception as e:
            return False, f"Ollama unavailable: {e}"
        matched = model in available or any(a.split(":")[0] == model for a in available)
        if not matched:
            return False, f"model '{model}' unavailable"
        return True, ""
    # Remote providers: probe with the actual configured model (a 1-token
    # call) instead of provider.health_check(), whose hard-coded model can
    # lag behind the vendor's catalog and 404 even when DESC_MODEL is valid.
    try:
        await provider.chat(
            [LLMMessage("user", "hi")], model=model, temperature=0.0, max_tokens=1
        )
    except Exception as e:
        return False, str(e)
    return True, ""


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

    available, reason = await _provider_available(provider, provider_name, model)
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
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

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
) -> dict:
    """
    Populate ``chunk.description`` in place for describable entities.

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
    base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    concurrency = concurrency or int(os.getenv("DESC_CONCURRENCY", "4"))
    cache = cache if cache is not None else DescriptionCache()

    targets = [c for c in remaining if _should_describe(c)]
    skipped = len(remaining) - len(targets)

    # Cache pass first — reuse descriptions for unchanged source (no LLM call).
    to_generate: list[tuple[CodeChunk, str]] = []
    cached = 0
    for chunk in targets:
        source_hash = _source_hash(chunk)
        hit = cache.get(source_hash)
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
