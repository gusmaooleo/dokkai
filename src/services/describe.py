"""
Tier 2 — per-entity micro-descriptions.

Generates a one-sentence natural-language summary for each *describable* code
entity with a small local Ollama model, so conceptual queries ("how does the
alarm flow work?") reach implementation code through the ``summary`` vector.

Cost control (a description is one LLM call — the pipeline's dominant cost):
  - **Selective**: skip tests, entities without source, and trivial one-liners.
  - **Cached by source hash**: unchanged source reuses its description across
    re-ingestions — pairs with the deterministic UUID for incremental ingest.
  - **Concurrent**: bounded parallel requests to Ollama.

If no model is configured (``DESC_MODEL`` unset) or the model/Ollama is
unavailable, description generation is skipped gracefully — ingestion still
succeeds, entities simply get no ``summary`` vector.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections.abc import Callable
from pathlib import Path

import httpx

from services.chunker import CodeChunk
from services.retriever import _is_test_path


DESC_SYSTEM_PROMPT = (
    "You summarize code for a search index. Reply with ONE concise sentence "
    "(max 25 words) describing what the code does. No preamble, no code, no "
    "bullet points — just the sentence."
)

_MIN_SOURCE_LINES = 3  # fewer non-blank lines ⇒ too trivial to be worth an LLM call
_GEN_OPTIONS = {"temperature": 0.1, "num_predict": 64, "num_ctx": 4096}
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
# Ollama generation
# ---------------------------------------------------------------------------

async def _model_available(client: httpx.AsyncClient, base_url: str, model: str) -> bool:
    try:
        resp = await client.get(f"{base_url}/api/tags")
        resp.raise_for_status()
        available = [m["name"] for m in resp.json().get("models", [])]
        return model in available or any(a.split(":")[0] == model for a in available)
    except Exception:
        return False


async def _generate_one(
    client: httpx.AsyncClient, base_url: str, model: str, chunk: CodeChunk, keep_alive: str
) -> str:
    resp = await client.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "system": DESC_SYSTEM_PROMPT,
            "prompt": f"{chunk.entity_type} {chunk.qualified_name}\n\n{chunk.source_code}",
            "stream": False,
            "keep_alive": keep_alive,
            "options": _GEN_OPTIONS,
        },
    )
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()


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

    model = model if model is not None else os.getenv("DESC_MODEL", "")
    skipped_total = len(remaining)
    if not model:
        return {"enabled": False, "reason": "DESC_MODEL not set",
                "describable": 0, "cached": 0, "generated": 0, "failed": 0,
                "skipped": skipped_total, "templated": templated}

    base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    concurrency = concurrency or int(os.getenv("DESC_CONCURRENCY", "4"))
    keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
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
    timeout = httpx.Timeout(float(os.getenv("OLLAMA_TIMEOUT", "600")), connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if to_generate and not await _model_available(client, base_url, model):
            return {"enabled": True, "reason": f"model '{model}' unavailable",
                    "model": model, "describable": len(targets),
                    "cached": cached, "generated": 0, "failed": 0,
                    "pending": len(to_generate), "skipped": skipped,
                    "templated": templated}

        sem = asyncio.Semaphore(concurrency)
        done = 0

        async def worker(chunk: CodeChunk, source_hash: str) -> None:
            nonlocal generated, failed, done
            async with sem:
                try:
                    desc = await _generate_one(client, base_url, model, chunk, keep_alive)
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
