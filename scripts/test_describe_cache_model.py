#!/usr/bin/env python
"""
Tests for the describe cache recording the producing provider+model
(feature 22, C8, decision ``22-cache-model``, revised after review):
``DescriptionCache`` keeps its pre-C8 on-disk shape — ``hash -> description``
bare strings — and adds a RESERVED SIDE-KEY, ``__models__: {hash: {"model":
..., "provider": ...}}``, rather than changing each entry's own shape. This
matters because:

  - the cache stays keyed on source hash ALONE — a provider/model switch
    must NEVER invalidate/re-describe an unchanged-source entry, only warn;
  - **no source hash can ever equal the literal string ``"__models__"``**
    (hashes are hex sha256; ``__models__`` isn't valid hex) — so a cache
    this code writes, read by ``main`` (the pre-C8 ``DescriptionCache``,
    which only ever addresses real hash keys), still yields plain strings
    for every entry: ``git checkout main`` against a cache this branch
    wrote is NOT a hard failure, unlike an earlier (reverted) design that
    changed each entry to a ``{"text": ..., "model": ...}`` dict;
  - a pre-C8/HEAD-written cache (no ``__models__`` at all) is read
    tolerantly: served as a normal hit, provider/model reported as
    unknown, and — the money-costing failure mode this whole commit exists
    to prevent — NEVER triggers an LLM call;
  - an unknown-producer (pre-C8/unrecorded) hit produces exactly ONE
    aggregate INFO line (not a warning, not per-entity) pointing at
    ``force: true`` — silence here would leave the one corpus that most
    needs the signal (every existing user's, today, 100% pre-C8) with no
    indication that a model upgrade requires an explicit re-describe;
  - a KNOWN provider/model mismatch (a cache hit recorded by a different
    provider and/or model — including the SAME model string served by a
    DIFFERENT provider, e.g. local ``ollama``+``qwen2.5-coder:3b`` vs a
    remote host serving the identically-named model) produces a single
    aggregate WARNING and is counted in ``stats["stale_model_hits"]``,
    while still being served as-is;
  - ``force=True`` still bypasses cache reads and regenerates, and the
    fresh result is written back WITH the currently configured
    provider+model recorded;
  - a malformed entry (wrong JSON type under a real hash key, or a
    malformed ``__models__``/per-hash producer record) is NEVER served —
    it's a safe miss (one redundant describe call), never a blank/garbage
    description silently indexed with no ``summary`` vector behind it;
  - a non-dict top-level JSON payload (corrupt file) doesn't raise out of
    ``DescriptionCache.__init__`` — same graceful-empty-cache behavior a
    JSON-decode failure already had;
  - ``DescriptionCache.save()``'s concurrent-writer merge keeps working
    when the two sides (on-disk vs in-memory) disagree about whether
    ``__models__`` is present at all, in both directions.

Offline only — ``services.describe.get_provider`` is monkeypatched (or, for
the tolerant-read/no-invalidation proofs, wired to RAISE if called at all —
a real behavioral check that a served cache hit never reaches the LLM).
Zero network, zero real API key. Never touches the real
``data/description_cache.json`` — every scenario uses a ``tempfile``
scratch path, and the real file's mtime is checked unchanged at the end.

Every scenario runs through ``_run_scenario()``, which converts ANY
exception — not just one raised inside an ``asyncio.run`` call, but also
one raised by, say, ``DescriptionCache.__init__`` itself reading a
mutant-corrupted file — into a single failing ``check()`` instead of
letting it escape (review item 7): one mutant-induced crash must never
abort the rest of the suite, in particular must never skip the final
"real cache untouched" guard.

Usage
-----
    uv run python scripts/test_describe_cache_model.py
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

_passed = 0
_failed = 0


def check(label: str, cond: bool, extra: object = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS: {label}")
    else:
        _failed += 1
        print(f"FAIL: {label} — {extra!r}")


def _run_scenario(name: str, fn) -> None:
    """
    Run one scenario function, converting ANY exception raised anywhere
    inside it (cache construction, save(), describe_chunks(), ...) into a
    single failing check instead of letting it propagate and kill every
    scenario that would have run after it.
    """
    try:
        fn()
    except Exception as e:
        check(f"{name}: scenario ran to completion without an unhandled exception", False, e)


def _clean_env() -> None:
    for k in list(os.environ):
        if k.endswith("_API_KEY") or k in (
            "DESC_PROVIDER", "DESC_MODEL", "DESC_BASE_URL",
            "DESC_CONCURRENCY", "OLLAMA_BASE_URL",
        ):
            os.environ.pop(k, None)


_clean_env()

import services.describe as describe  # noqa: E402
from services.chunker import CodeChunk  # noqa: E402

# Never touch the real cache — assert it's untouched at the very end too.
_REAL_CACHE_PATH = describe._default_cache_path()
_real_cache_mtime_before = (
    _REAL_CACHE_PATH.stat().st_mtime if _REAL_CACHE_PATH.exists() else None
)


def _make_chunk(source_code: str, name: str) -> CodeChunk:
    return CodeChunk(
        node_id=1,
        entity_type="Function",
        name=name,
        qualified_name=f"mod.{name}",
        imports=[],
        file_path=f"src/{name}.py",
        absolute_path=f"/repo/src/{name}.py",
        start_line=1,
        end_line=10,
        project_name="proj",
        module_name="mod",
        parent_class=None,
        source_code=source_code,
    )


class _StubProvider:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def health_check(self, model=None):
        return True, "ok"

    async def chat(self, messages, *, model, temperature=0.1, max_tokens=64):
        return self._reply


def _raising_get_provider(name, *, api_key="", base_url=None):
    raise AssertionError(
        f"get_provider must NEVER be called for a served cache hit (name={name!r})"
    )


def _run_describe(chunks, **kwargs):
    """
    Run describe_chunks(), capturing describe.logger's INFO+ output
    (level name included, so a test can tell an INFO line from a WARNING
    one). Exceptions are intentionally allowed to propagate here — the
    caller runs inside a ``_run_scenario()`` wrapper, which is the single
    place a crash is turned into a check() failure.
    """
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    handler.setLevel(logging.INFO)
    describe.logger.addHandler(handler)
    describe.logger.setLevel(logging.INFO)
    try:
        stats = asyncio.run(describe.describe_chunks(chunks, **kwargs))
    finally:
        describe.logger.removeHandler(handler)
    return stats, stream.getvalue()


class _HeadCacheSim:
    """
    Stand-in for ``main``'s (pre-C8) ``DescriptionCache`` — bare
    ``hash -> description`` strings only, no concept of ``__models__`` —
    reproduced inline (rather than checking out a different git ref
    mid-test) to prove the on-disk format is compatible in BOTH
    directions: this class reading a file the current code wrote, and the
    current code reading a file this class wrote (including after this
    class's own merge-on-save, which is oblivious to ``__models__`` and
    must pass it through untouched, not drop it).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
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
# Scenario 1 — pre-C8/HEAD-shaped cache file: bare hash -> string, NO
# __models__ side-key at all. Must be read tolerantly — served as a normal
# hit, ZERO LLM calls, provider/model reported as unknown, counted in
# unknown_model_hits (NOT stale_model_hits — there's nothing to compare),
# exactly ONE aggregate INFO line (not a warning), and the on-disk file
# left byte-for-byte untouched (describe_chunks only calls cache.save()
# when something was GENERATED — nothing is here).
# ---------------------------------------------------------------------------
def scenario_1_old_format_tolerant_no_invalidation() -> None:
    real_get_provider = describe.get_provider
    try:
        _clean_env()
        os.environ["DESC_PROVIDER"] = "fixtureprov"
        source1 = "def oldfmt():\n    x = 1\n    return x\n"
        chunk1 = _make_chunk(source1, "oldfmt")
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            hash1 = describe._source_hash(chunk1)
            cache_path.write_text(json.dumps({hash1: "an old cached description"}))
            before_bytes = cache_path.read_bytes()

            describe.get_provider = _raising_get_provider
            cache = describe.DescriptionCache(path=cache_path)
            stats1, log1 = _run_describe([chunk1], model="new-model", cache=cache)

            check(
                "old-format cache hit: description served as-is, zero LLM calls",
                chunk1.description == "an old cached description",
                chunk1.description,
            )
            check(
                "old-format cache hit: stats — 1 cached, 0 generated",
                stats1["cached"] == 1 and stats1["generated"] == 0,
                stats1,
            )
            check(
                "old-format (unknown-producer) cache hit: counted as unknown, NOT stale",
                stats1["unknown_model_hits"] == 1 and stats1["stale_model_hits"] == 0,
                stats1,
            )
            check(
                "old-format cache hit: exactly one INFO line, no WARNING",
                log1.count("INFO:") == 1 and "WARNING:" not in log1,
                log1,
            )
            check(
                "old-format cache hit: the INFO line points at force=True",
                "force=True" in log1 or "force: true" in log1.lower(),
                log1,
            )
            check(
                "old-format cache file is left byte-for-byte untouched on disk (nothing generated -> no save)",
                cache_path.read_bytes() == before_bytes,
                (before_bytes, cache_path.read_bytes()),
            )
        # structural guarantee the whole design leans on, checked here
        # while chunk1/hash1 are in scope.
        check(
            "structural guarantee: a real source hash (64 hex chars) can never equal '__models__' (10 chars)",
            len(hash1) == 64 and hash1 != "__models__",
            hash1,
        )
    finally:
        describe.get_provider = real_get_provider


# ---------------------------------------------------------------------------
# Scenario 2 — known provider/model mismatch: a cache entry with a
# __models__ record under a DIFFERENT provider+model than currently
# configured — served as-is (no re-describe, no LLM call), but DOES warn
# and DOES count in stats["stale_model_hits"]. Includes the feature-22
# headline case: the SAME model string served by a DIFFERENT provider
# (local ollama vs a remote host) must still be flagged. Control: matching
# provider+model -> no warning, count 0 (proves the check isn't
# unconditional).
# ---------------------------------------------------------------------------
def scenario_2_known_model_mismatch_warns_no_regen() -> None:
    real_get_provider = describe.get_provider
    try:
        _clean_env()
        os.environ["DESC_PROVIDER"] = "fixtureprov"
        source2 = "def switchfn():\n    y = 2\n    return y\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            chunk2a = _make_chunk(source2, "switchfn")
            hash2 = describe._source_hash(chunk2a)
            cache_path.write_text(json.dumps({
                hash2: "described by model A",
                "__models__": {hash2: {"model": "model-a", "provider": "ollama"}},
            }))

            describe.get_provider = _raising_get_provider
            cache = describe.DescriptionCache(path=cache_path)
            stats2, log2 = _run_describe([chunk2a], model="model-b", cache=cache)
            check(
                "model-mismatch cache hit: served as-is, zero LLM calls",
                chunk2a.description == "described by model A",
                chunk2a.description,
            )
            check(
                "model-mismatch cache hit: no re-description (generated stays 0)",
                stats2["cached"] == 1 and stats2["generated"] == 0,
                stats2,
            )
            check(
                "model-mismatch cache hit: counted in stats['stale_model_hits'], not unknown_model_hits",
                stats2["stale_model_hits"] == 1 and stats2["unknown_model_hits"] == 0,
                stats2,
            )
            check(
                "model-mismatch cache hit: WARNING logged, names old provider/model and new configured model",
                "WARNING:" in log2 and "model-a" in log2 and "model-b" in log2,
                log2,
            )

            # SAME model string, DIFFERENT provider — the feature-22
            # headline scenario (local ollama+qwen2.5-coder:3b vs a remote
            # host serving the identically-named model). Must still be
            # flagged stale.
            os.environ["DESC_PROVIDER"] = "ollama"
            chunk2c = _make_chunk(source2, "switchfn")
            cache_path.write_text(json.dumps({
                hash2: "described by ollama locally",
                "__models__": {hash2: {"model": "qwen2.5-coder:3b", "provider": "remote-gateway"}},
            }))
            cache_same_model = describe.DescriptionCache(path=cache_path)
            stats2c, _log2c = _run_describe(
                [chunk2c], model="qwen2.5-coder:3b", cache=cache_same_model
            )
            check(
                "same-model-string, DIFFERENT provider: still flagged stale (provider is compared too)",
                stats2c["stale_model_hits"] == 1,
                stats2c,
            )

            # Control: SAME configured provider+model as recorded -> no
            # mismatch, no warning.
            os.environ["DESC_PROVIDER"] = "fixtureprov"
            cache_path.write_text(json.dumps({
                hash2: "described by model A",
                "__models__": {hash2: {"model": "model-a", "provider": "fixtureprov"}},
            }))
            chunk2b = _make_chunk(source2, "switchfn")
            cache_same = describe.DescriptionCache(path=cache_path)
            stats2b, log2b = _run_describe([chunk2b], model="model-a", cache=cache_same)
            check(
                "control — configured provider+model matches recorded: no stale hit, no warning",
                stats2b["stale_model_hits"] == 0 and "WARNING:" not in log2b,
                (stats2b, log2b),
            )
    finally:
        describe.get_provider = real_get_provider


# ---------------------------------------------------------------------------
# Scenario 3 — force=True: bypasses the cache read (even for a
# pre-existing, old-format entry), regenerates via the provider, and the
# fresh result is written back WITH the currently configured
# provider+model recorded under __models__ — not the old value, not
# unrecorded.
# ---------------------------------------------------------------------------
def scenario_3_force_regenerates_and_records_producer() -> None:
    real_get_provider = describe.get_provider
    try:
        _clean_env()
        os.environ["DESC_PROVIDER"] = "fixtureprov"
        source3 = "def forcefn():\n    z = 3\n    return z\n"
        chunk3 = _make_chunk(source3, "forcefn")
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            hash3 = describe._source_hash(chunk3)
            cache_path.write_text(json.dumps({hash3: "stale text from before"}))

            calls: list[str] = []

            def stub_get_provider(name, *, api_key="", base_url=None):
                calls.append(name)
                return _StubProvider("fresh from model B")

            describe.get_provider = stub_get_provider
            cache = describe.DescriptionCache(path=cache_path)
            stats3, _log3 = _run_describe(
                [chunk3], model="model-b", cache=cache, force=True
            )
            check(
                "force=True: bypasses cache, actually calls the provider",
                len(calls) == 1,
                calls,
            )
            check(
                "force=True: regenerates (generated=1, cached=0), fresh text wins over stale cached text",
                stats3["generated"] == 1 and stats3["cached"] == 0
                and chunk3.description == "fresh from model B",
                (stats3, chunk3.description),
            )
            on_disk = json.loads(cache_path.read_text())
            check(
                "force=True: cache write preserves the entry as a BARE STRING (HEAD-compatible on-disk shape)",
                on_disk.get(hash3) == "fresh from model B",
                on_disk.get(hash3),
            )
            check(
                "force=True: __models__ side-key records the current provider AND model — not the old text, not unrecorded",
                on_disk.get("__models__", {}).get(hash3) == {"model": "model-b", "provider": "fixtureprov"},
                on_disk.get("__models__"),
            )
    finally:
        describe.get_provider = real_get_provider


# ---------------------------------------------------------------------------
# Scenario 4a — save() merge, direction 1: on-disk predates __models__
# entirely (HEAD-shaped — no side-key at all), in-memory has a fresh
# entry WITH a recorded producer (from set()). Both must survive the
# merge; the old entry stays a bare string with no producer recorded
# (still "unknown", not invented).
# ---------------------------------------------------------------------------
def scenario_4a_merge_direction_head_on_disk() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache.json"
        cache_path.write_text(json.dumps({"hashA": "textA old"}))

        cache = describe.DescriptionCache(path=cache_path)
        cache.set("hashB", "textB new", "model-x", "provider-x")
        cache.save()

        on_disk = json.loads(cache_path.read_text())
        check(
            "save() merge, direction 1 (on-disk has NO __models__, in-memory does): "
            "old entry survives as a bare string, still no producer recorded",
            on_disk.get("hashA") == "textA old" and "hashA" not in on_disk.get("__models__", {}),
            (on_disk.get("hashA"), on_disk.get("__models__")),
        )
        check(
            "save() merge, direction 1: new entry's text AND producer both persisted",
            on_disk.get("hashB") == "textB new"
            and on_disk.get("__models__", {}).get("hashB") == {"model": "model-x", "provider": "provider-x"},
            on_disk,
        )


# ---------------------------------------------------------------------------
# Scenario 4b — save() merge, direction 2: on-disk ALREADY has __models__
# (a concurrent writer's earlier save), and this process's own fresh
# in-memory entry CONFLICTS with one of those recorded hashes — the
# in-memory (fresher) text AND producer must both win together, not just
# the text. A non-conflicting recorded entry from the concurrent writer
# survives untouched.
# ---------------------------------------------------------------------------
def scenario_4b_merge_direction_concurrent_new_format() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache.json"
        cache = describe.DescriptionCache(path=cache_path)
        cache.set("hashC", "textC mine", "model-y", "provider-y")

        # Concurrent writer (already __models__-aware) lands a file with a
        # conflicting key (hashC, different text+producer) and a
        # non-conflicting key (hashD, its own recorded producer).
        cache_path.write_text(json.dumps({
            "hashC": "textC concurrent-stale",
            "hashD": "textD other",
            "__models__": {
                "hashC": {"model": "model-stale", "provider": "provider-stale"},
                "hashD": {"model": "model-d", "provider": "provider-d"},
            },
        }))

        cache.save()
        on_disk = json.loads(cache_path.read_text())
        check(
            "save() merge, direction 2 (on-disk already __models__-aware): "
            "in-memory text AND producer both win the conflicting key",
            on_disk.get("hashC") == "textC mine"
            and on_disk.get("__models__", {}).get("hashC") == {"model": "model-y", "provider": "provider-y"},
            (on_disk.get("hashC"), on_disk.get("__models__", {}).get("hashC")),
        )
        check(
            "save() merge, direction 2: the concurrent writer's non-conflicting recorded entry survives untouched",
            on_disk.get("hashD") == "textD other"
            and on_disk.get("__models__", {}).get("hashD") == {"model": "model-d", "provider": "provider-d"},
            (on_disk.get("hashD"), on_disk.get("__models__", {}).get("hashD")),
        )


# ---------------------------------------------------------------------------
# Scenario 5 — malformed entries are NEVER served: a safe miss (one
# redundant describe call), never a blank/garbage description. Covers the
# exact inversion the review flagged: a mutant that serves a corrupt entry
# as a hit with blank text (silently producing no summary vector for that
# entity) must go red here. Also covers a malformed __models__ side-key
# and a malformed per-hash producer record — never raise, never crash
# reading the descriptions themselves.
# ---------------------------------------------------------------------------
def scenario_5_malformed_entries_never_served() -> None:
    real_get_provider = describe.get_provider
    try:
        _clean_env()
        os.environ["DESC_PROVIDER"] = "fixtureprov"
        source5 = "def malformedfn():\n    w = 5\n    return w\n"
        chunk5 = _make_chunk(source5, "malformedfn")
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            hash5 = describe._source_hash(chunk5)
            # A non-string value under a real hash key (corruption / a
            # leftover from the reverted per-entry-shape design).
            cache_path.write_text(json.dumps({hash5: 12345}))

            gen_calls: list[str] = []

            def stub_get_provider_5(name, *, api_key="", base_url=None):
                gen_calls.append(name)
                return _StubProvider("freshly generated, not blank")

            describe.get_provider = stub_get_provider_5
            cache5 = describe.DescriptionCache(path=cache_path)
            stats5, _log5 = _run_describe([chunk5], model="model-b", cache=cache5)
            check(
                "malformed entry (non-string value): treated as a MISS, provider IS called",
                len(gen_calls) == 1,
                gen_calls,
            )
            check(
                "malformed entry: never served blank/garbage — description is the freshly generated text",
                chunk5.description == "freshly generated, not blank" and stats5["cached"] == 0
                and stats5["generated"] == 1,
                (chunk5.description, stats5),
            )
    finally:
        describe.get_provider = real_get_provider

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache.json"
        cache_path.write_text(json.dumps({
            "hashE": "textE",
            "hashF": "textF",
            "__models__": "not-a-dict-at-all",
        }))
        cache_e = describe.DescriptionCache(path=cache_path)
        check(
            "malformed __models__ (wrong type): descriptions still read fine, producer reported as unknown",
            cache_e.get("hashE") == ("textE", None, None),
            cache_e.get("hashE"),
        )

        cache_path.write_text(json.dumps({
            "hashG": "textG",
            "__models__": {"hashG": "not-a-dict-either"},
        }))
        cache_g = describe.DescriptionCache(path=cache_path)
        check(
            "malformed per-hash producer record: descriptions still read fine, producer reported as unknown",
            cache_g.get("hashG") == ("textG", None, None),
            cache_g.get("hashG"),
        )


# ---------------------------------------------------------------------------
# Scenario 6 — non-dict top-level JSON payload: DescriptionCache.__init__
# must NOT raise (same graceful-empty-cache behavior a JSON-decode failure
# already had). Exercised both as a direct constructor check and inside a
# full describe_chunks() run (must regenerate, not crash).
# ---------------------------------------------------------------------------
def scenario_6_nondict_toplevel_json_is_graceful() -> None:
    for label, bad_payload in (
        ("a JSON array", [1, 2, 3]),
        ("a bare JSON string", "oops, not an object"),
        ("a JSON number", 42),
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            cache_path.write_text(json.dumps(bad_payload))
            try:
                cache_bad = describe.DescriptionCache(path=cache_path)
                raised = False
            except Exception:
                raised = True
                cache_bad = None
            check(f"non-dict top-level JSON ({label}): constructor does not raise", not raised)
            if not raised:
                check(
                    f"non-dict top-level JSON ({label}): reads as an empty cache (miss for everything)",
                    cache_bad.get("anything") is None,
                    cache_bad.get("anything"),
                )

    real_get_provider = describe.get_provider
    try:
        _clean_env()
        os.environ["DESC_PROVIDER"] = "fixtureprov"
        source6 = "def nondictfn():\n    v = 6\n    return v\n"
        chunk6 = _make_chunk(source6, "nondictfn")
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            cache_path.write_text(json.dumps([1, 2, 3]))

            describe.get_provider = lambda name, **kw: _StubProvider("generated over a corrupt cache file")
            cache6 = describe.DescriptionCache(path=cache_path)
            stats6, _log6 = _run_describe([chunk6], model="model-b", cache=cache6)
            check(
                "describe_chunks() over a non-dict-JSON cache file: regenerates instead of crashing",
                stats6["generated"] == 1 and chunk6.description == "generated over a corrupt cache file",
                (stats6, chunk6.description),
            )
    finally:
        describe.get_provider = real_get_provider


# ---------------------------------------------------------------------------
# Scenario 7 — format compatibility, BOTH directions, using an inline
# stand-in for main's (pre-C8) DescriptionCache: no real git checkout
# needed — the point is the ON-DISK SHAPE, and _HeadCacheSim reproduces
# main's actual read/get/save logic verbatim.
# ---------------------------------------------------------------------------
def scenario_7_bidirectional_format_compat_with_head() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache.json"

        # This branch's code writes a cache entry WITH a recorded producer.
        c8_cache = describe.DescriptionCache(path=cache_path)
        hash_c8 = "a" * 64  # hex-shaped, like a real sha256 hex digest
        c8_cache.set(hash_c8, "text one", "model-1", "provider-1")
        c8_cache.save()

        head_cache = _HeadCacheSim(cache_path)
        check(
            "direction A — HEAD reading a file THIS code wrote: the real entry is a bare string, exactly the text",
            isinstance(head_cache.get(hash_c8), str) and head_cache.get(hash_c8) == "text one",
            head_cache.get(hash_c8),
        )

        # HEAD generates and writes its own new entry (its own save(), its
        # own merge logic — no knowledge of __models__ at all), then this
        # code reads the result: hash_c8's producer record must have
        # survived HEAD's round trip untouched, and the new HEAD-origin
        # entry is read fine too, reported as producer-unknown.
        hash_head = "b" * 64
        head_cache.set(hash_head, "text two - from head")
        head_cache.save()

        c8_cache_after = describe.DescriptionCache(path=cache_path)
        check(
            "direction A cont'd — HEAD's own save() preserves __models__ untouched (passes it through as an ordinary key)",
            c8_cache_after.get(hash_c8) == ("text one", "model-1", "provider-1"),
            c8_cache_after.get(hash_c8),
        )
        check(
            "direction B — this code reading a file HEAD wrote (after a round trip through HEAD's save()): "
            "HEAD-origin entry reads fine, producer reported as unknown",
            c8_cache_after.get(hash_head) == ("text two - from head", None, None),
            c8_cache_after.get(hash_head),
        )


# ---------------------------------------------------------------------------
# Scenario 8 — the real data/description_cache.json is never touched by
# any of the above — every scenario used a tempfile path.
# ---------------------------------------------------------------------------
def scenario_8_real_cache_file_untouched() -> None:
    real_cache_mtime_after = (
        _REAL_CACHE_PATH.stat().st_mtime if _REAL_CACHE_PATH.exists() else None
    )
    check(
        "the real data/description_cache.json was not touched by this test run",
        real_cache_mtime_after == _real_cache_mtime_before,
        (_real_cache_mtime_before, real_cache_mtime_after),
    )


def main() -> None:
    _run_scenario("scenario 1 (old-format tolerant read)", scenario_1_old_format_tolerant_no_invalidation)
    _run_scenario("scenario 2 (known model mismatch)", scenario_2_known_model_mismatch_warns_no_regen)
    _run_scenario("scenario 3 (force=True)", scenario_3_force_regenerates_and_records_producer)
    _run_scenario("scenario 4a (merge, HEAD-shaped on disk)", scenario_4a_merge_direction_head_on_disk)
    _run_scenario("scenario 4b (merge, __models__-aware on disk)", scenario_4b_merge_direction_concurrent_new_format)
    _run_scenario("scenario 5 (malformed entries)", scenario_5_malformed_entries_never_served)
    _run_scenario("scenario 6 (non-dict top-level JSON)", scenario_6_nondict_toplevel_json_is_graceful)
    _run_scenario("scenario 7 (bidirectional format compat)", scenario_7_bidirectional_format_compat_with_head)
    _run_scenario("scenario 8 (real cache untouched)", scenario_8_real_cache_file_untouched)

    print(f"\n{_passed} passed, {_failed} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
