#!/usr/bin/env python
"""
Tests for ``services.describe``'s provider generalization (feature 22, C4):
``_resolve_provider``/``_descriptor_base_url`` resolve through the shared
registry in ``services.llm_provider`` (built-in, ``config/providers.json``
id, or a base_url-paired custom id) instead of ``describe.py``'s old three
hardcoded ``if provider_name == "openai"/"anthropic"/"ollama"`` branches —
while preserving:

  - the graceful-skip contract: ``_resolve_provider`` returns
    ``(None, reason)``, never raises — only ``_require_provider`` /
    ``ensure_descriptor_available`` / ``describe_chunks`` turn a skip into
    :class:`~services.describe.DescriptorError`;
  - the Ollama path, byte-identical (native branch, ``OLLAMA_BASE_URL``,
    ``effective_base_url()`` container rewriting);
  - the description cache key (source hash) — completely
    provider-independent, so switching ``DESC_PROVIDER`` between runs can
    never trigger a spurious re-describe of unchanged source;
  - ``--no-describe`` (``process_and_store(describe=False)``) stays zero
    LLM calls — a REAL behavioral check (``describe_chunks`` wired to
    raise if invoked), not a source-text grep.

Also covers (review round 2): ``DESC_BASE_URL`` routed through the same
``validate_base_url()`` every other caller-supplied base URL in this
codebase uses — scheme-less, hostless, embedded control characters,
malformed IPv6, whitespace-only — always a graceful skip, never a raised
``httpx.InvalidURL``/opaque SDK error; ``DESC_API_KEY`` precedence over
both a built-in's own env var and a file-registered provider's own
``${ENV}``-resolved key, and as the only way to authenticate an
unregistered remote id; and the warning logged (never the key) when a
built-in provider resolves with a ``DESC_BASE_URL`` override.

Offline only: ``services.describe.get_provider`` is monkeypatched for
every in-process scenario the ``key`` param below labels "stub" — zero
network, zero real API key. The scenarios labeled "real get_provider" call
the REAL ``services.llm_provider.get_provider`` but only exercise its
pure-validation failure paths (missing key / missing base_url), which
raise ``ValueError`` before any network call — still offline. The one
scenario needing an actual ``config/providers.json`` load (a
file-registered provider) runs in a subprocess with
``DOKKAI_PROVIDERS_FILE`` pointed at a scratch fixture (mirroring
``scripts/test_providers_file.py``) and never sets a real key, only a
fixture ``${ENV}`` var.

This script never loads ``.env`` and explicitly strips every real
``*_API_KEY`` from its own process env before running, so a developer's
real credentials never leak into these key-precedence checks.

Usage
-----
    uv run python scripts/test_describe_provider.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
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


# Strip every real *_API_KEY and every DESC_*/OLLAMA_BASE_URL var a
# developer's shell/.env may already have exported, BEFORE importing
# services.describe (which imports services.llm_provider, whose
# providers.json loader also reads DOKKAI_PROVIDERS_FILE — left alone here
# since no fixture is set for the in-process scenarios below).
def _clean_env() -> None:
    for k in list(os.environ):
        if k.endswith("_API_KEY") or k in (
            "DESC_PROVIDER", "DESC_MODEL", "DESC_BASE_URL",
            "DESC_CONCURRENCY", "OLLAMA_BASE_URL",
        ):
            os.environ.pop(k, None)


_clean_env()

import services.describe as describe  # noqa: E402
import services.llm_provider as llm_provider  # noqa: E402
from services.chunker import CodeChunk  # noqa: E402


def _make_chunk(source_code: str, name: str = "foo", entity_type: str = "Function") -> CodeChunk:
    return CodeChunk(
        node_id=1,
        entity_type=entity_type,
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
    """Never touches the network."""

    def __init__(self, name: str, reply: str = "a description") -> None:
        self.provider_name = name
        self._reply = reply

    async def health_check(self, model: str | None = None):
        return True, "ok"

    async def chat(self, messages, *, model, temperature=0.1, max_tokens=64):
        return self._reply


def _raising_get_provider(name, *, api_key="", base_url=None):
    raise AssertionError(f"get_provider must not be called here (name={name!r})")


def main() -> None:
    real_get_provider = describe.get_provider
    calls: list[tuple] = []

    def stub_get_provider(name, *, api_key="", base_url=None):
        calls.append((name, api_key, base_url))
        return _StubProvider(name)

    # -----------------------------------------------------------------
    # 1. Built-in with its standard env var set — key resolved via
    #    key_env_for(), not a hardcoded per-provider branch.
    # -----------------------------------------------------------------
    _clean_env()
    os.environ["OPENAI_API_KEY"] = "fixture-openai-key-not-real"
    describe.get_provider = stub_get_provider
    calls.clear()
    provider, reason = describe._resolve_provider("openai", "")
    check("built-in with key set: resolves, empty reason", provider is not None and reason == "", reason)
    check(
        "built-in with key set: get_provider receives the env-resolved key, no base_url",
        calls == [("openai", "fixture-openai-key-not-real", None)],
        calls,
    )

    _clean_env()
    os.environ["ANTHROPIC_API_KEY"] = "fixture-anthropic-key-not-real"
    calls.clear()
    provider, reason = describe._resolve_provider("anthropic", "")
    check("built-in anthropic with key set: resolves", provider is not None and reason == "", reason)
    check(
        "built-in anthropic: key comes from ANTHROPIC_API_KEY via key_env_for(), not a hardcoded branch",
        calls == [("anthropic", "fixture-anthropic-key-not-real", None)],
        calls,
    )

    # -----------------------------------------------------------------
    # 2. Built-in missing its key — graceful skip (None, reason), NEVER a
    #    raised exception. Real get_provider: the ValueError it raises for
    #    a missing built-in key is pure validation, no network call.
    # -----------------------------------------------------------------
    _clean_env()
    describe.get_provider = real_get_provider
    try:
        provider, reason = describe._resolve_provider("openai", "")
        raised = False
    except Exception:
        raised = True
        provider, reason = None, ""
    check("built-in missing key: _resolve_provider never raises", not raised)
    check("built-in missing key: graceful skip, provider is None", provider is None, provider)
    check("built-in missing key: reason names 'API key'", "API key" in reason, reason)

    # -----------------------------------------------------------------
    # 3. Custom (unregistered) id + DESC_BASE_URL — resolves keylessly
    #    (LM Studio/vLLM-style local server), no key required.
    # -----------------------------------------------------------------
    _clean_env()
    describe.get_provider = stub_get_provider
    calls.clear()
    provider, reason = describe._resolve_provider("mycustom", "http://localhost:9999/v1")
    check("custom id + base_url: resolves, empty reason", provider is not None and reason == "", reason)
    check(
        "custom id + base_url: no key required, base_url passed through",
        calls == [("mycustom", "", "http://localhost:9999/v1")],
        calls,
    )

    # -----------------------------------------------------------------
    # 4. Custom (unregistered) id WITHOUT a base_url — graceful skip, not
    #    a raise. Real get_provider's "Unknown provider" ValueError is
    #    pure validation (no network).
    # -----------------------------------------------------------------
    _clean_env()
    describe.get_provider = real_get_provider
    try:
        provider, reason = describe._resolve_provider("mycustom", "")
        raised = False
    except Exception:
        raised = True
        provider, reason = None, ""
    check("unregistered id, no base_url: _resolve_provider never raises", not raised)
    check("unregistered id, no base_url: graceful skip", provider is None, provider)
    check(
        "unregistered id, no base_url: reason names 'Unknown provider' (get_provider's own message)",
        "Unknown provider" in reason,
        reason,
    )

    # -----------------------------------------------------------------
    # 5. DESC_BASE_URL naming/precedence: Ollama keeps reading
    #    OLLAMA_BASE_URL (unaffected by DESC_BASE_URL); every other
    #    provider reads DESC_BASE_URL.
    # -----------------------------------------------------------------
    _clean_env()
    os.environ["OLLAMA_BASE_URL"] = "http://ollama-fixture:11434"
    os.environ["DESC_BASE_URL"] = "http://other-fixture:8000/v1"
    check(
        "_descriptor_base_url('ollama') reads OLLAMA_BASE_URL, ignores DESC_BASE_URL",
        describe._descriptor_base_url("ollama") == "http://ollama-fixture:11434",
        describe._descriptor_base_url("ollama"),
    )
    check(
        "_descriptor_base_url('anything-else') reads DESC_BASE_URL",
        describe._descriptor_base_url("groq") == "http://other-fixture:8000/v1",
        describe._descriptor_base_url("groq"),
    )
    _clean_env()
    check(
        "_descriptor_base_url('ollama') default unchanged (http://localhost:11434)",
        describe._descriptor_base_url("ollama") == "http://localhost:11434",
        describe._descriptor_base_url("ollama"),
    )
    check(
        "_descriptor_base_url for a non-ollama provider defaults to '' (unset), not ollama's default",
        describe._descriptor_base_url("groq") == "",
        describe._descriptor_base_url("groq"),
    )

    # -----------------------------------------------------------------
    # 6. Ollama path byte-identical: native branch, real get_provider,
    #    OllamaProvider construction/base_url match calling
    #    llm_provider.get_provider('ollama', base_url=...) directly — the
    #    exact call the OLD _resolve_provider's ollama branch made too
    #    (see git history: `if provider_name == "ollama": return
    #    get_provider("ollama", base_url=base_url), ""` is untouched by C4).
    # -----------------------------------------------------------------
    _clean_env()
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    describe.get_provider = real_get_provider
    base_url = describe._descriptor_base_url("ollama")
    provider, reason = describe._resolve_provider("ollama", base_url)
    check("ollama: resolves via the native path, empty reason", reason == "" and provider is not None, reason)
    check(
        "ollama: provider class is OllamaProvider (not routed through a shape)",
        type(provider).__name__ == "OllamaProvider",
        type(provider).__name__,
    )
    reference = llm_provider.get_provider("ollama", base_url=base_url)
    check(
        "ollama: constructed provider's _base_url matches get_provider('ollama', base_url=...) directly",
        provider._base_url == reference._base_url,
        (provider._base_url, reference._base_url),
    )
    check(
        "ollama: _num_ctx/_keep_alive/_timeout match the reference construction (env-derived, untouched by C4)",
        provider._num_ctx == reference._num_ctx
        and provider._keep_alive == reference._keep_alive
        and provider._timeout.read == reference._timeout.read,
        (provider._num_ctx, provider._keep_alive),
    )
    # DOKKAI_IN_CONTAINER rewrite (effective_base_url) — proves the
    # container-URL-rewrite path is untouched: same localhost -> host
    # .docker.internal rewrite as before, driven by the shared
    # effective_base_url() helper, not anything new in describe.py.
    os.environ["DOKKAI_IN_CONTAINER"] = "1"
    try:
        provider, _ = describe._resolve_provider("ollama", describe._descriptor_base_url("ollama"))
        reference = llm_provider.get_provider("ollama", base_url=describe._descriptor_base_url("ollama"))
        check(
            "ollama: DOKKAI_IN_CONTAINER rewrite (localhost -> host.docker.internal) matches the reference exactly",
            provider._base_url == reference._base_url and "host.docker.internal" in provider._base_url,
            provider._base_url,
        )
    finally:
        os.environ.pop("DOKKAI_IN_CONTAINER", None)

    # -----------------------------------------------------------------
    # 7. Cache key (source hash) is provider-independent: switching
    #    DESC_PROVIDER between two runs over the SAME source must never
    #    trigger a spurious re-describe — the highest-consequence failure
    #    this change could introduce, per the spec.
    # -----------------------------------------------------------------
    _clean_env()
    source = "def foo():\n    x = 1\n    return x\n"
    chunk_a = _make_chunk(source, name="samefn")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache.json"

        os.environ["DESC_PROVIDER"] = "stuba"
        describe.get_provider = lambda name, **kw: _StubProvider(name, reply="description from provider A")
        cache1 = describe.DescriptionCache(path=cache_path)
        stats1 = asyncio.run(
            describe.describe_chunks([chunk_a], model="fixture-model", cache=cache1)
        )
        check(
            "cache round trip, run 1: generated the description via provider A",
            chunk_a.description == "description from provider A" and stats1["generated"] == 1,
            (chunk_a.description, stats1),
        )
        check("cache round trip, run 1: cache file persisted", cache_path.exists())

        # Second run: DIFFERENT provider, SAME source. get_provider must
        # NEVER even be called (cache hit short-circuits provider
        # resolution entirely) — enforced by making it raise if it is.
        os.environ["DESC_PROVIDER"] = "stubb"
        describe.get_provider = _raising_get_provider
        chunk_b = _make_chunk(source, name="samefn")  # fresh instance, description=""
        cache2 = describe.DescriptionCache(path=cache_path)
        stats2 = asyncio.run(
            describe.describe_chunks([chunk_b], model="fixture-model", cache=cache2)
        )
        check(
            "cache round trip, run 2 (different DESC_PROVIDER, same source): cache HIT, no provider call",
            chunk_b.description == "description from provider A" and stats2["cached"] == 1 and stats2["generated"] == 0,
            (chunk_b.description, stats2),
        )

        # Control: the cache key really is the source hash, independent of
        # provider — direct proof on _source_hash() itself.
        os.environ["DESC_PROVIDER"] = "stuba"
        hash_a = describe._source_hash(chunk_a)
        os.environ["DESC_PROVIDER"] = "stubb"
        hash_b = describe._source_hash(chunk_a)
        check("source hash is identical across a DESC_PROVIDER change", hash_a == hash_b, (hash_a, hash_b))

    describe.get_provider = real_get_provider

    # -----------------------------------------------------------------
    # 8. --no-describe stays zero LLM calls: describe=False bypass lives
    #    in services.vectorize.process_and_store, not services.describe.
    #    LOAD-BEARING: actually calls process_and_store(describe=False)
    #    with describe_chunks wired to RAISE if invoked — a real
    #    behavioral check, not a source-text grep (which stays green even
    #    if the guard branch is deleted/inverted). The describe=True
    #    control run proves the wiring isn't just vacuously never-called.
    # -----------------------------------------------------------------
    import services.vectorize as vectorize

    class _FakeWeaviateClient:
        def close(self) -> None:
            pass

    def _fake_upsert_chunks(client, chunks, ingestion_id, on_purge=None):
        if on_purge:
            on_purge(0)
        return len(chunks)

    def _raising_describe_chunks(*args, **kwargs):
        raise AssertionError("describe_chunks must NOT be called when describe=False")

    _describe_calls = {"n": 0}

    async def _counting_describe_chunks(chunks, **kwargs):
        _describe_calls["n"] += 1
        return {"enabled": True, "model": "fixture-model", "describable": 0,
                "cached": 0, "generated": 0, "failed": 0, "pending": 0,
                "skipped": 0, "templated": 0}

    async def _run_process_and_store(describe_flag: bool, describe_stub):
        fake_chunk = _make_chunk("def guarded():\n    x = 1\n    return x\n", name="guarded")
        saved = {
            "chunk_graph": vectorize.chunk_graph,
            "describe_chunks": vectorize.describe_chunks,
            "get_client": vectorize.get_client,
            "ensure_collection": vectorize.ensure_collection,
            "upsert_chunks": vectorize.upsert_chunks,
        }
        vectorize.chunk_graph = lambda path, source_root=None: [fake_chunk]
        vectorize.describe_chunks = describe_stub
        vectorize.get_client = lambda: _FakeWeaviateClient()
        vectorize.ensure_collection = lambda client, recreate=False: None
        vectorize.upsert_chunks = _fake_upsert_chunks
        try:
            return await vectorize.process_and_store("fixture/graph.json", describe=describe_flag)
        finally:
            for name, fn in saved.items():
                setattr(vectorize, name, fn)

    result_no_describe = asyncio.run(_run_process_and_store(False, _raising_describe_chunks))
    check(
        "--no-describe: describe_chunks is never invoked (0 LLM calls) — real behavioral check",
        result_no_describe["descriptions"] == {"enabled": False, "reason": "descriptions disabled for this ingestion"},
        result_no_describe,
    )

    _describe_calls["n"] = 0
    result_describe = asyncio.run(_run_process_and_store(True, _counting_describe_chunks))
    check(
        "control: describe=True DOES invoke describe_chunks exactly once (proves the guard isn't vacuous)",
        _describe_calls["n"] == 1 and result_describe["descriptions"]["enabled"] is True,
        (_describe_calls["n"], result_describe),
    )

    # -----------------------------------------------------------------
    # 9. File-registered provider (config/providers.json id): needs no
    #    DESC_*-specific key — its ${ENV} key (or literal) resolves
    #    exactly the way get_provider() already resolves it for any other
    #    consumer. Runs in a subprocess (the loader is an import-time
    #    side effect — see scripts/test_providers_file.py).
    # -----------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        providers_path = Path(tmpdir) / "providers.json"
        providers_path.write_text(json.dumps({
            "providers": {
                "fixtureprov": {
                    "api": "openai-completions",
                    "baseUrl": "http://fixture.invalid/v1",
                    "apiKey": "${DESCRIBE_TEST_FIXTURE_KEY}",
                }
            }
        }))

        probe = (
            "import os, sys\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "import services.describe as describe\n"
            "\n"
            "class _Stub:\n"
            "    async def health_check(self, model=None):\n"
            "        return True, 'ok'\n"
            "\n"
            "captured = {}\n"
            "def _stub_get_provider(name, *, api_key='', base_url=None):\n"
            "    captured['args'] = (name, api_key, base_url)\n"
            "    return _Stub()\n"
            "describe.get_provider = _stub_get_provider\n"
            "\n"
            "provider, reason = describe._resolve_provider('fixtureprov', '')\n"
            "print('reason:', reason)\n"
            "print('resolved:', provider is not None)\n"
            "print('args:', captured.get('args'))\n"
        )
        env = {
            k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")
        }
        env["PYTHONPATH"] = str(SRC_DIR)
        env["DOKKAI_PROVIDERS_FILE"] = str(providers_path)
        env["DESCRIBE_TEST_FIXTURE_KEY"] = "fixture-file-registered-key-not-real"
        r = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, env=env, timeout=30,
        )
        check(
            "file-registered provider: subprocess runs cleanly",
            r.returncode == 0,
            (r.returncode, r.stdout, r.stderr),
        )
        check(
            "file-registered provider: resolves with no DESC_BASE_URL/DESC-specific key needed",
            "reason: \nresolved: True" in r.stdout,
            r.stdout,
        )
        check(
            "file-registered provider: get_provider called with NO api_key from describe.py (resolved internally by get_provider itself from the file's own ${ENV})",
            "args: ('fixtureprov', '', None)" in r.stdout,
            r.stdout,
        )

        # Same file, but the ${ENV} var is NOT set — graceful skip, not a
        # raise, using the REAL get_provider (no stub) so its own
        # diagnostic ValueError is what surfaces.
        probe_unset = (
            "import os, sys\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "import services.describe as describe\n"
            "try:\n"
            "    provider, reason = describe._resolve_provider('fixtureprov', '')\n"
            "    print('raised: False')\n"
            "except Exception as e:\n"
            "    print('raised:', True)\n"
            "    provider, reason = None, ''\n"
            "print('provider_none:', provider is None)\n"
            "print('reason:', reason)\n"
        )
        env2 = dict(env)
        env2.pop("DESCRIBE_TEST_FIXTURE_KEY", None)
        r2 = subprocess.run(
            [sys.executable, "-c", probe_unset],
            capture_output=True, text=True, env=env2, timeout=30,
        )
        check(
            "file-registered provider, unresolved ${ENV} key: never raises",
            r2.returncode == 0 and "raised: False" in r2.stdout,
            (r2.returncode, r2.stdout, r2.stderr),
        )
        check(
            "file-registered provider, unresolved ${ENV} key: graceful skip (provider is None)",
            "provider_none: True" in r2.stdout,
            r2.stdout,
        )
        check(
            "file-registered provider, unresolved ${ENV} key: reason names the missing var",
            "DESCRIBE_TEST_FIXTURE_KEY" in r2.stdout,
            r2.stdout,
        )
        check(
            # get_builtin_provider_ids() at _resolve_provider's enrichment
            # check (not get_registered_provider_ids()) is what makes
            # THIS the message that surfaces: get_provider's own SPECIFIC
            # diagnostic, which names the config/providers.json path that
            # references the unresolved var — swapping in
            # get_registered_provider_ids() there would make a
            # file-registered id match too (it IS "registered"), silently
            # replacing this with describe.py's own GENERIC "requires an
            # API key — set DESC_API_KEY, or VAR" message, which never
            # names the file. Both mention the var name, so a check that
            # only looked for that (the one above) can't tell them apart;
            # this one can.
            "file-registered provider, unresolved ${ENV} key: reason is get_provider's "
            "SPECIFIC diagnostic (names the providers.json path), not describe.py's "
            "generic built-in-only fallback message",
            str(providers_path) in r2.stdout,
            r2.stdout,
        )

    # -----------------------------------------------------------------
    # 10. DESC_BASE_URL validation (review round 2, point 1): routed
    #     through the SAME validate_base_url() every other caller-
    #     supplied base URL in this codebase uses — never handed to the
    #     SDK unchecked, never escapes as an unattributed httpx.InvalidURL/
    #     HTTP 500. Every rejection is a graceful (None, reason) skip.
    # -----------------------------------------------------------------
    _clean_env()
    describe.get_provider = real_get_provider

    def _assert_graceful_invalid(label: str, base_url: str, must_contain: str) -> None:
        try:
            provider, reason = describe._resolve_provider("mycustom", base_url)
            raised = False
        except Exception as e:
            raised, provider, reason = True, None, str(e)
        check(f"{label}: never raises", not raised, reason)
        check(f"{label}: graceful skip (provider is None)", provider is None, provider)
        check(f"{label}: reason names DESC_BASE_URL and echoes the validator's own message", (
            reason.startswith("invalid DESC_BASE_URL: ") and must_contain in reason
        ), reason)

    _assert_graceful_invalid("scheme-less DESC_BASE_URL", "localhost:1234/v1", "http")
    _assert_graceful_invalid("hostless DESC_BASE_URL", "http://", "host")
    _assert_graceful_invalid("hostless DESC_BASE_URL (single-slash typo)", "http:/localhost:1234/v1", "host")
    _assert_graceful_invalid("DESC_BASE_URL with an embedded tab", "http://local\thost:8000/v1", "tab or line break")
    _assert_graceful_invalid("DESC_BASE_URL with an embedded newline", "ht\ntp://x/v1", "tab or line break")
    _assert_graceful_invalid("malformed IPv6 DESC_BASE_URL", "http://[::1:8000/v1", "Invalid base_url")

    # Whitespace-only DESC_BASE_URL: _descriptor_base_url() already strips
    # to "" — treated exactly like unset (no validation error at all), not
    # a rejection with a confusing message about an "empty" URL.
    os.environ["DESC_BASE_URL"] = "   "
    check(
        "whitespace-only DESC_BASE_URL: _descriptor_base_url() normalizes to '' (treated as unset)",
        describe._descriptor_base_url("mycustom") == "",
        describe._descriptor_base_url("mycustom"),
    )
    del os.environ["DESC_BASE_URL"]

    # Valid base_url: still accepted (control — the validator isn't
    # rejecting everything).
    provider, reason = describe._resolve_provider("mycustom", "http://localhost:9999/v1")
    check("a genuinely valid DESC_BASE_URL is still accepted", provider is not None and reason == "", reason)

    # -----------------------------------------------------------------
    # 11. DESC_API_KEY precedence (review round 2, point 2) — all three
    #     levels: explicit DESC_API_KEY wins over EVERYTHING (a built-in's
    #     own env var, and a file-registered provider's own resolved
    #     key); with no DESC_API_KEY, each of those falls back to its own
    #     mechanism (already proven in sections 1 and 9 above).
    # -----------------------------------------------------------------
    _clean_env()
    describe.get_provider = stub_get_provider

    # Level 1 vs level 2: DESC_API_KEY beats a built-in's own env var.
    os.environ["OPENAI_API_KEY"] = "fixture-openai-own-key"
    os.environ["DESC_API_KEY"] = "fixture-desc-api-key-override"
    calls.clear()
    provider, reason = describe._resolve_provider("openai", "")
    check(
        "DESC_API_KEY overrides a built-in's own OPENAI_API_KEY",
        calls == [("openai", "fixture-desc-api-key-override", None)],
        calls,
    )

    # No DESC_API_KEY: falls back to the built-in's own env var (level 2).
    _clean_env()
    os.environ["OPENAI_API_KEY"] = "fixture-openai-own-key"
    calls.clear()
    provider, reason = describe._resolve_provider("openai", "")
    check(
        "no DESC_API_KEY: falls back to the built-in's own OPENAI_API_KEY",
        calls == [("openai", "fixture-openai-own-key", None)],
        calls,
    )

    # DESC_API_KEY authenticates an UNREGISTERED remote id that otherwise
    # has no way to receive a key at all (the Groq-style repro from the
    # review: DESC_PROVIDER=groq, DESC_BASE_URL set, no providers.json
    # entry — without DESC_API_KEY this silently sends api_key="").
    _clean_env()
    os.environ["DESC_API_KEY"] = "fixture-desc-api-key-for-unregistered"
    calls.clear()
    provider, reason = describe._resolve_provider("groq", "https://api.groq.example/openai/v1")
    check(
        "DESC_API_KEY authenticates an unregistered remote id (no other way to supply a key)",
        calls == [("groq", "fixture-desc-api-key-for-unregistered", "https://api.groq.example/openai/v1")],
        calls,
    )
    _clean_env()
    calls.clear()
    provider, reason = describe._resolve_provider("groq", "https://api.groq.example/openai/v1")
    check(
        "control: with no DESC_API_KEY, that same unregistered id resolves KEYLESS (proves it was DESC_API_KEY doing the work above)",
        calls == [("groq", "", "https://api.groq.example/openai/v1")],
        calls,
    )
    describe.get_provider = real_get_provider

    # Level 1 vs level 3: DESC_API_KEY beats a file-registered provider's
    # own ${ENV}-resolved key. Subprocess — the file load is import-time.
    with tempfile.TemporaryDirectory() as tmpdir:
        providers_path = Path(tmpdir) / "providers_desc_api_key.json"
        providers_path.write_text(json.dumps({
            "providers": {
                "fixtureprov2": {
                    "api": "openai-completions",
                    "baseUrl": "http://fixture2.invalid/v1",
                    "apiKey": "${DESC_API_KEY_TEST_FILE_FIXTURE_KEY}",
                }
            }
        }))
        probe_precedence = (
            "import os, sys\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "import services.describe as describe\n"
            "captured = {}\n"
            "def _stub_get_provider(name, *, api_key='', base_url=None):\n"
            "    captured['args'] = (name, api_key, base_url)\n"
            "    class _Stub:\n"
            "        async def health_check(self, model=None):\n"
            "            return True, 'ok'\n"
            "    return _Stub()\n"
            "describe.get_provider = _stub_get_provider\n"
            "provider, reason = describe._resolve_provider('fixtureprov2', '')\n"
            "print('args:', captured.get('args'))\n"
        )
        env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
        env["PYTHONPATH"] = str(SRC_DIR)
        env["DOKKAI_PROVIDERS_FILE"] = str(providers_path)
        env["DESC_API_KEY_TEST_FILE_FIXTURE_KEY"] = "fixture-file-own-key-not-real"
        env["DESC_API_KEY"] = "fixture-desc-api-key-beats-file"
        r = subprocess.run([sys.executable, "-c", probe_precedence], capture_output=True, text=True, env=env, timeout=30)
        check(
            "DESC_API_KEY overrides a file-registered provider's own ${ENV}-resolved key",
            r.returncode == 0 and "args: ('fixtureprov2', 'fixture-desc-api-key-beats-file', None)" in r.stdout,
            (r.returncode, r.stdout, r.stderr),
        )

    # -----------------------------------------------------------------
    # 11b. DESC_API_KEY whitespace-only fall-through (review round 3,
    #      point 2a): a blank/whitespace value is treated as UNSET, never
    #      sent verbatim as a garbage credential — it must fall all the
    #      way through to the next precedence level (the built-in's own
    #      env var, or keyless for an id with no other key source).
    # -----------------------------------------------------------------
    _clean_env()
    describe.get_provider = stub_get_provider
    os.environ["DESC_API_KEY"] = "   "
    os.environ["OPENAI_API_KEY"] = "fixture-openai-own-key-whitespace-test"
    calls.clear()
    provider, reason = describe._resolve_provider("openai", "")
    check(
        "whitespace-only DESC_API_KEY falls through to the built-in's own OPENAI_API_KEY (not used verbatim)",
        calls == [("openai", "fixture-openai-own-key-whitespace-test", None)],
        calls,
    )

    _clean_env()
    os.environ["DESC_API_KEY"] = "   "
    calls.clear()
    provider, reason = describe._resolve_provider("mycustom", "http://localhost:9999/v1")
    check(
        "whitespace-only DESC_API_KEY, no other key source: resolves KEYLESS, not with a whitespace string as the key "
        "(the exact silent-garbage-credential failure mode DESC_API_KEY's .strip() exists to prevent)",
        calls == [("mycustom", "", "http://localhost:9999/v1")],
        calls,
    )
    describe.get_provider = real_get_provider

    # -----------------------------------------------------------------
    # 12. DESC_BASE_URL-override warning (review round 2 point 5, round 3
    #     point 3): logged for ANY REGISTERED provider — every built-in
    #     (openai, gemini, AND anthropic — round 3 explicitly flagged that
    #     testing only 'openai' let a "shrink _BUILTIN_IDS" mutant survive)
    #     as well as a FILE-REGISTERED id (round 3 point 3: its own
    #     baseUrl is exactly as leftover-prone as a built-in's default).
    #     Names the provider and the HOST, never the key. No warning for a
    #     built-in with no override, and no warning for a genuinely
    #     UNREGISTERED id (DESC_BASE_URL is its only source of a base URL
    #     at all — there is nothing to "override").
    # -----------------------------------------------------------------
    import io
    import logging as _logging

    def _capture_warnings(fn):
        stream = io.StringIO()
        handler = _logging.StreamHandler(stream)
        handler.setLevel(_logging.WARNING)
        describe.logger.addHandler(handler)
        describe.logger.setLevel(_logging.WARNING)
        try:
            fn()
        finally:
            describe.logger.removeHandler(handler)
        return stream.getvalue()

    _clean_env()
    describe.get_provider = stub_get_provider
    os.environ["OPENAI_API_KEY"] = "fixture-openai-key-never-logged-xyz123"
    log_output = _capture_warnings(lambda: describe._resolve_provider("openai", "https://attacker.invalid/v1"))
    check(
        "built-in (openai) + base_url override: a warning IS logged",
        "DESC_BASE_URL override" in log_output,
        log_output,
    )
    check("warning names the provider ('openai')", "openai" in log_output, log_output)
    check("warning names the host ('attacker.invalid')", "attacker.invalid" in log_output, log_output)
    check(
        "warning NEVER includes the key value",
        "fixture-openai-key-never-logged-xyz123" not in log_output,
        log_output,
    )

    # Round 3, point 2b: gemini and anthropic too — not just openai. A
    # mutant shrinking the warning gate to {"openai"} only must go red on
    # BOTH of these, not just be silently unexercised.
    _clean_env()
    os.environ["GEMINI_API_KEY"] = "fixture-gemini-key-never-logged"
    log_output_gemini = _capture_warnings(lambda: describe._resolve_provider("gemini", "https://attacker2.invalid/v1"))
    check(
        "built-in (gemini) + base_url override: a warning IS logged, names gemini and the host",
        "DESC_BASE_URL override" in log_output_gemini
        and "gemini" in log_output_gemini
        and "attacker2.invalid" in log_output_gemini,
        log_output_gemini,
    )

    _clean_env()
    os.environ["ANTHROPIC_API_KEY"] = "fixture-anthropic-key-never-logged"
    log_output_anthropic = _capture_warnings(lambda: describe._resolve_provider("anthropic", "https://attacker3.invalid/v1"))
    check(
        "built-in (anthropic) + base_url override: a warning IS logged, names anthropic and the host",
        "DESC_BASE_URL override" in log_output_anthropic
        and "anthropic" in log_output_anthropic
        and "attacker3.invalid" in log_output_anthropic,
        log_output_anthropic,
    )

    _clean_env()
    os.environ["OPENAI_API_KEY"] = "fixture-openai-key"
    log_output_no_override = _capture_warnings(lambda: describe._resolve_provider("openai", ""))
    check("built-in with NO base_url override: no warning logged", log_output_no_override == "", log_output_no_override)

    log_output_custom = _capture_warnings(lambda: describe._resolve_provider("mycustom", "http://localhost:9999/v1"))
    check(
        "genuinely UNREGISTERED id + base_url: no warning logged (DESC_BASE_URL is its only source, not an override)",
        log_output_custom == "",
        log_output_custom,
    )
    describe.get_provider = real_get_provider

    # Round 3, point 3: a FILE-REGISTERED id (not a built-in) ALSO gets the
    # warning — its own baseUrl is exactly as leftover-prone as a
    # built-in's default. Subprocess (the file load is import-time).
    with tempfile.TemporaryDirectory() as tmpdir:
        providers_path = Path(tmpdir) / "providers_warning.json"
        providers_path.write_text(json.dumps({
            "providers": {
                "fixtureprov3": {
                    "api": "openai-completions",
                    "baseUrl": "https://fixture3.invalid/v1",
                    "apiKey": "${FIXTUREPROV3_FIXTURE_KEY}",
                }
            }
        }))
        probe_warning = (
            "import io, logging, os, sys\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "import services.describe as describe\n"
            "class _Stub:\n"
            "    async def health_check(self, model=None):\n"
            "        return True, 'ok'\n"
            "describe.get_provider = lambda name, **kw: _Stub()\n"
            "stream = io.StringIO()\n"
            "h = logging.StreamHandler(stream)\n"
            "describe.logger.addHandler(h); describe.logger.setLevel(logging.WARNING)\n"
            "describe._resolve_provider('fixtureprov3', 'http://localhost:4321/v1')\n"
            "print('LOG:', stream.getvalue())\n"
        )
        env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
        env["PYTHONPATH"] = str(SRC_DIR)
        env["DOKKAI_PROVIDERS_FILE"] = str(providers_path)
        env["FIXTUREPROV3_FIXTURE_KEY"] = "fixture-prov3-key-not-real"
        r = subprocess.run([sys.executable, "-c", probe_warning], capture_output=True, text=True, env=env, timeout=30)
        check(
            "a FILE-REGISTERED provider + DESC_BASE_URL override also gets the warning",
            r.returncode == 0
            and "DESC_BASE_URL override" in r.stdout
            and "fixtureprov3" in r.stdout
            and "localhost" in r.stdout,
            (r.returncode, r.stdout, r.stderr),
        )

    # -----------------------------------------------------------------
    # 12b. Missing-key skip reason names BOTH DESC_API_KEY and the
    #      built-in's own variable (review round 3, point 4) — HEAD said
    #      'OPENAI_API_KEY not set'; C4 must not regress to a message that
    #      names NEITHER variable, now that there are two ways to supply
    #      one. A file-registered id's own missing-${ENV} diagnostic (already
    #      more specific — names the var AND the config file) is untouched.
    # -----------------------------------------------------------------
    _clean_env()
    describe.get_provider = real_get_provider
    for builtin_id, key_var in (("openai", "OPENAI_API_KEY"), ("gemini", "GEMINI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")):
        provider, reason = describe._resolve_provider(builtin_id, "")
        check(
            f"missing-key reason for '{builtin_id}' names DESC_API_KEY",
            provider is None and "DESC_API_KEY" in reason,
            reason,
        )
        check(
            f"missing-key reason for '{builtin_id}' names its own var ({key_var})",
            provider is None and key_var in reason,
            reason,
        )

    # -----------------------------------------------------------------
    # 13. WIRING (review round 3, point 1 — BLOCKING): ensure_descriptor_
    #     available() (describe.py, the pre-flight controllers/instances.py
    #     runs on EVERY ingest and describe-refresh) and describe_chunks()
    #     (the generation pass itself) actually pass DESC_BASE_URL — not
    #     OLLAMA_BASE_URL — into _resolve_provider() for a remote
    #     DESC_PROVIDER. Sections 5/6/10 above test _descriptor_base_url()
    #     and _resolve_provider() each in ISOLATION; this proves the two
    #     lines that wire them together at the only two call sites that
    #     matter during a real ingest, by monkeypatching
    #     describe._resolve_provider itself (called by name from
    #     _require_provider, which both entry points share) and capturing
    #     exactly what base_url reaches it.
    # -----------------------------------------------------------------
    class _WiringProvider:
        async def health_check(self, model=None):
            return True, "ok"

        async def chat(self, messages, *, model, temperature=0.1, max_tokens=64):
            return "a wiring-test description"

    def _make_capturing_resolve_provider(sink):
        def _stub(provider_name, base_url):
            sink.append((provider_name, base_url))
            return _WiringProvider(), ""
        return _stub

    real_resolve_provider = describe._resolve_provider

    # 13a. ensure_descriptor_available(), remote provider: DESC_BASE_URL
    # reaches _resolve_provider; OLLAMA_BASE_URL (also set) is ignored.
    _clean_env()
    os.environ["DESC_PROVIDER"] = "groq"
    os.environ["DESC_MODEL"] = "fixture-model"
    os.environ["DESC_BASE_URL"] = "https://api.groq.example/openai/v1"
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    sink = []
    describe._resolve_provider = _make_capturing_resolve_provider(sink)
    try:
        asyncio.run(describe.ensure_descriptor_available())
    finally:
        describe._resolve_provider = real_resolve_provider
    check(
        "ensure_descriptor_available() wires DESC_BASE_URL (not OLLAMA_BASE_URL) into _resolve_provider for a remote DESC_PROVIDER",
        sink == [("groq", "https://api.groq.example/openai/v1")],
        sink,
    )

    # 13b. ensure_descriptor_available(), DESC_PROVIDER=ollama (default):
    # OLLAMA_BASE_URL reaches _resolve_provider, DESC_BASE_URL (also set)
    # is ignored — control proving 13a isn't "whatever's set wins".
    _clean_env()
    os.environ["DESC_MODEL"] = "fixture-model"
    os.environ["DESC_BASE_URL"] = "http://should-be-ignored.invalid/v1"
    os.environ["OLLAMA_BASE_URL"] = "http://ollama-fixture:11434"
    sink = []
    describe._resolve_provider = _make_capturing_resolve_provider(sink)
    try:
        asyncio.run(describe.ensure_descriptor_available())
    finally:
        describe._resolve_provider = real_resolve_provider
    check(
        "ensure_descriptor_available() wires OLLAMA_BASE_URL (not DESC_BASE_URL) into _resolve_provider for DESC_PROVIDER=ollama",
        sink == [("ollama", "http://ollama-fixture:11434")],
        sink,
    )

    # 13c. describe_chunks(), remote provider: same wiring, at the actual
    # generation call site (only reached when there's something to
    # generate — a cache/template-only run never calls _resolve_provider
    # at all, already proven in section 7 above).
    _clean_env()
    os.environ["DESC_PROVIDER"] = "groq"
    os.environ["DESC_BASE_URL"] = "https://api.groq.example/openai/v1"
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    sink = []
    describe._resolve_provider = _make_capturing_resolve_provider(sink)
    wiring_chunk = _make_chunk("def wiring():\n    y = 2\n    return y\n", name="wiringcall")
    try:
        with tempfile.TemporaryDirectory() as td:
            cache = describe.DescriptionCache(path=Path(td) / "wiring_cache.json")
            asyncio.run(describe.describe_chunks([wiring_chunk], model="fixture-model", cache=cache))
    finally:
        describe._resolve_provider = real_resolve_provider
    check(
        "describe_chunks() wires DESC_BASE_URL (not OLLAMA_BASE_URL) into _resolve_provider for a remote DESC_PROVIDER",
        sink == [("groq", "https://api.groq.example/openai/v1")],
        sink,
    )

    # 13d. describe_chunks(), DESC_PROVIDER=ollama (default): control.
    _clean_env()
    os.environ["OLLAMA_BASE_URL"] = "http://ollama-fixture2:11434"
    os.environ["DESC_BASE_URL"] = "http://should-be-ignored-2.invalid/v1"
    sink = []
    describe._resolve_provider = _make_capturing_resolve_provider(sink)
    wiring_chunk2 = _make_chunk("def wiring2():\n    z = 3\n    return z\n", name="wiringcall2")
    try:
        with tempfile.TemporaryDirectory() as td:
            cache = describe.DescriptionCache(path=Path(td) / "wiring_cache2.json")
            asyncio.run(describe.describe_chunks([wiring_chunk2], model="fixture-model", cache=cache))
    finally:
        describe._resolve_provider = real_resolve_provider
    check(
        "describe_chunks() wires OLLAMA_BASE_URL (not DESC_BASE_URL) into _resolve_provider for DESC_PROVIDER=ollama",
        sink == [("ollama", "http://ollama-fixture2:11434")],
        sink,
    )

    # -----------------------------------------------------------------
    # 15. validate_base_url() (services.llm_provider, shared by DESC_
    #     BASE_URL / /config/llm / providers.json) redacts embedded
    #     userinfo credentials from its error message (review round 3,
    #     point 5) — a malformed DESC_BASE_URL with Basic-auth-style
    #     credentials must not echo the password back into the graceful-
    #     skip reason (which surfaces as an HTTP 400 body / job log). Uses
    #     a FIXTURE placeholder value; the ``extra`` shown on a FAILING
    #     check below is deliberately a fixed placeholder, never the
    #     actual reason string, so a broken guard can't leak the secret
    #     into this test's own output either.
    # -----------------------------------------------------------------
    _clean_env()
    describe.get_provider = real_get_provider
    _fixture_secret = "URLSECRET123FIXTURE"
    _hostless_with_creds = f"https://bob:{_fixture_secret}@"
    provider, reason = describe._resolve_provider("mycustom", _hostless_with_creds)
    check(
        "DESC_BASE_URL with embedded userinfo credentials: graceful skip (invalid — hostless)",
        provider is None,
        "<redacted, see next check>",
    )
    check(
        "DESC_BASE_URL with embedded userinfo credentials: password is NEVER echoed in the skip reason",
        _fixture_secret not in reason,
        "<redacted — the point of this check is that the secret never appears>",
    )
    check(
        "DESC_BASE_URL with embedded userinfo credentials: reason still names the actionable rejection cause",
        "host" in reason,
        reason,
    )

    print(f"\n{_passed} passed, {_failed} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
