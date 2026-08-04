#!/usr/bin/env python
"""
Unit tests for ``services.llm_config.validate_and_save_config``'s remote-
branch validation (feature 22, C2): registry resolution, ``base_url``
normalization/validation (scheme, host, surrounding whitespace, malformed
IPv6), key-required-vs-optional, the ``ollama``-on-remote rejection, empty
``provider_name``, and the "reject before any network call" guarantee.

Offline only — ``services.llm_config.get_provider`` and
``save_persisted_config`` are monkeypatched, so this needs no live API key
and no Postgres/network at all.

Usage
-----
    uv run python scripts/test_llm_config_validation.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import services.llm_config as llm_config  # noqa: E402
from services.llm_config import get_config_store, validate_and_save_config  # noqa: E402

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


class _StubProvider:
    """Never touches the network — health_check just returns a canned result."""

    def __init__(self, healthy: bool = True, msg: str = "ok") -> None:
        self._healthy = healthy
        self._msg = msg

    async def health_check(self, model: str | None = None):
        return self._healthy, self._msg


async def _noop_save_persisted_config(config) -> None:
    """Replaces the real Postgres write-through — this test never touches Postgres."""
    return None


async def save(**kwargs):
    return await validate_and_save_config(is_local=False, **kwargs)


async def main() -> None:
    llm_config.save_persisted_config = _noop_save_persisted_config

    provider_calls = {"n": 0}

    def counting_get_provider(name, *, api_key="", base_url=None):
        provider_calls["n"] += 1
        return _StubProvider(healthy=True)

    llm_config.get_provider = counting_get_provider

    # -----------------------------------------------------------------
    # Empty / whitespace-only provider_name
    # -----------------------------------------------------------------
    ok, msg = await save(provider_name="", model="m", key="")
    check("empty provider_name rejected", not ok and "provider_name" in msg, msg)

    ok, msg = await save(provider_name="   ", model="m", key="")
    check("whitespace-only provider_name rejected", not ok and "provider_name" in msg, msg)

    # -----------------------------------------------------------------
    # ollama is only reachable through is_local=true
    # -----------------------------------------------------------------
    ok, msg = await save(
        provider_name="ollama", model="qwen2.5-coder", key="",
        base_url="http://localhost:11434",
    )
    check(
        "ollama on the remote branch rejected, points at is_local=true",
        not ok and "is_local=true" in msg,
        msg,
    )

    # -----------------------------------------------------------------
    # Built-in accepted with key; case/whitespace-insensitive id
    # -----------------------------------------------------------------
    ok, msg = await save(provider_name="openai", model="gpt-4o-mini", key="sk-fake")
    check("built-in openai accepted with key", ok, msg)

    ok, msg = await save(provider_name=" OpenAI ", model="gpt-4o-mini", key="sk-fake")
    check("built-in id resolution is case/whitespace insensitive", ok, msg)

    # -----------------------------------------------------------------
    # Built-in rejected without key
    # -----------------------------------------------------------------
    ok, msg = await save(provider_name="openai", model="gpt-4o-mini", key="")
    check("built-in openai rejected without key", not ok and "API key is required" in msg, msg)

    ok, msg = await save(provider_name="gemini", model="gemini-flash-latest", key="")
    check("built-in gemini rejected without key", not ok and "API key is required" in msg, msg)

    ok, msg = await save(provider_name="anthropic", model="claude-x", key="")
    check("built-in anthropic rejected without key", not ok and "API key is required" in msg, msg)

    # -----------------------------------------------------------------
    # Custom id: base_url required, key optional
    # -----------------------------------------------------------------
    ok, msg = await save(provider_name="groq", model="llama-3.3-70b", key="")
    check("custom provider without base_url rejected", not ok and "base_url" in msg, msg)

    ok, msg = await save(
        provider_name="lmstudio", model="local-model", key="",
        base_url="http://localhost:1234/v1",
    )
    check("custom provider with base_url and no key accepted", ok, msg)

    # -----------------------------------------------------------------
    # Scheme-less base_url
    # -----------------------------------------------------------------
    for bad in ("localhost:1234/v1", "api.groq.com/openai/v1"):
        ok, msg = await save(provider_name="groq", model="m", key="k", base_url=bad)
        check(f"scheme-less base_url rejected: {bad!r}", not ok and "http" in msg, msg)

    # -----------------------------------------------------------------
    # Hostless base_url, including the single-slash typo
    # -----------------------------------------------------------------
    for bad in ("http:/x", "http://", "http:///v1", "http:/localhost:1234/v1"):
        ok, msg = await save(provider_name="lmstudio", model="m", key="", base_url=bad)
        check(f"hostless base_url rejected: {bad!r}", not ok and "host" in msg, msg)

    # -----------------------------------------------------------------
    # Malformed IPv6 — urlsplit() itself raises ValueError, must become a
    # clean 400-style rejection here rather than propagating (see
    # controllers/config.py, which has no try around this call).
    # -----------------------------------------------------------------
    ok, msg = await save(
        provider_name="lmstudio", model="m", key="",
        base_url="http://[::1:8000/v1",
    )
    check(
        "unbalanced IPv6 bracket rejected cleanly, not a raw exception",
        not ok and "Invalid base_url" in msg,
        msg,
    )

    # Control case: a *valid* IPv6 host must still be accepted.
    ok, msg = await save(provider_name="lmstudio", model="m", key="", base_url="http://[::1]:8000/v1")
    check("valid IPv6 host accepted", ok, msg)

    # -----------------------------------------------------------------
    # Tab/CR/LF INSIDE the string. urlsplit() deletes these before
    # parsing while .strip() only touches the ends, so without an
    # explicit guard the string that gets validated is not the string
    # that gets stored and dialled. httpx rejects them either way, so
    # this is about owning the error message rather than handing the
    # user an SDK-worded one — and about the docstring's "validated ==
    # persisted == passed to get_provider" guarantee being literally
    # true. Note "ht\ntp://..." is exactly the shape that slipped
    # through before the guard existed.
    # -----------------------------------------------------------------
    for bad in ("ht\ntp://localhost:8000/v1", "http://local\thost:8000/v1", "https://h.test/v\r1"):
        ok, msg = await save(provider_name="lmstudio", model="m", key="", base_url=bad)
        check(
            f"embedded tab/CR/LF rejected here, not by the SDK: {bad!r}",
            not ok and "tab or line break" in msg,
            msg,
        )

    # -----------------------------------------------------------------
    # Whitespace normalization: the validated, persisted, and
    # get_provider()-passed base_url must all be the exact same
    # (stripped) string.
    # -----------------------------------------------------------------
    captured = {}

    def capturing_get_provider(name, *, api_key="", base_url=None):
        captured["base_url"] = base_url
        return _StubProvider(healthy=True)

    llm_config.get_provider = capturing_get_provider
    ok, msg = await save(
        provider_name="lmstudio", model="m", key="",
        base_url=" http://localhost:1234/v1\n",
    )
    check("surrounding whitespace stripped, config accepted", ok, msg)
    check(
        "normalized base_url passed to get_provider has no surrounding whitespace",
        captured.get("base_url") == "http://localhost:1234/v1",
        captured,
    )
    persisted = get_config_store().get_llm_config()
    check(
        "persisted base_url is the exact same normalized string passed to get_provider",
        persisted is not None and persisted.base_url == captured.get("base_url"),
        persisted,
    )

    # -----------------------------------------------------------------
    # No config-time rejection ever calls get_provider (i.e. never
    # touches the network) — a zero-attempt trip-wire across every
    # rejected shape above.
    # -----------------------------------------------------------------
    provider_calls["n"] = 0
    llm_config.get_provider = counting_get_provider
    rejected_cases = [
        dict(provider_name="", model="m", key=""),
        dict(provider_name="ollama", model="m", key="", base_url="http://localhost:11434"),
        dict(provider_name="openai", model="m", key=""),
        dict(provider_name="groq", model="m", key=""),
        dict(provider_name="groq", model="m", key="k", base_url="localhost:1234/v1"),
        dict(provider_name="lmstudio", model="m", key="", base_url="http://"),
        dict(provider_name="lmstudio", model="m", key="", base_url="http:/localhost:1234/v1"),
        dict(provider_name="lmstudio", model="m", key="", base_url="http://[::1:8000/v1"),
    ]
    for kwargs in rejected_cases:
        await save(**kwargs)
    check(
        "no config-time rejection ever constructs a provider / touches the network",
        provider_calls["n"] == 0,
        provider_calls,
    )

    print(f"\n{_passed} passed, {_failed} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
