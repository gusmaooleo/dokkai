#!/usr/bin/env python
"""
Tests for ``GET /config/llm/providers`` (feature 22, C5) —
``controllers.config.list_providers``'s shape and the display decisions it
encodes:

- ``key_required`` mirrors ``services.llm_provider.key_env_for`` exactly
  (the same predicate ``services.llm_config.validate_and_save_config``
  consults), so the UI never demands a key the server wouldn't also
  require, and never skips one the server would reject.
- ``registered_via_file`` and ``label`` are derived from
  ``services.llm_provider.get_builtin_provider_ids()``/``display_name_for()``
  — never a second, driftable copy of the three built-in ids, and never a
  caller-side ``str.title()`` mangling of a file-registered id's original
  casing.
- ``default_base_url`` is populated ONLY for a built-in id (from
  ``default_base_url_for()``) — always ``null`` for a file-registered
  provider, even one whose own ``baseUrl`` is real and non-null, because
  that value may legitimately carry embedded Basic-auth credentials and
  this endpoint is readable by any authenticated user, not just an admin.

Most scenarios run the REAL production endpoint function
(``controllers.config.list_providers``) in a subprocess (``python -c``,
fresh interpreter per scenario — the ``config/providers.json`` loader is a
module-import-time side effect, same reason ``test_providers_file.py``
uses this shape), with ``DOKKAI_PROVIDERS_FILE`` pointed at a scratch
fixture. This is the exact function FastAPI dispatches to for the route
(no test client needed there — the function takes no request data, so
calling it directly IS exercising the real route body).

The auth-posture check is the one exception: a real, in-process FastAPI
``TestClient`` mounting the actual router, with ``services.auth.
resolve_session`` monkeypatched (no DB) — a behavioral proof (401/200/403)
rather than a source-text scan, which would pass even if the dependency
just moved somewhere a text search wouldn't catch it.

Offline only: no network, no Postgres, no real API key. Every ``${...}``
this test resolves is a fabricated fixture var, never a real ``*_API_KEY``
— ``run()`` strips those from the child environment before adding
fixture-only vars back in (same discipline as ``test_providers_file.py``).

Usage
-----
    uv run python scripts/test_config_providers_endpoint.py
"""

from __future__ import annotations

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


def write_fixture(tmpdir: str, name: str, data: object) -> str:
    path = os.path.join(tmpdir, name)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def write_raw_fixture(tmpdir: str, name: str, text: str) -> str:
    path = os.path.join(tmpdir, name)
    with open(path, "w") as f:
        f.write(text)
    return path


CALL_ENDPOINT = (
    "import asyncio, json; "
    "from controllers.config import list_providers; "
    "r = asyncio.run(list_providers()); "
    "print(json.dumps(r.model_dump()))"
)


def run(
    providers_file: str | None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run the real endpoint function in a fresh interpreter."""
    env = {
        k: v
        for k, v in os.environ.items()
        # Never let a real .env-loaded *_API_KEY leak into a child whose
        # whole point is testing ${VAR} resolution against fixture vars.
        if not k.endswith("_API_KEY")
    }
    env["PYTHONPATH"] = str(SRC_DIR)
    if providers_file is not None:
        env["DOKKAI_PROVIDERS_FILE"] = providers_file
    else:
        env.pop("DOKKAI_PROVIDERS_FILE", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", CALL_ENDPOINT],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=30,
    )


def by_id(providers: list[dict], pid: str) -> dict | None:
    return next((p for p in providers if p["id"] == pid), None)


def test_endpoint_scenarios(tmpdir: str) -> None:
    # -----------------------------------------------------------------
    # No providers file: exactly the three built-ins, each requiring a
    # key, none file-registered, sorted by id. Gemini carries its
    # non-guessable default_base_url; openai/anthropic don't override
    # the SDK's own official default, so theirs is null.
    # -----------------------------------------------------------------
    absent_path = os.path.join(tmpdir, "does-not-exist.json")
    r = run(absent_path)
    check("no file: exits 0", r.returncode == 0, (r.returncode, r.stdout, r.stderr))
    body = json.loads(r.stdout) if r.returncode == 0 else {"providers": []}
    providers = body.get("providers", [])
    check(
        "no file: exactly the three built-in ids, sorted",
        [p["id"] for p in providers] == ["anthropic", "gemini", "openai"],
        providers,
    )
    check(
        "no file: every built-in requires a key",
        all(p["key_required"] for p in providers),
        providers,
    )
    check(
        "no file: no built-in is marked file-registered",
        not any(p["registered_via_file"] for p in providers),
        providers,
    )
    check(
        "no file: 'ollama' never appears in this listing",
        by_id(providers, "ollama") is None,
        providers,
    )
    check(
        "no file: built-in labels are human-readable, not raw ids",
        by_id(providers, "openai")["label"] == "OpenAI"
        and by_id(providers, "gemini")["label"] == "Gemini"
        and by_id(providers, "anthropic")["label"] == "Anthropic",
        providers,
    )
    check(
        "no file: gemini carries its non-guessable default_base_url",
        by_id(providers, "gemini")["default_base_url"]
        == "https://generativelanguage.googleapis.com/v1beta/openai/",
        by_id(providers, "gemini"),
    )
    check(
        "no file: openai/anthropic have no default_base_url override (SDK's own official default applies)",
        by_id(providers, "openai")["default_base_url"] is None
        and by_id(providers, "anthropic")["default_base_url"] is None,
        providers,
    )
    check(
        "no file: response schema has no key-value field at all (only key_required, a bool)",
        "apiKey" not in r.stdout and '"key":' not in r.stdout,
        r.stdout,
    )
    check(
        "no file: every built-in has an empty models[] (they never declare one)",
        all(p["models"] == [] for p in providers),
        providers,
    )

    # -----------------------------------------------------------------
    # A file-registered provider whose declared ${ENV_VAR} key is NOT
    # currently set: key_required=true (matches key_env_for /
    # validate_and_save_config exactly — the UI must still demand one),
    # registered_via_file=true, default_base_url=null EVEN THOUGH the
    # entry has a real, non-null baseUrl of its own — this endpoint must
    # never expose a file-registered provider's own base URL (it may
    # legitimately carry embedded credentials).
    # -----------------------------------------------------------------
    unresolved_path = write_fixture(
        tmpdir,
        "unresolved.json",
        {
            "providers": {
                "groq": {
                    "api": "openai-completions",
                    "baseUrl": "https://api.groq.com/openai/v1",
                    "apiKey": "${SCRATCH_TEST_GROQ_KEY}",
                    "models": ["llama-3.3-70b-versatile"],
                }
            }
        },
    )
    r = run(unresolved_path)
    body = json.loads(r.stdout) if r.returncode == 0 else {"providers": []}
    groq = by_id(body.get("providers", []), "groq")
    check("unresolved ${ENV} key: groq appears in the listing", groq is not None, body)
    check(
        "unresolved ${ENV} key: key_required is true (env var not set)",
        groq is not None and groq["key_required"] is True,
        groq,
    )
    check(
        "unresolved ${ENV} key: registered_via_file is true",
        groq is not None and groq["registered_via_file"] is True,
        groq,
    )
    check(
        "unresolved ${ENV} key: default_base_url is null despite having a real baseUrl of its own",
        groq is not None and groq["default_base_url"] is None,
        groq,
    )
    check(
        "unresolved ${ENV} key: built-ins still present alongside it",
        {p["id"] for p in body.get("providers", [])} == {"anthropic", "gemini", "groq", "openai"},
        body,
    )
    check(
        "unresolved ${ENV} key: groq's own base URL string never appears in the response",
        "api.groq.com" not in r.stdout,
        r.stdout,
    )
    check(
        "unresolved ${ENV} key: groq's declared models[] is delivered (unlike default_base_url, "
        "not restricted to built-ins — a model name carries no secret)",
        groq is not None and groq["models"] == ["llama-3.3-70b-versatile"],
        groq,
    )

    # Same fixture, but with the env var actually set (live re-check,
    # not a load-time snapshot) — flips key_required to false, matching
    # key_env_for's own "checked live" contract.
    r = run(unresolved_path, extra_env={"SCRATCH_TEST_GROQ_KEY": "fixture-only-not-a-real-key"})
    body = json.loads(r.stdout) if r.returncode == 0 else {"providers": []}
    groq = by_id(body.get("providers", []), "groq")
    check(
        "${ENV} key now set: key_required flips to false, same process semantics as key_env_for",
        groq is not None and groq["key_required"] is False,
        groq,
    )
    check(
        "${ENV} key now set: still registered_via_file",
        groq is not None and groq["registered_via_file"] is True,
        groq,
    )
    check(
        "${ENV} key now set: fixture key value never echoed in the response",
        "fixture-only-not-a-real-key" not in r.stdout,
        r.stdout,
    )

    # -----------------------------------------------------------------
    # A file-registered provider with a literal apiKey AND a baseUrl that
    # legitimately embeds Basic-auth userinfo: key_required=false, and
    # NEITHER the literal key NOR the userinfo-bearing base URL ever
    # appears in the response — default_base_url stays null for it.
    # -----------------------------------------------------------------
    literal_key_path = write_fixture(
        tmpdir,
        "literal_key.json",
        {
            "providers": {
                "fixedkey": {
                    "api": "openai-completions",
                    "baseUrl": "http://svc-user:svc-pass@localhost:9999/v1",
                    "apiKey": "sk-fixture-literal-not-a-real-key",
                }
            }
        },
    )
    r = run(literal_key_path)
    body = json.loads(r.stdout) if r.returncode == 0 else {"providers": []}
    fixedkey = by_id(body.get("providers", []), "fixedkey")
    check("literal apiKey: key_required is false", fixedkey is not None and fixedkey["key_required"] is False, fixedkey)
    check("literal apiKey: registered_via_file is true", fixedkey is not None and fixedkey["registered_via_file"] is True, fixedkey)
    check(
        "literal apiKey: default_base_url is null (its own baseUrl, even with embedded userinfo, is never exposed)",
        fixedkey is not None and fixedkey["default_base_url"] is None,
        fixedkey,
    )
    check(
        "literal apiKey: the literal key value never appears in the response",
        "sk-fixture-literal-not-a-real-key" not in r.stdout,
        r.stdout,
    )
    check(
        "literal apiKey: the embedded Basic-auth userinfo never appears in the response",
        "svc-user" not in r.stdout and "svc-pass" not in r.stdout,
        r.stdout,
    )

    # -----------------------------------------------------------------
    # A deliberately keyless file-registered provider (no apiKey field at
    # all) with a MIXED-CASE id as written in the file: key_required is
    # false, and the label preserves the file's original casing — NOT a
    # caller-side str.title() of the normalized lowercase id, which
    # mangles anything with internal capitals ('OpenRouter' -> would
    # wrongly become 'Openrouter').
    # -----------------------------------------------------------------
    keyless_path = write_fixture(
        tmpdir,
        "keyless.json",
        {
            "providers": {
                "OpenRouter": {
                    "api": "openai-completions",
                    "baseUrl": "https://openrouter.ai/api/v1",
                }
            }
        },
    )
    r = run(keyless_path)
    body = json.loads(r.stdout) if r.returncode == 0 else {"providers": []}
    openrouter = by_id(body.get("providers", []), "openrouter")
    check("keyless provider: key_required is false", openrouter is not None and openrouter["key_required"] is False, openrouter)
    check("keyless provider: registered_via_file is true", openrouter is not None and openrouter["registered_via_file"] is True, openrouter)
    check(
        "keyless provider: label preserves the file's original casing ('OpenRouter', not 'Openrouter')",
        openrouter is not None and openrouter["label"] == "OpenRouter",
        openrouter,
    )
    check(
        "keyless provider with no declared models[]: models is empty (free-text field), not omitted/null",
        openrouter is not None and openrouter["models"] == [],
        openrouter,
    )

    # -----------------------------------------------------------------
    # Sort-order regression guard, with enough fixture volume that a
    # regression removing sorted() is caught deterministically rather
    # than probabilistically. get_registered_provider_ids() returns a
    # frozenset — iterating it directly (no sorted()) is near-uniform
    # random per process (hash randomization), so with only the 3
    # built-ins a broken build still happens to print sorted order ~1/6
    # of the time. With 15 total ids the coincidence probability is
    # 1/15! — not exercisable in practice.
    # -----------------------------------------------------------------
    many_ids = [f"scratch-provider-{i:02d}" for i in range(12)]
    # Insert in a deliberately UN-sorted order, so an accidental
    # "insertion order happened to already be alphabetical" false-negative
    # is also ruled out.
    shuffled = many_ids[::-1]
    many_path = write_fixture(
        tmpdir,
        "many.json",
        {
            "providers": {
                pid: {"api": "openai-completions", "baseUrl": f"http://localhost:9/{pid}"}
                for pid in shuffled
            }
        },
    )
    r = run(many_path)
    body = json.loads(r.stdout) if r.returncode == 0 else {"providers": []}
    all_ids = [p["id"] for p in body.get("providers", [])]
    check(
        "sort-order guard (15 ids): response is in exact sorted order",
        all_ids == sorted(["anthropic", "gemini", "openai", *many_ids]),
        all_ids,
    )


def test_malformed_file_boot_failure(tmpdir: str) -> None:
    """
    A malformed config/providers.json fails the WHOLE process at import —
    the endpoint never gets a chance to serve a silently empty/partial
    listing. Complements test_providers_file.py's loader-level coverage
    by proving it through this specific caller (the endpoint function).
    """
    malformed_path = write_raw_fixture(tmpdir, "malformed.json", "{not valid json")
    r = run(malformed_path)
    check(
        "malformed providers.json: process fails at import (nonzero exit), never prints a response",
        r.returncode != 0 and r.stdout.strip() == "",
        (r.returncode, r.stdout, r.stderr),
    )
    check(
        "malformed providers.json: failure is loud and names the problem, not a silent empty listing",
        "malformed JSON" in r.stderr,
        r.stderr,
    )


def test_auth_posture() -> None:
    """
    Real behavioral proof of GET /config/llm/providers' auth posture — an
    in-process FastAPI TestClient over the actual router, with
    services.auth.resolve_session monkeypatched (no DB, no network).
    Deliberately NOT a source-text scan: that would still pass if
    require_admin moved to the router level or the decorator were
    reformatted, neither of which changes real behavior but both of which
    would defeat a substring check.
    """
    import controllers.config as config_controller
    import services.auth as auth_module
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    users = {
        "admin-token": {"id": 1, "username": "admin", "role": "admin"},
        "viewer-token": {"id": 2, "username": "viewer", "role": "viewer"},
    }

    async def fake_resolve_session(token: str):
        return users.get(token)

    auth_module.resolve_session = fake_resolve_session

    app = FastAPI()
    app.include_router(config_controller.router)
    client = TestClient(app)

    r = client.get("/config/llm/providers")
    check("no token: GET /config/llm/providers is 401", r.status_code == 401, r.status_code)

    r = client.get("/config/llm/providers", headers={"Authorization": "Bearer viewer-token"})
    check(
        "viewer token: GET /config/llm/providers is 200 (a read, not admin-gated)",
        r.status_code == 200,
        (r.status_code, r.text),
    )

    r = client.post(
        "/config/llm",
        headers={"Authorization": "Bearer viewer-token"},
        json={"is_local": False, "provider_data": {"provider_name": "openai", "model": "m", "key": "k"}},
    )
    check(
        "viewer token: POST /config/llm is 403 (writes stay admin-gated, unlike the new GET)",
        r.status_code == 403,
        (r.status_code, r.text),
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        test_endpoint_scenarios(tmpdir)
        test_malformed_file_boot_failure(tmpdir)
    test_auth_posture()

    print(f"\n{_passed} passed, {_failed} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
