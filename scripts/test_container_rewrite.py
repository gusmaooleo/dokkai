#!/usr/bin/env python
"""
Unit test for ``services.llm_provider.effective_base_url`` — the
localhost -> host.docker.internal rewrite applied when the dokkai API runs
inside its own container (``DOKKAI_IN_CONTAINER=1``) against an Ollama
``base_url`` persisted from a host run.

Pure function, no Docker/network/Postgres required.

Usage
-----
    uv run python scripts/test_container_rewrite.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.llm_provider import effective_base_url  # noqa: E402

_passed = 0


def check(label: str, actual, expected) -> None:
    global _passed
    if actual != expected:
        raise AssertionError(f"FAIL: {label} — got {actual!r}, expected {expected!r}")
    _passed += 1
    print(f"PASS: {label}")


def main() -> None:
    # DOKKAI_IN_CONTAINER unset/false — never rewrite, regardless of host.
    os.environ.pop("DOKKAI_IN_CONTAINER", None)
    check(
        "no-op when DOKKAI_IN_CONTAINER unset",
        effective_base_url("http://localhost:11434"),
        "http://localhost:11434",
    )

    os.environ["DOKKAI_IN_CONTAINER"] = "1"

    check(
        "localhost -> host.docker.internal, port preserved",
        effective_base_url("http://localhost:11434"),
        "http://host.docker.internal:11434",
    )
    check(
        "127.0.0.1 -> host.docker.internal, port preserved",
        effective_base_url("http://127.0.0.1:11434"),
        "http://host.docker.internal:11434",
    )
    check(
        "https scheme preserved",
        effective_base_url("https://localhost:11434"),
        "https://host.docker.internal:11434",
    )
    check(
        "path/query preserved",
        effective_base_url("http://localhost:11434/v1?x=1"),
        "http://host.docker.internal:11434/v1?x=1",
    )
    check(
        "non-localhost host untouched (already host.docker.internal)",
        effective_base_url("http://host.docker.internal:11434"),
        "http://host.docker.internal:11434",
    )
    check(
        "non-localhost host untouched (remote provider)",
        effective_base_url("https://api.openai.com/v1"),
        "https://api.openai.com/v1",
    )
    check(
        "no port — rewritten host, no port appended",
        effective_base_url("http://localhost"),
        "http://host.docker.internal",
    )

    # Falsy values for the flag must not trigger a rewrite.
    for falsy in ("0", "false", "no", ""):
        os.environ["DOKKAI_IN_CONTAINER"] = falsy
        check(
            f"DOKKAI_IN_CONTAINER={falsy!r} treated as off",
            effective_base_url("http://localhost:11434"),
            "http://localhost:11434",
        )

    os.environ.pop("DOKKAI_IN_CONTAINER", None)

    # -----------------------------------------------------------------
    # Save-vs-use asymmetry (validate_and_save_config's Ollama reachability
    # probe, services/llm_config.py): the value PERSISTED must stay exactly
    # what the caller supplied, while the value the PROBE actually connects
    # to (`f"{effective_base_url(resolved_base_url)}/api/tags"`) is
    # rewritten in-container. This mirrors the fix for the save-time 400
    # ("Cannot reach Ollama at localhost:11434") that used to happen when
    # POSTing /config/llm through the containerized API with a
    # UI-defaulted localhost base_url.
    # -----------------------------------------------------------------
    os.environ["DOKKAI_IN_CONTAINER"] = "1"
    resolved_base_url = "http://localhost:11434"
    probe_url = f"{effective_base_url(resolved_base_url)}/api/tags"
    check(
        "persisted value is NOT rewritten (storage-time invariant)",
        resolved_base_url,
        "http://localhost:11434",
    )
    check(
        "probe URL IS rewritten (use-time, save-vs-use asymmetry fix)",
        probe_url,
        "http://host.docker.internal:11434/api/tags",
    )

    os.environ.pop("DOKKAI_IN_CONTAINER", None)
    # Outside a container, probe and persisted value coincide (no rewrite).
    resolved_base_url = "http://localhost:11434"
    probe_url = f"{effective_base_url(resolved_base_url)}/api/tags"
    check(
        "outside a container, probe URL matches persisted value",
        probe_url,
        f"{resolved_base_url}/api/tags",
    )

    print(f"\n{_passed} checks passed.")


if __name__ == "__main__":
    main()
