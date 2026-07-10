#!/usr/bin/env python
"""
Smoke test for the review routine's 'skills' stage (decision 9d, sub-part B,
step B6) — ``services.routines.review``'s model-driven skill selection +
injection into the analyze system prompt. This repo has no test framework
yet, so this is a plain assert-and-print script, mirroring
``scripts/test_review_playbooks.py``.

Talks to the REAL running API (``./dev.sh``, port 8000 by default unless
``DOKKAI_API_URL`` is set) with the REAL saffira corpus repo/project (branch
``fix/center-to-box`` — a TypeScript-only diff), real Postgres/Weaviate/
Ollama up, and qwen2.5-coder:3b already persisted via ``POST /config/llm``
(no model seeded by this script — same precondition as
``test_review_analyze.py``). READ-ONLY on the repo.

Scope: MECHANICS only, not judgment — review has no launch-time 'skills'
param (unlike 'playbooks'): skills are model-PULLED from the full catalog
every run. A 3b model may or may not actually select the relevant skill —
this script only asserts things that don't depend on model judgment:
  - the 'skills' stage fires (catalog is non-empty) — announced via
    'selecting skills from catalog of N'
  - the selection is either parsed (stats.skills_used is a subset of the
    catalog names) OR degrades gracefully on a parse failure (stats.skills_used
    == [] and a 'parse_failed: skill selection' warning event) — asserted as
    "stats.skills_used exists" per the plan, since which outcome occurs is
    nondeterministic with a 3b model
  - 'loaded skill <name>' events fire ONLY for names in stats.skills_used
    (no event for the unselected skill)
  - stats.skills_used is reflected consistently with the loaded-skill events

Cleanup: deletes every scratch ``skills``/``routine_runs``/``jobs`` row this
script creates.

Usage
-----
    ./dev.sh &                      # or: PORT=8000 ./dev.sh &
    uv run python scripts/test_review_skills.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import httpx

# Make the `services` package importable (the app roots absolute imports at src/) —
# needed for the fake-provider unit tests below, which call review's internals
# directly rather than going through the API.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from services.routines.review import (  # noqa: E402
    MAX_SELECTED_SKILLS,
    SKILL_INJECTION_MAX_CHARS,
    _build_analyze_system_prompt,
    _select_skills,
)

BASE_URL = os.getenv("DOKKAI_API_URL", "http://localhost:8000")
PROJECT = "saffira_back-end"
TARGET_REF = "fix/center-to-box"
ADMIN_USERNAME = os.getenv("DOKKAI_ROOT_USER", "admin")
ADMIN_PASSWORD = os.getenv("DOKKAI_ROOT_PASSWORD", "admin")

SKILL_RELEVANT = "typescript-review"
SKILL_IRRELEVANT = "terraform-review"
ALL_SKILL_NAMES = [SKILL_RELEVANT, SKILL_IRRELEVANT]

_passed = 0
_created_run_ids: list[str] = []
_created_skill_names: list[str] = []


def check(label: str, condition: bool) -> None:
    global _passed
    if not condition:
        raise AssertionError(f"FAIL: {label}")
    _passed += 1
    print(f"PASS: {label}")


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def login(client: httpx.AsyncClient, username: str, password: str) -> str:
    resp = await client.post("/auth/login", json={"username": username, "password": password})
    resp.raise_for_status()
    return resp.json()["token"]


async def poll_run_detail(
    client: httpx.AsyncClient, token: str, run_id: str, timeout: float = 600.0
) -> dict:
    elapsed = 0.0
    while elapsed < timeout:
        resp = await client.get(f"/routines/runs/{run_id}", headers=auth_headers(token))
        resp.raise_for_status()
        data = resp.json()
        if data["status"] in ("done", "failed"):
            return data
        await asyncio.sleep(1.0)
        elapsed += 1.0
    raise TimeoutError(f"run {run_id} did not finish in {timeout}s")


async def get_job_events(client: httpx.AsyncClient, token: str, job_id: str) -> list[dict]:
    resp = await client.get(f"/instances/jobs/{job_id}", headers=auth_headers(token))
    resp.raise_for_status()
    return resp.json().get("events") or []


async def cleanup(client: httpx.AsyncClient, token: str) -> None:
    for run_id in list(_created_run_ids):
        try:
            await client.delete(f"/routines/runs/{run_id}", headers=auth_headers(token))
        except Exception as e:
            print(f"cleanup: failed to delete run {run_id} (non-fatal): {e}")
    _created_run_ids.clear()

    for name in list(_created_skill_names):
        try:
            await client.delete(f"/routines/skills/{name}", headers=auth_headers(token))
        except Exception as e:
            print(f"cleanup: failed to delete skill {name} (non-fatal): {e}")
    _created_skill_names.clear()


async def create_skill(
    client: httpx.AsyncClient, token: str, name: str, description: str, content: str
) -> None:
    resp = await client.post(
        "/routines/skills",
        json={"name": name, "description": description, "content": content},
        headers=auth_headers(token),
    )
    resp.raise_for_status()
    _created_skill_names.append(name)


async def launch_review(client: httpx.AsyncClient, token: str) -> httpx.Response:
    payload = {"kind": "review", "project": PROJECT, "target_ref": TARGET_REF}
    return await client.post("/routines/runs", json=payload, headers=auth_headers(token))


# ---------------------------------------------------------------------------
# Unit tests (no live API/DB/model needed) — a fake provider exercises
# _select_skills' parse-degrade, cap, and missing-skill-name paths directly,
# and _build_analyze_system_prompt's skill-injection cap. review's structure
# passes `provider`/`catalog` as plain arguments, so these are cheap to
# inject without touching Ollama or Postgres.
# ---------------------------------------------------------------------------

_DIFF_STATS = {"files_reviewable": 1, "insertions": 1, "deletions": 1}


class _FakeProvider:
    provider_name = "fake"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def chat(self, messages, *, model, temperature=0.3, max_tokens=4096):
        self.calls += 1
        idx = min(self.calls - 1, len(self._responses) - 1)
        return self._responses[idx]


def _collecting_emit():
    events: list[tuple[str, str]] = []

    def emit(stage: str, message: str) -> None:
        events.append((stage, message))

    return events, emit


async def test_selection_parse_degrade() -> None:
    """Both the initial call and the retry return invalid JSON — selection
    degrades to no skills, with a warning event, never raising."""
    provider = _FakeProvider(["not json at all", "still not json"])
    catalog = [{"name": "a", "description": "d"}]
    events, emit = _collecting_emit()

    names, llm_calls = await _select_skills(provider, "fake-model", catalog, [], _DIFF_STATS, emit)

    check("parse-degrade: names == []", names == [])
    check("parse-degrade: llm_calls == 2 (initial + one retry)", llm_calls == 2)
    check(
        "parse-degrade: 'parse_failed: skill selection' warning event emitted",
        any(s == "skills" and m.startswith("parse_failed: skill selection") for s, m in events),
    )


async def test_selection_cap() -> None:
    """A selection response naming more than MAX_SELECTED_SKILLS catalog
    entries is capped to the first MAX_SELECTED_SKILLS."""
    catalog = [{"name": n, "description": "d"} for n in ("sA", "sB", "sC", "sD", "sE")]
    provider = _FakeProvider(['["sA", "sB", "sC", "sD", "sE"]'])
    events, emit = _collecting_emit()

    names, llm_calls = await _select_skills(provider, "fake-model", catalog, [], _DIFF_STATS, emit)

    check(f"cap: exactly {MAX_SELECTED_SKILLS} names selected", len(names) == MAX_SELECTED_SKILLS)
    check("cap: names are the FIRST N in response order", names == ["sA", "sB", "sC"])
    check("cap: single llm_call (no retry needed — valid JSON)", llm_calls == 1)


async def test_selection_missing_skill_name() -> None:
    """A model-invented name not present in the catalog is dropped, with a
    warning, while the valid name in the same response is kept."""
    catalog = [{"name": "sA", "description": "d"}, {"name": "sB", "description": "d"}]
    provider = _FakeProvider(['["sA", "does-not-exist"]'])
    events, emit = _collecting_emit()

    names, llm_calls = await _select_skills(provider, "fake-model", catalog, [], _DIFF_STATS, emit)

    check("missing-name: known name kept", names == ["sA"])
    check(
        "missing-name: unknown-skill warning event emitted",
        any(
            s == "skills" and m == "skill selection named unknown skill(s), ignoring: does-not-exist"
            for s, m in events
        ),
    )


def test_injection_cap() -> None:
    """Skills share their OWN cap (SKILL_INJECTION_MAX_CHARS), separate from
    the playbook cap — the last skill in selection order is truncated first
    once that budget is spent, injected AFTER the playbook section."""
    skills = [
        {"name": "big1", "content": "y" * (SKILL_INJECTION_MAX_CHARS - 100)},
        {"name": "big2", "content": "z" * 5_000},
    ]
    prompt, playbooks_truncated, skills_truncated = _build_analyze_system_prompt([], skills)

    check("injection-cap: playbooks_truncated is False (no playbooks given)", playbooks_truncated is False)
    check("injection-cap: skills_truncated is True (combined bodies exceed the cap)", skills_truncated is True)
    check(
        "injection-cap: '## Loaded skills (analysis capabilities)' section present",
        "## Loaded skills (analysis capabilities)" in prompt,
    )
    check("injection-cap: first (higher-priority) skill body fully present", ("y" * 50) in prompt)
    check(
        "injection-cap: second (lower-priority) skill body truncated (fewer than 5000 z's)",
        prompt.count("z") < 5_000,
    )

    # No playbooks + no skills => prompt unchanged (byte-identical), per the
    # existing playbook-only contract extended to skills.
    unchanged_prompt, pb_trunc, sk_trunc = _build_analyze_system_prompt([], [])
    check("injection-cap: no playbooks/skills => truncated flags both False", pb_trunc is False and sk_trunc is False)
    check(
        "injection-cap: no playbooks/skills => '## Loaded skills' section absent",
        "## Loaded skills (analysis capabilities)" not in unchanged_prompt,
    )


async def run_unit_tests() -> None:
    print("--- unit tests (fake provider, no live API/DB) ---")
    await test_selection_parse_degrade()
    await test_selection_cap()
    await test_selection_missing_skill_name()
    test_injection_cap()


async def main() -> None:
    await run_unit_tests()

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=600.0) as client:
        admin_token = await login(client, ADMIN_USERNAME, ADMIN_PASSWORD)
        check("admin login", bool(admin_token))

        try:
            # --- scratch skills: one relevant to the TypeScript diff, one not ---
            await create_skill(
                client,
                admin_token,
                SKILL_RELEVANT,
                "TypeScript code review conventions: type safety, async/await correctness, "
                "and idiomatic TS patterns.",
                "---\ntitle: typescript review\n---\n"
                "When reviewing TypeScript changes, flag any implicit `any`, unchecked "
                "non-null assertions (`!`), and missing return-type annotations on exported "
                "functions.",
            )
            await create_skill(
                client,
                admin_token,
                SKILL_IRRELEVANT,
                "Terraform/HCL infrastructure-as-code review conventions.",
                "---\ntitle: terraform review\n---\n"
                "When reviewing Terraform changes, flag any hardcoded credentials, missing "
                "`lifecycle { prevent_destroy = true }` on stateful resources, and unpinned "
                "provider versions.",
            )

            # --- mechanics run: no launch-time skills param — the catalog drives it ---
            started = time.monotonic()
            resp = await launch_review(client, admin_token)
            check("launch review 202", resp.status_code == 202)
            launch = resp.json()
            run_id, job_id = launch["run_id"], launch["job_id"]
            _created_run_ids.append(run_id)

            run = await poll_run_detail(client, admin_token, run_id)
            wall = time.monotonic() - started
            print(f"\nwall time (launch -> run terminal): {wall:.1f}s")
            check("review run status == done", run["status"] == "done")

            events = await get_job_events(client, admin_token, job_id)
            stages = [e["stage"] for e in events]
            print(f"\n--- event sequence ({len(events)} events) ---")
            for e in events:
                print(f"  [{e['stage']}] {e['message']}")

            check("'skills' stage event fired", "skills" in stages)
            check(
                "'selecting skills from catalog of 2' event present",
                any(
                    e["stage"] == "skills" and e["message"] == "selecting skills from catalog of 2"
                    for e in events
                ),
            )

            stats = run["stats"]
            print(f"\n--- stats ---\n{stats}")
            check("stats: skills_used exists", "skills_used" in stats)
            skills_used = stats["skills_used"]
            check(
                "stats: skills_used is a list, subset of the catalog names",
                isinstance(skills_used, list) and all(n in ALL_SKILL_NAMES for n in skills_used),
            )

            parse_failed = any(
                e["stage"] == "skills" and e["message"].startswith("parse_failed: skill selection")
                for e in events
            )
            if parse_failed:
                check("skill selection parse failure degrades to skills_used == []", skills_used == [])
                print("NOTE: skill selection JSON parse failed (both attempts) — degraded gracefully.")
            else:
                print(f"NOTE: skill selection parsed successfully — selected: {skills_used!r}")

            loaded_skill_events = {
                e["message"].removeprefix("loaded skill ")
                for e in events
                if e["stage"] == "skills" and e["message"].startswith("loaded skill ")
            }
            check(
                "loaded-skill events == stats.skills_used (only selected ones loaded)",
                loaded_skill_events == set(skills_used),
            )
            check(
                "no loaded-skill event for an unselected skill",
                all(name in skills_used for name in loaded_skill_events),
            )

            # --- llm_calls includes the selection call(s) ---
            check("stats: llm_calls >= 3 (>=1 selection + 1 analyze + 1 summarize)", stats["llm_calls"] >= 3)

        finally:
            await cleanup(client, admin_token)

    print(f"\n{_passed} checks PASSED")


if __name__ == "__main__":
    asyncio.run(main())
