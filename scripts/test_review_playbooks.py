#!/usr/bin/env python
"""
Smoke test for playbook injection into the review routine (B3) —
``services.routines.review``'s 'playbooks' stage + system-prompt injection,
and ``controllers/routines.py``'s launch-time playbook validation (decision
9d). This repo has no test framework yet, so this is a plain assert-and-print
script, mirroring ``scripts/test_routines_api.py``.

Talks to the REAL running API (``./dev.sh``, port 8000 by default unless
``DOKKAI_API_URL`` is set) with the REAL saffira corpus repo/project (branch
``fix/center-to-box``), real Postgres/Weaviate/Ollama up, and
qwen2.5-coder:3b already persisted via ``POST /config/llm`` (no model seeded
by this script — same precondition as ``test_review_analyze.py``). READ-ONLY
on the repo.

Scope: MECHANICS only, not judgment (a 3b model may or may not actually obey
a playbook's instructions — the A/B acceptance demonstration with the 14b
model is a later step). This script only asserts things that don't depend on
model compliance: the 'playbooks' stage event fires with 'loaded playbook
<name>'; ``stats.playbooks_used`` reflects the injected names (proving the
system-prompt injection happened, without needing a debug hook); a
near-16KB playbook triggers the truncation warning event; and the launch
validations (unknown playbook 400, >4 playbooks 400, non-review-applicable
playbook 400) fire correctly.

Cleanup: deletes every scratch ``playbooks``/``routine_runs``/``jobs`` row
this script creates.

Usage
-----
    ./dev.sh &                      # or: PORT=8000 ./dev.sh &
    uv run python scripts/test_review_playbooks.py
"""

from __future__ import annotations

import asyncio
import os
import time

import httpx

BASE_URL = os.getenv("DOKKAI_API_URL", "http://localhost:8000")
PROJECT = "saffira_back-end"
TARGET_REF = "fix/center-to-box"
ADMIN_USERNAME = os.getenv("DOKKAI_ROOT_USER", "admin")
ADMIN_PASSWORD = os.getenv("DOKKAI_ROOT_PASSWORD", "admin")

PB_NORMAL = "test-review-playbooks-house-rules"
PB_LARGE = "test-review-playbooks-oversized"
PB_BUGHUNT_ONLY = "test-review-playbooks-bughunt-only"
ALL_PLAYBOOK_NAMES = [PB_NORMAL, PB_LARGE, PB_BUGHUNT_ONLY]

_passed = 0
_created_run_ids: list[str] = []
_created_playbook_names: list[str] = []


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

    for name in list(_created_playbook_names):
        try:
            await client.delete(f"/routines/playbooks/{name}", headers=auth_headers(token))
        except Exception as e:
            print(f"cleanup: failed to delete playbook {name} (non-fatal): {e}")
    _created_playbook_names.clear()


async def create_playbook(
    client: httpx.AsyncClient, token: str, name: str, content: str, routines: list[str]
) -> None:
    resp = await client.post(
        "/routines/playbooks",
        json={"name": name, "content": content, "routines": routines},
        headers=auth_headers(token),
    )
    resp.raise_for_status()
    _created_playbook_names.append(name)


async def launch_review(client: httpx.AsyncClient, token: str, playbooks: list[str] | None = None) -> httpx.Response:
    payload = {"kind": "review", "project": PROJECT, "target_ref": TARGET_REF}
    if playbooks is not None:
        payload["playbooks"] = playbooks
    return await client.post("/routines/runs", json=payload, headers=auth_headers(token))


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=600.0) as client:
        admin_token = await login(client, ADMIN_USERNAME, ADMIN_PASSWORD)
        check("admin login", bool(admin_token))

        try:
            # --- scratch playbooks ---
            await create_playbook(
                client,
                admin_token,
                PB_NORMAL,
                "---\ntitle: house rules\n---\nEvery finding title must start with [HOUSE].",
                ["review", "bughunt"],
            )
            await create_playbook(
                client,
                admin_token,
                PB_LARGE,
                "Rule: " + ("x" * 15_000),  # under the 16KB per-playbook limit, over the 12k injection cap
                ["review"],
            )
            await create_playbook(
                client,
                admin_token,
                PB_BUGHUNT_ONLY,
                "# bughunt-only playbook",
                ["bughunt"],
            )

            # --- negative: unknown playbook name, 400 ---
            resp = await launch_review(client, admin_token, playbooks=["does-not-exist-playbook-xyz"])
            check("unknown playbook launch 400", resp.status_code == 400)
            check(
                "unknown playbook launch 400 message",
                resp.json()["detail"] == "playbook 'does-not-exist-playbook-xyz' not found",
            )

            # --- negative: more than 4 playbooks, 400 ---
            resp = await launch_review(client, admin_token, playbooks=["a", "b", "c", "d", "e"])
            check(">4 playbooks launch 400", resp.status_code == 400)
            check(
                ">4 playbooks launch 400 message",
                resp.json()["detail"] == "a run can use at most 4 playbooks",
            )

            # --- negative: playbook that doesn't apply to 'review', 400 ---
            resp = await launch_review(client, admin_token, playbooks=[PB_BUGHUNT_ONLY])
            check("non-review playbook launch 400", resp.status_code == 400)
            check(
                "non-review playbook launch 400 message",
                resp.json()["detail"] == f"playbook '{PB_BUGHUNT_ONLY}' does not apply to review routines",
            )

            # --- mechanics run: one small playbook ---
            started = time.monotonic()
            resp = await launch_review(client, admin_token, playbooks=[PB_NORMAL])
            check("launch review with playbook 202", resp.status_code == 202)
            launch = resp.json()
            run_id, job_id = launch["run_id"], launch["job_id"]
            _created_run_ids.append(run_id)

            run = await poll_run_detail(client, admin_token, run_id)
            wall = time.monotonic() - started
            print(f"\nwall time (launch -> run terminal, PB_NORMAL): {wall:.1f}s")
            check("review+playbook run status == done", run["status"] == "done")

            events = await get_job_events(client, admin_token, job_id)
            stages = [e["stage"] for e in events]
            check("'playbooks' stage event fired", "playbooks" in stages)
            check(
                "'loaded playbook <name>' event present",
                any(
                    e["stage"] == "playbooks" and e["message"] == f"loaded playbook {PB_NORMAL}"
                    for e in events
                ),
            )
            print(f"\n--- event sequence ({len(events)} events) ---")
            for e in events:
                print(f"  [{e['stage']}] {e['message']}")

            stats = run["stats"]
            check("stats: playbooks_used present", "playbooks_used" in stats)
            check("stats: playbooks_used == [PB_NORMAL]", stats["playbooks_used"] == [PB_NORMAL])
            print(f"\n--- stats ---\n{stats}")

            # --- mechanics run: an oversized playbook triggers the truncation warning ---
            started = time.monotonic()
            resp = await launch_review(client, admin_token, playbooks=[PB_LARGE])
            check("launch review with oversized playbook 202", resp.status_code == 202)
            launch2 = resp.json()
            run_id2, job_id2 = launch2["run_id"], launch2["job_id"]
            _created_run_ids.append(run_id2)

            run2 = await poll_run_detail(client, admin_token, run_id2)
            wall2 = time.monotonic() - started
            print(f"\nwall time (launch -> run terminal, PB_LARGE): {wall2:.1f}s")
            check("review+oversized-playbook run status == done", run2["status"] == "done")

            events2 = await get_job_events(client, admin_token, job_id2)
            check(
                "truncation warning event present",
                any(
                    e["stage"] == "playbooks"
                    and e["message"] == "playbook content truncated to fit the context budget"
                    for e in events2
                ),
            )
            check(
                "'loaded playbook <name>' event present (oversized run)",
                any(
                    e["stage"] == "playbooks" and e["message"] == f"loaded playbook {PB_LARGE}"
                    for e in events2
                ),
            )
            check("stats: playbooks_used == [PB_LARGE] (oversized run)", run2["stats"]["playbooks_used"] == [PB_LARGE])

            # --- no-playbook run: prompt/stats stay byte-identical to sub-part A ---
            started = time.monotonic()
            resp = await launch_review(client, admin_token)
            check("launch review without playbooks 202", resp.status_code == 202)
            launch3 = resp.json()
            run_id3, job_id3 = launch3["run_id"], launch3["job_id"]
            _created_run_ids.append(run_id3)

            run3 = await poll_run_detail(client, admin_token, run_id3)
            wall3 = time.monotonic() - started
            print(f"\nwall time (launch -> run terminal, no playbooks): {wall3:.1f}s")
            check("no-playbook run status == done", run3["status"] == "done")
            check("stats: playbooks_used == [] (no playbooks)", run3["stats"]["playbooks_used"] == [])

            events3 = await get_job_events(client, admin_token, job_id3)
            check(
                "no 'playbooks' stage event fired when no playbooks given",
                not any(e["stage"] == "playbooks" for e in events3),
            )

        finally:
            await cleanup(client, admin_token)

    print(f"\n{_passed} checks PASSED")


if __name__ == "__main__":
    asyncio.run(main())
