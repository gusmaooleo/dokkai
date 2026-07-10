#!/usr/bin/env python
"""
Smoke test for the routines HTTP API (``controllers/routines.py``) — this
repo has no test framework yet, so this is a plain assert-and-print script,
mirroring ``scripts/test_review_analyze.py``.

Talks to the REAL running API (``./dev.sh``, port 8000 by default unless
``DOKKAI_API_URL`` is set) with the REAL saffira corpus repo/project (branch
``fix/center-to-box``), real Postgres/Weaviate/Ollama up, and the model
already persisted via ``POST /config/llm`` (no model seeded by this script —
same precondition as ``test_review_analyze.py``). Exercises auth (admin
login, a scratch viewer role), the launch pre-flights, the shared job
system, and cleans up every run/job/user it creates. READ-ONLY on the repo.

Usage
-----
    ./dev.sh &                      # or: PORT=8000 ./dev.sh &
    uv run python scripts/test_routines_api.py
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
VIEWER_USERNAME = "test-routines-api-viewer"
VIEWER_PASSWORD = "viewer-pw-123"

_passed = 0
_created_run_ids: list[str] = []


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


async def cleanup_runs(client: httpx.AsyncClient, token: str) -> None:
    for run_id in list(_created_run_ids):
        try:
            await client.delete(f"/routines/runs/{run_id}", headers=auth_headers(token))
        except Exception as e:
            print(f"cleanup: failed to delete run {run_id} (non-fatal): {e}")
    _created_run_ids.clear()


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=600.0) as client:
        admin_token = await login(client, ADMIN_USERNAME, ADMIN_PASSWORD)
        check("admin login", bool(admin_token))

        try:
            # --- branches ---
            resp = await client.get(
                "/routines/git/branches", params={"project": PROJECT}, headers=auth_headers(admin_token)
            )
            check("branches 200", resp.status_code == 200)
            branches_data = resp.json()
            branch_names = {b["name"] for b in branches_data["branches"]}
            check("fix/center-to-box present", TARGET_REF in branch_names)
            check("default_base == master", branches_data["default_base"] == "master")

            # --- negative: unknown project, 404 verbatim (branches) ---
            resp = await client.get(
                "/routines/git/branches",
                params={"project": "does-not-exist-project"},
                headers=auth_headers(admin_token),
            )
            check("unknown project branches 404", resp.status_code == 404)
            check(
                "unknown project branches 404 verbatim message",
                "no graph found for project 'does-not-exist-project'" in resp.json()["detail"],
            )

            # --- negative: unknown project, 404 verbatim (launch) ---
            resp = await client.post(
                "/routines/runs",
                json={"kind": "review", "project": "does-not-exist-project", "target_ref": "master"},
                headers=auth_headers(admin_token),
            )
            check("unknown project launch 404", resp.status_code == 404)
            check(
                "unknown project launch 404 verbatim message",
                "no graph found for project 'does-not-exist-project'" in resp.json()["detail"],
            )

            # --- negative: bughunt missing scope -> 422 (DTO validator, mirrors review's target_ref) ---
            resp = await client.post(
                "/routines/runs",
                json={"kind": "bughunt", "project": PROJECT},
                headers=auth_headers(admin_token),
            )
            check("bughunt missing scope 422", resp.status_code == 422)

            # --- negative: bughunt non-ollama provider override -> 400 ---
            resp = await client.post(
                "/routines/runs",
                json={
                    "kind": "bughunt",
                    "project": PROJECT,
                    "scope": "error handling",
                    "provider": "openai",
                },
                headers=auth_headers(admin_token),
            )
            check("bughunt non-ollama provider 400", resp.status_code == 400)

            # --- launch a review run (default/persisted model) ---
            started = time.monotonic()
            resp = await client.post(
                "/routines/runs",
                json={"kind": "review", "project": PROJECT, "target_ref": TARGET_REF},
                headers=auth_headers(admin_token),
            )
            check("launch review 202", resp.status_code == 202)
            launch = resp.json()
            run_id, job_id = launch["run_id"], launch["job_id"]
            _created_run_ids.append(run_id)

            # --- shared job system: the routine job is visible while running ---
            resp = await client.get("/instances/jobs", headers=auth_headers(admin_token))
            check("instances/jobs 200", resp.status_code == 200)
            jobs = resp.json()
            check("routine job visible in GET /instances/jobs", any(j["id"] == job_id for j in jobs))

            # --- poll run detail until terminal ---
            run = await poll_run_detail(client, admin_token, run_id)
            wall = time.monotonic() - started
            print(f"\nwall time (launch -> run terminal): {wall:.1f}s")
            check("review run status == done", run["status"] == "done")
            check("findings is a list", isinstance(run["findings"], list))
            findings_count = len(run["findings"])
            print(f"findings count: {findings_count}")
            if findings_count:
                f = run["findings"][0]
                for key in (
                    "id", "file_path", "start_line", "end_line", "severity", "category",
                    "title", "body", "suggestion", "evidence", "anchored", "created_at",
                ):
                    check(f"finding has '{key}'", key in f)

            # --- severity_counts on the list endpoint ---
            resp = await client.get(
                "/routines/runs", params={"project": PROJECT}, headers=auth_headers(admin_token)
            )
            check("list runs 200", resp.status_code == 200)
            listed_run = next((r for r in resp.json() if r["id"] == run_id), None)
            check("run present in list", listed_run is not None)
            check("severity_counts is a dict", isinstance(listed_run["severity_counts"], dict))
            check(
                "severity_counts total matches findings count",
                sum(listed_run["severity_counts"].values()) == findings_count,
            )

            # --- viewer role: GETs work, POST is forbidden with the verbatim message ---
            resp = await client.post(
                "/auth/users",
                json={"username": VIEWER_USERNAME, "password": VIEWER_PASSWORD, "role": "viewer"},
                headers=auth_headers(admin_token),
            )
            check("viewer user created", resp.status_code == 201)
            viewer_id = resp.json()["id"]

            try:
                viewer_token = await login(client, VIEWER_USERNAME, VIEWER_PASSWORD)
                resp = await client.get(
                    "/routines/runs", params={"project": PROJECT}, headers=auth_headers(viewer_token)
                )
                check("viewer can GET /routines/runs", resp.status_code == 200)

                resp = await client.get(
                    "/routines/git/branches",
                    params={"project": PROJECT},
                    headers=auth_headers(viewer_token),
                )
                check("viewer can GET /routines/git/branches", resp.status_code == 200)

                resp = await client.post(
                    "/routines/runs",
                    json={"kind": "review", "project": PROJECT, "target_ref": TARGET_REF},
                    headers=auth_headers(viewer_token),
                )
                check("viewer POST /routines/runs forbidden", resp.status_code == 403)
                check(
                    "viewer 403 verbatim role message",
                    resp.json()["detail"] == "role 'viewer' cannot perform this action",
                )
            finally:
                resp = await client.delete(f"/auth/users/{viewer_id}", headers=auth_headers(admin_token))
                check("viewer user deleted", resp.status_code == 200)

            # --- delete the run -> 404 on re-GET ---
            resp = await client.delete(f"/routines/runs/{run_id}", headers=auth_headers(admin_token))
            check("delete run 200", resp.status_code == 200)
            _created_run_ids.remove(run_id)

            resp = await client.get(f"/routines/runs/{run_id}", headers=auth_headers(admin_token))
            check("re-GET deleted run 404", resp.status_code == 404)
            check(
                "delete-then-get 404 message",
                resp.json()["detail"] == f"routine run '{run_id}' not found",
            )

        finally:
            await cleanup_runs(client, admin_token)

    print(f"\n{_passed} checks PASSED")


if __name__ == "__main__":
    asyncio.run(main())
