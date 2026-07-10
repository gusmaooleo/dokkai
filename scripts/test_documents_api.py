#!/usr/bin/env python
"""
Smoke test for the playbooks/skills HTTP API
(``controllers/routines.py``'s ``/routines/playbooks`` and
``/routines/skills`` endpoints, B2) — this repo has no test framework yet,
so this is a plain assert-and-print script, mirroring
``scripts/test_routines_api.py``.

Talks to the REAL running API (``./dev.sh``, port 8000 by default unless
``DOKKAI_API_URL`` is set) with real Postgres up. Exercises auth (admin
login, a scratch viewer role) and cleans up every playbook/skill/user it
creates.

Usage
-----
    ./dev.sh &                      # or: PORT=8000 ./dev.sh &
    uv run python scripts/test_documents_api.py
"""

from __future__ import annotations

import asyncio
import os

import httpx

BASE_URL = os.getenv("DOKKAI_API_URL", "http://localhost:8000")
ADMIN_USERNAME = os.getenv("DOKKAI_ROOT_USER", "admin")
ADMIN_PASSWORD = os.getenv("DOKKAI_ROOT_PASSWORD", "admin")
VIEWER_USERNAME = "test-documents-api-viewer"
VIEWER_PASSWORD = "viewer-pw-123"

PLAYBOOK_NAME = "test-documents-api-scratch-playbook"
SKILL_NAME = "test-documents-api-scratch-skill"

_passed = 0


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


async def cleanup(client: httpx.AsyncClient, admin_token: str, viewer_id: int | None) -> None:
    try:
        await client.delete(
            f"/routines/playbooks/{PLAYBOOK_NAME}", headers=auth_headers(admin_token)
        )
    except Exception as e:
        print(f"cleanup: failed to delete playbook (non-fatal): {e}")
    try:
        await client.delete(f"/routines/skills/{SKILL_NAME}", headers=auth_headers(admin_token))
    except Exception as e:
        print(f"cleanup: failed to delete skill (non-fatal): {e}")
    if viewer_id is not None:
        try:
            await client.delete(f"/auth/users/{viewer_id}", headers=auth_headers(admin_token))
        except Exception as e:
            print(f"cleanup: failed to delete viewer user (non-fatal): {e}")


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        admin_token = await login(client, ADMIN_USERNAME, ADMIN_PASSWORD)
        check("admin login", bool(admin_token))

        viewer_id = None
        try:
            # ================= playbooks =================

            playbook_content = "---\ntitle: scratch\n---\n# Playbook body"
            resp = await client.post(
                "/routines/playbooks",
                json={"name": PLAYBOOK_NAME, "content": playbook_content, "routines": ["review"]},
                headers=auth_headers(admin_token),
            )
            check("create playbook 201", resp.status_code == 201)
            created = resp.json()
            check("create playbook: content stored verbatim", created["content"] == playbook_content)
            check("create playbook: routines stored", created["routines"] == ["review"])

            resp = await client.get("/routines/playbooks", headers=auth_headers(admin_token))
            check("list playbooks 200", resp.status_code == 200)
            listed = next((p for p in resp.json() if p["name"] == PLAYBOOK_NAME), None)
            check("list playbooks: scratch playbook present", listed is not None)
            check("list playbooks: no 'content' field", "content" not in listed)
            check(
                "list playbooks: content_bytes matches",
                listed["content_bytes"] == len(playbook_content.encode("utf-8")),
            )

            resp = await client.get(
                f"/routines/playbooks/{PLAYBOOK_NAME}", headers=auth_headers(admin_token)
            )
            check("get playbook by name 200", resp.status_code == 200)
            check("get playbook by name: full content", resp.json()["content"] == playbook_content)

            resp = await client.patch(
                f"/routines/playbooks/{PLAYBOOK_NAME}",
                json={"content": "# Playbook body v2"},
                headers=auth_headers(admin_token),
            )
            check("patch playbook 200", resp.status_code == 200)
            check("patch playbook: content updated", resp.json()["content"] == "# Playbook body v2")
            check("patch playbook: routines untouched", resp.json()["routines"] == ["review"])

            resp = await client.post(
                "/routines/playbooks",
                json={"name": PLAYBOOK_NAME, "content": "dup"},
                headers=auth_headers(admin_token),
            )
            check("duplicate playbook name 409", resp.status_code == 409)
            check(
                "duplicate playbook name 409 verbatim message",
                resp.json()["detail"] == f"playbook '{PLAYBOOK_NAME}' already exists",
            )

            oversized = "x" * 16_385
            resp = await client.post(
                "/routines/playbooks",
                json={"name": "test-documents-api-oversized-playbook", "content": oversized},
                headers=auth_headers(admin_token),
            )
            check("oversized playbook content 400", resp.status_code == 400)
            check("oversized playbook content 400 message mentions 16KB", "16KB" in resp.json()["detail"])

            resp = await client.post(
                "/routines/playbooks",
                json={
                    "name": "test-documents-api-bad-routine-kind",
                    "content": "x",
                    "routines": ["not-a-real-kind"],
                },
                headers=auth_headers(admin_token),
            )
            check("unknown routine kind 400", resp.status_code == 400)
            check(
                "unknown routine kind 400 message",
                "unknown routine kind 'not-a-real-kind'" in resp.json()["detail"],
            )

            resp = await client.get(
                "/routines/playbooks/does-not-exist-playbook", headers=auth_headers(admin_token)
            )
            check("get unknown playbook 404", resp.status_code == 404)
            check(
                "get unknown playbook 404 verbatim message",
                resp.json()["detail"] == "playbook 'does-not-exist-playbook' not found",
            )

            # a '/' or '\' in the name would be forever unreachable via
            # GET/PATCH/DELETE-by-name (FastAPI path params don't match a
            # literal slash, even URL-encoded) — rejected at create time.
            resp = await client.post(
                "/routines/playbooks",
                json={"name": "a/b", "content": "x"},
                headers=auth_headers(admin_token),
            )
            check("slash-containing playbook name 400", resp.status_code == 400)
            check(
                "slash-containing playbook name 400 message",
                resp.json()["detail"] == "name must not contain slashes or control characters",
            )

            resp = await client.post(
                "/routines/playbooks",
                json={"name": "a\\b", "content": "x"},
                headers=auth_headers(admin_token),
            )
            check("backslash-containing playbook name 400", resp.status_code == 400)

            resp = await client.post(
                "/routines/playbooks",
                json={"name": "a\tb", "content": "x"},
                headers=auth_headers(admin_token),
            )
            check("tab-containing playbook name 400", resp.status_code == 400)

            # ================= skills =================

            skill_content = "---\ntitle: scratch\n---\n# Skill body"
            resp = await client.post(
                "/routines/skills",
                json={"name": SKILL_NAME, "description": "Use this for scratch tests", "content": skill_content},
                headers=auth_headers(admin_token),
            )
            check("create skill 201", resp.status_code == 201)
            created_skill = resp.json()
            check("create skill: content stored verbatim", created_skill["content"] == skill_content)
            check(
                "create skill: description stored",
                created_skill["description"] == "Use this for scratch tests",
            )

            resp = await client.get("/routines/skills", headers=auth_headers(admin_token))
            check("list skills 200", resp.status_code == 200)
            listed_skill = next((s for s in resp.json() if s["name"] == SKILL_NAME), None)
            check("list skills: scratch skill present", listed_skill is not None)
            check("list skills: no 'content' field", "content" not in listed_skill)
            check(
                "list skills: content_bytes matches",
                listed_skill["content_bytes"] == len(skill_content.encode("utf-8")),
            )

            resp = await client.get(f"/routines/skills/{SKILL_NAME}", headers=auth_headers(admin_token))
            check("get skill by name 200", resp.status_code == 200)
            check("get skill by name: full content", resp.json()["content"] == skill_content)

            resp = await client.patch(
                f"/routines/skills/{SKILL_NAME}",
                json={"content": "# Skill body v2"},
                headers=auth_headers(admin_token),
            )
            check("patch skill 200", resp.status_code == 200)
            check("patch skill: content updated", resp.json()["content"] == "# Skill body v2")
            check(
                "patch skill: description untouched",
                resp.json()["description"] == "Use this for scratch tests",
            )

            resp = await client.post(
                "/routines/skills",
                json={"name": SKILL_NAME, "description": "dup", "content": "dup"},
                headers=auth_headers(admin_token),
            )
            check("duplicate skill name 409", resp.status_code == 409)
            check(
                "duplicate skill name 409 verbatim message",
                resp.json()["detail"] == f"skill '{SKILL_NAME}' already exists",
            )

            resp = await client.post(
                "/routines/skills",
                json={"name": "test-documents-api-missing-description", "content": "x"},
                headers=auth_headers(admin_token),
            )
            print(f"missing 'description' field on create skill -> HTTP {resp.status_code}")
            check(
                "missing skill description rejected (422, Pydantic-level: field is required)",
                resp.status_code == 422,
            )

            resp = await client.post(
                "/routines/skills",
                json={"name": "test-documents-api-empty-description", "description": "", "content": "x"},
                headers=auth_headers(admin_token),
            )
            check("empty skill description 400", resp.status_code == 400)
            check("empty skill description 400 message mentions description", "description" in resp.json()["detail"])

            resp = await client.get(
                "/routines/skills/does-not-exist-skill", headers=auth_headers(admin_token)
            )
            check("get unknown skill 404", resp.status_code == 404)
            check(
                "get unknown skill 404 verbatim message",
                resp.json()["detail"] == "skill 'does-not-exist-skill' not found",
            )

            # ================= viewer role: GETs work, mutations forbidden =================

            resp = await client.post(
                "/auth/users",
                json={"username": VIEWER_USERNAME, "password": VIEWER_PASSWORD, "role": "viewer"},
                headers=auth_headers(admin_token),
            )
            check("viewer user created", resp.status_code == 201)
            viewer_id = resp.json()["id"]

            viewer_token = await login(client, VIEWER_USERNAME, VIEWER_PASSWORD)

            resp = await client.get("/routines/playbooks", headers=auth_headers(viewer_token))
            check("viewer can GET /routines/playbooks", resp.status_code == 200)

            resp = await client.get(f"/routines/skills/{SKILL_NAME}", headers=auth_headers(viewer_token))
            check("viewer can GET /routines/skills/{name}", resp.status_code == 200)

            resp = await client.post(
                "/routines/playbooks",
                json={"name": "test-documents-api-viewer-should-not-create", "content": "x"},
                headers=auth_headers(viewer_token),
            )
            check("viewer POST /routines/playbooks forbidden", resp.status_code == 403)
            check(
                "viewer 403 verbatim role message",
                resp.json()["detail"] == "role 'viewer' cannot perform this action",
            )

            resp = await client.post(
                "/routines/skills",
                json={"name": "test-documents-api-viewer-should-not-create", "description": "x", "content": "x"},
                headers=auth_headers(viewer_token),
            )
            check("viewer POST /routines/skills forbidden", resp.status_code == 403)

            # ================= delete -> 404 on re-GET =================

            resp = await client.delete(
                f"/routines/playbooks/{PLAYBOOK_NAME}", headers=auth_headers(admin_token)
            )
            check("delete playbook 200", resp.status_code == 200)
            resp = await client.get(
                f"/routines/playbooks/{PLAYBOOK_NAME}", headers=auth_headers(admin_token)
            )
            check("re-GET deleted playbook 404", resp.status_code == 404)

            resp = await client.delete(f"/routines/skills/{SKILL_NAME}", headers=auth_headers(admin_token))
            check("delete skill 200", resp.status_code == 200)
            resp = await client.get(f"/routines/skills/{SKILL_NAME}", headers=auth_headers(admin_token))
            check("re-GET deleted skill 404", resp.status_code == 404)

        finally:
            await cleanup(client, admin_token, viewer_id)

    print(f"\n{_passed} checks PASSED")


if __name__ == "__main__":
    asyncio.run(main())
