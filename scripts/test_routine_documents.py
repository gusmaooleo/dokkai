#!/usr/bin/env python
"""
Smoke test for playbook/skill persistence (``services.routines.documents``)
— this repo has no test framework yet, so this is a plain assert-and-print
script rather than a new pytest dependency, mirroring
``scripts/test_routine_engine.py``.

Talks to the REAL local Postgres (``docker compose up -d``). Creates and
tears down its own ``playbooks``/``skills`` rows under scratch names —
nothing else on the machine is touched.

Usage
-----
    uv run python scripts/test_routine_documents.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make the `services` package importable (the app roots absolute imports at src/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import asyncpg  # noqa: E402

from services.db import get_pool, init_db  # noqa: E402
from services.routines import documents  # noqa: E402

_passed = 0

PLAYBOOK_NAME = "test-routine-documents-scratch-playbook"
PLAYBOOK_NAME_2 = "test-routine-documents-scratch-playbook-2"
SKILL_NAME = "test-routine-documents-scratch-skill"
SKILL_NAME_2 = "test-routine-documents-scratch-skill-2"
ALL_PLAYBOOK_NAMES = [PLAYBOOK_NAME, PLAYBOOK_NAME_2]
ALL_SKILL_NAMES = [SKILL_NAME, SKILL_NAME_2]


def check(label: str, condition: bool) -> None:
    global _passed
    if not condition:
        raise AssertionError(f"FAIL: {label}")
    _passed += 1
    print(f"PASS: {label}")


async def cleanup() -> None:
    """Best-effort teardown of every row this script may have created."""
    try:
        pool = await get_pool()
        await pool.execute("DELETE FROM playbooks WHERE name = ANY($1::text[])", ALL_PLAYBOOK_NAMES)
        await pool.execute("DELETE FROM skills WHERE name = ANY($1::text[])", ALL_SKILL_NAMES)
        print("cleanup: removed scratch playbooks/skills rows")
    except Exception as e:
        print(f"cleanup: best-effort teardown failed (non-fatal): {e}")


def test_split_frontmatter() -> None:
    meta, body = documents.split_frontmatter("---\ntitle: x\n---\nbody text")
    check("frontmatter: meta extracted", meta == "title: x\n")
    check("frontmatter: body extracted", body == "body text")

    meta, body = documents.split_frontmatter("just body, no frontmatter")
    check("no frontmatter: meta is None", meta is None)
    check("no frontmatter: body unchanged", body == "just body, no frontmatter")

    meta, body = documents.split_frontmatter("---\nnot closed at all")
    check("malformed leading ---: meta is None", meta is None)
    check("malformed leading ---: body unchanged", body == "---\nnot closed at all")

    meta, body = documents.split_frontmatter("---\r\ntitle: x\r\n---\r\nbody\r\ncontinues")
    check("CRLF frontmatter: meta extracted", meta == "title: x\r\n")
    check("CRLF frontmatter: body extracted", body == "body\r\ncontinues")

    meta, body = documents.split_frontmatter("---\n---\nempty meta body")
    check("empty frontmatter block: meta extracted", meta == "")
    check("empty frontmatter block: body extracted", body == "empty meta body")


async def test_playbook_crud() -> None:
    created = await documents.create_playbook(PLAYBOOK_NAME, "# Playbook one")
    check("create_playbook: default routines", created["routines"] == ["review", "bughunt"])
    check("create_playbook: content stored", created["content"] == "# Playbook one")

    created2 = await documents.create_playbook(
        PLAYBOOK_NAME_2, "# Playbook two", routines=["bughunt"]
    )
    check("create_playbook: explicit routines stored", created2["routines"] == ["bughunt"])

    listed = await documents.list_playbooks()
    listed_names = {p["name"] for p in listed}
    check("list_playbooks: includes both scratch playbooks", {PLAYBOOK_NAME, PLAYBOOK_NAME_2} <= listed_names)

    fetched = await documents.get_playbook(PLAYBOOK_NAME)
    check("get_playbook: found", fetched is not None and fetched["name"] == PLAYBOOK_NAME)

    missing = await documents.get_playbook("does-not-exist-playbook")
    check("get_playbook: missing name returns None", missing is None)

    updated = await documents.update_playbook(PLAYBOOK_NAME, content="# Playbook one v2")
    check("update_playbook: content updated", updated["content"] == "# Playbook one v2")
    check("update_playbook: routines untouched", updated["routines"] == ["review", "bughunt"])

    updated = await documents.update_playbook(PLAYBOOK_NAME, routines=["review"])
    check("update_playbook: routines-only update", updated["routines"] == ["review"])
    check("update_playbook: content untouched by routines-only update", updated["content"] == "# Playbook one v2")

    missing_update = await documents.update_playbook("does-not-exist-playbook", content="x")
    check("update_playbook: missing name returns None", missing_update is None)

    deleted = await documents.delete_playbook(PLAYBOOK_NAME_2)
    check("delete_playbook: returns True on success", deleted is True)
    gone = await documents.get_playbook(PLAYBOOK_NAME_2)
    check("delete_playbook: row actually gone", gone is None)
    deleted_again = await documents.delete_playbook(PLAYBOOK_NAME_2)
    check("delete_playbook: returns False when already gone", deleted_again is False)

    # re-create #2 for the sync-reader tests below
    await documents.create_playbook(PLAYBOOK_NAME_2, "# Playbook two again", routines=["bughunt"])


async def test_skill_crud() -> None:
    created = await documents.create_skill(SKILL_NAME, "Use this for X", "# Skill one")
    check("create_skill: description stored", created["description"] == "Use this for X")
    check("create_skill: content stored", created["content"] == "# Skill one")

    await documents.create_skill(SKILL_NAME_2, "Use this for Y", "# Skill two")

    listed = await documents.list_skills()
    listed_names = {s["name"] for s in listed}
    check("list_skills: includes both scratch skills", {SKILL_NAME, SKILL_NAME_2} <= listed_names)

    fetched = await documents.get_skill(SKILL_NAME)
    check("get_skill: found", fetched is not None and fetched["name"] == SKILL_NAME)

    missing = await documents.get_skill("does-not-exist-skill")
    check("get_skill: missing name returns None", missing is None)

    updated = await documents.update_skill(SKILL_NAME, description="Use this for X, updated")
    check("update_skill: description updated", updated["description"] == "Use this for X, updated")
    check("update_skill: content untouched", updated["content"] == "# Skill one")

    updated = await documents.update_skill(SKILL_NAME, content="# Skill one v2")
    check("update_skill: content-only update", updated["content"] == "# Skill one v2")
    check(
        "update_skill: description untouched by content-only update",
        updated["description"] == "Use this for X, updated",
    )

    deleted = await documents.delete_skill(SKILL_NAME)
    check("delete_skill: returns True on success", deleted is True)
    gone = await documents.get_skill(SKILL_NAME)
    check("delete_skill: row actually gone", gone is None)

    # re-create for the sync-reader tests below
    await documents.create_skill(SKILL_NAME, "Use this for X", "# Skill one")


async def test_validation() -> None:
    oversized = "x" * (documents._MAX_CONTENT_BYTES + 1)
    try:
        await documents.create_playbook("oversized-playbook-should-not-be-created", oversized)
        check("oversized playbook content rejected", False)
    except ValueError as e:
        check("oversized playbook content raises ValueError with actionable message", "16KB" in str(e))

    try:
        await documents.create_skill("oversized-skill-should-not-be-created", "desc", oversized)
        check("oversized skill content rejected", False)
    except ValueError as e:
        check("oversized skill content raises ValueError with actionable message", "16KB" in str(e))

    for bad_name, label in [
        ("", "empty name"),
        ("   ", "whitespace-only name"),
        (" leading-space", "leading-whitespace name"),
        ("trailing-space ", "trailing-whitespace name"),
        ("x" * (documents._MAX_NAME_LEN + 1), "too-long name"),
    ]:
        try:
            await documents.create_playbook(bad_name, "content")
            check(f"{label} rejected", False)
        except ValueError:
            check(f"{label} raises ValueError", True)

    try:
        await documents.create_skill("skill-without-description", "", "content")
        check("empty skill description rejected", False)
    except ValueError as e:
        check("empty skill description raises ValueError", "description" in str(e))

    # unknown routine kinds are rejected (typo-proofing the B3 injection match)
    try:
        await documents.create_playbook("bad-routines-pb", "content", routines=["reveiw"])
        check("unknown routine kind rejected on create", False)
    except ValueError as e:
        check("unknown routine kind rejected on create", "unknown routine kind 'reveiw'" in str(e))
    try:
        await documents.update_playbook("any", routines=["nope"])
        check("unknown routine kind rejected on update", False)
    except ValueError as e:
        check("unknown routine kind rejected on update", "unknown routine kind 'nope'" in str(e))


async def test_duplicate_name() -> None:
    try:
        await documents.create_playbook(PLAYBOOK_NAME, "duplicate content")
        check("duplicate playbook name raises UniqueViolationError", False)
    except asyncpg.UniqueViolationError:
        check("duplicate playbook name raises UniqueViolationError", True)

    try:
        await documents.create_skill(SKILL_NAME, "dup desc", "duplicate content")
        check("duplicate skill name raises UniqueViolationError", False)
    except asyncpg.UniqueViolationError:
        check("duplicate skill name raises UniqueViolationError", True)


def test_sync_readers() -> None:
    # Run from a worker thread (via asyncio.to_thread below), mirroring how
    # the routine engine actually calls these — each uses asyncio.run()
    # internally and must not be called from a thread with a running loop.
    # Order preservation: request in reverse-of-creation order, including one
    # unknown name that must be silently skipped.
    fetched = documents.fetch_playbooks_sync([PLAYBOOK_NAME_2, "unknown-playbook-name", PLAYBOOK_NAME])
    check("fetch_playbooks_sync: count excludes unknown name", len(fetched) == 2)
    check(
        "fetch_playbooks_sync: preserves requested order",
        [p["name"] for p in fetched] == [PLAYBOOK_NAME_2, PLAYBOOK_NAME],
    )

    empty = documents.fetch_playbooks_sync([])
    check("fetch_playbooks_sync: empty input returns empty list", empty == [])

    catalog = documents.fetch_skills_catalog_sync()
    catalog_by_name = {c["name"]: c for c in catalog}
    check("fetch_skills_catalog_sync: includes scratch skill", SKILL_NAME in catalog_by_name)
    check(
        "fetch_skills_catalog_sync: shape is name+description only",
        set(catalog_by_name[SKILL_NAME].keys()) == {"name", "description"},
    )

    full_skill = documents.fetch_skill_sync(SKILL_NAME)
    check("fetch_skill_sync: found, includes content", full_skill is not None and full_skill["content"] == "# Skill one")

    missing_skill = documents.fetch_skill_sync("unknown-skill-name")
    check("fetch_skill_sync: missing name returns None", missing_skill is None)


async def main() -> None:
    await init_db()
    await cleanup()  # in case a previous failed run left rows behind

    test_split_frontmatter()

    try:
        await test_playbook_crud()
        await test_skill_crud()
        await test_validation()
        await test_duplicate_name()
        await asyncio.to_thread(test_sync_readers)
    finally:
        await cleanup()

    print(f"\n{_passed} checks PASSED")


if __name__ == "__main__":
    asyncio.run(main())
