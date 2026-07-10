#!/usr/bin/env python
"""
Smoke test for ``services.routines.review.run_review``'s first two stages
(diff, context) — this repo has no test framework yet, so this is a plain
assert-and-print script, mirroring ``scripts/test_routine_engine.py``.

Runs the REAL review routine (via ``submit_routine``) against the REAL
saffira corpus repo, project ``saffira_back-end``, with the real Postgres
and Weaviate up (``docker compose up -d``). READ-ONLY on the repo (see
``services.git_repo``'s module docstring) — nothing is checked out/mutated.

Covers:
  - a small real diff (``fix/center-to-box``): stages emit in order, stats
    are sane, context excerpts are line-numbered, graph entities matched.
  - the >60-files guard (``feat/cross-detection``, ~99 files changed).
  - the empty-diff guard (``master`` against itself).

Cleanup: deletes every ``routine_runs``/``jobs`` row this script creates.

Usage
-----
    uv run python scripts/test_review_stages.py
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

# Make the `services` package importable (the app roots absolute imports at src/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from services.db import get_pool, init_db  # noqa: E402
from services.jobs import get_job_store  # noqa: E402
from services.routines import store  # noqa: E402
from services.routines.engine import submit_routine  # noqa: E402
from services.routines.review import _stage_context, _stage_diff, run_review  # noqa: E402

SAFFIRA_REPO = "/Users/leonardo/saffira/saffira_back-end"
# Project name MUST match the ingested graph ("ingested/saffira_back-end.json")
# for the context stage to find graph entities — this is a real project name,
# not a scratch one, so cleanup below deletes only the specific run/job rows
# this script creates (tracked in _created), never a project-wide wipe.
PROJECT = "saffira_back-end"

_created: list[tuple[str, str]] = []  # (run_id, job_id) pairs to clean up
_passed = 0


def check(label: str, condition: bool) -> None:
    global _passed
    if not condition:
        raise AssertionError(f"FAIL: {label}")
    _passed += 1
    print(f"PASS: {label}")


async def poll_job(job_id: str, timeout: float = 60.0):
    store_ = get_job_store()
    elapsed = 0.0
    while elapsed < timeout:
        job = store_.get(job_id)
        if job is not None and job.status in ("succeeded", "failed"):
            return job
        await asyncio.sleep(0.2)
        elapsed += 0.2
    raise TimeoutError(f"job {job_id} did not finish in {timeout}s")


async def poll_run(run_id: str, timeout: float = 60.0):
    elapsed = 0.0
    while elapsed < timeout:
        run = await store.get_run(run_id)
        if run is not None and run["status"] in ("done", "failed"):
            return run
        await asyncio.sleep(0.2)
        elapsed += 0.2
    raise TimeoutError(f"run {run_id} did not finish in {timeout}s")


async def cleanup() -> None:
    """
    Best-effort teardown of exactly the run/job rows this script created
    (tracked in ``_created``) — ``project`` here is the real
    ``saffira_back-end`` name, so cleanup must never wildcard-delete by
    project the way the scratch-project tests do.
    """
    if not _created:
        return
    try:
        pool = await get_pool()
        removed_runs = removed_jobs = 0
        for run_id, job_id in _created:
            removed_runs += int(
                (await pool.execute("DELETE FROM routine_runs WHERE id = $1", uuid.UUID(run_id)))
                == "DELETE 1"
            )
            removed_jobs += int(
                (await pool.execute("DELETE FROM jobs WHERE id = $1", uuid.UUID(job_id))) == "DELETE 1"
            )
        print(f"cleanup: removed {removed_runs} routine_runs row(s), {removed_jobs} jobs row(s)")
        _created.clear()
    except Exception as e:
        print(f"cleanup: best-effort teardown failed (non-fatal): {e}")


async def test_small_real_diff() -> None:
    params = {"repo_path": SAFFIRA_REPO, "target_ref": "fix/center-to-box"}
    result = await submit_routine("review", PROJECT, SAFFIRA_REPO, params, run_review)
    _created.append((result["run_id"], result["job_id"]))
    job = await poll_job(result["job_id"])
    check("small-diff job status == succeeded", job.status == "succeeded")

    events = job.events or []
    stages = [e["stage"] for e in events]
    check("diff events precede context events", stages.index("diff") < stages.index("context"))
    check("resolving-refs event present", any("resolving refs" in e["message"] for e in events))
    check("diffstat event present", any(e["message"].startswith("diffstat:") for e in events))
    check(
        "per-file progress events present",
        any(e["stage"] == "context" and "context:" in e["message"] and "/" in e["message"] for e in events),
    )
    print(f"\n--- event sequence ({len(events)} events) ---")
    for e in events:
        print(f"  [{e['stage']}] {e['message']}")

    run = await poll_run(result["run_id"])
    check("small-diff run status == done", run["status"] == "done")
    stats = run["stats"]
    print(f"\n--- stats ---\n{stats}")
    check("stats: 2 files changed", stats["files_changed"] == 2)
    check("stats: 2 files reviewable", stats["files_reviewable"] == 2)
    check("stats: 0 files skipped", stats["files_skipped"] == 0)
    check("stats: hunks > 0", stats["hunks"] > 0)
    check("stats: insertions/deletions present", stats["insertions"] > 0 and stats["deletions"] > 0)
    check("stats: entities_matched > 0", stats["entities_matched"] > 0)
    check("stats: context_chars_total > 0", stats["context_chars_total"] > 0)
    print(
        f"\nentities_matched={stats['entities_matched']} "
        f"context_chars_total={stats['context_chars_total']}"
    )

    job_result = job.result or {}
    check("job result summary is the stage-1/2 placeholder", job_result.get("summary") == "analysis stages not yet implemented")
    check("job result findings empty", job_result.get("findings") == [])


def test_stage_context_contents() -> None:
    """
    Calls ``_stage_diff``/``_stage_context`` directly (no job/DB involved) to
    inspect the assembled :class:`ReviewFileContext` objects — the job/run
    result only carries aggregate stats, not the per-file context itself.
    """
    events: list[tuple[str, str]] = []
    emit = lambda stage, message: events.append((stage, message))  # noqa: E731

    _base, target, reviewable, diff_stats = _stage_diff(
        SAFFIRA_REPO, None, "fix/center-to-box", emit
    )
    check("diff stage found the 2 reviewable files", len(reviewable) == 2)

    contexts, context_chars_total, entities_matched = _stage_context(
        SAFFIRA_REPO, target, reviewable, PROJECT, emit
    )
    check("context stage returned one context per reviewable file", len(contexts) == len(reviewable))

    service_ctx = next(c for c in contexts if c.path.endswith("DetectionPipelineService.ts"))
    check("service file has target content excerpts", len(service_ctx.target_content_excerpts) > 0)
    first_excerpt_line = service_ctx.target_content_excerpts[0].splitlines()[0]
    print(f"\nfirst excerpt line: {first_excerpt_line!r}")
    check(
        "excerpts are line-numbered (e.g. '457: ...')",
        first_excerpt_line.split(":", 1)[0].strip().isdigit(),
    )
    check("service file matched graph entities", len(service_ctx.graph_entities) > 0)
    print(f"graph_entities matched for service file: {service_ctx.graph_entities}")
    check("service file has retrieved (Weaviate) entries", len(service_ctx.retrieved) > 0)
    check("retrieved entries capped at MAX_RETRIEVED_PER_FILE", len(service_ctx.retrieved) <= 5)

    gitignore_ctx = next(c for c in contexts if c.path == ".gitignore")
    check(".gitignore has no graph entities", gitignore_ctx.graph_entities == [])

    check("aggregate entities_matched == sum over files", entities_matched == sum(len(c.graph_entities) for c in contexts))
    check("aggregate context_chars_total == sum over files", context_chars_total == sum(len(x) for c in contexts for x in c.target_content_excerpts))
    print(f"context_chars_total={context_chars_total} entities_matched={entities_matched}")


async def test_guard_too_many_files() -> None:
    params = {"repo_path": SAFFIRA_REPO, "target_ref": "feat/cross-detection"}
    result = await submit_routine("review", PROJECT, SAFFIRA_REPO, params, run_review)
    _created.append((result["run_id"], result["job_id"]))
    job = await poll_job(result["job_id"])
    check("too-many-files job status == failed", job.status == "failed")
    check("too-many-files error mentions the file limit", "60" in (job.error or "") and "narrow" in (job.error or ""))
    print(f"\ntoo-many-files error: {job.error}")

    run = await poll_run(result["run_id"])
    check("too-many-files run status == failed", run["status"] == "failed")


async def test_guard_empty_diff() -> None:
    params = {"repo_path": SAFFIRA_REPO, "base_ref": "master", "target_ref": "master"}
    result = await submit_routine("review", PROJECT, SAFFIRA_REPO, params, run_review)
    _created.append((result["run_id"], result["job_id"]))
    job = await poll_job(result["job_id"])
    check("empty-diff job status == failed", job.status == "failed")
    check("empty-diff error mentions no changes", "no changes between" in (job.error or ""))
    print(f"\nempty-diff error: {job.error}")

    run = await poll_run(result["run_id"])
    check("empty-diff run status == failed", run["status"] == "failed")


async def main() -> None:
    await init_db()

    try:
        await test_small_real_diff()
        test_stage_context_contents()
        await test_guard_too_many_files()
        await test_guard_empty_diff()
    finally:
        await cleanup()

    print(f"\n{_passed} checks PASSED")


if __name__ == "__main__":
    asyncio.run(main())
