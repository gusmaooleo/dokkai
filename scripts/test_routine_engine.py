#!/usr/bin/env python
"""
Smoke test for the routine engine plumbing (``services.jobs`` external-kind
support + ``services.routines.store``/``engine``) — this repo has no test
framework yet, so this is a plain assert-and-print script rather than a new
pytest dependency, mirroring ``scripts/test_git_repo.py``.

Talks to the REAL local Postgres (``docker compose up -d``). Creates and
tears down its own ``routine_runs``/``findings``/``jobs`` rows for two
scratch project names — nothing else on the machine is touched.

Usage
-----
    uv run python scripts/test_routine_engine.py
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

# Make the `services` package importable (the app roots absolute imports at src/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from services.db import get_pool, init_db  # noqa: E402
from services.jobs import Job, ProjectJobConflict, add_job_event, get_job_store  # noqa: E402
from services.routines import store  # noqa: E402
from services.routines.engine import submit_routine  # noqa: E402

_passed = 0

SUCCESS_PROJECT = "test-routine-engine-scratch"
FAIL_PROJECT = "test-routine-engine-scratch-fail"
BADFINDING_PROJECT = "test-routine-engine-scratch-badfinding"
LOCK_PROJECT = "test-routine-engine-scratch-lock"
ALL_TEST_PROJECTS = [SUCCESS_PROJECT, FAIL_PROJECT, LOCK_PROJECT, BADFINDING_PROJECT]


def check(label: str, condition: bool) -> None:
    global _passed
    if not condition:
        raise AssertionError(f"FAIL: {label}")
    _passed += 1
    print(f"PASS: {label}")


async def ok_runner(job, run_id, params, emit) -> dict:
    for i in range(5):
        emit("step", f"doing thing {i}")
        await asyncio.sleep(0.05)
    return {
        "summary": "looked fine",
        "stats": {"files": 3},
        "findings": [
            {
                "file_path": "foo.py",
                "start_line": 1,
                "end_line": 2,
                "severity": "low",
                "category": "style",
                "title": "nit",
                "body": "minor nit",
                "suggestion": None,
                "evidence": {"note": "test"},
                "anchored": True,
            }
        ],
    }


async def bad_finding_runner(job, run_id, params, emit) -> dict:
    # A severity outside the CHECK constraint makes insert_findings fail,
    # exercising the run-failed / job-succeeded divergence.
    return {
        "summary": "ran fine but findings are unstorable",
        "stats": {},
        "findings": [
            {
                "file_path": "foo.py",
                "start_line": 1,
                "end_line": 1,
                "severity": "not-a-real-severity",
                "category": "style",
                "title": "bad",
                "body": "bad",
                "suggestion": None,
                "evidence": None,
                "anchored": True,
            }
        ],
    }


async def failing_runner(job, run_id, params, emit) -> dict:
    emit("step", "about to blow up")
    raise RuntimeError("boom")


async def slow_runner(job, run_id, params, emit) -> dict:
    await asyncio.sleep(1.5)
    return {"summary": "done", "stats": {}}


async def poll_job(job_id: str, timeout: float = 10.0) -> Job:
    store_ = get_job_store()
    elapsed = 0.0
    while elapsed < timeout:
        job = store_.get(job_id)
        if job is not None and job.status in ("succeeded", "failed"):
            return job
        await asyncio.sleep(0.1)
        elapsed += 0.1
    raise TimeoutError(f"job {job_id} did not finish in {timeout}s")


async def poll_run(run_id: str, timeout: float = 10.0) -> dict:
    elapsed = 0.0
    while elapsed < timeout:
        run = await store.get_run(run_id)
        if run is not None and run["status"] in ("done", "failed"):
            return run
        await asyncio.sleep(0.1)
        elapsed += 0.1
    raise TimeoutError(f"run {run_id} did not finish in {timeout}s")


async def cleanup() -> None:
    """Best-effort teardown of every row this script may have created."""
    try:
        pool = await get_pool()
        run_rows = await pool.fetch(
            "SELECT id FROM routine_runs WHERE project = ANY($1::text[])", ALL_TEST_PROJECTS
        )
        for row in run_rows:
            await pool.execute("DELETE FROM routine_runs WHERE id = $1", row["id"])
        job_rows = await pool.fetch(
            "SELECT id FROM jobs WHERE project = ANY($1::text[])", ALL_TEST_PROJECTS
        )
        for row in job_rows:
            await pool.execute("DELETE FROM jobs WHERE id = $1", row["id"])
        print(
            f"cleanup: removed {len(run_rows)} routine_runs row(s), "
            f"{len(job_rows)} jobs row(s) for scratch test projects"
        )
    except Exception as e:
        print(f"cleanup: best-effort teardown failed (non-fatal): {e}")


def test_event_bounding() -> None:
    """Unit-style check of add_job_event's cap/monotonic-seq behavior, with
    no DB involved — 250 events pushed, only the last 200 survive, and seq
    keeps counting up across the drop so gaps are detectable."""
    job = Job(id=str(uuid.uuid4()), repo_path="", kind="review", project="unit-test")
    for i in range(250):
        add_job_event(job, "step", f"event {i}")
    check("event log capped at 200", len(job.events) == 200)
    seqs = [e["seq"] for e in job.events]
    check("event seqs still monotonic after drop", seqs == list(range(50, 250)))
    check("oldest surviving event is #50", job.events[0]["message"] == "event 50")
    check("newest event is #249", job.events[-1]["message"] == "event 249")


async def test_success_path() -> None:
    result = await submit_routine(
        "review", SUCCESS_PROJECT, "/tmp/does-not-matter", {"branch": "main"}, ok_runner
    )
    job_id, run_id = result["job_id"], result["run_id"]

    # Immediately after submit, the routine (5 * 0.05s sleeps) cannot have
    # finished yet — the job must still be queued/running, not terminal.
    live_job = get_job_store().get(job_id)
    check("job not yet terminal right after submit", live_job.status in ("queued", "running"))

    job = await poll_job(job_id)
    check("success job status == succeeded", job.status == "succeeded")
    check("success job events present", job.events is not None and len(job.events) == 5)
    seqs = [e["seq"] for e in job.events]
    check("success job events ordered/monotonic", seqs == list(range(5)))

    run = await poll_run(run_id)
    check("success run status == done", run["status"] == "done")
    check("success run summary persisted", run["summary"] == "looked fine")
    check("success run stats persisted", run["stats"] == {"files": 3})

    pool = await get_pool()
    findings_rows = await pool.fetch("SELECT * FROM findings WHERE run_id = $1", uuid.UUID(run_id))
    check("findings inserted", len(findings_rows) == 1)
    check("finding title persisted", findings_rows[0]["title"] == "nit")


async def test_failure_path() -> None:
    result = await submit_routine("bughunt", FAIL_PROJECT, "/tmp/does-not-matter", {}, failing_runner)
    job = await poll_job(result["job_id"])
    check("failing job status == failed", job.status == "failed")
    check("failing job error mentions boom", "boom" in (job.error or ""))
    check("failing job kept its one event before the raise", job.events is not None and len(job.events) == 1)

    run = await poll_run(result["run_id"])
    check("failing run status == failed", run["status"] == "failed")
    check("failing run error mentions boom", "boom" in (run["error"] or ""))


async def test_unstorable_findings_path() -> None:
    result = await submit_routine(
        "review", BADFINDING_PROJECT, "/tmp/does-not-matter", {}, bad_finding_runner
    )
    job = await poll_job(result["job_id"])
    check("bad-finding job still succeeds (raw result in snapshot)", job.status == "succeeded")

    run = await poll_run(result["run_id"])
    check("bad-finding run is failed", run["status"] == "failed")
    check(
        "bad-finding run error explains the persistence failure",
        "findings could not be persisted" in (run["error"] or ""),
    )


async def test_project_lock() -> None:
    result1 = await submit_routine("review", LOCK_PROJECT, "/tmp/does-not-matter", {}, slow_runner)
    try:
        await submit_routine("review", LOCK_PROJECT, "/tmp/does-not-matter", {}, slow_runner)
        check("second submit on locked project raises ProjectJobConflict", False)
    except ProjectJobConflict as e:
        check(
            "lock conflict message matches the ingestion 409 format",
            str(e) == f"another job is running for project '{LOCK_PROJECT}'",
        )

    # The rejected submit must not have left an orphaned routine_runs row.
    runs = await store.list_runs(project=LOCK_PROJECT)
    check("no orphaned run row from the rejected submit", len(runs) == 1)

    await poll_job(result1["job_id"])  # let it finish so it doesn't linger


async def main() -> None:
    await init_db()
    await cleanup()  # in case a previous failed run left rows behind

    test_event_bounding()

    try:
        await test_success_path()
        await test_failure_path()
        await test_unstorable_findings_path()
        await test_project_lock()
    finally:
        await cleanup()

    print(f"\n{_passed} checks PASSED")


if __name__ == "__main__":
    asyncio.run(main())
