#!/usr/bin/env python
"""
Smoke test for ``services.routines.review.run_review``'s analyze/summarize
stages (A5) — this repo has no test framework yet, so this is a plain
assert-and-print script, mirroring ``scripts/test_review_stages.py``.

Runs the FULL review routine (via ``submit_routine``) against the REAL
saffira corpus repo, project ``saffira_back-end``, branch
``fix/center-to-box`` (2 reviewable files — cheap even on a 3b model), with
the real Postgres + Weaviate up (``docker compose up -d``) and Ollama
running the model already persisted via ``POST /config/llm`` (checked
directly through ``services.llm_config.load_persisted_config`` — no model
seeded by this script). READ-ONLY on the repo.

Covers: run completes; stats carry llm_calls/findings/parse_failures/model/
provider; findings rows land in Postgres with valid enums; every finding's
``anchored`` flag is cross-checked against an INDEPENDENT recompute from the
real diff's changed-line ranges; the run summary is non-empty markdown;
analyze/summarize events are present.

Cleanup: deletes every ``routine_runs``/``jobs`` row this script creates.

Usage
-----
    uv run python scripts/test_review_analyze.py
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path

# Make the `services` package importable (the app roots absolute imports at src/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from services.db import get_pool, init_db  # noqa: E402
from services.jobs import get_job_store  # noqa: E402
from services.llm_config import get_config_store, load_persisted_config  # noqa: E402
from services.routines import store  # noqa: E402
from services.routines.engine import submit_routine  # noqa: E402
from services.routines.review import _stage_diff, run_review  # noqa: E402

SAFFIRA_REPO = "/Users/leonardo/saffira/saffira_back-end"
PROJECT = "saffira_back-end"
TARGET_REF = "fix/center-to-box"

_VALID_SEVERITIES = {"critical", "high", "medium", "low", "info"}
_VALID_CATEGORIES = {"bug", "security", "performance", "maintainability", "style", "other"}
_ANCHOR_TOLERANCE = 2

_created: list[tuple[str, str]] = []  # (run_id, job_id) pairs to clean up
_passed = 0


def check(label: str, condition: bool) -> None:
    global _passed
    if not condition:
        raise AssertionError(f"FAIL: {label}")
    _passed += 1
    print(f"PASS: {label}")


async def poll_job(job_id: str, timeout: float = 600.0):
    store_ = get_job_store()
    elapsed = 0.0
    while elapsed < timeout:
        job = store_.get(job_id)
        if job is not None and job.status in ("succeeded", "failed"):
            return job
        await asyncio.sleep(0.5)
        elapsed += 0.5
    raise TimeoutError(f"job {job_id} did not finish in {timeout}s")


async def poll_run(run_id: str, timeout: float = 600.0):
    elapsed = 0.0
    while elapsed < timeout:
        run = await store.get_run(run_id)
        if run is not None and run["status"] in ("done", "failed"):
            return run
        await asyncio.sleep(0.5)
        elapsed += 0.5
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


async def test_full_review() -> None:
    params = {"repo_path": SAFFIRA_REPO, "target_ref": TARGET_REF}

    started = time.monotonic()
    result = await submit_routine("review", PROJECT, SAFFIRA_REPO, params, run_review)
    _created.append((result["run_id"], result["job_id"]))

    job = await poll_job(result["job_id"])
    wall = time.monotonic() - started
    print(f"\nwall time (submit -> job terminal): {wall:.1f}s")
    check("review job status == succeeded", job.status == "succeeded")

    events = job.events or []
    stages = [e["stage"] for e in events]
    check("analyze events present", "analyze" in stages)
    check("summarize events present", "summarize" in stages)
    check(
        "per-file 'analyzing' progress events present",
        any(e["stage"] == "analyze" and e["message"].startswith("analyzing ") for e in events),
    )
    check(
        "summarize 'generating run summary' event present",
        any(e["stage"] == "summarize" and "generating run summary" in e["message"] for e in events),
    )
    print(f"\n--- event sequence ({len(events)} events) ---")
    for e in events:
        print(f"  [{e['stage']}] {e['message']}")

    run = await poll_run(result["run_id"])
    check("review run status == done", run["status"] == "done")

    stats = run["stats"]
    print(f"\n--- stats ---\n{stats}")
    check(
        "stats: llm_calls >= 3 (2 analyze + 1 summarize, +retries maybe)",
        stats["llm_calls"] >= 3,
    )
    check("stats: findings_total present", "findings_total" in stats)
    check("stats: findings_anchored present", "findings_anchored" in stats)
    check("stats: parse_failures present", "parse_failures" in stats)
    check("stats: prompt_chars_total > 0", stats["prompt_chars_total"] > 0)
    check("stats: wall_seconds_analyze present", "wall_seconds_analyze" in stats)
    check("stats: wall_seconds_summarize present", "wall_seconds_summarize" in stats)
    check("stats: model recorded", bool(stats.get("model")))
    check("stats: provider recorded", stats.get("provider") == "ollama")

    summary = run["summary"]
    check("summary is non-empty markdown", bool(summary and summary.strip()))
    print(f"\n--- summary ---\n{summary}")

    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT * FROM findings WHERE run_id = $1 ORDER BY id", uuid.UUID(result["run_id"])
    )
    check("findings row count matches stats.findings_total", len(rows) == stats["findings_total"])

    # Recompute the diff's changed-line ranges INDEPENDENTLY (a fresh
    # _stage_diff call) to cross-check every finding's `anchored` flag
    # rather than trusting the routine's own bookkeeping.
    _base, _target, reviewable, _diff_stats = _stage_diff(
        SAFFIRA_REPO, None, TARGET_REF, lambda *_: None
    )
    ranges_by_file = {fd.path: fd.changed_line_ranges for fd in reviewable}

    print(f"\n--- {len(rows)} finding(s) produced by the model ---")
    for r in rows:
        check(
            f"finding {r['id']} severity '{r['severity']}' is a valid enum",
            r["severity"] in _VALID_SEVERITIES,
        )
        check(
            f"finding {r['id']} category '{r['category']}' is a valid enum",
            r["category"] in _VALID_CATEGORIES,
        )
        expected_anchored = False
        if r["start_line"] is not None:
            ranges = ranges_by_file.get(r["file_path"], [])
            expected_anchored = any(
                (rs - _ANCHOR_TOLERANCE) <= r["start_line"] <= (re_ + _ANCHOR_TOLERANCE)
                for rs, re_ in ranges
            )
        check(
            f"finding {r['id']} anchored={r['anchored']} matches independent recompute",
            r["anchored"] == expected_anchored,
        )
        print(
            f"  #{r['id']} [{r['severity']}/{r['category']}] {r['file_path']}:"
            f"{r['start_line']}-{r['end_line']} anchored={r['anchored']}\n"
            f"      {r['title']}\n"
            f"      {r['body']}"
        )
        if r["suggestion"]:
            print(f"      suggestion: {r['suggestion']}")


async def main() -> None:
    await init_db()

    persisted = await load_persisted_config()
    if persisted is None:
        raise SystemExit(
            "no persisted LLM config found — POST to /config/llm with a local Ollama "
            "model (e.g. qwen2.5-coder:3b) first"
        )
    get_config_store().set_llm_config(persisted)
    print(f"active LLM config: {persisted.provider_name}/{persisted.model} @ {persisted.base_url}")

    try:
        await test_full_review()
    finally:
        await cleanup()

    print(f"\n{_passed} checks PASSED")


if __name__ == "__main__":
    asyncio.run(main())
