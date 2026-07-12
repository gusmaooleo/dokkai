#!/usr/bin/env python
"""
Smoke test for the bug-hunt routine (B5) — ``services.routines.bughunt`` and
``controllers/routines.py``'s ``kind='bughunt'`` launch support. This repo
has no test framework yet, so this is a plain assert-and-print script,
mirroring ``scripts/test_routines_api.py``/``scripts/test_review_playbooks.py``.

Two kinds of coverage:

  - HTTP-level MECHANICS (not judged on finding quality — a 3b model may or
    may not report anything real for a given scope): talks to the REAL
    running API (``./dev.sh``), the REAL sample corpus (project
    ``sample_back-end``, already ingested), real Postgres/Weaviate, and
    Ollama running ``qwen2.5-coder:3b`` (already persisted via
    ``POST /config/llm`` — same precondition as ``test_review_analyze.py``).
    Launches one bughunt run with a playbook + path_prefix attached and
    asserts the stage/event sequence, stats shape, and launch validations
    (missing scope 422, non-ollama provider 400, playbook applicability).
  - In-process UNIT cases, no HTTP/LLM/Weaviate involved: the JSON-verdict
    parse-failure degrade path (a FAKE provider, ``services.routines.bughunt.
    run_bughunt`` called directly with ``_stage_resolve``/``get_client``
    monkeypatched) and the finding-anchoring rule
    (``services.routines.bughunt._validate_finding`` called directly against
    a synthetic ``graph_store.Graph``).

Cleanup: deletes every scratch ``playbooks``/``routine_runs``/``jobs`` row
this script creates.

Usage
-----
    ./dev.sh &                      # or: PORT=8000 ./dev.sh &
    uv run python scripts/test_bughunt.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

import httpx

# Make the `services` package importable (the app roots absolute imports at src/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from services import graph_store  # noqa: E402
from services.jobs import Job  # noqa: E402
from services.routines import bughunt  # noqa: E402
from services.routines.agent_loop import extract_fallback_tool_call  # noqa: E402

BASE_URL = os.getenv("DOKKAI_API_URL", "http://localhost:8000")
PROJECT = "sample_back-end"
SCOPE = "error handling in the alarms services"
PATH_PREFIX = "src/features/alarms"
ADMIN_USERNAME = os.getenv("DOKKAI_ROOT_USER", "admin")
ADMIN_PASSWORD = os.getenv("DOKKAI_ROOT_PASSWORD", "admin")

PB_BUGHUNT = "test-bughunt-house-rules"
PB_REVIEW_ONLY = "test-bughunt-review-only"

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


def launch_bughunt_payload(
    scope: str | None = SCOPE,
    path_prefix: str | None = PATH_PREFIX,
    playbooks: list[str] | None = None,
    provider: str | None = None,
) -> dict:
    payload: dict = {"kind": "bughunt", "project": PROJECT}
    if scope is not None:
        payload["scope"] = scope
    if path_prefix is not None:
        payload["path_prefix"] = path_prefix
    if playbooks is not None:
        payload["playbooks"] = playbooks
    if provider is not None:
        payload["provider"] = provider
    return payload


# ---------------------------------------------------------------------------
# HTTP-level mechanics + launch validations
# ---------------------------------------------------------------------------


async def test_http() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=600.0) as client:
        admin_token = await login(client, ADMIN_USERNAME, ADMIN_PASSWORD)
        check("admin login", bool(admin_token))

        try:
            await create_playbook(
                client,
                admin_token,
                PB_BUGHUNT,
                "---\ntitle: house rules\n---\nAlways cite exact file/line evidence.",
                ["bughunt"],
            )
            await create_playbook(
                client, admin_token, PB_REVIEW_ONLY, "# review-only playbook", ["review"]
            )

            # --- negative: missing scope -> 422 (DTO validator) ---
            resp = await client.post(
                "/routines/runs",
                json={"kind": "bughunt", "project": PROJECT},
                headers=auth_headers(admin_token),
            )
            check("missing scope 422", resp.status_code == 422)

            # --- negative: non-ollama provider override -> 400 ---
            resp = await client.post(
                "/routines/runs",
                json=launch_bughunt_payload(provider="openai"),
                headers=auth_headers(admin_token),
            )
            check("non-ollama provider 400", resp.status_code == 400)

            # --- negative: review-only playbook doesn't apply to bughunt -> 400 ---
            resp = await client.post(
                "/routines/runs",
                json=launch_bughunt_payload(playbooks=[PB_REVIEW_ONLY]),
                headers=auth_headers(admin_token),
            )
            check("review-only playbook on bughunt 400", resp.status_code == 400)
            check(
                "review-only playbook on bughunt 400 message",
                resp.json()["detail"] == f"playbook '{PB_REVIEW_ONLY}' does not apply to bughunt routines",
            )

            # --- negative: unknown project, 404 verbatim (mirrors review) ---
            resp = await client.post(
                "/routines/runs",
                json={"kind": "bughunt", "project": "does-not-exist-project", "scope": "x"},
                headers=auth_headers(admin_token),
            )
            check("unknown project bughunt launch 404", resp.status_code == 404)

            # --- mechanics run: scope + path_prefix + a bughunt-applicable playbook ---
            started = time.monotonic()
            resp = await client.post(
                "/routines/runs",
                json=launch_bughunt_payload(playbooks=[PB_BUGHUNT]),
                headers=auth_headers(admin_token),
            )
            check("launch bughunt 202", resp.status_code == 202)
            launch = resp.json()
            run_id, job_id = launch["run_id"], launch["job_id"]
            _created_run_ids.append(run_id)

            run = await poll_run_detail(client, admin_token, run_id)
            wall = time.monotonic() - started
            print(f"\nwall time (launch -> run terminal): {wall:.1f}s")
            check("bughunt run status == done", run["status"] == "done")
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

            events = await get_job_events(client, admin_token, job_id)
            stages = [e["stage"] for e in events]
            print(f"\n--- event sequence ({len(events)} events) ---")
            for e in events:
                print(f"  [{e['stage']}] {e['message']}")

            check("'resolve' stage event fired", "resolve" in stages)
            check("'playbooks' stage event fired", "playbooks" in stages)
            check(
                "'loaded playbook <name>' event present",
                any(e["stage"] == "playbooks" and e["message"] == f"loaded playbook {PB_BUGHUNT}" for e in events),
            )
            check("'hunt' stage event fired", "hunt" in stages)
            if any(e["stage"] == "agent" and e["message"].startswith("→ ") for e in events):
                check("at least one '→ tool' agent event fired during hunt", True)
            else:
                # A small local model occasionally answers directly without
                # calling a tool first (documented quirk, see
                # scripts/test_agent_loop.py's test_live_smoke) — advisory,
                # not a loop/routine bug; mechanics of tool dispatch itself
                # are already pinned by test_agent_loop.py's deterministic
                # fake-provider tests.
                print(
                    "\nADVISORY: no '→ tool' agent event this run — the 3b model answered "
                    "directly without calling a tool (known small-model quirk, not a bug)."
                )
            # 'resolve' must come before 'hunt', which must come before any
            # terminal 'summarize'/parse-degrade — a loose ordering check
            # (indices), not a strict full-sequence match (stage repeats are
            # normal, e.g. multiple 'agent' tool-call events).
            resolve_idx = stages.index("resolve")
            hunt_idx = stages.index("hunt")
            check("'resolve' precedes 'hunt'", resolve_idx < hunt_idx)
            if "findings" in stages:
                check("'hunt' precedes 'findings'", hunt_idx < stages.index("findings"))

            stats = run["stats"]
            print(f"\n--- stats ---\n{stats}")
            for key in (
                "rounds", "tool_calls_made", "tool_payload_chars", "loaded_skills",
                "model", "provider", "playbooks_used", "parse_failures", "llm_calls",
            ):
                check(f"stats has '{key}'", key in stats)
            check("stats.playbooks_used == [PB_BUGHUNT]", stats["playbooks_used"] == [PB_BUGHUNT])
            check("stats.provider == 'ollama'", stats["provider"] == "ollama")
            if stats["parse_failures"] == 0:
                check("stats has findings_total/anchored", "findings_total" in stats and "findings_anchored" in stats)
                check("summarize stage event fired", "summarize" in stages)
            else:
                check(
                    "parse-degrade: summary prefixed 'agent answer (unparsed): '",
                    (run["summary"] or "").startswith("agent answer (unparsed): "),
                )

        finally:
            await cleanup(client, admin_token)


# ---------------------------------------------------------------------------
# Unit case: JSON-parse degrade path (fake provider, no LLM/Weaviate/DB)
# ---------------------------------------------------------------------------


class _FakeNonJsonProvider:
    provider_name = "ollama"

    async def chat_with_tools(self, messages, *, model, tools=None, temperature=0.3, max_tokens=4096):
        return {"role": "assistant", "content": "I looked around but this is not JSON, sorry."}

    async def chat(self, messages, *, model, temperature=0.3, max_tokens=4096):
        return "Still not JSON, sorry."


class _FakeCollections:
    def get(self, name):
        return object()


class _FakeWeaviateClient:
    collections = _FakeCollections()

    def close(self) -> None:
        pass


async def test_parse_degrade_unit() -> None:
    async def fake_stage_resolve(params, emit):
        emit("resolve", "resolve: using fake/ollama")
        return _FakeNonJsonProvider(), "fake-model", "ollama"

    def fake_get_client():
        return _FakeWeaviateClient()

    original_stage_resolve = bughunt._stage_resolve
    original_get_client = bughunt.get_client
    bughunt._stage_resolve = fake_stage_resolve
    bughunt.get_client = fake_get_client
    try:
        events: list[tuple[str, str]] = []
        job = Job(id="fake-job", repo_path="/tmp", kind="bughunt", project="fake-project")
        result = await bughunt.run_bughunt(
            job, "fake-run-id", {"repo_path": "/tmp", "scope": "anything"}, lambda s, m: events.append((s, m))
        )
    finally:
        bughunt._stage_resolve = original_stage_resolve
        bughunt.get_client = original_get_client

    check("parse-degrade: findings == []", result["findings"] == [])
    check("parse-degrade: stats.parse_failures == 1", result["stats"]["parse_failures"] == 1)
    check(
        "parse-degrade: summary prefixed 'agent answer (unparsed): '",
        result["summary"] == "agent answer (unparsed): I looked around but this is not JSON, sorry.",
    )
    check(
        "parse-degrade: parse_failed event emitted",
        any(msg.startswith("parse_failed:") for _, msg in events),
    )
    # Stats key-set parity (reviewer minor #2): 'llm_calls' must be present
    # on the degrade path too, not just the success path — here it's exactly
    # 1 (the one retry chat() call the failed parse triggered).
    check("parse-degrade: stats has 'llm_calls'", "llm_calls" in result["stats"])
    check("parse-degrade: stats.llm_calls == 1 (the retry call)", result["stats"]["llm_calls"] == 1)


# ---------------------------------------------------------------------------
# Unit case: read-before-report pushback guard (fake provider, no LLM/Weaviate/DB)
# ---------------------------------------------------------------------------


class _ScriptedBughuntProvider:
    """Returns each entry of *script* in order, one per ``chat_with_tools``
    call; ``chat()`` (used by the JSON-reformat retry / summarize stage)
    always returns a fixed non-JSON string — fine since none of the pushback
    test cases below exercise the parse-retry path (every scripted answer is
    already valid JSON)."""

    provider_name = "ollama"

    def __init__(self, script: list[dict]) -> None:
        self._script = list(script)
        self.calls: list[dict] = []

    async def chat_with_tools(self, messages, *, model, tools=None, temperature=0.3, max_tokens=4096):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        if self._script:
            return self._script.pop(0)
        return {"role": "assistant", "content": '{"findings": [], "notes": "script exhausted"}'}

    async def chat(self, messages, *, model, temperature=0.3, max_tokens=4096):
        return "### Summary\n\n- ok"


async def _run_bughunt_with_fake_provider(provider) -> tuple[dict, list[tuple[str, str]]]:
    async def fake_stage_resolve(params, emit):
        emit("resolve", "resolve: using fake/ollama")
        return provider, "fake-model", "ollama"

    def fake_get_client():
        return _FakeWeaviateClient()

    original_stage_resolve = bughunt._stage_resolve
    original_get_client = bughunt.get_client
    bughunt._stage_resolve = fake_stage_resolve
    bughunt.get_client = fake_get_client
    try:
        events: list[tuple[str, str]] = []
        job = Job(id="fake-job", repo_path="/tmp", kind="bughunt", project="fake-project")
        result = await bughunt.run_bughunt(
            job, "fake-run-id", {"repo_path": "/tmp", "scope": "anything"}, lambda s, m: events.append((s, m))
        )
    finally:
        bughunt._stage_resolve = original_stage_resolve
        bughunt.get_client = original_get_client
    return result, events


_FINDING_JSON = (
    '{"findings": [{"file_path": "src/features/alarms/fake.ts", "start_line": 1, "end_line": 1, '
    '"severity": "medium", "category": "bug", "title": "t", "body": "b", "suggestion": null}], '
    '"notes": "n"}'
)
_NO_FINDING_JSON = '{"findings": [], "notes": "clean"}'


async def test_pushback_findings_without_reads() -> None:
    """A verdict with findings but zero tool calls at all (so certainly no
    get_entity/get_file) must trigger exactly ONE pushback round, and the
    pushback's own answer (even if it's ALSO reads-free) is accepted as
    final — 'accept whatever comes back' (no second pushback)."""
    provider = _ScriptedBughuntProvider(
        [
            {"role": "assistant", "content": _FINDING_JSON},  # round 1: findings, no tool calls at all
            {"role": "assistant", "content": _FINDING_JSON},  # pushback round: re-answers, still no reads
        ]
    )
    result, events = await _run_bughunt_with_fake_provider(provider)

    check("pushback (no reads): stats.pushback_used is True", result["stats"]["pushback_used"] is True)
    check(
        "pushback (no reads): pushback event emitted",
        any(msg.startswith("pushback: findings without code reads") for _, msg in events),
    )
    check(
        "pushback (no reads): 'agent re-answered' event emitted",
        any(msg.startswith("pushback: agent re-answered") for _, msg in events),
    )
    check(
        "pushback (no reads): exactly 2 chat_with_tools calls made (1 + the 1 pushback round)",
        len(provider.calls) == 2,
    )
    check("pushback (no reads): stats.tools_called == []", result["stats"]["tools_called"] == [])


async def test_pushback_findings_with_reads() -> None:
    """A verdict with findings where a get_entity call happened earlier in
    the SAME loop (even one that errored, since the fake Weaviate client
    can't really serve it) must NOT trigger a pushback — the mechanical
    guard only checks whether the tool was called, not whether it
    succeeded."""
    provider = _ScriptedBughuntProvider(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "get_entity", "arguments": {"qualified_name": "x.y"}}}],
            },
            {"role": "assistant", "content": _FINDING_JSON},  # round 2: final verdict after the read attempt
        ]
    )
    result, events = await _run_bughunt_with_fake_provider(provider)

    check("pushback (with reads): stats.pushback_used is False", result["stats"]["pushback_used"] is False)
    check(
        "pushback (with reads): no pushback event emitted",
        not any(msg.startswith("pushback:") for _, msg in events),
    )
    check(
        "pushback (with reads): stats.tools_called == ['get_entity']",
        result["stats"]["tools_called"] == ["get_entity"],
    )
    check(
        "pushback (with reads): exactly 2 chat_with_tools calls made (no pushback round)",
        len(provider.calls) == 2,
    )


async def test_pushback_zero_findings() -> None:
    """A clean (zero-finding) verdict must never trigger a pushback,
    regardless of whether any tool was called."""
    provider = _ScriptedBughuntProvider([{"role": "assistant", "content": _NO_FINDING_JSON}])
    result, events = await _run_bughunt_with_fake_provider(provider)

    check("pushback (zero findings): stats.pushback_used is False", result["stats"]["pushback_used"] is False)
    check(
        "pushback (zero findings): no pushback event emitted",
        not any(msg.startswith("pushback:") for _, msg in events),
    )
    check("pushback (zero findings): exactly 1 chat_with_tools call made", len(provider.calls) == 1)


# ---------------------------------------------------------------------------
# Unit case: embedded-extraction shape disambiguation (approved follow-up) —
# no LLM/Weaviate/DB, direct function calls.
# ---------------------------------------------------------------------------


def test_embedded_extraction_shape_disambiguation() -> None:
    """A message containing BOTH a findings-shaped block and a tool-call-
    shaped block (findings block placed FIRST, to prove this is shape-
    driven, not 'just take the first candidate'): the loop's own fallback
    tool-call parser must pick the tool-call-shaped block, and bughunt's
    verdict parser must pick the findings-shaped block — same shared
    ``find_json_object_candidates``, different shape predicates."""
    content = (
        'Here is my verdict: {"findings": [], "notes": "clean"}\n'
        "Actually let me check one more thing first: "
        '{"name": "get_file", "arguments": {"path": "a.ts"}}'
    )

    tool_call = extract_fallback_tool_call(content)
    check(
        "shape disambiguation: tool-loop picks the tool-call-shaped block",
        tool_call == {"name": "get_file", "arguments": {"path": "a.ts"}},
    )

    verdict = bughunt._parse_hunt_answer(content)
    check(
        "shape disambiguation: verdict parser picks the findings-shaped block",
        verdict == {"findings": [], "notes": "clean"},
    )


def test_parse_hunt_answer_embedded_extraction() -> None:
    """``_parse_hunt_answer`` itself recovers a findings verdict embedded in
    prose (regression for the naive first-'{'-to-last-'}' span it used to
    use, which a leading/trailing unrelated brace could break)."""
    pure = bughunt._parse_hunt_answer('{"findings": [], "notes": "ok"}')
    check("parse_hunt_answer: pure object (regression)", pure == {"findings": [], "notes": "ok"})

    fenced = bughunt._parse_hunt_answer('```json\n{"findings": [], "notes": "ok"}\n```')
    check("parse_hunt_answer: fenced block (regression)", fenced == {"findings": [], "notes": "ok"})

    in_prose = bughunt._parse_hunt_answer(
        'I investigated and concluded: {"findings": [], "notes": "clean"} — that is my final answer.'
    )
    check("parse_hunt_answer: bare object mid-prose", in_prose == {"findings": [], "notes": "clean"})

    check(
        "parse_hunt_answer: no findings-shaped candidate -> None (degrade preserved)",
        bughunt._parse_hunt_answer('I looked around: {"name": "get_file", "arguments": {}} but that is all.')
        is None,
    )


# ---------------------------------------------------------------------------
# Unit case: finding anchoring rule (synthetic graph, no LLM/Weaviate/DB)
# ---------------------------------------------------------------------------


def _synthetic_graph() -> graph_store.Graph:
    return graph_store.Graph(
        project_name="fake-anchor-project",
        path=Path("/tmp/fake-anchor-project.json"),
        nodes=[
            {
                "node_id": 1,
                "labels": ["Function"],
                "properties": {
                    "path": "src/features/alarms/service.ts",
                    "qualified_name": "alarms.service.handleError",
                    "start_line": 10,
                    "end_line": 20,
                },
            }
        ],
        relationships=[],
        metadata={},
    )


def test_anchoring_unit() -> None:
    graph = _synthetic_graph()
    original_get_graph = graph_store.get_graph
    graph_store.get_graph = lambda project_name: graph

    def raising_get_graph(project_name):
        raise AssertionError("get_graph must not be called when the file isn't in the project graph")

    try:
        # --- case A: file in graph, start_line inside the entity's span -> anchored ---
        original_resolve_file = bughunt.resolve_file
        bughunt.resolve_file = lambda project, path: {"kind": "File", "path": path}
        try:
            raw = {
                "file_path": "src/features/alarms/service.ts",
                "start_line": 15,
                "end_line": 15,
                "severity": "high",
                "category": "bug",
                "title": "t",
                "body": "b",
            }
            finding, notes = bughunt._validate_finding(raw, "p", "/tmp", None, "m", "ollama")
            check("case A: kept", finding is not None)
            check("case A: anchored == True", finding["anchored"] is True)
            check(
                "case A: evidence.entities cites the matched entity",
                finding["evidence"]["entities"] == ["alarms.service.handleError"],
            )
            check(
                "case A: stored file_path unchanged (already relative — regression)",
                finding["file_path"] == "src/features/alarms/service.ts",
            )

            # --- case I: file_path given ABSOLUTE (repo_path-prefixed, as tool
            # results/get_file echo) but pointing at the SAME entity/line as case
            # A -> normalized to repo-relative BEFORE anchoring, so it anchors
            # exactly like case A, AND the STORED file_path is the relative form
            # (matches review's storage convention / what the UI renders) ---
            raw_abs_anchored = {**raw, "file_path": "/tmp/src/features/alarms/service.ts"}
            finding_i, notes_i = bughunt._validate_finding(raw_abs_anchored, "p", "/tmp", None, "m", "ollama")
            check("case I: absolute in-repo path — kept", finding_i is not None)
            check("case I: absolute in-repo path — anchored == True", finding_i["anchored"] is True)
            check(
                "case I: stored file_path normalized to repo-relative",
                finding_i["file_path"] == "src/features/alarms/service.ts",
            )
            check(
                "case I: evidence.entities cites the same matched entity as case A",
                finding_i["evidence"]["entities"] == ["alarms.service.handleError"],
            )

            # --- case B: file in graph, start_line OUTSIDE the entity's span -> unanchored, kept ---
            raw_outside = {**raw, "start_line": 999, "end_line": 999}
            finding_b, notes_b = bughunt._validate_finding(raw_outside, "p", "/tmp", None, "m", "ollama")
            check("case B: kept", finding_b is not None)
            check("case B: anchored == False", finding_b["anchored"] is False)
            check("case B: unanchored note present", any("unanchored" in n for n in notes_b))

            # --- case G: path_prefix boundary — a sibling dir that merely shares the
            # prefix STRING (not a path segment) must be rejected, not bare-startswith-admitted ---
            raw_sibling = {
                "file_path": "src/features/alarms_v2/x.ts",
                "start_line": 1,
                "end_line": 1,
                "severity": "low",
                "category": "style",
                "title": "t",
                "body": "b",
            }
            finding_g, notes_g = bughunt._validate_finding(
                raw_sibling, "p", "/tmp", "src/features/alarms", "m", "ollama"
            )
            check("case G: sibling dir sharing prefix string dropped (segment boundary)", finding_g is None)
            check("case G: dropped note mentions path_prefix", any("outside path_prefix" in n for n in notes_g))

            # --- case H: a true subpath of path_prefix is accepted by the boundary check ---
            raw_within = {**raw_sibling, "file_path": "src/features/alarms/x.ts"}
            finding_h, _notes_h = bughunt._validate_finding(
                raw_within, "p", "/tmp", "src/features/alarms", "m", "ollama"
            )
            check("case H: file within path_prefix kept", finding_h is not None)
        finally:
            bughunt.resolve_file = original_resolve_file

        # --- case C: file NOT in graph but exists on disk -> kept, unanchored ---
        bughunt.resolve_file = lambda project, path: None
        try:
            with tempfile.TemporaryDirectory() as repo_path:
                real_rel = "foo/bar.py"
                real_abs = Path(repo_path) / real_rel
                real_abs.parent.mkdir(parents=True, exist_ok=True)
                real_abs.write_text("# real file\n")

                raw_disk = {
                    "file_path": real_rel,
                    "start_line": 1,
                    "end_line": 1,
                    "severity": "low",
                    "category": "style",
                    "title": "t",
                    "body": "b",
                }
                graph_store.get_graph = raising_get_graph  # must NOT be reached (file not in graph)
                finding_c, notes_c = bughunt._validate_finding(raw_disk, "p", repo_path, None, "m", "ollama")
                check("case C: kept (exists on disk)", finding_c is not None)
                check("case C: anchored == False", finding_c["anchored"] is False)
                check(
                    "case C: 'exists on disk but not in graph' note present",
                    any("exists on disk but is not a node in the project graph" in n for n in notes_c),
                )

                # --- case D: file NOT in graph and NOT on disk -> dropped (hallucination) ---
                raw_ghost = {**raw_disk, "file_path": "totally/made/up/ghost.py"}
                finding_d, notes_d = bughunt._validate_finding(raw_ghost, "p", repo_path, None, "m", "ollama")
                check("case D: dropped (hallucinated)", finding_d is None)
                check("case D: dropped note mentions hallucinated", any("hallucinated" in n for n in notes_d))

                # --- case E: absolute file_path escaping repo_path (e.g. '/etc/passwd', which
                # DOES exist on disk) -> dropped, not treated as "exists on disk". Also verifies
                # order of operations for the up-front repo-relative normalization:
                # _relative_to_repo returns an escaping absolute path UNCHANGED (os.path.relpath
                # would start with '..'), so the containment guard below still evaluates the
                # untouched escaping path exactly as before normalization was introduced ---
                check("precondition: /etc/passwd exists on this machine", os.path.isfile("/etc/passwd"))
                raw_abs = {**raw_disk, "file_path": "/etc/passwd"}
                finding_e, notes_e = bughunt._validate_finding(raw_abs, "p", repo_path, None, "m", "ollama")
                check("case E: absolute path escaping repo_path dropped", finding_e is None)
                check("case E: dropped note mentions hallucinated", any("hallucinated" in n for n in notes_e))

                # --- case F: '../' traversal escaping repo_path -> dropped ---
                raw_traversal = {**raw_disk, "file_path": "../../../../../../../../etc/passwd"}
                finding_f, notes_f = bughunt._validate_finding(raw_traversal, "p", repo_path, None, "m", "ollama")
                check("case F: '../' traversal escaping repo_path dropped", finding_f is None)
                check("case F: dropped note mentions hallucinated", any("hallucinated" in n for n in notes_f))
        finally:
            bughunt.resolve_file = original_resolve_file
    finally:
        graph_store.get_graph = original_get_graph


async def main() -> None:
    test_anchoring_unit()
    print()
    await test_parse_degrade_unit()
    print()
    await test_pushback_findings_without_reads()
    print()
    await test_pushback_findings_with_reads()
    print()
    await test_pushback_zero_findings()
    print()
    test_embedded_extraction_shape_disambiguation()
    print()
    test_parse_hunt_answer_embedded_extraction()
    print()
    await test_http()

    print(f"\n{_passed} checks PASSED")


if __name__ == "__main__":
    asyncio.run(main())
