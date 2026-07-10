#!/usr/bin/env python
"""
Smoke test for the in-process agentic tool loop (B4) —
``services.routines.agent_loop``. This repo has no test framework yet, so
this is a plain assert-and-print script, mirroring
``scripts/test_review_analyze.py``/``scripts/test_routine_documents.py``.

Deterministic guards (max-rounds forcing, dedupe, load_skill cap-3, fallback
parser, non-ollama rejection) are exercised against a FAKE provider — no
model quirks in the loop's own control flow. Exactly ONE live smoke talks to
the REAL stack: Weaviate (project ``saffira_back-end``, already ingested)
and Ollama running ``qwen2.5-coder:3b`` (already pulled) — fine for
MECHANICS (does it call tools, emit events, produce a non-empty answer),
not judged for correctness of the answer's content.

Talks to the REAL local Postgres too (``docker compose up -d``) — the
load_skill/catalog test creates and tears down its own scratch ``skills``
rows.

Usage
-----
    docker compose up -d   # Postgres + Weaviate
    ollama serve            # (if not already running)
    uv run python scripts/test_agent_loop.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make the `services` package importable (the app roots absolute imports at src/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from services.db import get_pool, init_db  # noqa: E402
from services.llm_provider import OllamaProvider  # noqa: E402
from services.retriever import Retriever  # noqa: E402
from services.routines import documents  # noqa: E402
from services.routines.agent_loop import (  # noqa: E402
    MAX_LOADED_SKILLS,
    SYSTEM_PROMPT,
    ToolSpec,
    build_tools,
    extract_fallback_tool_call,
    find_json_object_candidates,
    run_agent_loop,
)
from services.weaviate_client import get_client  # noqa: E402

PROJECT = "saffira_back-end"
MODEL = "qwen2.5-coder:3b"

SKILL_1 = "test-agent-loop-scratch-skill-1"
SKILL_2 = "test-agent-loop-scratch-skill-2"
SKILL_3 = "test-agent-loop-scratch-skill-3"
SKILL_4_NEW = "test-agent-loop-scratch-skill-4-not-created"
ALL_SKILL_NAMES = [SKILL_1, SKILL_2, SKILL_3, SKILL_4_NEW]

_passed = 0


def check(label: str, condition: bool) -> None:
    global _passed
    if not condition:
        raise AssertionError(f"FAIL: {label}")
    _passed += 1
    print(f"PASS: {label}")


async def cleanup() -> None:
    try:
        pool = await get_pool()
        await pool.execute("DELETE FROM skills WHERE name = ANY($1::text[])", ALL_SKILL_NAMES)
        print("cleanup: removed scratch skills rows")
    except Exception as e:
        print(f"cleanup: best-effort teardown failed (non-fatal): {e}")


def make_event_collector() -> tuple[list[tuple[str, str]], callable]:
    events: list[tuple[str, str]] = []

    def emit(stage: str, message: str) -> None:
        events.append((stage, message))

    return events, emit


def _fake_tools() -> list[ToolSpec]:
    """Minimal stateless tools for the deterministic (no-infra) guard tests."""

    async def search_handler(args: dict) -> str:
        return f"1. [Function] fake_entity — fake/path.py:1-2 (score 0.90) — query={args.get('query')}"

    async def get_entity_handler(args: dict) -> str:
        return f"[Function] {args.get('qualified_name')}\nFile: fake/path.py"

    async def unused_load_skill_handler(args: dict) -> str:
        # run_agent_loop special-cases "load_skill" (cap tracking, per
        # _dispatch_load_skill) and must never fall through to a tool's own
        # registered handler for it — this ToolSpec exists only so
        # "load_skill" is a recognized tool name for the fake registry.
        raise AssertionError("load_skill's registered ToolSpec.handler must not be invoked directly")

    return [
        ToolSpec(
            name="search",
            description="fake search",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=search_handler,
        ),
        ToolSpec(
            name="get_entity",
            description="fake get_entity",
            parameters={
                "type": "object",
                "properties": {"qualified_name": {"type": "string"}},
                "required": ["qualified_name"],
            },
            handler=get_entity_handler,
        ),
        ToolSpec(
            name="load_skill",
            description="fake load_skill",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=unused_load_skill_handler,
        ),
    ]


# ---------------------------------------------------------------------------
# Fake providers
# ---------------------------------------------------------------------------


class NonOllamaProvider:
    provider_name = "openai"

    async def chat_with_tools(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("chat_with_tools must not be called — rejected before any chat")


class ScriptedOllamaProvider:
    """Returns each entry of *script* in order, one per ``chat_with_tools``
    call; records every call's (messages, tools) for later inspection."""

    provider_name = "ollama"

    def __init__(self, script: list[dict]) -> None:
        self._script = list(script)
        self.calls: list[dict] = []

    async def chat_with_tools(self, messages, *, model, tools=None, temperature=0.3, max_tokens=4096):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        if self._script:
            return self._script.pop(0)
        return {"role": "assistant", "content": "(script exhausted)"}


class AlwaysToolCallProvider:
    """Never stops calling tools on its own — used to prove the max-rounds
    forcing guard actually terminates the loop."""

    provider_name = "ollama"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat_with_tools(self, messages, *, model, tools=None, temperature=0.3, max_tokens=4096):
        self.calls.append({"tools": tools})
        return {
            "role": "assistant",
            "content": "I want to keep searching",
            "tool_calls": [{"function": {"name": "search", "arguments": {"query": "loop forever"}}}],
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_non_ollama_rejected() -> None:
    events, emit = make_event_collector()
    try:
        await run_agent_loop(NonOllamaProvider(), "gpt-4o", "sys", "hi", _fake_tools(), emit)
        check("non-ollama provider raises NotImplementedError", False)
    except NotImplementedError as e:
        check("non-ollama provider raises NotImplementedError", True)
        check("non-ollama error names the provider", "'openai'" in str(e))
        check("non-ollama error mentions Ollama-native", "Ollama-native tool calling" in str(e))


def test_fallback_parser() -> None:
    plain = extract_fallback_tool_call('{"name": "search", "arguments": {"query": "x"}}')
    check("fallback parser: plain JSON", plain == {"name": "search", "arguments": {"query": "x"}})

    tagged = extract_fallback_tool_call(
        '<tool_call>{"name": "get_entity", "arguments": {"qualified_name": "a.b"}}</tool_call>'
    )
    check(
        "fallback parser: <tool_call> tags",
        tagged == {"name": "get_entity", "arguments": {"qualified_name": "a.b"}},
    )

    fenced = extract_fallback_tool_call(
        '```json\n{"name": "search", "arguments": {"query": "y"}}\n```'
    )
    check("fallback parser: markdown fence", fenced == {"name": "search", "arguments": {"query": "y"}})

    check("fallback parser: invalid JSON -> None", extract_fallback_tool_call("not json at all") is None)
    check(
        "fallback parser: valid JSON, wrong shape -> None",
        extract_fallback_tool_call('{"foo": "bar"}') is None,
    )

    # --- embedded-extraction (approved follow-up): content richer than one
    # bare object — a fenced block or a bare object surrounded by prose ---
    fenced_in_prose = extract_fallback_tool_call(
        'Let me look that up.\n```json\n{"name": "search", "arguments": {"query": "z"}}\n```\nDone.'
    )
    check(
        "fallback parser: fenced block in prose",
        fenced_in_prose == {"name": "search", "arguments": {"query": "z"}},
    )

    bare_in_prose = extract_fallback_tool_call(
        'I will call:\n{"name": "get_file", "arguments": {"path": "a.ts"}}\nThat should work.'
    )
    check(
        "fallback parser: bare object mid-prose",
        bare_in_prose == {"name": "get_file", "arguments": {"path": "a.ts"}},
    )

    multiple = extract_fallback_tool_call(
        'First:\n{"name": "search", "arguments": {"query": "first"}}\n'
        'Then:\n{"name": "search", "arguments": {"query": "second"}}\n'
    )
    check(
        "fallback parser: multiple embedded blocks — first wins",
        multiple == {"name": "search", "arguments": {"query": "first"}},
    )

    no_args = extract_fallback_tool_call('{"name": "search"}')
    check(
        "fallback parser: missing 'arguments' defaults to {}",
        no_args == {"name": "search", "arguments": {}},
    )

    check(
        "fallback parser: no valid candidate amid unrelated JSON -> None",
        extract_fallback_tool_call('Some notes: {"foo": "bar"} and {"baz": 1}.') is None,
    )

    # --- pathological input: must not crash or hang, bounded scan ---
    unbalanced = extract_fallback_tool_call("{" * 5000 + "no closing brace at all")
    check("fallback parser: unbalanced braces -> None, no crash", unbalanced is None)

    many_small = "".join(f'noise {{"n": {i}}} ' for i in range(500))
    check(
        "fallback parser: many small non-matching objects -> None, bounded (doesn't scan past cap)",
        extract_fallback_tool_call(many_small) is None,
    )
    check(
        "find_json_object_candidates: bounded at _MAX_JSON_CANDIDATES even with 500 balanced objects",
        len(find_json_object_candidates(many_small)) <= 20,
    )


async def test_dedupe_guard() -> None:
    events, emit = make_event_collector()
    provider = ScriptedOllamaProvider(
        [
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search", "arguments": {"query": "foo"}}}]},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search", "arguments": {"query": "foo"}}}]},
            {"role": "assistant", "content": "Final answer using the cached result."},
        ]
    )
    result = await run_agent_loop(provider, MODEL, "sys", "hi", _fake_tools(), emit)

    check("dedupe: rounds == 3", result["rounds"] == 3)
    check("dedupe: only the first call actually executed", result["tool_calls_made"] == 1)
    check("dedupe: final answer non-empty", bool(result["answer"]))

    round3_incoming = provider.calls[2]["messages"][-1]["content"]
    check("dedupe: round 2's duplicate call got an 'already called' tool error", "already called" in round3_incoming)
    check(
        "dedupe: an 'already called' event was NOT double-executed (no 2nd search event content)",
        sum(1 for _, msg in events if msg.startswith("→ search")) == 2,
    )


async def test_max_rounds_forcing() -> None:
    events, emit = make_event_collector()
    provider = AlwaysToolCallProvider()
    result = await run_agent_loop(provider, MODEL, "sys", "hi", _fake_tools(), emit)

    check("max-rounds: forced at round 12", result["rounds"] == 12)
    check("max-rounds: forced answer is non-empty (last-round content honored)", result["answer"] == "I want to keep searching")
    check("max-rounds: last round offered no tools (forced, tool-less)", provider.calls[-1]["tools"] is None)
    check("max-rounds: exactly 12 provider calls made", len(provider.calls) == 12)


async def test_load_skill_catalog_and_cap() -> None:
    await documents.create_skill(SKILL_1, "First test skill", "---\ntitle: one\n---\nSkill one body.")
    await documents.create_skill(SKILL_2, "Second test skill", "Skill two body (no frontmatter).")
    await documents.create_skill(SKILL_3, "Third test skill", "Skill three body.")

    events, emit = make_event_collector()
    provider = ScriptedOllamaProvider(
        [
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "load_skill", "arguments": {"name": SKILL_1}}}]},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "load_skill", "arguments": {"name": SKILL_2}}}]},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "load_skill", "arguments": {"name": SKILL_3}}}]},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "load_skill", "arguments": {"name": SKILL_4_NEW}}}]},
            {"role": "assistant", "content": "Loaded 3 skills, that's enough."},
        ]
    )
    result = await run_agent_loop(provider, MODEL, "sys", "hi", _fake_tools(), emit)

    check("load_skill: loaded_skills == [1, 2, 3]", result["loaded_skills"] == [SKILL_1, SKILL_2, SKILL_3])
    check(
        "load_skill: 'loaded skill' events fired for each",
        [msg for _, msg in events if msg.startswith("loaded skill")]
        == [f"loaded skill {SKILL_1}", f"loaded skill {SKILL_2}", f"loaded skill {SKILL_3}"],
    )

    round5_incoming = provider.calls[4]["messages"][-1]["content"]
    check(
        f"load_skill: 4th (new) skill refused — cap {MAX_LOADED_SKILLS}",
        f"maximum of {MAX_LOADED_SKILLS}" in round5_incoming,
    )

    system_prompt = provider.calls[0]["messages"][0]["content"]
    check("load_skill: system prompt carries the skills catalog", SKILL_1 in system_prompt and "First test skill" in system_prompt)
    check("load_skill: system prompt instructs a 3-skill cap", f"at most {MAX_LOADED_SKILLS} skills" in system_prompt)

    # frontmatter-stripped body actually reached the model as a tool result
    skill1_result = provider.calls[1]["messages"][-1]["content"]
    check("load_skill: frontmatter stripped from skill body", "Skill one body." in skill1_result and "title: one" not in skill1_result)


_LIVE_SMOKE_QUESTION = "What does the authMiddleware do? Use search then get_entity then answer."


async def test_grep_project_mcp_wording() -> None:
    """
    Deterministic (no LLM involved) parity check: ``grep_project``'s
    no-matches wording must match ``src/mcp_server.py``'s ``grep_project``
    tool byte-for-byte, including the empty-pattern case (mcp treats a
    blank pattern as a no-matches result, not a validation error).
    """
    client = get_client()
    try:
        tools = {t.name: t for t in build_tools(PROJECT, client)}
        grep = tools["grep_project"]

        empty_text = await grep.handler({"pattern": ""})
        check(
            "grep_project: empty pattern returns mcp's no-matches wording (not an error)",
            empty_text == f"no matches for '' in project '{PROJECT}'",
        )

        missing_text = await grep.handler({})
        check(
            "grep_project: missing pattern arg behaves like an empty pattern",
            missing_text == f"no matches for '' in project '{PROJECT}'",
        )

        # A real corpus can't reliably be forced to yield zero BM25 hits (this
        # dataset's tokenizer partially matches even nonsense patterns), so
        # the "chunks came back empty" branch is exercised deterministically
        # by monkeypatching Retriever.search_bm25 rather than depending on
        # empirical corpus contents.
        original_search_bm25 = Retriever.search_bm25
        Retriever.search_bm25 = lambda self, *a, **kw: []
        try:
            pattern = "some-pattern"
            no_hit_text = await grep.handler({"pattern": pattern})
            check(
                "grep_project: no-hit pattern returns mcp's exact no-matches wording",
                no_hit_text == f"no matches for '{pattern}' in project '{PROJECT}'",
            )
        finally:
            Retriever.search_bm25 = original_search_bm25
    finally:
        client.close()


async def test_live_smoke() -> None:
    """
    MECHANICS-only live smoke against the real 3b model — NOT judged on
    whether the model chooses to call a tool. A small local model
    occasionally answers directly in round 1 despite the system prompt's
    "do not answer without calling a tool first" instruction (an observed
    qwen2.5-coder:3b quirk, same family as the content-vs-tool_calls quirk
    documented in ``scripts/mcp_harness.py``'s docstring) — the loop
    correctly returns whatever the model gives it either way, so a bare
    "no tool call" is not a loop bug. Retried once for stability; if still
    no tool call after the retry, that's reported as an advisory note
    rather than a failure. The only assertion that must always hold is
    that the loop terminates with a non-empty final answer.
    """
    client = get_client()
    try:
        tools = build_tools(PROJECT, client)
        provider = OllamaProvider()

        attempt = 1
        events, emit = make_event_collector()
        result = await run_agent_loop(
            provider, MODEL, SYSTEM_PROMPT, _LIVE_SMOKE_QUESTION, tools, emit
        )

        if result["tool_calls_made"] == 0:
            print(
                "\n(live smoke: attempt 1 used no tool — retrying once; small local models "
                "occasionally answer directly, see docstring)"
            )
            attempt = 2
            events, emit = make_event_collector()
            result = await run_agent_loop(
                provider, MODEL, SYSTEM_PROMPT, _LIVE_SMOKE_QUESTION, tools, emit
            )
    finally:
        client.close()

    check("live smoke: final answer non-empty", bool(result["answer"].strip()))

    if result["tool_calls_made"] >= 1:
        check("live smoke: events were emitted for the executed tool call(s)", len(events) >= 1)
    else:
        print(
            "\nADVISORY: live smoke made no tool call in either attempt — the loop still "
            "terminated correctly with a (possibly ungrounded) answer; this is a known small-"
            "model limitation, not a loop bug. Mechanics (tool dispatch/events/dedupe/rounds) "
            "are already pinned by the deterministic fake-provider tests above."
        )

    print(f"\n--- live smoke: attempt={attempt} rounds={result['rounds']} tool_calls_made={result['tool_calls_made']} "
          f"tool_payload_chars={result['tool_payload_chars']} loaded_skills={result['loaded_skills']} ---")
    print("--- tool-call event sequence ---")
    for stage, msg in events:
        print(f"  [{stage}] {msg}")
    print(f"\n--- final answer (first 500 chars) ---\n{result['answer'][:500]}")


async def main() -> None:
    await init_db()
    try:
        test_fallback_parser()
        await test_non_ollama_rejected()
        await test_dedupe_guard()
        await test_max_rounds_forcing()
        await test_load_skill_catalog_and_cap()
        await test_grep_project_mcp_wording()
        await test_live_smoke()
    finally:
        await cleanup()

    print(f"\n{_passed} checks PASSED")


if __name__ == "__main__":
    asyncio.run(main())
