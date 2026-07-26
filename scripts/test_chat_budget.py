#!/usr/bin/env python
"""
Smoke test for the C4 prompt-budget fix (``services.chat``) — this repo has
no test framework yet, so this is a plain assert-and-print script, mirroring
``scripts/test_retriever_seeding.py``. PURE Python: no server, no Weaviate,
no live LLM call — exercises the pure helpers directly (one check spawns a
subprocess to construct a real ``OllamaProvider`` against an env var, same
pattern ``test_retriever_seeding.py`` uses for env-sensitive module state).

Covers:
  1. ``_fit_messages``: everything fits -> no drop, chronological order kept.
  2. ``_fit_messages``: overflow -> oldest history dropped first, order of
     the KEPT messages preserved, dropped count correct.
  3. ``_fit_messages``: degenerate budget (system+current alone exceed it)
     -> both kept anyway (fail-open), all history dropped, no crash.
  4. ``_fit_messages``: a window that would start with an assistant turn has
     that leading assistant message dropped too (post-review fix — providers
     like Anthropic require the first non-system message to be from the
     user).
  5. ``_fit_messages`` / ``chat_with_codebase`` sizing formula: reproduces
     the reviewer's OLLAMA_NUM_CTX=8192 scenario end to end (realistic
     4-message history incl. two ~1800-char assistant answers, context sized
     via the real ``history_reserve_tokens`` formula) -> the history window
     is NOT dead code (post-review fix; previously context consumed 100% of
     the budget and 0 of 4 turns survived), and no degenerate-budget WARNING
     fires on this ordinary-sized request.
  6. ``_retrieval_query``: an identifier-shaped current message is returned
     unchanged (preserves the C2 fast path; history is never even looked at).
  7. ``_retrieval_query``: a natural-language follow-up gets the last 2
     previous USER turns prepended (oldest first); assistant turns are
     excluded.
  8. ``_retrieval_query``: no history -> current message returned unchanged.
  9. ``_retrieval_query``: over the char cap -> front-truncated, so the tail
     (the current message's own ending) survives.
  10. ``_retrieval_query``: whitespace-only history entries are filtered out
      before taking the last 2 user turns (post-review fix — a blank turn
      no longer evicts a real one from the window or leaves a blank line).
  11. ``OllamaProvider.context_window``: OLLAMA_NUM_CTX=0 ("let Ollama pick")
      reports a conservative 4096 — NOT None (None would route to the large
      remote-provider constant, re-opening the overflow this step closes)
      and NOT a phantom 0-token window.
  12. ``_fitted_context_chars``: first turn (no history) skips the history
      reserve — measurably more context than the has-history case at the
      same budget.

Usage
-----
    uv run python scripts/test_chat_budget.py
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from services.chat import (  # noqa: E402
    _CHARS_PER_TOKEN,
    _DEFAULT_CONTEXT_MAX_CHARS,
    _HISTORY_RESERVE_TOKENS_CAP,
    _MIN_CONTEXT_CHARS,
    _RESERVED_OUTPUT_TOKENS,
    _estimate_tokens,
    _fit_messages,
    _fitted_context_chars,
    _get_system_prompt,
    _retrieval_query,
    _MAX_RETRIEVAL_QUERY_CHARS,
)
from services.chat_store import ChatMessage  # noqa: E402
from services.llm_provider import LLMMessage  # noqa: E402
from services.retriever import _looks_like_identifier  # noqa: E402

_passed = 0


def check(label: str, condition: bool) -> None:
    global _passed
    if not condition:
        raise AssertionError(f"FAIL: {label}")
    _passed += 1
    print(f"PASS: {label}")


class _CaptureHandler(logging.Handler):
    """Collects log records without needing a configured root handler."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_fit_messages_all_fit() -> None:
    print("\n_fit_messages: everything fits")
    system_text = "S" * 40  # 10 tokens
    history = [LLMMessage(role="user" if i % 2 == 0 else "assistant", content="H" * 40)
               for i in range(3)]  # 10 tokens each
    current_text = "C" * 40  # 10 tokens
    messages, dropped = _fit_messages(system_text, history, current_text, budget_tokens=1000)

    check("no messages dropped", dropped == 0)
    check("all 3 history messages kept + system + current", len(messages) == 5)
    check("system message first", messages[0].role == "system" and messages[0].content == system_text)
    check("history order preserved", [m.content for m in messages[1:4]] == [h.content for h in history])
    check("current message last", messages[-1].content == current_text)


def test_fit_messages_overflow_drops_oldest() -> None:
    print("\n_fit_messages: overflow drops oldest history first")
    system_text = "S" * 8  # 2 tokens
    current_text = "C" * 8  # 2 tokens
    history = [LLMMessage(role="user", content=f"turn{i}" + "H" * 96) for i in range(5)]
    # Each history message: 100 chars -> 25 tokens. protected = 4 tokens.
    # budget=64 -> remaining=60 -> newest 2 fit (50), 3rd doesn't (75 > 60).
    messages, dropped = _fit_messages(system_text, history, current_text, budget_tokens=64)

    check("3 oldest history messages dropped", dropped == 3)
    check("messages = system + 2 newest history + current", len(messages) == 4)
    kept_contents = [m.content for m in messages[1:-1]]
    check(
        "kept history is the 2 NEWEST turns, in chronological order",
        kept_contents == [history[3].content, history[4].content],
    )
    check("system message protected", messages[0].content == system_text)
    check("current message protected", messages[-1].content == current_text)


def test_fit_messages_degenerate_budget() -> None:
    print("\n_fit_messages: degenerate budget (system+current alone overflow)")
    system_text = "S" * 10_000  # 2500 tokens
    current_text = "C" * 10_000  # 2500 tokens
    history = [LLMMessage(role="user", content="H" * 40)]
    messages, dropped = _fit_messages(system_text, history, current_text, budget_tokens=100)

    check("system+current kept despite exceeding budget (fail-open)", len(messages) == 2)
    check("system message present", messages[0].role == "system" and messages[0].content == system_text)
    check("current message present", messages[1].role == "user" and messages[1].content == current_text)
    check("all history dropped", dropped == len(history))


def test_fit_messages_drops_leading_assistant() -> None:
    print("\n_fit_messages: a window starting with an assistant turn drops it")
    system_text = "S" * 8   # 2 tokens
    current_text = "C" * 8  # 2 tokens
    history = [
        LLMMessage(role="user", content="U" * 40),        # 10 tokens
        LLMMessage(role="assistant", content="A" * 40),    # 10 tokens
        LLMMessage(role="user", content="u" * 40),         # 10 tokens
        LLMMessage(role="assistant", content="a" * 40),    # 10 tokens
    ]
    # protected=4, remaining=35 -> raw newest-first walk keeps 3 (30 <= 35,
    # the 4th would make 40 > 35): [assistant(a), user(u), assistant(A)],
    # i.e. an ODD-count window starting with an assistant turn (A).
    messages, dropped = _fit_messages(system_text, history, current_text, budget_tokens=39)

    kept = messages[1:-1]
    check("the leading assistant turn was dropped, not just the budget cut", len(kept) == 2)
    check("kept window starts with a USER turn", kept[0].role == "user")
    check("dropped count includes the extra leading-assistant drop", dropped == 2)
    check("kept window is the 2 newest turns", [m.content for m in kept] == ["u" * 40, "a" * 40])


def test_fit_messages_8192_scenario_history_not_dead() -> None:
    print("\n_fit_messages: reproduces reviewer's OLLAMA_NUM_CTX=8192 scenario")
    handler = _CaptureHandler()
    chat_logger = logging.getLogger("services.chat")
    chat_logger.addHandler(handler)
    chat_logger.setLevel(logging.DEBUG)
    try:
        # Exercises the REAL sizing helper (load-bearing: a formula change
        # in chat.py fails here, instead of a replicated copy passing).
        budget_tokens = 8192 - _RESERVED_OUTPUT_TOKENS
        system_prompt = _get_system_prompt("developer")
        message = "and where is it validated?"
        fitted_max_chars = _fitted_context_chars(
            budget_tokens, system_prompt, message, has_history=True
        )
        print(f"  budget_tokens={budget_tokens} fitted_max_chars={fitted_max_chars}")
        check("context budget NOT floored at ordinary num_ctx=8192", fitted_max_chars > _MIN_CONTEXT_CHARS)
        check("fitted context is in the reviewer's ~10-11k ballpark", 9_000 <= fitted_max_chars <= 12_000)

        context = "X" * fitted_max_chars  # simulates build_graph_context's output at that budget
        system_text = f"{system_prompt}\n\n=== CODEBASE CONTEXT ===\n{context}\n=== END CONTEXT ==="

        history = [
            ChatMessage(role="user", content="how does auth work?"),
            ChatMessage(role="assistant", content="A" * 1800),
            ChatMessage(role="user", content="and how are tokens issued?"),
            ChatMessage(role="assistant", content="B" * 1800),
        ]
        history_messages = [LLMMessage(role=m.role, content=m.content) for m in history]

        messages, dropped = _fit_messages(system_text, history_messages, message, budget_tokens)
        kept = messages[1:-1]
        print(f"  kept {len(kept)}/{len(history)} history messages, dropped={dropped}")

        check(
            "history window is NOT dead code — at least the 2 most recent turns kept",
            len(kept) >= 2,
        )
        check(
            "the most recent turn made it into the window",
            kept[-1].content == history[-1].content,
        )
        check(
            "second-most-recent turn made it into the window too",
            kept[-2].content == history[-2].content,
        )
        check(
            "no degenerate-budget WARNING fired on this ordinary-sized request",
            not any(r.levelno >= logging.WARNING for r in handler.records),
        )
    finally:
        chat_logger.removeHandler(handler)


def test_retrieval_query_identifier_unchanged() -> None:
    print("\n_retrieval_query: identifier-shaped current is unchanged")
    history = [ChatMessage(role="user", content="how does auth work?")]
    current = "validate_token"
    got = _retrieval_query(history, current)
    check("identifier query returned unchanged", got == current)


def test_retrieval_query_followup_includes_user_turns() -> None:
    print("\n_retrieval_query: follow-up prepends previous USER turns, excludes assistant")
    history = [
        ChatMessage(role="user", content="Where is auth handled?"),
        ChatMessage(role="assistant", content="It's in auth.py, see authenticate()."),
        ChatMessage(role="user", content="how is the token issued?"),
    ]
    current = "and where is it validated?"
    got = _retrieval_query(history, current)

    check("current query is NOT identifier-shaped (takes the history-aware path)", not _looks_like_identifier(current))
    check("both previous USER turns present", "Where is auth handled?" in got and "how is the token issued?" in got)
    check("assistant turn excluded", "authenticate()" not in got)
    check("oldest-first ordering before the current message", got.index("Where is auth handled?") < got.index("how is the token issued?") < got.index(current))
    check("current message present verbatim at the end", got.endswith(current))


def test_retrieval_query_no_history() -> None:
    print("\n_retrieval_query: no history -> current unchanged")
    current = "how does the payment flow work?"
    got = _retrieval_query([], current)
    check("no-history query returned unchanged", got == current)


def test_retrieval_query_cap_front_truncates() -> None:
    print("\n_retrieval_query: over the cap -> front-truncated, tail intact")
    # 750 chars total, distinguishable head/tail, no history (isolates the
    # bare-current truncation path): current alone already exceeds the cap.
    current = "A" * 100 + "B" * 650
    got = _retrieval_query([], current)

    check("result respects the char cap", len(got) == _MAX_RETRIEVAL_QUERY_CHARS)
    check("front of the concatenation was dropped ('A's gone)", "A" not in got)
    check("tail (end of current message) survives intact", got == current[-_MAX_RETRIEVAL_QUERY_CHARS:])
    check("result is a suffix of current", current.endswith(got))

    print("\n_retrieval_query: cap forces history to be dropped entirely first")
    history = [
        ChatMessage(role="user", content="X" * 400),
        ChatMessage(role="user", content="Y" * 400),
    ]
    current2 = "Z" * 590  # fits alone under the cap, but not with any history turn
    got2 = _retrieval_query(history, current2)
    check("all history dropped once even 1 turn + current can't fit", got2 == current2)


def test_retrieval_query_filters_blank_history() -> None:
    print("\n_retrieval_query: whitespace-only history entries are filtered out")
    history = [
        ChatMessage(role="user", content="first real question"),
        ChatMessage(role="assistant", content="a1"),
        ChatMessage(role="user", content="second real question"),
        ChatMessage(role="assistant", content="a2"),
        ChatMessage(role="user", content="   "),  # blank -- the MOST RECENT user turn
    ]
    current = "anaphoric follow-up?"
    got = _retrieval_query(history, current)

    check(
        "the blank turn didn't evict a real turn from the 2-turn window",
        "first real question" in got and "second real question" in got,
    )
    check("no blank line left behind by the filtered-out turn", not any(line.strip() == "" for line in got.split("\n")))
    check("current message still present verbatim at the end", got.endswith(current))


def test_first_turn_skips_history_reserve() -> None:
    print("\n_fitted_context_chars: first turn (no history) skips the reserve")
    budget_tokens = 8192 - _RESERVED_OUTPUT_TOKENS
    system_prompt = _get_system_prompt("developer")
    message = "how does authentication work?"
    with_history = _fitted_context_chars(
        budget_tokens, system_prompt, message, has_history=True
    )
    without_history = _fitted_context_chars(
        budget_tokens, system_prompt, message, has_history=False
    )
    print(f"  with_history={with_history} without_history={without_history}")
    check(
        "no-history context is exactly the reserve bigger",
        without_history - with_history
        == min(budget_tokens // 4, _HISTORY_RESERVE_TOKENS_CAP) * _CHARS_PER_TOKEN,
    )
    check("no-history context still capped at the default", without_history <= _DEFAULT_CONTEXT_MAX_CHARS)


def test_ollama_context_window_zero_is_unknown() -> None:
    print("\nOllamaProvider.context_window: OLLAMA_NUM_CTX=0 reports conservative 4096")
    script = f"""
import sys, json
sys.path.insert(0, {str(SRC)!r})
from services.llm_provider import OllamaProvider
print(json.dumps({{"context_window": OllamaProvider().context_window}}))
"""
    import os
    env = os.environ.copy()
    env["OLLAMA_NUM_CTX"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"subprocess failed: {result.stderr}")
    data = json.loads(result.stdout.strip().splitlines()[-1])
    check("OLLAMA_NUM_CTX=0 -> conservative 4096, not None and not 0", data["context_window"] == 4096)

    env["OLLAMA_NUM_CTX"] = "8192"
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"subprocess failed: {result.stderr}")
    data = json.loads(result.stdout.strip().splitlines()[-1])
    check("a real OLLAMA_NUM_CTX still reports its value", data["context_window"] == 8192)


def main() -> None:
    test_fit_messages_all_fit()
    test_fit_messages_overflow_drops_oldest()
    test_fit_messages_degenerate_budget()
    test_fit_messages_drops_leading_assistant()
    test_fit_messages_8192_scenario_history_not_dead()
    test_retrieval_query_identifier_unchanged()
    test_retrieval_query_followup_includes_user_turns()
    test_retrieval_query_no_history()
    test_retrieval_query_cap_front_truncates()
    test_retrieval_query_filters_blank_history()
    test_first_turn_skips_history_reserve()
    test_ollama_context_window_zero_is_unknown()
    print(f"\n{_passed} checks PASSED")


if __name__ == "__main__":
    main()
