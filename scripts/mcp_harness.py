#!/usr/bin/env python
"""
Small-model MCP harness — proves a SMALL LOCAL MODEL can answer a real
codebase question using ONLY the dokkai MCP tools, within a documented token
budget (roadmap feature 03, acceptance criterion 2 / decision N-07).

It spawns the dokkai MCP server as a stdio subprocess (mcp SDK client), mirrors
its live tool schemas into Ollama's `/api/chat` tool-calling format, and runs an
agent loop: the model asks a question, calls tools, reads the results, and
answers. Every tool call is logged with its response size; the report shows a
per-call table, the total tool-response payload (chars/4 token approximation),
and PASS/FAIL against `--budget`.

Usage
-----
    uv run python scripts/mcp_harness.py \
        [--question "how does the alarm flow work?"] \
        [--project saffira_back-end] \
        [--model qwen2.5-coder:3b] \
        [--base-url http://localhost:11434] \
        [--budget 5000] \
        [--max-rounds 8]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Tool

REPO_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_QUESTION = "how does the alarm flow work?"

SYSTEM_PROMPT = (
    "You are answering questions about a codebase. You have no knowledge of "
    "this code beyond what the tools return — use them. Call context(query) or "
    "search(query) ONCE to find relevant entities; if you need more detail on a "
    "specific entity, call get_entity() or neighbors() on it. Never call the "
    "same tool with the same arguments twice — you already have that result. "
    "As soon as the tool results answer the question, STOP calling tools and "
    "write your final answer as plain English text (not JSON), citing the file "
    "paths you used. Do not answer without calling at least one tool first. "
    "When you call a tool, output ONLY the JSON for that one call and nothing "
    "else — no explanation, no example result. You do not have real results "
    "until the system gives them back to you; never invent or guess a tool's "
    "output yourself."
)

_TOOL_RESULT_REMINDER = (
    "\n\n[If the information above answers the question, reply now with your "
    "final answer as plain English text — do not call any more tools.]"
)


def _tokens(text: str) -> int:
    """Cheap chars/4 token approximation (same convention as decision N-07)."""
    return len(text) // 4


def _extract_fallback_tool_call(content: str) -> dict | None:
    """
    Best-effort fallback for models that emit a well-formed tool-call JSON as
    plain ``content`` instead of Ollama's structured ``tool_calls`` field
    (observed with qwen2.5-coder:3b, which never wraps its output in the
    ``<tool_call>`` tags Ollama's parser expects — a model quirk, not a schema
    issue: reproduced even with a trivial single-tool example).

    Strips markdown code fences / ``<tool_call>`` tags if present, then parses
    ``{"name": ..., "arguments": {...}}``.
    """
    text = content.strip()
    if "<tool_call>" in text:
        start = text.find("<tool_call>") + len("<tool_call>")
        end = text.find("</tool_call>")
        text = text[start : end if end != -1 else None].strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(obj, dict) and isinstance(obj.get("name"), str) and isinstance(obj.get("arguments"), dict):
        return obj
    return None


def _mcp_tool_to_ollama(tool: Tool) -> dict:
    """Mirror a live MCP tool schema into Ollama's chat tool-calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


async def _chat(
    client: httpx.AsyncClient, base_url: str, model: str, messages: list[dict], tools: list[dict]
) -> dict:
    try:
        resp = await client.post(
            f"{base_url}/api/chat",
            json={"model": model, "messages": messages, "tools": tools, "stream": False},
        )
        resp.raise_for_status()
    except httpx.ConnectError:
        raise SystemExit(f"✗ Cannot reach Ollama at {base_url} — is it running?")
    except httpx.HTTPStatusError as e:
        raise SystemExit(
            f"✗ Ollama rejected the request ({e.response.status_code}) — "
            f"is model '{model}' pulled? Try: ollama pull {model}"
        )
    return resp.json()


async def run(args: argparse.Namespace) -> bool:
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "--directory", str(REPO_ROOT), "python", "src/mcp_server.py"],
    )

    calls: list[dict] = []
    final_answer: str | None = None
    rounds_used = 0

    timeout = httpx.Timeout(120.0, connect=10.0)
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            ollama_tools = [_mcp_tool_to_ollama(t) for t in tools_result.tools]
            print(f"MCP server ready — {len(ollama_tools)} tools: "
                  f"{', '.join(t['function']['name'] for t in ollama_tools)}", file=sys.stderr)

            messages: list[dict] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Project: {args.project}\n\nQuestion: {args.question}"},
            ]

            async with httpx.AsyncClient(timeout=timeout) as http_client:
                for round_no in range(1, args.max_rounds + 1):
                    rounds_used = round_no
                    data = await _chat(http_client, args.base_url, args.model, messages, ollama_tools)
                    message = data["message"]
                    messages.append(message)

                    tool_calls = message.get("tool_calls") or []
                    used_fallback = False
                    if not tool_calls:
                        fallback = _extract_fallback_tool_call(message.get("content") or "")
                        if fallback:
                            tool_calls = [{"function": fallback}]
                            used_fallback = True

                    if not tool_calls:
                        final_answer = message.get("content", "")
                        break

                    for tc in tool_calls:
                        fn = tc["function"]
                        name = fn["name"]
                        raw_args = fn.get("arguments") or {}
                        if isinstance(raw_args, str):
                            raw_args = json.loads(raw_args)
                        note = " (parsed from raw content, no native tool_calls)" if used_fallback else ""
                        print(f"[round {round_no}] tool call: {name}({raw_args}){note}", file=sys.stderr)

                        try:
                            result = await session.call_tool(name, raw_args)
                            text = "\n".join(
                                c.text for c in result.content if hasattr(c, "text")
                            )
                            if result.isError:
                                text = f"ERROR: {text}"
                        except Exception as exc:  # noqa: BLE001 — report any client-side failure as a tool error
                            text = f"ERROR: {exc}"

                        chars = len(text)
                        toks = _tokens(text)
                        calls.append({"name": name, "args": raw_args, "chars": chars, "tokens": toks})
                        print(f"  -> {chars} chars (~{toks} tokens)", file=sys.stderr)

                        messages.append({"role": "tool", "content": text + _TOOL_RESULT_REMINDER})

    # ---- report ----
    total_chars = sum(c["chars"] for c in calls)
    total_tokens = sum(c["tokens"] for c in calls)

    print("\n" + "=" * 72)
    print("REPORT")
    print("=" * 72)
    print(f"Question : {args.question}")
    print(f"Project  : {args.project}")
    print(f"Model    : {args.model}")
    print(f"Rounds   : {rounds_used} (max {args.max_rounds})\n")

    if calls:
        print(f"{'#':<3} {'tool':<14} {'args':<40} {'chars':>8} {'~tokens':>8}")
        print("-" * 76)
        for i, c in enumerate(calls, 1):
            args_str = json.dumps(c["args"])
            if len(args_str) > 40:
                args_str = args_str[:37] + "..."
            print(f"{i:<3} {c['name']:<14} {args_str:<40} {c['chars']:>8} {c['tokens']:>8}")
        print("-" * 76)
    else:
        print("(no tool calls made)")

    print(f"\nTotal tool payload : {total_chars} chars (~{total_tokens} tokens)")
    print(f"Budget              : {args.budget} tokens")

    called_any_tool = len(calls) > 0
    within_budget = total_tokens <= args.budget
    # The acceptance criterion is answering under budget — a run that loops
    # tools without ever producing a final answer must not report PASS.
    passed = called_any_tool and within_budget and final_answer is not None

    print(f"Result              : {'PASS' if passed else 'FAIL'}")
    if not called_any_tool:
        print("  reason: model never called a tool")
    elif not within_budget:
        print("  reason: total tool payload exceeds budget")

    print(f"\nFinal answer:\n{final_answer or '(none — max rounds reached without an answer)'}")

    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=CANONICAL_QUESTION, help="Question to ask the model")
    parser.add_argument("--project", default="saffira_back-end", help="Project name to pass to the tools")
    parser.add_argument("--model", default="qwen2.5-coder:3b", help="Ollama model to use")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--budget", type=int, default=5000, help="Total tool-payload token budget")
    parser.add_argument("--max-rounds", type=int, default=8, help="Max agent loop rounds")
    args = parser.parse_args()

    passed = asyncio.run(run(args))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
