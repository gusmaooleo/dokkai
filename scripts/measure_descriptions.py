#!/usr/bin/env python
"""
Measure the real cost of generating per-entity micro-descriptions (Tier 2) on
*this* machine, so the description strategy (model, selectivity, parallelism,
prompt shape) is decided with numbers instead of guesses.

It reads an ingested graph JSON, samples real entities, asks a local Ollama
model to describe each one, and reports tokens/sec + per-entity wall-clock,
then extrapolates to the whole repository. With ``--prompt both`` it A/Bs the
full vs. trimmed prompt (decision 1c) on the same sample of entities.

Usage
-----
    uv run python scripts/measure_descriptions.py \
        --model qwen2.5-coder:3b \
        --samples 15 \
        [--prompt full|trimmed|both] \
        [--json ingested/<file>.json] \
        [--source-root /path/to/repo] \
        [--concurrency 4] \
        [--base-url http://localhost:11434]

Ollama returns exact timing fields (prompt_eval_count / eval_count and their
nanosecond durations), so the measured tok/s reflects your actual hardware and
model — not an estimate. The prompt itself is built via
``services.describe.build_describe_prompt`` — the same function the describe
service uses — so this script can never drift from what's actually shipped.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx

# Make the `services` package importable (the app roots absolute imports at src/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from services.chunker import chunk_graph, CodeChunk  # noqa: E402
from services.describe import build_describe_prompt  # noqa: E402
from services.retriever import _is_test_path  # noqa: E402


SYSTEM_PROMPT = (
    "You summarize code for a search index. Reply with ONE concise sentence "
    "(max 25 words) describing what the code does. No preamble, no code, no "
    "bullet points — just the sentence."
)

# Options tuned for a short, deterministic description of a single entity.
_GEN_OPTIONS = {"temperature": 0.1, "num_predict": 64, "num_ctx": 4096}

# How many sample descriptions to keep for eyeballing (side-by-side in --prompt both).
_MAX_EXAMPLES = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_json() -> str | None:
    """Pick the most recent JSON under ingested/ if --json isn't given."""
    ingested = Path(__file__).resolve().parent.parent / "ingested"
    files = sorted(ingested.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0]) if files else None


def _describable(chunks: list[CodeChunk]) -> list[CodeChunk]:
    """Entities we'd actually describe: have source and aren't tests."""
    return [c for c in chunks if c.source_code and not _is_test_path(c.file_path)]


def _sample(items: list[CodeChunk], n: int) -> list[CodeChunk]:
    """Evenly spaced sample, so we cover small and large entities alike."""
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


async def _generate(
    client: httpx.AsyncClient, base_url: str, model: str, chunk: CodeChunk, mode: str
) -> dict:
    """One /api/generate call; returns Ollama's response incl. timing fields."""
    resp = await client.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "system": SYSTEM_PROMPT,
            "prompt": build_describe_prompt(chunk, mode),
            "stream": False,
            "keep_alive": "5m",
            "options": _GEN_OPTIONS,
        },
    )
    resp.raise_for_status()
    return resp.json()


def _ns_to_s(ns: int | None) -> float:
    return (ns or 0) / 1e9


async def _ensure_model(client: httpx.AsyncClient, base_url: str, model: str) -> None:
    resp = await client.get(f"{base_url}/api/tags")
    resp.raise_for_status()
    available = [m["name"] for m in resp.json().get("models", [])]
    # Ollama tags are like "qwen2.5:3b"; accept a bare name matching any tag.
    if model in available or any(a.split(":")[0] == model for a in available):
        return
    print(f"✗ Model '{model}' is not pulled in Ollama.\n")
    print("  Available models:")
    for a in available:
        print(f"    - {a}")
    print(f"\n  Pull it with:  ollama pull {model}")
    sys.exit(1)


def _fmt_dur(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


# ---------------------------------------------------------------------------
# Serial measurement (one prompt mode, over a fixed sample)
# ---------------------------------------------------------------------------

async def _measure_serial(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    model: str,
    sample: list[CodeChunk],
    mode: str,
    *,
    label_mode: bool = False,
) -> dict:
    """Serial pass measuring one prompt ``mode`` over ``sample``; returns stats."""
    prompt_tokens = gen_tokens = 0
    prompt_secs = gen_secs = 0.0
    prompt_chars = 0
    per_entity_wall: list[float] = []
    examples: list[tuple[str, str]] = []

    for i, chunk in enumerate(sample, 1):
        t0 = time.perf_counter()
        data = await _generate(client, args.base_url, model, chunk, mode)
        per_entity_wall.append(time.perf_counter() - t0)

        prompt_tokens += data.get("prompt_eval_count", 0)
        gen_tokens += data.get("eval_count", 0)
        prompt_secs += _ns_to_s(data.get("prompt_eval_duration"))
        gen_secs += _ns_to_s(data.get("eval_duration"))
        prompt_chars += len(build_describe_prompt(chunk, mode))
        if len(examples) < _MAX_EXAMPLES:
            examples.append((chunk.qualified_name, data.get("response", "").strip()))

        prefix = f"[{mode:>7}] " if label_mode else ""
        print(f"  {prefix}[{i}/{len(sample)}] {per_entity_wall[-1]:5.2f}s  {chunk.qualified_name[:70]}")

    n = len(sample)
    uses_tokens = prompt_tokens > 0
    return {
        "mode": mode,
        "avg_prompt_size": (prompt_tokens if uses_tokens else prompt_chars) / n,
        "uses_tokens": uses_tokens,
        "avg_out": gen_tokens / n,
        "prompt_tps": prompt_tokens / prompt_secs if prompt_secs else 0.0,
        "gen_tps": gen_tokens / gen_secs if gen_secs else 0.0,
        "avg_wall": sum(per_entity_wall) / n,
        "examples": examples,
    }


def _print_mode_report(stats: dict, n_desc: int) -> None:
    unit = "tokens" if stats["uses_tokens"] else "chars"
    print(f"\n  Mode: {stats['mode']}")
    print(f"    avg prompt size : {stats['avg_prompt_size']:6.0f} {unit}/entity")
    print(f"    avg output      : {stats['avg_out']:6.0f} tokens/entity")
    print(f"    prompt eval     : {stats['prompt_tps']:6.0f} tok/s")
    print(f"    generation      : {stats['gen_tps']:6.0f} tok/s")
    print(f"    per-entity wall : {stats['avg_wall']:6.2f} s (serial)")
    projected = stats["avg_wall"] * n_desc
    print(f"    projected total : {_fmt_dur(projected)}  (for {n_desc} describable entities)")


def _print_ab_examples(full_stats: dict, trimmed_stats: dict) -> None:
    print("\n" + "-" * 64)
    print("SAMPLE DESCRIPTIONS — full vs trimmed (eyeball the quality):")
    for (qn, full_desc), (_, trimmed_desc) in zip(full_stats["examples"], trimmed_stats["examples"]):
        print(f"\n  • {qn}")
        print(f"    full    → {full_desc}")
        print(f"    trimmed → {trimmed_desc}")


# ---------------------------------------------------------------------------
# Main measurement
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    json_path = args.json or _default_json()
    if not json_path:
        print("✗ No graph JSON found under ingested/. Pass --json <path>.")
        sys.exit(1)

    print(f"Graph      : {json_path}")
    chunks = chunk_graph(json_path, source_root=args.source_root)
    describable = _describable(chunks)
    total, n_desc = len(chunks), len(describable)
    print(f"Entities   : {total} total · {n_desc} describable (has source, not test)")

    if n_desc == 0:
        print("✗ Nothing to describe (no source found). Pass --source-root <repo>.")
        sys.exit(1)

    sample = _sample(describable, args.samples)
    print(f"Model      : {args.model}")
    print(f"Prompt     : {args.prompt}")
    print(f"Sample     : {len(sample)} entities\n")

    timeout = httpx.Timeout(120.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        await _ensure_model(client, args.base_url, args.model)

        # Warm up (first call cold-loads the model — exclude from stats).
        print("Warming up the model…")
        warmup_mode = "full" if args.prompt == "both" else args.prompt
        await _generate(client, args.base_url, args.model, sample[0], warmup_mode)

        if args.prompt == "both":
            print("\nMeasuring (serial) — mode=full…")
            full_stats = await _measure_serial(client, args, args.model, sample, "full", label_mode=True)
            print("\nMeasuring (serial) — mode=trimmed…")
            trimmed_stats = await _measure_serial(
                client, args, args.model, sample, "trimmed", label_mode=True
            )
            concurrent_wall = None
        else:
            print("Measuring (serial)…\n")
            stats = await _measure_serial(client, args, args.model, sample, args.prompt)

            # ---- optional concurrent pass: measured real speedup ----
            concurrent_wall = None
            if args.concurrency > 1:
                print(f"\nMeasuring (concurrency={args.concurrency})…")
                sem = asyncio.Semaphore(args.concurrency)

                async def _one(c: CodeChunk) -> None:
                    async with sem:
                        await _generate(client, args.base_url, args.model, c, args.prompt)

                t0 = time.perf_counter()
                await asyncio.gather(*(_one(c) for c in sample))
                concurrent_wall = time.perf_counter() - t0

    # ---- report ----
    print("\n" + "=" * 64)
    print("RESULTS")
    print("=" * 64)

    if args.prompt == "both":
        _print_mode_report(full_stats, n_desc)
        _print_mode_report(trimmed_stats, n_desc)
        if trimmed_stats["avg_wall"]:
            speedup = full_stats["avg_wall"] / trimmed_stats["avg_wall"]
            print(f"\n  trimmed vs full  : {speedup:.2f}× faster per entity")
        _print_ab_examples(full_stats, trimmed_stats)
    else:
        _print_mode_report(stats, n_desc)
        if concurrent_wall:
            serial_total_sample = stats["avg_wall"] * len(sample)
            speedup = serial_total_sample / concurrent_wall
            serial_total = stats["avg_wall"] * n_desc
            print(f"\n  concurrency={args.concurrency:<2}     : "
                  f"{_fmt_dur(serial_total / speedup)}  (measured {speedup:.1f}× speedup)")

        print("\n" + "-" * 64)
        print("SAMPLE DESCRIPTIONS (eyeball the quality):")
        for qn, desc in stats["examples"][:3]:
            print(f"\n  • {qn}\n    → {desc}")

    print(f"\n  Note: this is a ONE-TIME cost. With source-hash caching + the\n"
          f"  deterministic UUID, re-ingestion only re-describes changed entities.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5-coder:3b", help="Ollama model for descriptions")
    parser.add_argument(
        "--prompt", choices=("full", "trimmed", "both"), default="full",
        help="Prompt mode to measure; 'both' A/Bs full vs trimmed on the same sample",
    )
    parser.add_argument("--json", default=None, help="Graph JSON (default: newest under ingested/)")
    parser.add_argument("--source-root", default=None, help="Repo root to read source from")
    parser.add_argument("--samples", type=int, default=15, help="How many entities to measure")
    parser.add_argument("--concurrency", type=int, default=1, help="Also measure this parallelism")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
