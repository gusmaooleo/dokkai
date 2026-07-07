#!/usr/bin/env python
"""
Measure the real cost of generating per-entity micro-descriptions (Tier 2) on
*this* machine, so the description strategy (model, selectivity, parallelism)
is decided with numbers instead of guesses.

It reads an ingested graph JSON, samples real entities, asks a local Ollama
model to describe each one, and reports tokens/sec + per-entity wall-clock,
then extrapolates to the whole repository.

Usage
-----
    uv run python scripts/measure_descriptions.py \
        --model qwen2.5:3b \
        --samples 15 \
        [--json ingested/<file>.json] \
        [--source-root /path/to/repo] \
        [--concurrency 4] \
        [--base-url http://localhost:11434]

Ollama returns exact timing fields (prompt_eval_count / eval_count and their
nanosecond durations), so the measured tok/s reflects your actual hardware and
model — not an estimate.
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
from services.retriever import _is_test_path  # noqa: E402


SYSTEM_PROMPT = (
    "You summarize code for a search index. Reply with ONE concise sentence "
    "(max 25 words) describing what the code does. No preamble, no code, no "
    "bullet points — just the sentence."
)

# Options tuned for a short, deterministic description of a single entity.
_GEN_OPTIONS = {"temperature": 0.1, "num_predict": 64, "num_ctx": 4096}


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


def _prompt_for(chunk: CodeChunk) -> str:
    return f"{chunk.entity_type} {chunk.qualified_name}\n\n{chunk.source_code}"


async def _generate(client: httpx.AsyncClient, base_url: str, model: str, chunk: CodeChunk) -> dict:
    """One /api/generate call; returns Ollama's response incl. timing fields."""
    resp = await client.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "system": SYSTEM_PROMPT,
            "prompt": _prompt_for(chunk),
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
    print(f"Sample     : {len(sample)} entities\n")

    timeout = httpx.Timeout(120.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        await _ensure_model(client, args.base_url, args.model)

        # Warm up (first call cold-loads the model — exclude from stats).
        print("Warming up the model…")
        await _generate(client, args.base_url, args.model, sample[0])

        # ---- serial pass: accurate per-entity tok/s from Ollama's fields ----
        prompt_tokens = gen_tokens = 0
        prompt_secs = gen_secs = 0.0
        per_entity_wall: list[float] = []
        examples: list[tuple[str, str]] = []

        print("Measuring (serial)…\n")
        for i, chunk in enumerate(sample, 1):
            t0 = time.perf_counter()
            data = await _generate(client, args.base_url, args.model, chunk)
            per_entity_wall.append(time.perf_counter() - t0)

            prompt_tokens += data.get("prompt_eval_count", 0)
            gen_tokens += data.get("eval_count", 0)
            prompt_secs += _ns_to_s(data.get("prompt_eval_duration"))
            gen_secs += _ns_to_s(data.get("eval_duration"))
            if len(examples) < 3:
                examples.append((chunk.qualified_name, data.get("response", "").strip()))
            print(f"  [{i}/{len(sample)}] {per_entity_wall[-1]:5.2f}s  {chunk.qualified_name[:70]}")

        avg_wall = sum(per_entity_wall) / len(per_entity_wall)
        prompt_tps = prompt_tokens / prompt_secs if prompt_secs else 0.0
        gen_tps = gen_tokens / gen_secs if gen_secs else 0.0
        avg_in = prompt_tokens / len(sample)
        avg_out = gen_tokens / len(sample)

        # ---- optional concurrent pass: measured real speedup ----
        concurrent_wall = None
        if args.concurrency > 1:
            print(f"\nMeasuring (concurrency={args.concurrency})…")
            sem = asyncio.Semaphore(args.concurrency)

            async def _one(c: CodeChunk) -> None:
                async with sem:
                    await _generate(client, args.base_url, args.model, c)

            t0 = time.perf_counter()
            await asyncio.gather(*(_one(c) for c in sample))
            concurrent_wall = time.perf_counter() - t0

    # ---- report ----
    print("\n" + "=" * 64)
    print("RESULTS")
    print("=" * 64)
    print(f"  avg input       : {avg_in:6.0f} tokens/entity")
    print(f"  avg output      : {avg_out:6.0f} tokens/entity")
    print(f"  prompt eval     : {prompt_tps:6.0f} tok/s")
    print(f"  generation      : {gen_tps:6.0f} tok/s")
    print(f"  per-entity wall : {avg_wall:6.2f} s (serial)")

    serial_total = avg_wall * n_desc
    print("\n  EXTRAPOLATION to all describable entities "
          f"({n_desc}):")
    print(f"    serial            : {_fmt_dur(serial_total)}")
    if concurrent_wall:
        speedup = (avg_wall * len(sample)) / concurrent_wall
        print(f"    concurrency={args.concurrency:<2}     : "
              f"{_fmt_dur(serial_total / speedup)}  (measured {speedup:.1f}× speedup)")
    print(f"\n  Note: this is a ONE-TIME cost. With source-hash caching + the\n"
          f"  deterministic UUID, re-ingestion only re-describes changed entities.")

    print("\n" + "-" * 64)
    print("SAMPLE DESCRIPTIONS (eyeball the quality):")
    for qn, desc in examples:
        print(f"\n  • {qn}\n    → {desc}")


def _fmt_dur(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:3b", help="Ollama model for descriptions")
    parser.add_argument("--json", default=None, help="Graph JSON (default: newest under ingested/)")
    parser.add_argument("--source-root", default=None, help="Repo root to read source from")
    parser.add_argument("--samples", type=int, default=15, help="How many entities to measure")
    parser.add_argument("--concurrency", type=int, default=1, help="Also measure this parallelism")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
