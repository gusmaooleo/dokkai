#!/usr/bin/env python
"""
Path connectivity eval (roadmap feature 19 / Lote 17 Q6a) — measures what
the MCP ``path()`` tool actually delivers on a golden dataset's FLOW
questions: for each flow question, the first and last ``expect`` targets are
taken as the endpoints, and ``navigation.get_path`` is run between them.

This is graph-only and fully OFFLINE: unlike ``eval_retrieval.py``, it never
opens a Weaviate connection (endpoints resolve via
``services.graph_store.resolve_entity`` against the ingested graph JSON, not
a corpus). ``navigation.get_path`` is called directly — never
``mcp_server._resolve_project``, which lists ingested graphs via a Weaviate
chunk-count call and would break the offline property.

Honest framing: ``path()`` connects two entities with typed, real hops — it
does NOT reconstruct the retrieval narrative between them. A flow question's
MIDDLE ``expect`` targets (between the first and last) are reported as a
"mid-target coverage" DIAGNOSTIC only, not a score — a low number is
expected and does not mean the path is wrong, only that the shortest
structural route doesn't happen to pass through every intermediate step the
question's author had in mind.

Datasets are the same JSON files ``eval_retrieval.py`` reads
(``scripts/eval_dataset_*.json`` by default); only ``category == "flow"``
questions are used. A dataset with no flow questions, or whose project has
no ingested graph (e.g. the fixture dataset — chunks-only, no graph JSON),
is skipped with an explicit message, never silently.

Usage
-----
    uv run python scripts/eval_path.py \
        [--dataset PATH ...] [--max-depth N] [--json FILE] [--out FILE.md]
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = _REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

# Load .env BEFORE importing services/mcp_server — same convention as
# eval_retrieval.py.
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# mcp_server is imported in-process purely for `_render_path` — REAL
# tool-output rendering/token counts, not a reimplementation. Safe: nothing
# at mcp_server's import time or in `_render_path` touches Weaviate (only
# the `@mcp.tool()`-wrapped functions this script never calls do).
import mcp_server  # noqa: E402
from services import graph_store, navigation  # noqa: E402

DEFAULT_DATASET_GLOB = str(Path(__file__).resolve().parent / "eval_dataset_*.json")
SCHEMA_VERSION = 1

# Public-safe, aggregate-only context line (roadmap feature 18's own eval)
# — hardcode nothing else corpus-specific here.
_RETRIEVAL_FLOW_BASELINE_NOTE = (
    "retrieval flow baseline (18): cov@10 0.41 best variant at ~2.4k context tokens"
)


# ---------------------------------------------------------------------------
# Minimal dataset schema — flow questions only. Full schema validation
# (target path/qname consistency, stale marking against a live corpus,
# negation rules, ...) lives in eval_retrieval.py and doesn't apply here:
# this script never touches Weaviate, so there is no live index to check
# targets against — resolution happens per-question against the graph
# instead (see measure_flow_question).
# ---------------------------------------------------------------------------

class DatasetError(Exception):
    """A dataset JSON violates the minimal schema this script needs."""


@dataclass
class Target:
    qualified_name: str | None = None
    path: str | None = None


@dataclass
class FlowQuestion:
    id: str
    query: str
    expect: list[Target]


@dataclass
class FlowDataset:
    path: str
    project: str
    questions: list[FlowQuestion]


def _parse_target(raw: Any, qid: str) -> Target:
    if not isinstance(raw, dict):
        raise DatasetError(f"question {qid!r}: expect entry must be an object, got {raw!r}")
    qn, path = raw.get("qualified_name"), raw.get("path")
    return Target(
        qualified_name=qn if isinstance(qn, str) and qn else None,
        path=path if isinstance(path, str) and path else None,
    )


def load_flow_dataset(path: str) -> FlowDataset:
    """Load *path* and keep only its ``category == "flow"`` questions."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise DatasetError(f"{path}: top level must be an object")
    project = raw.get("project")
    if not isinstance(project, str) or not project:
        raise DatasetError(f"{path}: 'project' is required and must be a non-empty string")
    raw_questions = raw.get("questions")
    if not isinstance(raw_questions, list):
        raise DatasetError(f"{path}: 'questions' is required and must be a list")

    questions: list[FlowQuestion] = []
    seen_ids: set[str] = set()
    for rq in raw_questions:
        if not isinstance(rq, dict) or rq.get("category") != "flow":
            continue
        qid = rq.get("id")
        if not isinstance(qid, str) or not qid:
            raise DatasetError(f"{path}: flow question missing a non-empty string 'id'")
        if qid in seen_ids:
            raise DatasetError(f"{path}: duplicate question id {qid!r}")
        seen_ids.add(qid)
        query = rq.get("query")
        raw_expect = rq.get("expect")
        if not isinstance(raw_expect, list):
            raise DatasetError(f"question {qid!r}: 'expect' is required and must be a list")
        expect = [_parse_target(t, qid) for t in raw_expect]
        questions.append(
            FlowQuestion(id=qid, query=query if isinstance(query, str) else "", expect=expect)
        )

    return FlowDataset(path=path, project=project, questions=questions)


# ---------------------------------------------------------------------------
# Per-question measurement
# ---------------------------------------------------------------------------

@dataclass
class FlowResult:
    id: str
    resolvable: bool
    error: str | None = None
    connected: bool | None = None
    phase: str | None = None  # "clean" | "via_external" | "unreachable"
    hops: int | None = None
    mid_matched: int = 0
    mid_total: int = 0
    output_tokens: int | None = None


def measure_flow_question(q: FlowQuestion, project: str, max_depth: int) -> FlowResult:
    """
    Take *q*'s first/last ``expect`` targets as the ``path()`` endpoints and
    measure the result. A target that doesn't resolve to an entity (or a
    question with fewer than 2 targets, or a first/last target with no
    ``qualified_name`` — ``path()`` needs one) is reported
    ``resolvable=False`` with the reason — excluded from aggregates, never
    scored as a 0 (drift-detector spirit of ``eval_retrieval.py``).
    """
    if len(q.expect) < 2:
        return FlowResult(
            id=q.id, resolvable=False,
            error=f"fewer than 2 expect targets ({len(q.expect)}) — no endpoint pair",
        )
    source_t, target_t = q.expect[0], q.expect[-1]
    if not source_t.qualified_name or not target_t.qualified_name:
        return FlowResult(
            id=q.id, resolvable=False,
            error="first/last expect target has no qualified_name — path() needs one",
        )

    try:
        payload = navigation.get_path(
            project, source_t.qualified_name, target_t.qualified_name, max_depth=max_depth
        )
    except ValueError as e:
        return FlowResult(id=q.id, resolvable=False, error=str(e))

    render = mcp_server._render_path(payload)
    output_tokens = len(render) // 4

    mid_targets = q.expect[1:-1]
    node_qnames = {n["qualified_name"] for n in payload["nodes"]}
    mid_matched = sum(
        1 for t in mid_targets if t.qualified_name is not None and t.qualified_name in node_qnames
    )

    if not payload["found"]:
        phase, hops = "unreachable", None
    else:
        phase = "via_external" if payload["via_external"] else "clean"
        hops = len(payload["hops"])

    return FlowResult(
        id=q.id, resolvable=True, connected=payload["found"], phase=phase, hops=hops,
        mid_matched=mid_matched, mid_total=len(mid_targets), output_tokens=output_tokens,
    )


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------

def _stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def aggregate_flow_results(results: list[FlowResult]) -> dict[str, Any]:
    hops_values = [r.hops for r in results if r.hops is not None]
    token_values = [r.output_tokens for r in results if r.output_tokens is not None]
    return {
        "n": len(results),
        "clean": sum(1 for r in results if r.phase == "clean"),
        "via_external": sum(1 for r in results if r.phase == "via_external"),
        "unreachable": sum(1 for r in results if r.phase == "unreachable"),
        "hops": _stats(hops_values),
        "tokens": _stats(token_values),
        "mid_target_coverage": {
            "matched": sum(r.mid_matched for r in results),
            "total": sum(r.mid_total for r in results),
        },
    }


def _fmt_stat(v: float | int | None) -> str:
    if v is None:
        return "-"
    return f"{v:.1f}" if isinstance(v, float) else str(v)


def _pct(n: int, total: int) -> str:
    return f"{n} ({n / total * 100:.1f}%)" if total else f"{n} (-)"


def _print_aggregate(agg: dict[str, Any]) -> None:
    n = agg["n"]
    print(f"\n  -- path connectivity ({n} scored) --")
    print(
        f"  clean: {_pct(agg['clean'], n)}  via_external: {_pct(agg['via_external'], n)}  "
        f"unreachable: {_pct(agg['unreachable'], n)}"
    )
    h, t = agg["hops"], agg["tokens"]
    print(f"  hops:   min={_fmt_stat(h['min'])} median={_fmt_stat(h['median'])} max={_fmt_stat(h['max'])}")
    print(f"  tokens: min={_fmt_stat(t['min'])} median={_fmt_stat(t['median'])} max={_fmt_stat(t['max'])}")
    cov = agg["mid_target_coverage"]
    cov_pct = f" ({cov['matched'] / cov['total'] * 100:.1f}%)" if cov["total"] else ""
    print(f"  mid-target coverage (diagnostic, not a score): {cov['matched']}/{cov['total']}{cov_pct}")


def _display_path(path: str) -> str:
    """
    Path as recorded in --json/--out output: relative to the REPO ROOT
    (never cwd), falling back to just the filename for a dataset outside the
    repo — never the raw absolute string. Copied from
    ``eval_retrieval.py``'s helper of the same name/docstring intent rather
    than imported, per this script's house-style isolation.
    """
    try:
        return str(Path(path).resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return Path(path).name


def run_dataset(path: str, max_depth: int) -> dict[str, Any] | None:
    print(f"\n{'=' * 72}\nDataset: {path}")
    try:
        dataset = load_flow_dataset(path)
    except (DatasetError, OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"  SCHEMA/LOAD ERROR — dataset aborted: {e}")
        return None

    if not dataset.questions:
        print("  SKIPPED — no flow-category questions in this dataset")
        return None

    try:
        graph_store.get_graph(dataset.project)
    except (graph_store.GraphNotFoundError, OSError, json.JSONDecodeError) as e:
        # A missing graph AND a damaged/truncated ingested/*.json are both
        # per-dataset skips — a corrupt file must not kill the healthy
        # datasets in the same run (mirrors graph_store.list_graphs's own
        # tolerance for unreadable graph files).
        print(
            f"  SKIPPED — project {dataset.project!r} has no readable ingested graph "
            f"(this script is graph-only; a chunks-only/fixture project has nothing to path over): {e.__class__.__name__}"
        )
        return None

    print(f"  project: {dataset.project} ({len(dataset.questions)} flow questions)")

    results = [measure_flow_question(q, dataset.project, max_depth) for q in dataset.questions]
    unresolvable = [r for r in results if not r.resolvable]
    scored = [r for r in results if r.resolvable]

    if unresolvable:
        print(f"  endpoint-unresolvable ({len(unresolvable)}, excluded from aggregates):")
        for r in unresolvable:
            print(f"    [{r.id}] {r.error}")

    for r in scored:
        print(
            f"  [{r.id}] connected={r.connected} phase={r.phase} hops={_fmt_stat(r.hops)} "
            f"mid_coverage={r.mid_matched}/{r.mid_total} tokens={r.output_tokens}"
        )

    agg = aggregate_flow_results(scored)
    _print_aggregate(agg)

    return {
        "path": _display_path(path),
        "project": dataset.project,
        "n_questions": len(dataset.questions),
        "unresolvable": [{"id": r.id, "error": r.error} for r in unresolvable],
        "questions": [
            {
                "id": r.id, "connected": r.connected, "phase": r.phase, "hops": r.hops,
                "mid_matched": r.mid_matched, "mid_total": r.mid_total,
                "output_tokens": r.output_tokens,
            }
            for r in scored
        ],
        "aggregate": agg,
    }


# ---------------------------------------------------------------------------
# --json / --out
# ---------------------------------------------------------------------------

def build_payload(dataset_results: list[dict[str, Any]], max_depth: int) -> dict[str, Any]:
    """The --json document: versioned, no timestamps, no absolute local
    paths (see _display_path) — determinism + privacy. Records max_depth
    because it changes the numbers: two artifacts from different depths
    must be distinguishable."""
    return {
        "schema_version": SCHEMA_VERSION,
        "max_depth": max_depth,
        "retrieval_flow_baseline_note": _RETRIEVAL_FLOW_BASELINE_NOTE,
        "datasets": dataset_results,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """
    AGGREGATE-ONLY report (same content policy as eval_retrieval.py's --out):
    no per-question rows, no qualified_names — those live in --json's
    "questions"/"unresolvable" sections only. A question id alone (never its
    qname/error text) may appear, mirroring eval_retrieval's stale-id
    precedent. Built purely from ``payload`` (no clock/env access), so two
    identical runs produce a byte-identical file.
    """
    lines: list[str] = ["# Path Connectivity Eval Report", ""]
    for ds in payload["datasets"]:
        lines.append(f"## Dataset: {ds['path']}")
        lines.append("")
        lines.append(f"- project: `{ds['project']}` ({ds['n_questions']} flow questions)")
        if ds["unresolvable"]:
            ids = ", ".join(u["id"] for u in ds["unresolvable"])
            lines.append(f"- endpoint-unresolvable ({len(ds['unresolvable'])}): {ids}")
        lines.append("")
        agg = ds["aggregate"]
        n = agg["n"]
        lines.append(f"n={n} scored")
        lines.append("")
        lines.append(f"- clean: {_pct(agg['clean'], n)}")
        lines.append(f"- via_external: {_pct(agg['via_external'], n)}")
        lines.append(f"- unreachable: {_pct(agg['unreachable'], n)}")
        h, t = agg["hops"], agg["tokens"]
        lines.append(f"- hops: min={_fmt_stat(h['min'])} median={_fmt_stat(h['median'])} max={_fmt_stat(h['max'])}")
        lines.append(f"- tokens: min={_fmt_stat(t['min'])} median={_fmt_stat(t['median'])} max={_fmt_stat(t['max'])}")
        cov = agg["mid_target_coverage"]
        lines.append(f"- mid-target coverage (diagnostic, not a score): {cov['matched']}/{cov['total']}")
        lines.append("")
    lines.append(f"_{_RETRIEVAL_FLOW_BASELINE_NOTE}_")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset", action="append", default=None,
        help="Dataset JSON path (repeatable; default: glob scripts/eval_dataset_*.json)",
    )
    parser.add_argument(
        "--max-depth", type=int, default=10,
        help="Max BFS hops passed to path() (default: 10, same as the MCP tool's default)",
    )
    parser.add_argument(
        "--json", default=None, metavar="FILE",
        help="Write the full machine-readable results (aggregates + per-question rows) to FILE",
    )
    parser.add_argument(
        "--out", default=None, metavar="FILE.md",
        help="Write an aggregate-only report to FILE.md (markdown)",
    )
    return parser.parse_args()


def _validate_writable(path: str, flag: str) -> None:
    """Prove *path* is writable BEFORE any real work runs — same convention
    as eval_retrieval.py's helper of the same name/intent."""
    try:
        with open(path, "a", encoding="utf-8"):
            pass
    except OSError as e:
        print(f"Cannot write {flag} path {path!r}: {e}")
        sys.exit(1)


def main() -> None:
    args = parse_args()

    if args.max_depth < 1 or args.max_depth > 20:
        print(f"--max-depth must be between 1 and 20, got {args.max_depth}")
        sys.exit(1)

    if args.json:
        _validate_writable(args.json, "--json")
    if args.out:
        _validate_writable(args.out, "--out")

    dataset_paths = args.dataset or sorted(glob.glob(DEFAULT_DATASET_GLOB))
    if not dataset_paths:
        print(f"No datasets found (default glob: {DEFAULT_DATASET_GLOB}); pass --dataset PATH")
        sys.exit(1)

    dataset_results: list[dict[str, Any]] = []
    for path in dataset_paths:
        result = run_dataset(path, args.max_depth)
        if result is not None:
            dataset_results.append(result)

    print(f"\n{_RETRIEVAL_FLOW_BASELINE_NOTE}")

    payload = build_payload(dataset_results, args.max_depth)

    if args.json:
        try:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
        except OSError as e:
            print(f"Cannot write --json path {args.json!r}: {e}")

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(render_markdown(payload))
        except OSError as e:
            print(f"Cannot write --out path {args.out!r}: {e}")


if __name__ == "__main__":
    main()
