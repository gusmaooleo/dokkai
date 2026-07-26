#!/usr/bin/env python
"""
Retrieval eval harness (roadmap feature 18, part 1) — rank-based quality
metrics for ``services.retriever`` variants against a real Weaviate corpus.
No LLM anywhere: this measures retrieval only (hit@k / MRR / coverage /
token cost), never an answer's correctness.

Datasets are plain JSON files (``scripts/eval_dataset_*.json`` by default),
each naming a Weaviate ``project_name`` and a list of questions with expected
targets (qualified_name and/or file path). Before scoring, every target is
resolved against a single live cursor scan of that project (a drift
detector): a target that no longer exists in the corpus marks its question
STALE — excluded from aggregates and listed loudly, never silently scored 0.

Weaviate hybrid scores are RELATIVE_SCORE-normalized per response, so they
are never comparable across queries or variants — every metric here is
rank-based (hit@k, MRR, coverage), not score-based.

A ``fixture: true`` dataset (e.g. ``scripts/eval_dataset_fixture.json``) is
self-seeding: its synthetic corpus (``scripts/eval_fixture.py``) is upserted
into the live collection under the dataset's ``project`` right before the
resolution pass, and removed again once the dataset finishes (``finally`` —
survives a crash mid-run). Pass ``--keep-fixture`` to leave it seeded for
manual inspection.

Usage
-----
    uv run python scripts/eval_retrieval.py \
        [--dataset PATH ...] [--variants seeds,bm25,...] [--verbose] [--keep-fixture]
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

# Load .env BEFORE importing services — same convention as
# scripts/test_retriever_seeding.py, so WEAVIATE_HOST/VECTORIZER_PROVIDER/...
# come from the user's real environment rather than process defaults.
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from services.retriever import (  # noqa: E402
    Retriever,
    RetrievedChunk,
    VECTOR_CODE,
    VECTOR_SUMMARY,
)
from services.weaviate_client import (  # noqa: E402
    COLLECTION_NAME,
    delete_project_chunks,
    ensure_collection,
    get_client,
    upsert_chunks,
)

CATEGORIES = ("identifier", "conceptual", "flow", "negation")
HIT_KS = (1, 3, 5, 10)
EVAL_WINDOW = 10  # W=10 -- ranks beyond this don't score, even for graph
                  # variants that may return up to 30 results.
DEFAULT_DATASET_GLOB = str(Path(__file__).resolve().parent / "eval_dataset_*.json")


# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------

class DatasetError(Exception):
    """A dataset JSON violates the schema — aborts THAT dataset only."""


@dataclass
class Target:
    qualified_name: str | None = None
    path: str | None = None


@dataclass
class Question:
    id: str
    category: str
    query: str
    expect: list[Target]
    notes: str = ""
    stale: bool = False
    stale_reason: str = ""


@dataclass
class Dataset:
    path: str
    project: str
    fixture: bool
    questions: list[Question]


def _parse_target(raw: Any, qid: str) -> Target:
    if not isinstance(raw, dict):
        raise DatasetError(f"question {qid!r}: expect entry must be an object, got {raw!r}")
    qn, path = raw.get("qualified_name"), raw.get("path")
    if qn is not None:
        if not isinstance(qn, str):
            raise DatasetError(f"question {qid!r}: expect.qualified_name must be a string, got {qn!r}")
        if not qn:
            # LOW-C: an empty string is a value, not "absent" — reject it
            # outright rather than silently leaving a target with no real
            # signal that still counts toward "(N scored)".
            raise DatasetError(f"question {qid!r}: expect.qualified_name must not be empty")
    if path is not None:
        if not isinstance(path, str):
            raise DatasetError(f"question {qid!r}: expect.path must be a string, got {path!r}")
        if not path:
            raise DatasetError(f"question {qid!r}: expect.path must not be empty")
    if qn is None and path is None:
        raise DatasetError(f"question {qid!r}: expect entry needs qualified_name and/or path")
    return Target(qualified_name=qn, path=path)


def load_dataset(path: str) -> Dataset:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise DatasetError(f"{path}: top level must be an object")
    project = raw.get("project")
    if not isinstance(project, str) or not project:
        raise DatasetError(f"{path}: 'project' is required and must be a non-empty string")
    fixture = bool(raw.get("fixture", False))
    if fixture:
        # MED-1(b)/LOW-3: a fixture dataset MUST target eval_fixture's own
        # project — otherwise a typo'd/foreign `project` would seed the
        # synthetic corpus under a real project name, and `run_dataset`'s
        # cleanup (which, per MED-1(d), always deletes eval_fixture's own
        # PROJECT_NAME) would silently miss the mismatch instead of
        # protecting it. Caught here as a normal SCHEMA/LOAD ERROR — a
        # one-dataset abort — rather than surfacing later as a raw
        # RuntimeError that kills the whole run. Local import: only a
        # `fixture: true` dataset ever needs eval_fixture.
        import eval_fixture

        if project != eval_fixture.PROJECT_NAME:
            raise DatasetError(
                f"{path}: fixture datasets must use project={eval_fixture.PROJECT_NAME!r}, got {project!r}"
            )
    raw_questions = raw.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise DatasetError(f"{path}: 'questions' is required and must be a non-empty list")

    questions: list[Question] = []
    seen_ids: set[str] = set()
    for rq in raw_questions:
        if not isinstance(rq, dict):
            raise DatasetError(f"{path}: question entry must be an object, got {rq!r}")
        qid = rq.get("id")
        if not isinstance(qid, str) or not qid:
            raise DatasetError(f"{path}: question missing a non-empty string 'id'")
        if qid in seen_ids:
            raise DatasetError(f"{path}: duplicate question id {qid!r}")
        seen_ids.add(qid)

        category = rq.get("category")
        if category not in CATEGORIES:
            raise DatasetError(f"question {qid!r}: category must be one of {CATEGORIES}, got {category!r}")

        query = rq.get("query")
        if not isinstance(query, str) or not query.strip():
            raise DatasetError(f"question {qid!r}: 'query' is required and must be non-empty")

        raw_expect = rq.get("expect")
        if not isinstance(raw_expect, list):
            raise DatasetError(f"question {qid!r}: 'expect' is required and must be a list")
        expect = [_parse_target(t, qid) for t in raw_expect]

        if category == "negation":
            if expect:
                raise DatasetError(f"question {qid!r}: negation questions must have expect: []")
        elif not expect:
            raise DatasetError(f"question {qid!r}: category {category!r} needs >=1 expect target")

        if category == "flow":
            # LOW-B: not a schema error (the target is otherwise valid) —
            # just a heads-up that coverage@10 is entity-only (LOW-5), so a
            # path-only target can never contribute to it.
            for t in expect:
                if t.qualified_name is None:
                    print(
                        f"  WARNING: question {qid!r} (flow) has a path-only target "
                        f"({t.path!r}) — it can never count toward cov@10, coverage is entity-only"
                    )

        questions.append(Question(
            id=qid, category=category, query=query, expect=expect,
            notes=rq.get("notes", ""),
        ))

    return Dataset(path=path, project=project, fixture=fixture, questions=questions)


# ---------------------------------------------------------------------------
# Resolution pass (drift detector) — one cursor scan per dataset's project.
# ---------------------------------------------------------------------------

@dataclass
class ProjectIndex:
    object_count: int
    qualified_names: set[str] = field(default_factory=set)
    paths: set[str] = field(default_factory=set)
    qn_to_path: dict[str, str] = field(default_factory=dict)
    qn_described: dict[str, bool] = field(default_factory=dict)


def resolve_project(client, project_name: str) -> ProjectIndex:
    """
    Single ``collection.iterator()`` cursor scan (minimal ``return_properties``,
    no vectors), matching ``project_name`` CLIENT-SIDE with exact ``==`` — this
    schema's TEXT properties use WORD tokenization, under which a server-side
    ``Filter...equal()`` matches by token superset, not literal string equality
    (e.g. it would match "sample_back-end" for project "back-end" — see
    ``weaviate_client._uuids_matching``'s docstring), so it is never safe here.
    """
    if not client.collections.exists(COLLECTION_NAME):
        return ProjectIndex(object_count=0)

    collection = client.collections.get(COLLECTION_NAME)
    index = ProjectIndex(object_count=0)
    for obj in collection.iterator(
        return_properties=["project_name", "qualified_name", "file_path", "description"],
    ):
        props = obj.properties
        if props.get("project_name") != project_name:
            continue
        index.object_count += 1
        qn = str(props.get("qualified_name") or "")
        path = str(props.get("file_path") or "")
        if qn:
            index.qualified_names.add(qn)
            # 32/3173 live objects carry file_path == "" — never register an
            # empty path as an entity's resolved path, or every falsy-path
            # target/result would spuriously "match" each other (MED-2).
            if path:
                index.qn_to_path[qn] = path
            index.qn_described[qn] = bool((props.get("description") or "").strip())
        if path:
            index.paths.add(path)
    return index


def _explicit_path(t: Target) -> str | None:
    """A target's own ``path`` field, with an empty string normalized to
    absent (MED-2) — never treated as a real, matchable path."""
    return t.path if t.path else None


def mark_stale(dataset: Dataset, index: ProjectIndex) -> None:
    """Any unresolvable target marks its WHOLE question stale (never partially
    scored) — excluded from aggregates, listed loudly in the report header.
    Also flags an internal inconsistency (LOW-4): a target that names BOTH a
    qualified_name and a path whose corpus-resolved path disagrees with it."""
    for q in dataset.questions:
        reasons: list[str] = []
        for t in q.expect:
            qn = t.qualified_name
            path = _explicit_path(t)
            if qn is not None and qn not in index.qualified_names:
                reasons.append(f"qualified_name not found: {qn}")
            if path is not None and path not in index.paths:
                reasons.append(f"path not found: {path}")
            if qn is not None and path is not None and qn in index.qualified_names:
                resolved = index.qn_to_path.get(qn)  # None when that entity's own file_path is falsy
                if resolved and resolved != path:
                    reasons.append(
                        f"target qn/path mismatch: {qn} resolves to {resolved!r}, target said {path!r}"
                    )
        if reasons:
            q.stale = True
            q.stale_reason = "; ".join(reasons)


def _target_path(t: Target, index: ProjectIndex) -> str | None:
    """A target's file path for FILE-mode matching: its own explicit path, or
    (entity-only target) the path resolved from its qualified_name via the
    project index map. An empty/falsy path — the target's own or the
    resolved one — is always treated as absent (MED-2), never as a real
    matchable path."""
    explicit = _explicit_path(t)
    if explicit is not None:
        return explicit
    if t.qualified_name is not None:
        return index.qn_to_path.get(t.qualified_name) or None
    return None


# ---------------------------------------------------------------------------
# Variant registry — one entry each, uniform (retriever, query, project)
# signature, so future features add entries without touching the runner.
# ---------------------------------------------------------------------------

class _Bm25SeedRetriever(Retriever):
    """
    ``search_graph`` seeds itself via ``self.search_seeds`` — overriding it
    here (same signature) swaps the seed lane for plain BM25, giving a fully
    deterministic, zero-embedding graph-search pipeline (``graph-bm25``).
    """

    def search_seeds(
        self,
        query: str,
        *,
        project_name: str | None = None,
        top_k: int = 8,
        alpha: float = 0.7,
    ) -> list[RetrievedChunk]:
        return self.search_bm25(query, project_name=project_name, top_k=top_k)


def _v_seeds(r: Retriever, query: str, project: str) -> list[RetrievedChunk]:
    return r.search_seeds(query, project_name=project, top_k=10)


def _v_seeds_nofast(r: Retriever, query: str, project: str) -> list[RetrievedChunk]:
    return r.search(
        query, project_name=project, top_k=10, target_vector=[VECTOR_SUMMARY, VECTOR_CODE],
    )


def _v_bm25(r: Retriever, query: str, project: str) -> list[RetrievedChunk]:
    return r.search_bm25(query, project_name=project, top_k=10)


def _v_graph(r: Retriever, query: str, project: str) -> list[RetrievedChunk]:
    # top_k_seeds=8 mirrors mcp_server.context()'s default k — all other
    # kwargs left at search_graph's own defaults (max_hops=2, max_nodes=30,
    # direction="both") for MCP `context` parity.
    return r.search_graph(query, project_name=project, top_k_seeds=8)


def _v_graph_bm25(r: Retriever, query: str, project: str) -> list[RetrievedChunk]:
    bm25_retriever = _Bm25SeedRetriever(r.client)
    return bm25_retriever.search_graph(query, project_name=project, top_k_seeds=8)


VARIANTS: dict[str, Callable[[Retriever, str, str], list[RetrievedChunk]]] = {
    "seeds": _v_seeds,
    "seeds-nofast": _v_seeds_nofast,
    "bm25": _v_bm25,
    "graph": _v_graph,
    "graph-bm25": _v_graph_bm25,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _match_ranks(
    results: list[RetrievedChunk], targets: list[Target], index: ProjectIndex, mode: str,
) -> list[int]:
    """1-based ranks (within ``results``, already windowed to W=10) where ANY
    target matches, under ``mode`` ('entity' or 'file')."""
    ranks: list[int] = []
    for i, r in enumerate(results):
        for t in targets:
            if mode == "entity":
                hit = t.qualified_name is not None and r.qualified_name == t.qualified_name
            else:
                tp = _target_path(t, index)
                # A falsy result file_path can never file-match, even a
                # falsy target path — _target_path already normalizes
                # falsy-to-None, but the result side needs its own guard
                # (MED-2: 32/3173 live objects carry file_path == "").
                hit = bool(tp) and bool(r.file_path) and r.file_path == tp
            if hit:
                ranks.append(i + 1)
                break
    return ranks


def evaluate(
    question: Question, results: list[RetrievedChunk], index: ProjectIndex, retriever: Retriever,
) -> dict[str, Any]:
    """Per-(question, variant) record. ``results`` is the variant's full
    ranked output (up to 30 for graph variants); rank-based metrics are
    windowed to EVAL_WINDOW, token metrics use the full list except
    raw_tokens@10 which is explicitly top-10 only.

    Entity and file are scored as two SEPARATE match modes throughout —
    hit@k, and (since MED-3) MRR@10 too, as ``mrr_entity``/``mrr_file``. A
    single "combined" MRR was tried and dropped: an entity match at rank i
    always implies a file match at rank <= i (entity match => same chunk =>
    same file_path), so combined MRR was empirically IDENTICAL to file-mode
    MRR in 100% of live cases — not a real second signal.
    """
    window = results[:EVAL_WINDOW]

    # LOW-1: build_graph_context([]) returns a non-empty placeholder string
    # ("(No relevant code found...)"), which would otherwise report ~10
    # phantom tokens for a variant that returned nothing at all.
    context_tokens = (
        len(retriever.build_graph_context(results, max_chars=10_000)) // 4 if results else 0
    )
    record: dict[str, Any] = {
        "context_tokens": context_tokens,  # MCP `context` parity: same call/args as mcp_server.context()
        "raw_tokens10": sum(len(c.chunk_text) // 4 for c in window),
    }

    if question.category == "negation":
        record["abstained"] = len(results) == 0
        record["waste_tokens"] = record["context_tokens"]
        return record

    entity_targets = [t for t in question.expect if t.qualified_name is not None]
    file_targets = [t for t in question.expect if _target_path(t, index) is not None]

    if entity_targets:
        e_ranks = _match_ranks(window, entity_targets, index, "entity")
        record["entity_hits"] = {k: int(bool(e_ranks) and min(e_ranks) <= k) for k in HIT_KS}
        record["mrr_entity"] = (1.0 / min(e_ranks)) if e_ranks else 0.0
    if file_targets:
        f_ranks = _match_ranks(window, file_targets, index, "file")
        record["file_hits"] = {k: int(bool(f_ranks) and min(f_ranks) <= k) for k in HIT_KS}
        record["mrr_file"] = (1.0 / min(f_ranks)) if f_ranks else 0.0

    if question.category == "flow":
        # LOW-5: entity-only. A same-file sibling is not the step's own
        # code — coverage counts a target as covered only when its OWN
        # qualified_name is in the top 10, never via a file-path match.
        # A path-only target (no qualified_name) can therefore never be
        # "covered" here — it still counts in the denominator.
        matched = sum(
            1 for t in question.expect
            if t.qualified_name is not None and any(r.qualified_name == t.qualified_name for r in window)
        )
        record["coverage"] = matched / len(question.expect)

    record["all_described"] = (
        all(index.qn_described.get(t.qualified_name, False) for t in entity_targets)
        if entity_targets else None
    )
    return record


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a (category, variant) cell's per-question records into
    report-ready means. Uniform across every category — no branching on
    category here: ``mean_of()`` naturally returns None when a key is absent
    from every record (e.g. "mrr_entity"/"coverage" for negation, "abstained"
    for non-negation), which the report renders as a blank ("-")."""
    agg: dict[str, Any] = {"n": len(records)}

    def mean_of(key: str, sub_key: int | None = None) -> float | None:
        vals = []
        for r in records:
            if key not in r:
                continue
            v = r[key] if sub_key is None else r[key].get(sub_key)
            if v is not None:
                vals.append(v)
        return _mean(vals)

    for k in HIT_KS:
        agg[f"entity_hit@{k}"] = mean_of("entity_hits", k)
        agg[f"file_hit@{k}"] = mean_of("file_hits", k)
    agg["mrr_entity10"] = mean_of("mrr_entity")
    agg["mrr_file10"] = mean_of("mrr_file")
    agg["coverage10"] = mean_of("coverage")
    agg["abstention_rate"] = mean_of("abstained")
    agg["waste_tokens"] = mean_of("waste_tokens")
    agg["context_tokens"] = mean_of("context_tokens")
    agg["raw_tokens10"] = mean_of("raw_tokens10")
    return agg


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(v: float | None, digits: int = 2) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _print_category_table(
    category: str, variant_names: list[str], rows: dict[str, dict], applicability: dict[str, int],
) -> None:
    n_scored = next(iter(rows.values()))["n"] if rows else 0
    print(f"\n-- {category} ({n_scored} scored) --")
    if category == "flow":
        # LOW-5: state the entity-only rule where it's read, not just in the
        # docstring — a same-file sibling never counts as covering a step.
        print("  cov@10 is entity-only: a same-file sibling does not count as covering the step")
    # LOW-A: eh@k/fh@k/MRR/coverage/negation each average over a DIFFERENT
    # subset of these n_scored questions (a question mixing entity-only and
    # file-only targets contributes to only one column each) — without this,
    # a mean over 1 question reads as a mean over all n_scored.
    print(
        f"  n= entity:{applicability['entity']} file:{applicability['file']} "
        f"coverage:{applicability['coverage']} negation:{applicability['negation']} "
        f"(of {n_scored} scored)"
    )

    # Uniform column set across every category table: "-" fills wherever a
    # category's questions don't carry that metric (coverage@10 is only ever
    # non-blank for flow, abstain/waste_tok only for negation, etc).
    headers = (
        ["variant"] + [f"eh@{k}" for k in HIT_KS] + [f"fh@{k}" for k in HIT_KS]
        + ["eMRR@10", "fMRR@10", "cov@10", "abstain", "waste_tok", "ctx_tok", "raw_tok10"]
    )
    # LOW-3: the variant column must be wide enough for the longest variant
    # NAME actually being printed (e.g. "seeds-nofast", 12 chars), not just
    # its header ("variant", 7 chars).
    widths = [max(len(h), 9) for h in headers]
    widths[0] = max(widths[0], max((len(v) for v in variant_names), default=0))
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for name in variant_names:
        agg = rows[name]
        cells = [name]
        cells += [_fmt(agg[f"entity_hit@{k}"]) for k in HIT_KS]
        cells += [_fmt(agg[f"file_hit@{k}"]) for k in HIT_KS]
        cells.append(_fmt(agg["mrr_entity10"], 3))
        cells.append(_fmt(agg["mrr_file10"], 3))
        cells.append(_fmt(agg["coverage10"]))
        cells.append(_fmt(agg["abstention_rate"]))
        cells.append(_fmt(agg["waste_tokens"], 0))
        cells.append(_fmt(agg["context_tokens"], 0))
        cells.append(_fmt(agg["raw_tokens10"], 0))
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))


def _seed_fixture(client) -> tuple[int, str]:
    """
    Seed the synthetic fixture corpus (``scripts/eval_fixture.py``) into the
    live collection under ``eval_fixture.PROJECT_NAME`` — ``load_dataset()``
    already enforced ``dataset.project == PROJECT_NAME`` for any
    ``fixture: true`` dataset, so this never takes a caller-supplied project
    name. Seeded via the same ``upsert_chunks`` plumbing real ingestion uses.
    """
    import eval_fixture  # local import: only needed for fixture datasets

    # LOW-1: only create/upgrade the collection when it doesn't already
    # exist. ensure_collection() unconditionally inherits the
    # DOKKAI_RECREATE_COLLECTION escape hatch, which DROPS THE ENTIRE
    # collection (every project, not just the fixture's) — never call it
    # when the collection is already there.
    if not client.collections.exists(COLLECTION_NAME):
        ensure_collection(client)

    chunks = eval_fixture.build_chunks()
    if not chunks:
        # MED-1(c): a programming error (or a hostile monkeypatch), not a
        # dataset problem — an empty list makes the per-chunk project_name
        # guard below vacuously true, which would otherwise let a broken
        # fixture "seed" 0 objects while still marking fixture_seeded and
        # triggering a (harmless here, but silently wrong) cleanup pass.
        raise RuntimeError("eval_fixture.build_chunks() returned no chunks")
    if any(c.project_name != eval_fixture.PROJECT_NAME for c in chunks):
        raise RuntimeError(
            f"eval_fixture.build_chunks() chunks must all carry project_name={eval_fixture.PROJECT_NAME!r}"
        )
    return upsert_chunks(client, chunks, ingestion_id="eval-fixture-seed"), eval_fixture.PROJECT_NAME


def run_dataset(
    path: str, client, retriever: Retriever, variant_names: list[str], *,
    verbose: bool, keep_fixture: bool = False,
) -> None:
    print(f"\n{'=' * 72}\nDataset: {path}")
    try:
        dataset = load_dataset(path)
    except (DatasetError, OSError, json.JSONDecodeError, UnicodeDecodeError, ImportError) as e:
        # MED-1: a nonexistent path (OSError), malformed JSON
        # (JSONDecodeError), a non-UTF-8 file (UnicodeDecodeError), a schema
        # violation (DatasetError) and a missing/broken eval_fixture module
        # (ImportError, from the lazy import in the fixture gate) must all
        # abort ONLY this dataset — never the whole run.
        print(f"  SCHEMA/LOAD ERROR — dataset aborted: {e}")
        return

    fixture_seeded = False
    try:
        if dataset.fixture:
            # LOW-2: set BEFORE calling _seed_fixture, not after it returns —
            # a partial failure (real chunks already upserted, then an
            # exception, e.g. from upsert_chunks' own failed_objects check)
            # must still be swept by the finally below. A no-op cleanup on
            # the rare case nothing was actually written yet is a cheap
            # price for that guarantee.
            fixture_seeded = True
            seeded, seeded_project = _seed_fixture(client)
            print(f"  seeded {seeded} fixture chunk(s) under project={seeded_project!r}")

        index = resolve_project(client, dataset.project)
        print(f"  project: {dataset.project} ({index.object_count} objects)")
        if index.object_count == 0:
            print("  SKIPPED — project has 0 objects in Weaviate")
            return

        mark_stale(dataset, index)
        stale = [q for q in dataset.questions if q.stale]
        scored_questions = [q for q in dataset.questions if not q.stale]
        if stale:
            print(f"  STALE questions ({len(stale)}, excluded from aggregates):")
            for q in stale:
                print(f"    [{q.id}] {q.stale_reason}")

        # records[category][variant] = list of per-question record dicts
        records: dict[str, dict[str, list[dict]]] = {c: {v: [] for v in variant_names} for c in CATEGORIES}

        for q in scored_questions:
            for vname in variant_names:
                results = VARIANTS[vname](retriever, q.query, dataset.project)
                record = evaluate(q, results, index, retriever)
                records[q.category][vname].append(record)
                if verbose:
                    bits = [f"[{q.id}] {q.category} variant={vname}"]
                    if q.category == "negation":
                        bits.append(f"abstained={record['abstained']}")
                    else:
                        if "entity_hits" in record:
                            bits.append(f"eh@1={record['entity_hits'][1]} eMRR={record['mrr_entity']:.3f}")
                        if "file_hits" in record:
                            bits.append(f"fh@1={record['file_hits'][1]} fMRR={record['mrr_file']:.3f}")
                        if "coverage" in record:
                            bits.append(f"cov={record['coverage']:.2f}")
                        bits.append(f"described={record['all_described']}")
                    bits.append(f"tok={record['context_tokens']}/{record['raw_tokens10']}")
                    print("  " + " ".join(bits))

        for category in CATEGORIES:
            cat_questions = [q for q in scored_questions if q.category == category]
            if not cat_questions:
                continue
            rows = {v: aggregate(records[category][v]) for v in variant_names}
            applicability = {
                "entity": sum(1 for q in cat_questions if any(t.qualified_name is not None for t in q.expect)),
                "file": sum(1 for q in cat_questions if any(_target_path(t, index) is not None for t in q.expect)),
                "coverage": len(cat_questions) if category == "flow" else 0,
                "negation": len(cat_questions) if category == "negation" else 0,
            }
            _print_category_table(category, variant_names, rows, applicability)
    finally:
        # Cleanup runs even if an exception blew up mid-run (a bad variant,
        # a crashed report, a partial _seed_fixture failure, ...) — a
        # seeded fixture must never leak into the live shared collection.
        if fixture_seeded:
            # MED-1(d): belt and suspenders on top of load_dataset()'s
            # project==PROJECT_NAME check — cleanup ALWAYS targets
            # eval_fixture's own constant, never dataset.project, so this
            # can never be tricked (or mistakenly edited) into deleting an
            # arbitrary project. Import is cheap here: eval_fixture is
            # already in sys.modules from _seed_fixture (or load_dataset).
            import eval_fixture

            if keep_fixture:
                print(f"  --keep-fixture: leaving seeded chunks under project={eval_fixture.PROJECT_NAME!r}")
            else:
                removed = delete_project_chunks(client, eval_fixture.PROJECT_NAME)
                print(
                    f"  fixture cleanup: removed {removed} chunk(s) "
                    f"from project={eval_fixture.PROJECT_NAME!r}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset", action="append", default=None,
        help="Dataset JSON path (repeatable; default: glob scripts/eval_dataset_*.json)",
    )
    parser.add_argument(
        "--variants", default=None,
        help=f"Comma-separated variant names to run (default: all — {','.join(VARIANTS)})",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-question rows")
    parser.add_argument(
        "--keep-fixture", action="store_true",
        help="Skip cleanup of a seeded fixture dataset's chunks (for manual inspection)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_paths = args.dataset or sorted(glob.glob(DEFAULT_DATASET_GLOB))
    if not dataset_paths:
        print(f"No datasets found (default glob: {DEFAULT_DATASET_GLOB}); pass --dataset PATH")
        sys.exit(1)

    variant_names = args.variants.split(",") if args.variants else list(VARIANTS)
    # LOW-2: dedupe order-preservingly — "--variants bm25,bm25" must not
    # double-count rows or corrupt the "(N scored)" header.
    seen: set[str] = set()
    variant_names = [v for v in variant_names if not (v in seen or seen.add(v))]
    unknown = [v for v in variant_names if v not in VARIANTS]
    if unknown:
        print(f"Unknown variant(s): {unknown}; available: {list(VARIANTS)}")
        sys.exit(1)

    client = get_client()
    try:
        retriever = Retriever(client)
        for path in dataset_paths:
            run_dataset(
                path, client, retriever, variant_names,
                verbose=args.verbose, keep_fixture=args.keep_fixture,
            )
    finally:
        client.close()


if __name__ == "__main__":
    main()
