#!/usr/bin/env python
"""
Smoke test for feature 21 (gitignore-aware ingestion, step C6) —
``services.ignore_rules`` — this repo has no test framework yet, so this is
a plain assert-and-print script, mirroring ``scripts/test_retriever_seeding.py``.

Covers:
  PURE (no live server needed):
    1. A real temp git repo (tracked source file, ``.gitignore``-d ``.next/``
       and ``secrets.txt``, a NOT-gitignored ``vendor.min.js`` caught only by
       the curated defaults, and a nested ``packages/app/.gitignore``) —
       ``classify_paths`` correctly buckets every path, the whole batch goes
       through exactly ONE ``git check-ignore`` subprocess (not one per
       path), and ``git_status == "ok"``.
    2. A non-git temp dir — defaults-only classification (git check-ignore
       fails open, ``files_filtered_gitignore == 0``, ``git_status ==
       "not-a-repo"``).
    3. ``filter_graph`` against a synthetic graph JSON shaped like the real
       cgr output (inspected from ``ingested/sample_back-end.json`` /
       ``ingested/tinyrepo.json``): removed nodes AND their dangling edges
       gone, kept nodes/edges intact, stats counts exact (incl.
       ``git_status``), a node whose path resolves OUTSIDE the repo root is
       left alone and counted (``nodes_outside_repo``) without crashing, and
       ``filtered["metadata"]``'s ``total_nodes``/``total_relationships`` are
       rewritten to the post-filter counts.
    4. ``services.weaviate_client._uuids_matching``/``_delete_by_uuids``
       against a fake collection: cursor-scan + exact client-side predicate
       collects the right uuids, and a per-batch ``failed > 0`` result RAISES
       with counts instead of silently under-reporting.
    5. Forced-failure probe for ``services.ingest._filter_and_promote_inplace``
       (monkeypatched ``filter_graph`` raising): the still-unfiltered
       timestamped file is gone afterwards — never lingers as the "newest
       candidate" for ``find_latest_graph_json``.

  LIVE (needs ``docker compose up -d`` — real Weaviate):
    6. Stale-chunk purge: 3 synthetic chunks upserted under ingestion_id A,
       then 2 of them re-upserted under ingestion_id B — the purge (built
       into ``services.weaviate_client.upsert_chunks``, feature 21 item 4)
       removes exactly the 1 stale (A-only) chunk; the 2 B chunks remain.
       Probe project cleaned up afterwards.
    7. Cross-project isolation: a SECOND probe project whose name shares
       every token of the first (``PROJECT + "-extra"``) — live-confirmed
       WORD-tokenized ``equal`` matches by token-superset, so a naive
       ``project_name`` filter alone would let a re-ingest of the first
       project's purge (or a ``delete_project_chunks`` call) wipe the
       second's chunks too. Both code paths must leave the other project
       fully intact.

Also re-runs (regression) ``test_retriever_seeding.py``, ``test_chunker_caps.py``,
and ``test_context_assembly.py`` as subprocesses and reports their outcome —
this script's own exit code is non-zero if any of them fail.

Usage
-----
    uv run python scripts/test_gitignore_ingest.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# Imports sanity for every module feature 21 touched.
from services.chunker import CodeChunk  # noqa: E402
from services.ignore_rules import classify_paths, filter_graph  # noqa: E402
from services import ingest  # noqa: E402
from services import jobs  # noqa: E402, F401
from services import vectorize  # noqa: E402, F401
from services.weaviate_client import (  # noqa: E402
    COLLECTION_NAME,
    _delete_by_uuids,
    _uuids_matching,
    delete_project_chunks,
    ensure_collection,
    fetch_project_chunk_index,
    get_client,
    upsert_chunks,
)

PROJECT = "_gitignore_ingest_probe"
# Deliberately shares EVERY token of PROJECT (hyphen-appended) — the exact
# shape of the live-confirmed cross-project contamination bug (finding 1):
# WORD-tokenized `equal(PROJECT)` matches PROJECT_EXTRA too (token superset).
PROJECT_EXTRA = f"{PROJECT}-extra"

_passed = 0


def check(label: str, condition: bool) -> None:
    global _passed
    if not condition:
        raise AssertionError(f"FAIL: {label}")
    _passed += 1
    print(f"PASS: {label}")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# 1) real git repo — full classification + batching
# ---------------------------------------------------------------------------


def test_git_repo_classification() -> None:
    print("\n-- classify_paths against a real temp git repo --")
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "Test")

        (repo / ".gitignore").write_text(".next/\nsecrets.txt\n")
        (repo / "src").mkdir()
        (repo / "src" / "app.py").write_text("print('hi')\n")
        (repo / ".next").mkdir()
        (repo / ".next" / "build.js").write_text("// build\n")
        (repo / "secrets.txt").write_text("shh\n")
        (repo / "vendor.min.js").write_text("// min\n")

        nested = repo / "packages" / "app"
        nested.mkdir(parents=True)
        (nested / ".gitignore").write_text("local.env\n")
        (nested / "local.env").write_text("SECRET=1\n")
        (nested / "index.py").write_text("x = 1\n")

        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "init")

        rel_paths = [
            "src/app.py",
            ".next/build.js",
            "secrets.txt",
            "vendor.min.js",
            "packages/app/local.env",
            "packages/app/index.py",
        ]

        call_count = {"n": 0}
        real_run = subprocess.run

        def counting_run(*args, **kwargs):
            argv = args[0] if args else kwargs.get("args")
            if argv[:2] == ["git", "check-ignore"]:
                call_count["n"] += 1
            return real_run(*args, **kwargs)

        with mock.patch("services.ignore_rules.subprocess.run", side_effect=counting_run):
            excluded, stats = classify_paths(str(repo), rel_paths)

        check("tracked source file kept", "src/app.py" not in excluded)
        check(
            ".next/build.js excluded (default dir segment — caught before git is even asked)",
            ".next/build.js" in excluded,
        )
        check("secrets.txt excluded (gitignore)", "secrets.txt" in excluded)
        check(
            "vendor.min.js excluded (curated default glob, NOT gitignored)",
            "vendor.min.js" in excluded,
        )
        check(
            "nested packages/app/.gitignore respected (local.env excluded)",
            "packages/app/local.env" in excluded,
        )
        check("packages/app/index.py kept", "packages/app/index.py" not in excluded)
        check(
            "exactly ONE git check-ignore subprocess for the whole batch (not N)",
            call_count["n"] == 1,
        )
        check("files_filtered_defaults == 2 (.next/build.js, vendor.min.js)",
              stats["files_filtered_defaults"] == 2)
        check("files_filtered_gitignore == 2 (secrets.txt, packages/app/local.env)",
              stats["files_filtered_gitignore"] == 2)
        check("git_status == 'ok' (git ran successfully against a real repo)",
              stats["git_status"] == "ok")


# ---------------------------------------------------------------------------
# 2) non-git dir — defaults-only
# ---------------------------------------------------------------------------


def test_non_git_defaults_only() -> None:
    print("\n-- classify_paths against a non-git temp dir (defaults-only) --")
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        (repo / "src").mkdir()
        (repo / "src" / "app.py").write_text("x = 1\n")
        (repo / "dist").mkdir()
        (repo / "dist" / "bundle.js").write_text("// bundle\n")
        (repo / "readme.min.js").write_text("// min\n")

        rel_paths = ["src/app.py", "dist/bundle.js", "readme.min.js"]
        excluded, stats = classify_paths(str(repo), rel_paths)

        check("non-git: normal source file kept", "src/app.py" not in excluded)
        check("non-git: dist/ dir segment excluded via defaults",
              "dist/bundle.js" in excluded)
        check("non-git: *.min.js basename excluded via defaults",
              "readme.min.js" in excluded)
        check("non-git: files_filtered_defaults == 2", stats["files_filtered_defaults"] == 2)
        check(
            "non-git: files_filtered_gitignore == 0 (git check-ignore fails open, exit 128)",
            stats["files_filtered_gitignore"] == 0,
        )
        check(
            "non-git: git_status == 'not-a-repo' (distinguishable from a real git error)",
            stats["git_status"] == "not-a-repo",
        )


# ---------------------------------------------------------------------------
# 3) filter_graph — synthetic graph JSON shaped like the real cgr output
# ---------------------------------------------------------------------------


def test_filter_graph_synthetic() -> None:
    print("\n-- filter_graph against a synthetic cgr-shaped graph JSON --")
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = tmp

        nodes = [
            {"node_id": 0, "labels": ["Project"], "properties": {"name": "proj"}},
            {
                "node_id": 1,
                "labels": ["Folder"],
                "properties": {"path": "src", "absolute_path": f"{repo_root}/src", "name": "src"},
            },
            {
                "node_id": 2,
                "labels": ["File"],
                "properties": {
                    "path": "src/app.py",
                    "absolute_path": f"{repo_root}/src/app.py",
                    "name": "app.py",
                    "extension": ".py",
                },
            },
            {
                "node_id": 3,
                "labels": ["Function"],
                "properties": {
                    "path": "src/app.py",
                    "absolute_path": f"{repo_root}/src/app.py",
                    "name": "main",
                    "qualified_name": "proj.src.app.main",
                    "start_line": 1,
                    "end_line": 3,
                    "is_exported": True,
                    "decorators": [],
                },
            },
            {
                "node_id": 4,
                "labels": ["Folder"],
                "properties": {"path": ".next", "absolute_path": f"{repo_root}/.next", "name": ".next"},
            },
            {
                "node_id": 5,
                "labels": ["File"],
                "properties": {
                    "path": ".next/build.js",
                    "absolute_path": f"{repo_root}/.next/build.js",
                    "name": "build.js",
                    "extension": ".js",
                },
            },
            {
                "node_id": 6,
                "labels": ["Function"],
                "properties": {
                    "path": ".next/build.js",
                    "absolute_path": f"{repo_root}/.next/build.js",
                    "name": "webpackBootstrap",
                    "qualified_name": "proj.next.build.webpackBootstrap",
                    "start_line": 1,
                    "end_line": 5,
                    "is_exported": False,
                    "decorators": [],
                },
            },
            {"node_id": 7, "labels": ["ExternalPackage"], "properties": {"name": "requests"}},
            {
                "node_id": 8,
                "labels": ["File"],
                "properties": {"path": "/etc/hosts", "absolute_path": "/etc/hosts", "name": "hosts"},
            },
        ]
        relationships = [
            {"from_id": 0, "to_id": 1, "type": "CONTAINS_FOLDER", "properties": {}},
            {"from_id": 1, "to_id": 2, "type": "CONTAINS_FILE", "properties": {}},
            {"from_id": 2, "to_id": 3, "type": "DEFINES", "properties": {}},
            {"from_id": 0, "to_id": 4, "type": "CONTAINS_FOLDER", "properties": {}},
            {"from_id": 4, "to_id": 5, "type": "CONTAINS_FILE", "properties": {}},
            {"from_id": 5, "to_id": 6, "type": "DEFINES", "properties": {}},
            {"from_id": 3, "to_id": 7, "type": "DEPENDS_ON_EXTERNAL", "properties": {}},
            {"from_id": 8, "to_id": 3, "type": "CALLS", "properties": {}},
        ]
        data = {
            "nodes": nodes,
            "relationships": relationships,
            "metadata": {
                "total_nodes": len(nodes),
                "total_relationships": len(relationships),
                "exported_at": "2026-01-01T00:00:00+00:00",
            },
        }

        filtered, stats = filter_graph(data, repo_root)

        kept_ids = {n["node_id"] for n in filtered["nodes"]}
        check(
            "kept: Project, Folder(src), File(app.py), Function(main), "
            "ExternalPackage, File(/etc/hosts, outside repo)",
            kept_ids == {0, 1, 2, 3, 7, 8},
        )
        check(
            ".next Folder/File/Function all removed",
            {4, 5, 6}.isdisjoint(kept_ids),
        )
        check("stats.nodes_removed == 3", stats["nodes_removed"] == 3)
        check(
            "stats.files_filtered_defaults == 2 (.next, .next/build.js — deduped across "
            "their nodes)",
            stats["files_filtered_defaults"] == 2,
        )
        check("stats.files_filtered_gitignore == 0 (non-git repo_root)",
              stats["files_filtered_gitignore"] == 0)

        remaining_pairs = {(r["from_id"], r["to_id"]) for r in filtered["relationships"]}
        check(
            "dangling edges into/out of removed nodes are gone",
            {(0, 4), (4, 5), (5, 6)}.isdisjoint(remaining_pairs),
        )
        check(
            "edges among kept nodes survive intact",
            {(0, 1), (1, 2), (2, 3), (3, 7), (8, 3)} <= remaining_pairs,
        )
        check("stats.edges_removed == 3", stats["edges_removed"] == 3)
        check(
            "node with path outside repo_root left alone (kept) and counted",
            8 in kept_ids and stats["nodes_outside_repo"] == 1,
        )
        check(
            "stats.git_status == 'not-a-repo' (non-git repo_root, 'src'/'src/app.py' "
            "still get checked and git reports not-a-repo)",
            stats["git_status"] == "not-a-repo",
        )
        check(
            "filtered metadata.total_nodes rewritten to the post-filter node count",
            filtered["metadata"]["total_nodes"] == len(kept_ids) == 6,
        )
        check(
            "filtered metadata.total_relationships rewritten to the post-filter edge count",
            filtered["metadata"]["total_relationships"] == len(remaining_pairs) == 5,
        )
        check(
            "other metadata keys (exported_at) pass through unchanged",
            filtered["metadata"]["exported_at"] == "2026-01-01T00:00:00+00:00",
        )


# ---------------------------------------------------------------------------
# 4) pure — _uuids_matching / _delete_by_uuids against a fake collection
# ---------------------------------------------------------------------------


class _FakeObj:
    def __init__(self, uuid: str, properties: dict) -> None:
        self.uuid = uuid
        self.properties = properties


class _FakeDeleteManyReturn:
    def __init__(self, successful: int, failed: int) -> None:
        self.successful = successful
        self.failed = failed


class _FakeDataOps:
    def __init__(self, results: list) -> None:
        self._results = iter(results)
        self.calls: list = []

    def delete_many(self, where):
        self.calls.append(where)
        return next(self._results)


class _FakeCollection:
    def __init__(self, objects: list, delete_results: list) -> None:
        self._objects = objects
        self.data = _FakeDataOps(delete_results)

    def iterator(self, return_properties=None):
        return iter(self._objects)


def test_delete_helpers_pure() -> None:
    print("\n-- pure: _uuids_matching / _delete_by_uuids against a fake collection --")

    u1 = "00000000-0000-0000-0000-000000000001"
    u2 = "00000000-0000-0000-0000-000000000002"
    u3 = "00000000-0000-0000-0000-000000000003"

    objects = [
        _FakeObj(u1, {"project_name": "p", "ingestion_id": "A"}),
        _FakeObj(u2, {"project_name": "p", "ingestion_id": "B"}),
        _FakeObj(u3, {"project_name": "OTHER", "ingestion_id": "A"}),
    ]
    fake = _FakeCollection(objects, delete_results=[])
    matched = _uuids_matching(
        fake,
        return_properties=["project_name", "ingestion_id"],
        predicate=lambda props: props.get("project_name") == "p" and props.get("ingestion_id") != "B",
    )
    check("_uuids_matching: exact predicate matches only u1 (not u2's B, not OTHER's u3)",
          matched == [u1])

    fake_ok = _FakeCollection([], delete_results=[_FakeDeleteManyReturn(successful=2, failed=0)])
    result = _delete_by_uuids(fake_ok, [u1, u2], context="test-ok")
    check("_delete_by_uuids: returns the successful count when nothing failed", result == 2)

    fake_fail = _FakeCollection([], delete_results=[_FakeDeleteManyReturn(successful=2, failed=1)])
    try:
        _delete_by_uuids(fake_fail, [u1, u2, u3], context="test-fail")
        check("_delete_by_uuids: raises when failed > 0", False)
    except RuntimeError as e:
        check("_delete_by_uuids: raises RuntimeError mentioning the failed/total counts",
              "1/3" in str(e) and "test-fail" in str(e))

    check("_delete_by_uuids: empty uuid list is a no-op (0, no delete_many call)",
          _delete_by_uuids(fake_ok, [], context="unused") == 0)


# ---------------------------------------------------------------------------
# 5) pure — forced-failure probe for ingest._filter_and_promote_inplace
# ---------------------------------------------------------------------------


def test_forced_filter_failure() -> None:
    print("\n-- pure: forced filter_graph failure never leaves an unfiltered candidate --")
    with tempfile.TemporaryDirectory() as tmp:
        output_json = str(Path(tmp) / "proj-20260101T000000.json")
        with open(output_json, "w") as f:
            json.dump({"nodes": [], "relationships": [], "metadata": {}}, f)
        tmp_json = output_json + ".filtering.tmp"

        with mock.patch("services.ingest.filter_graph", side_effect=RuntimeError("boom")):
            try:
                ingest._filter_and_promote_inplace(output_json, tmp)
                check("forced filter_graph failure propagates", False)
            except RuntimeError as e:
                check("forced filter_graph failure propagates", str(e) == "boom")

        check(
            "the still-unfiltered timestamped candidate is removed on failure "
            "(never wins as the newest candidate for find_latest_graph_json)",
            not Path(output_json).exists(),
        )
        check("the temp file is also cleaned up on failure", not Path(tmp_json).exists())

        # Happy path, same helper: filtered payload atomically replaces output_json.
        with open(output_json, "w") as f:
            json.dump({"nodes": [], "relationships": [], "metadata": {}}, f)
        stats = ingest._filter_and_promote_inplace(output_json, tmp)
        check("happy path: output_json exists after a successful filter", Path(output_json).exists())
        check("happy path: no leftover temp file", not Path(tmp_json).exists())
        check("happy path: returns the filter stats dict", "git_status" in stats)


# ---------------------------------------------------------------------------
# 6) LIVE — stale-chunk purge on re-ingest
# ---------------------------------------------------------------------------


def _chunk(**kw) -> CodeChunk:
    defaults = dict(
        node_id=0, imports=[], project_name=PROJECT, parent_class=None,
        source_code="", doc="", description="",
    )
    defaults.update(kw)
    return CodeChunk(**defaults)


def test_live_purge() -> None:
    print("\n-- LIVE: stale-chunk purge (ingestion_id != current run) --")
    client = get_client()
    try:
        ensure_collection(client)

        chunks_a = [
            _chunk(
                node_id=i, entity_type="Function", name=f"fn{i}",
                qualified_name=f"probe.fn{i}", file_path="probe.py",
                absolute_path="/repo/probe.py", start_line=1, end_line=2,
                module_name="probe",
                chunk_text=f"[Function] fn{i}\ndef fn{i}(): pass",
            )
            for i in (1, 2, 3)
        ]
        inserted_a = upsert_chunks(client, chunks_a, ingestion_id="gitignore-ingest-probe-A")
        check("3 chunks inserted under ingestion_id A", inserted_a == 3)

        chunks_b = chunks_a[:2]  # fn1, fn2 re-upserted under B; fn3 becomes stale
        purge_stats = {"count": None}
        inserted_b = upsert_chunks(
            client,
            chunks_b,
            ingestion_id="gitignore-ingest-probe-B",
            on_purge=lambda n: purge_stats.__setitem__("count", n),
        )
        check("2 chunks (re-)inserted under ingestion_id B", inserted_b == 2)
        check("purge removed exactly the 1 stale chunk (fn3, ingestion_id A)",
              purge_stats["count"] == 1)

        index = fetch_project_chunk_index(client, PROJECT)
        remaining_qnames = {v["qualified_name"] for v in index.values()}
        check(
            "only fn1/fn2 (ingestion_id B) remain — fn3 (ingestion_id A) is gone",
            remaining_qnames == {"probe.fn1", "probe.fn2"},
        )
    finally:
        try:
            removed = delete_project_chunks(client, PROJECT)
            print(f"cleanup: removed {removed} probe chunk(s)")
        finally:
            client.close()


# ---------------------------------------------------------------------------
# 7) LIVE — cross-project isolation (purge + delete_project_chunks)
# ---------------------------------------------------------------------------


def test_live_cross_project_isolation() -> None:
    print("\n-- LIVE: cross-project isolation (purge + delete_project_chunks) --")
    print(f"   PROJECT={PROJECT!r}  PROJECT_EXTRA={PROJECT_EXTRA!r} (shares every token)")
    client = get_client()
    try:
        ensure_collection(client)

        main_chunk = _chunk(
            node_id=1, entity_type="Function", name="main_fn",
            qualified_name="probe.main_fn", file_path="probe.py",
            absolute_path="/repo/probe.py", start_line=1, end_line=2,
            module_name="probe", chunk_text="[Function] main_fn",
            project_name=PROJECT,
        )
        extra_chunk = _chunk(
            node_id=1, entity_type="Function", name="extra_fn",
            qualified_name="probe.extra_fn", file_path="probe.py",
            absolute_path="/repo/probe.py", start_line=1, end_line=2,
            module_name="probe", chunk_text="[Function] extra_fn",
            project_name=PROJECT_EXTRA,
        )
        upsert_chunks(client, [main_chunk], ingestion_id="iso-A")
        upsert_chunks(client, [extra_chunk], ingestion_id="iso-A")

        def exact_count(project_name: str) -> int:
            # _uuids_matching does an exact client-side match — unlike
            # fetch_project_chunk_index (still `equal`-filtered, out of this
            # step's scope), so this is reliable for BOTH project names,
            # regardless of which one's tokens are a subset of the other's.
            coll = client.collections.get(COLLECTION_NAME)
            return len(_uuids_matching(
                coll, return_properties=["project_name"],
                predicate=lambda props: props.get("project_name") == project_name,
            ))

        # 1) purge path: re-ingest PROJECT under a NEW ingestion_id with the
        # SAME chunk (nothing stale for PROJECT itself) — must NOT purge
        # PROJECT_EXTRA's chunk even though its project_name shares every
        # token with PROJECT.
        upsert_chunks(client, [main_chunk], ingestion_id="iso-B")
        check("purge: PROJECT still has its 1 chunk", exact_count(PROJECT) == 1)
        check(
            "purge: PROJECT_EXTRA's chunk untouched by PROJECT's purge "
            "(token-sharing project NOT wiped)",
            exact_count(PROJECT_EXTRA) == 1,
        )

        # 2) delete_project_chunks path: deleting PROJECT must not remove
        # PROJECT_EXTRA's chunk.
        removed = delete_project_chunks(client, PROJECT)
        check("delete_project_chunks(PROJECT) removed exactly its own 1 chunk", removed == 1)
        check(
            "delete_project_chunks(PROJECT): PROJECT_EXTRA's chunk survives",
            exact_count(PROJECT_EXTRA) == 1,
        )
        check("delete_project_chunks(PROJECT): PROJECT itself is now empty", exact_count(PROJECT) == 0)
    finally:
        try:
            r1 = delete_project_chunks(client, PROJECT)
            r2 = delete_project_chunks(client, PROJECT_EXTRA)
            print(f"cleanup: removed {r1} + {r2} probe chunk(s)")
        finally:
            client.close()


# ---------------------------------------------------------------------------
# Regression: re-run the other scripts this step touches transitively
# ---------------------------------------------------------------------------


def _run_regression_script(name: str) -> None:
    print(f"\n-- regression: scripts/{name} --")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / name)],
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    check(f"scripts/{name} exits 0", result.returncode == 0)


def main() -> None:
    test_git_repo_classification()
    test_non_git_defaults_only()
    test_filter_graph_synthetic()
    test_delete_helpers_pure()
    test_forced_filter_failure()
    test_live_purge()
    test_live_cross_project_isolation()

    _run_regression_script("test_retriever_seeding.py")
    _run_regression_script("test_chunker_caps.py")
    _run_regression_script("test_context_assembly.py")

    print(f"\n{_passed} checks PASSED")


if __name__ == "__main__":
    main()
