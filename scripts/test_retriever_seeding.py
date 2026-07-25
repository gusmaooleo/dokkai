#!/usr/bin/env python
"""
Smoke test for the C1+C2 retriever seeding fixes (``services.retriever``) —
this repo has no test framework yet, so this is a plain assert-and-print
script, mirroring ``scripts/test_review_analyze.py``. Runs against the REAL
Weaviate 1.28.4 (``docker compose up -d``).

Deviation from the original brief: the brief assumed a live corpus already
ingested under project ``saffira_back-end`` (~2770 chunks). At run time the
collection was empty (``GET /v1/schema`` -> zero classes — a fresh/reset
volume), and re-ingesting that project would run code-graph-rag + write into
``ingested/`` (off limits for this change). So this script seeds a small,
self-contained synthetic project directly through the same
``services.weaviate_client.upsert_chunks`` plumbing the real pipeline uses —
real server, real Ollama embeddings, real "no summary vector when
description is empty" code path — and cleans it up afterwards.

Covers:
  1. ``search_seeds`` (multi-target hybrid, C1 defect 1 fix) for a natural
     -language query and an identifier-ish query: non-empty, no duplicate
     qualified_names, len<=k, scores present.
  2. The single multi-target hybrid call executes and returns results that
     include an entity with NO ``summary`` vector (empty description)
     alongside one that has both vectors — confirming the fix doesn't drop
     vector-less entities.
  3. ``search_graph`` (built on top of ``search_seeds``) still works after
     ``_merge_seeds`` was removed.
  4. BM25 test-file penalty (C1 defect 3 fix): a test-path chunk with a
     stronger raw BM25 match than a non-test chunk on the same term ranks
     ABOVE it with ``RETRIEVAL_TEST_PENALTY=1.0`` (penalty disabled) but
     BELOW it with the default penalty — checked via two subprocess
     invocations, since the env var is read at import time.
  5. ``_looks_like_identifier`` (C2): the strong-signal heuristic returns
     True/False on a set of literal-identifier vs. natural-language probes.
  6. Identifier fast path (C2), mention exclusion: a mention-heavy CALLER
     whose body repeats an identifier more often than the identifier's own
     definition chunk must NOT out-rank — or even appear alongside — the
     definition.
  7. Identifier fast path (C2), separator-boundary guard: a differently
     -named entity whose qualified_name merely ENDS WITH the query as a raw
     substring (``queue.helpers._send_notification`` for query
     "send_notification") must NOT be admitted as a definition match.
  8. Identifier fast path (C2), score normalization: fast-path scores stay
     in 0..1 like every other path (the frontend source card renders
     ``score * 100``%; graph expansion assumes a bounded parent score) —
     checked across the fast, hybrid, and fallback paths.
  9. A natural-language query still takes the (unchanged) hybrid path.
  10. Fallback correctness (C2): an identifier-shaped query matching nothing
      definition-ish still returns real hybrid results on this corpus (not
      a vacuous "didn't crash" check).
  11. Strip idempotency (C2): a trailing newline (as an MCP client might
      send) must not silently disable the fast path.

Cleanup: deletes every chunk this script inserts (via
``delete_project_chunks``), even on failure.

Usage
-----
    uv run python scripts/test_retriever_seeding.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

# Load .env BEFORE importing services: retriever reads RETRIEVAL_TEST_PENALTY
# at import time, and ensure_collection() below must see the user's real
# VECTORIZER_PROVIDER/EMBED_MODEL — otherwise a fresh volume would get the
# collection created with process-env defaults and a later real ingest would
# silently inherit the wrong vectorizer.
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from services.chunker import CodeChunk  # noqa: E402
from services.retriever import Retriever, _looks_like_identifier  # noqa: E402
from services.weaviate_client import (  # noqa: E402
    delete_project_chunks,
    ensure_collection,
    get_client,
    upsert_chunks,
)

PROJECT = "_retriever_seeding_probe"

_passed = 0


def check(label: str, condition: bool) -> None:
    global _passed
    if not condition:
        raise AssertionError(f"FAIL: {label}")
    _passed += 1
    print(f"PASS: {label}")


def _chunk(**kw) -> CodeChunk:
    defaults = dict(
        node_id=0, imports=[], project_name=PROJECT, parent_class=None,
        source_code="", doc="", description="",
    )
    defaults.update(kw)
    return CodeChunk(**defaults)


def build_seed_chunks() -> list[CodeChunk]:
    return [
        # HAS a summary vector (real description) — natural-language seed.
        _chunk(
            node_id=1, entity_type="Function", name="authenticate",
            qualified_name="auth.login_service.authenticate",
            file_path="src/services/login_service.py",
            absolute_path="/repo/src/services/login_service.py",
            start_line=10, end_line=20, module_name="login_service",
            description="Authenticates a user given a username and password, "
                         "returning a signed session token.",
            calls=["auth.token_helper.validate_token"],
            chunk_text="[Function] authenticate\n"
                        "def authenticate(username, password):\n"
                        "    return validate_token(issue_token(username, password))",
        ),
        # NO description -> NO summary vector — reachable only via the code
        # vector / BM25 lane. Also the target of the `calls` edge above, so
        # search_graph has something to expand into.
        _chunk(
            node_id=2, entity_type="Function", name="validate_token",
            qualified_name="auth.token_helper.validate_token",
            file_path="src/services/token_helper.py",
            absolute_path="/repo/src/services/token_helper.py",
            start_line=1, end_line=8, module_name="token_helper",
            called_by=["auth.login_service.authenticate"],
            chunk_text="[Function] validate_token\n"
                        "def validate_token(token):\n"
                        "    return token in ISSUED_TOKENS",
        ),
        # Non-test chunk carrying the BM25 probe term ONCE.
        _chunk(
            node_id=3, entity_type="Function", name="frobnicate",
            qualified_name="widget_service.frobnicate",
            file_path="src/services/widget_service.py",
            absolute_path="/repo/src/services/widget_service.py",
            start_line=1, end_line=4, module_name="widget_service",
            chunk_text="[Function] frobnicate\n"
                        "def frobnicate(x):\n"
                        "    return frobnicate_zzqx(x)",
        ),
        # Test-path chunk carrying the SAME term repeated more times, so its
        # raw BM25 score is HIGHER than the non-test chunk's — the penalty
        # must invert that ordering.
        _chunk(
            node_id=4, entity_type="Function", name="test_frobnicate",
            qualified_name="tests.test_widget_service.test_frobnicate",
            file_path="tests/test_widget_service.py",
            absolute_path="/repo/tests/test_widget_service.py",
            start_line=1, end_line=6, module_name="test_widget_service",
            chunk_text="[Function] test_frobnicate\n"
                        "def test_frobnicate():\n"
                        "    # frobnicate_zzqx frobnicate_zzqx frobnicate_zzqx\n"
                        "    assert frobnicate_zzqx(1) == frobnicate_zzqx(1)",
        ),
        # Filler entities, unrelated to either query, so search_seeds'
        # top_k truncation is a real cut, not a vacuous len<=k.
        _chunk(
            node_id=5, entity_type="Function", name="log_message",
            qualified_name="logging.logger.log_message",
            file_path="src/services/logger.py",
            absolute_path="/repo/src/services/logger.py",
            start_line=1, end_line=3, module_name="logger",
            chunk_text="[Function] log_message\ndef log_message(msg):\n    print(msg)",
        ),
        _chunk(
            node_id=6, entity_type="Function", name="get_value",
            qualified_name="cache.store.get_value",
            file_path="src/services/cache.py",
            absolute_path="/repo/src/services/cache.py",
            start_line=1, end_line=3, module_name="cache",
            chunk_text="[Function] get_value\ndef get_value(key):\n    return CACHE.get(key)",
        ),
        # --- C2 identifier fast-path probe ---------------------------------
        # The DEFINITION: mentions its own identifier only twice (header +
        # signature).
        _chunk(
            node_id=7, entity_type="Function", name="send_notification",
            qualified_name="notifications.sender.send_notification",
            file_path="src/services/notification_sender.py",
            absolute_path="/repo/src/services/notification_sender.py",
            start_line=1, end_line=4, module_name="notification_sender",
            called_by=["notifications.batch.notify_all"],
            chunk_text="[Function] send_notification\n"
                        "def send_notification(msg):\n"
                        "    return DISPATCH.send(msg)",
        ),
        # A CALLER whose body repeats the identifier MORE OFTEN than the
        # definition itself — the raw BM25 term-frequency winner, but not a
        # definition, so the fast-path filter must exclude it.
        _chunk(
            node_id=8, entity_type="Function", name="notify_all",
            qualified_name="notifications.batch.notify_all",
            file_path="src/services/notification_batch.py",
            absolute_path="/repo/src/services/notification_batch.py",
            start_line=1, end_line=8, module_name="notification_batch",
            calls=["notifications.sender.send_notification"],
            chunk_text="[Function] notify_all\n"
                        "def notify_all(msgs):\n"
                        "    # send_notification send_notification send_notification"
                        " send_notification\n"
                        "    for msg in msgs:\n"
                        "        send_notification(msg)\n"
                        "    return len(msgs)",
        ),
        # A DIFFERENTLY-NAMED wrapper whose qualified_name merely ENDS WITH
        # the query as a raw substring — "queue.helpers._send_notification"
        # ends with "send_notification" character-for-character, so a naive
        # `.endswith(query_lower)` (no separator boundary) wrongly admits it.
        # Its body also repeats the bare term 3x, so it would win on raw BM25
        # term frequency too if not excluded structurally.
        _chunk(
            node_id=9, entity_type="Function", name="_send_notification",
            qualified_name="queue.helpers._send_notification",
            file_path="src/services/queue_helpers.py",
            absolute_path="/repo/src/services/queue_helpers.py",
            start_line=1, end_line=6, module_name="queue_helpers",
            chunk_text="[Function] _send_notification\n"
                        "def _send_notification(msg):\n"
                        "    # send_notification send_notification send_notification\n"
                        "    return QUEUE.push(msg)",
        ),
    ]


def check_looks_like_identifier() -> None:
    """No live server needed — pure heuristic, checked up front."""
    print("\n_looks_like_identifier probes:")
    for token, expected in [
        ("validate_token", True),
        ("TokenService.authenticate", True),
        ("camelCaseName", True),
        ("login", False),                # plain lowercase word -> semantic search
        ("how does auth work", False),    # natural language, has spaces
        ("a_b", True),                    # len 3 with underscore -> strong signal
        ("how_does auth", False),         # contains whitespace, disqualified
    ]:
        got = _looks_like_identifier(token)
        print(f"  {token!r:30} -> {got}")
        check(f"_looks_like_identifier({token!r}) == {expected}", got == expected)


def run_bm25_subprocess(penalty_env: str | None) -> list[list]:
    """Run search_bm25 for the probe term in a fresh subprocess (the test
    penalty is read from RETRIEVAL_TEST_PENALTY at import time, so comparing
    default vs disabled needs two separate interpreter processes)."""
    script = f"""
import sys, json
sys.path.insert(0, {str(SRC)!r})
from services.retriever import Retriever
from services.weaviate_client import get_client
client = get_client()
try:
    chunks = Retriever(client).search_bm25(
        "frobnicate_zzqx", project_name={PROJECT!r}, top_k=5,
    )
    print(json.dumps([[c.qualified_name, c.score] for c in chunks]))
finally:
    client.close()
"""
    env = os.environ.copy()
    if penalty_env is not None:
        env["RETRIEVAL_TEST_PENALTY"] = penalty_env
    else:
        env.pop("RETRIEVAL_TEST_PENALTY", None)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"subprocess failed: {result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> None:
    check_looks_like_identifier()

    client = get_client()
    try:
        ensure_collection(client)
        chunks = build_seed_chunks()
        inserted = upsert_chunks(client, chunks, ingestion_id="retriever-seeding-probe-1")
        print(f"seeded {inserted} synthetic chunk(s) under project={PROJECT!r}")
        check("all seed chunks inserted", inserted == len(chunks))

        retriever = Retriever(client)

        # --- 1) search_seeds: natural-language query -----------------------
        nl_results = retriever.search_seeds(
            "how does a user log in and get authenticated",
            project_name=PROJECT, top_k=3,
        )
        print(f"\nnatural-language seeds ({len(nl_results)}):")
        for c in nl_results:
            print(f"  score={c.score!r}  {c.qualified_name}")
        check("nl seeds non-empty", len(nl_results) > 0)
        check("nl seeds len <= top_k", len(nl_results) <= 3)
        nl_qnames = [c.qualified_name for c in nl_results]
        check("nl seeds no duplicate qualified_names", len(nl_qnames) == len(set(nl_qnames)))
        check("nl seeds all carry a score", all(c.score is not None for c in nl_results))

        # --- 1b) search_seeds: identifier-ish query -------------------------
        id_results = retriever.search_seeds(
            "validate_token", project_name=PROJECT, top_k=3,
        )
        print(f"\nidentifier-ish seeds ({len(id_results)}):")
        for c in id_results:
            print(f"  score={c.score!r}  {c.qualified_name}")
        check("identifier seeds non-empty", len(id_results) > 0)
        check("identifier seeds len <= top_k", len(id_results) <= 3)
        id_qnames = [c.qualified_name for c in id_results]
        check("identifier seeds no duplicate qualified_names", len(id_qnames) == len(set(id_qnames)))
        check("identifier seeds all carry a score", all(c.score is not None for c in id_results))

        # --- 2) multi-target hybrid includes the no-summary-vector entity --
        union_results = retriever.search_seeds(
            "token", project_name=PROJECT, top_k=6,
        )
        union_qnames = {c.qualified_name for c in union_results}
        print(f"\nunion seeds for 'token': {sorted(union_qnames)}")
        check(
            "multi-target hybrid returns the described entity (has summary vector)",
            "auth.login_service.authenticate" in union_qnames,
        )
        check(
            "multi-target hybrid ALSO returns the undescribed entity (no summary vector)",
            "auth.token_helper.validate_token" in union_qnames,
        )

        # --- 3) search_graph still works (built on search_seeds) -----------
        # top_k_seeds=1 so validate_token can ONLY appear via the calls edge
        # expansion (not as a seed) — a real test of graph traversal after
        # _merge_seeds was removed, not just seed overlap.
        graph_results = retriever.search_graph(
            "how does authentication work", project_name=PROJECT,
            top_k_seeds=1, max_hops=2, max_nodes=10,
        )
        print(f"\nsearch_graph results ({len(graph_results)}):")
        for c in graph_results:
            print(f"  hop={c.hop} via={c.via!r} score={c.score!r} {c.qualified_name}")
        check("search_graph returns results", len(graph_results) > 0)
        check("search_graph includes a seed (hop 0)", any(c.hop == 0 for c in graph_results))
        validate_token_node = next(
            (c for c in graph_results if c.qualified_name == "auth.token_helper.validate_token"),
            None,
        )
        check(
            "search_graph expanded to validate_token via the calls edge (hop >= 1)",
            validate_token_node is not None
            and validate_token_node.hop >= 1
            and "calls" in validate_token_node.via,
        )

        # --- 4) BM25 test-file penalty (defect 3) ---------------------------
        default_rows = run_bm25_subprocess(penalty_env=None)
        disabled_rows = run_bm25_subprocess(penalty_env="1.0")
        print(f"\nBM25 'frobnicate_zzqx' — default penalty: {default_rows}")
        print(f"BM25 'frobnicate_zzqx' — penalty disabled (1.0): {disabled_rows}")

        default_order = [row[0] for row in default_rows]
        disabled_order = [row[0] for row in disabled_rows]
        TEST_QN = "tests.test_widget_service.test_frobnicate"
        NONTEST_QN = "widget_service.frobnicate"

        # Membership first, so a missing qname fails as a labeled check
        # instead of a raw ValueError from list.index().
        check(
            "both chunks present in BM25 results (penalty disabled)",
            TEST_QN in disabled_order and NONTEST_QN in disabled_order,
        )
        check(
            "both chunks present in BM25 results (default penalty)",
            TEST_QN in default_order and NONTEST_QN in default_order,
        )
        check(
            "penalty disabled: test chunk (stronger raw match) ranks ABOVE non-test",
            disabled_order.index(TEST_QN) < disabled_order.index(NONTEST_QN),
        )
        check(
            "default penalty: non-test chunk now ranks ABOVE the test chunk",
            default_order.index(NONTEST_QN) < default_order.index(TEST_QN),
        )
        default_test_score = next(s for qn, s in default_rows if qn == TEST_QN)
        disabled_test_score = next(s for qn, s in disabled_rows if qn == TEST_QN)
        check(
            "default penalty measurably lowered the test chunk's score",
            default_test_score < disabled_test_score,
        )

        # --- 6) C2 identifier fast path: definition beats mention-heavy caller,
        # AND a same-suffix-but-different entity (separator-boundary guard)
        DEFINITION_QN = "notifications.sender.send_notification"
        CALLER_QN = "notifications.batch.notify_all"
        BOUNDARY_TRAP_QN = "queue.helpers._send_notification"
        fastpath_results = retriever.search_seeds(
            "send_notification", project_name=PROJECT, top_k=5,
        )
        print(f"\nfast-path seeds for 'send_notification' ({len(fastpath_results)}):")
        for c in fastpath_results:
            print(f"  score={c.score!r}  {c.qualified_name}")
        fastpath_qnames = [c.qualified_name for c in fastpath_results]
        check("fast path returns results", len(fastpath_results) > 0)
        check(
            "fast path ranks the DEFINITION first, not the mention-heavy caller",
            fastpath_qnames[0] == DEFINITION_QN,
        )
        check(
            "fast path excludes the mention-only caller entirely",
            CALLER_QN not in fastpath_qnames,
        )

        # --- 7) C2 separator-boundary guard: reviewer repro ------------------
        check(
            "fast path excludes the same-suffix-but-different entity "
            "(queue.helpers._send_notification)",
            BOUNDARY_TRAP_QN not in fastpath_qnames,
        )

        # --- 8) C2 score normalization: every path stays in 0..1 -------------
        # (hybrid_results / fallback_results computed in sections 9 / 10
        # below; checked together with fastpath_results once all three exist)

        # --- 9) C2 natural-language query still takes the hybrid path -------
        hybrid_results = retriever.search_seeds(
            "how does a user log in", project_name=PROJECT, top_k=3,
        )
        print(f"\nhybrid-path seeds for 'how does a user log in' ({len(hybrid_results)}):")
        for c in hybrid_results:
            print(f"  score={c.score!r}  {c.qualified_name}")
        check(
            "'how does a user log in' is NOT identifier-shaped",
            not _looks_like_identifier("how does a user log in"),
        )
        check("hybrid-path seeds non-empty", len(hybrid_results) > 0)
        check("hybrid-path seeds all carry a score", all(c.score is not None for c in hybrid_results))

        # --- 10) C2 fallback: identifier-shaped query, nothing definition-ish
        check(
            "'zz_nonexistent_zz' IS identifier-shaped",
            _looks_like_identifier("zz_nonexistent_zz"),
        )
        fallback_results = retriever.search_seeds(
            "zz_nonexistent_zz", project_name=PROJECT, top_k=3,
        )
        print(f"\nfallback seeds for 'zz_nonexistent_zz' ({len(fallback_results)}):")
        for c in fallback_results:
            print(f"  score={c.score!r}  {c.qualified_name}")
        check(
            "no-match identifier query falls back to hybrid and returns real results "
            "(not a vacuous isinstance check)",
            len(fallback_results) > 0,
        )

        # --- 8, continued) score normalization, now that all three exist ----
        all_scored = fastpath_results + hybrid_results + fallback_results
        check(
            "every returned seed score (fast path, hybrid, fallback) is <= 1.0",
            all((c.score or 0.0) <= 1.0 + 1e-9 for c in all_scored),
        )

        # --- 11) C2 strip idempotency: trailing newline == stripped query ---
        newline_query = "validate_token\n"
        check(
            "'validate_token\\n' is still identifier-shaped after stripping",
            _looks_like_identifier(newline_query),
        )
        newline_results = retriever.search_seeds(
            newline_query, project_name=PROJECT, top_k=3,
        )
        plain_results = retriever.search_seeds(
            "validate_token", project_name=PROJECT, top_k=3,
        )
        print(
            f"\n'validate_token\\n' seeds: "
            f"{[c.qualified_name for c in newline_results]}"
        )
        check(
            "trailing-newline query takes the identical fast path as the "
            "stripped query",
            [c.qualified_name for c in newline_results]
            == [c.qualified_name for c in plain_results],
        )

    finally:
        # Nested so the client closes even if the chunk cleanup itself raises.
        try:
            removed = delete_project_chunks(client, PROJECT)
            print(f"\ncleanup: removed {removed} probe chunk(s)")
        finally:
            client.close()

    print(f"\n{_passed} checks PASSED")


if __name__ == "__main__":
    main()
