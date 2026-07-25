#!/usr/bin/env python
"""
Smoke test for the C5 fixes — this repo has no test framework yet, so this is
a plain assert-and-print script, mirroring ``scripts/test_retriever_seeding.py``.

Two defects fixed in this step:

  1. Hub pollution in the EMBEDDED text: ``CodeChunk.build_text`` now caps
     EACH relation list (Methods/Inherits/Implements/Overrides/Calls/Called
     by) at ``_MAX_RELATION_ITEMS`` in the rendered ``chunk_text`` (the
     STORED array properties on the object are untouched — this only affects
     what gets embedded/keyword-indexed).
  2. ``imports`` (already extracted by the chunker) is now persisted: a new
     ``imports`` TEXT_ARRAY property on the Weaviate schema, included in the
     upsert payload, migrated onto pre-existing collections via
     ``ensure_collection``'s additive ``add_property`` guard, and plumbed
     through ``RetrievedChunk``.

Covers:
  PURE (no live server needed):
    a. 20 calls + 20 called_by -> each embedded relation line shows exactly
       the first 16 items followed by ``(+4 more)``; items beyond the cap
       are absent from the embedded text.
    b. A relation list AT/UNDER the cap renders unchanged — no ``(+N more)``
       marker, all items present verbatim.
    c. A chunk with NO relations at all: no relation line is emitted for any
       of the six relation kinds (can't diff against pre-change output, so
       this asserts structure instead — matches the untouched pre-change
       behavior for the empty-list case).
    d. The FULL, uncapped lists remain on the ``CodeChunk`` object itself
       after ``build_text()`` runs — the cap is render-time only.

  LIVE (needs ``docker compose up -d`` — real Weaviate, real Ollama
  embeddings):
    e. ``ensure_collection`` against the real (already-existing) collection
       adds ``imports`` if missing; a probe chunk with
       ``imports=["pkg.a", "pkg.b"]`` upserted through
       ``services.weaviate_client.upsert_chunks`` and re-fetched through
       ``Retriever.get_by_qualified_name`` round-trips ``imports`` intact.
       Cleans up its probe project afterwards.
    f. Fresh-create path: a throwaway collection (via a ``COLLECTION_NAME``
       env override, in a subprocess — ``COLLECTION_NAME`` is read at
       import time, same reason ``RETRIEVAL_TEST_PENALTY`` needs a
       subprocess in ``test_retriever_seeding.py``) that does NOT exist yet
       is created fresh via ``ensure_collection`` — the property is present
       from creation (``_PROPERTIES`` already includes it). Drops the
       throwaway collection afterwards.
    g. Migration path: a throwaway collection is created WITHOUT the
       ``imports`` property (bypassing ``ensure_collection``, using
       ``_PROPERTIES`` minus ``imports``), then ``ensure_collection`` is run
       against it — the additive guard adds the property, and a subsequent
       upsert with ``imports`` succeeds and round-trips. Drops the throwaway
       collection afterwards.

Usage
-----
    uv run python scripts/test_chunker_caps.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

# Load .env BEFORE importing services — mirrors test_retriever_seeding.py:
# ensure_collection() below must see the user's real VECTORIZER_PROVIDER/
# EMBED_MODEL, not process-env defaults.
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from services.chunker import CodeChunk, _MAX_RELATION_ITEMS  # noqa: E402
from services.retriever import Retriever  # noqa: E402
from services.weaviate_client import (  # noqa: E402
    COLLECTION_NAME,
    delete_project_chunks,
    ensure_collection,
    get_client,
    upsert_chunks,
)

PROJECT = "_chunker_caps_probe"

_passed = 0


def check(label: str, condition: bool) -> None:
    global _passed
    if not condition:
        raise AssertionError(f"FAIL: {label}")
    _passed += 1
    print(f"PASS: {label}")


def _base_chunk(**kw) -> CodeChunk:
    defaults = dict(
        node_id=0, imports=[], file_path="pkg.py", absolute_path="/repo/pkg.py",
        start_line=1, end_line=5, project_name="p", module_name="pkg",
        parent_class=None, source_code="def f(): pass",
    )
    defaults.update(kw)
    return CodeChunk(**defaults)


# --- PURE tests --------------------------------------------------------


def test_relation_cap_over_limit() -> None:
    print("\n-- relation cap: 20 calls + 20 called_by -> capped at 16 with marker --")
    calls = [f"pkg.call_{i}" for i in range(20)]
    called_by = [f"pkg.caller_{i}" for i in range(20)]
    chunk = _base_chunk(
        entity_type="Function", name="hub_fn", qualified_name="pkg.hub_fn",
        calls=calls, called_by=called_by,
    )
    text = chunk.build_text()
    print(text)

    check("_MAX_RELATION_ITEMS is 16", _MAX_RELATION_ITEMS == 16)
    check(
        "Calls: line shows exactly the first 16 items + '(+4 more)'",
        f"Calls: {', '.join(calls[:16])} (+4 more)" in text,
    )
    check(
        "Called by: line shows exactly the first 16 items + '(+4 more)'",
        f"Called by: {', '.join(called_by[:16])} (+4 more)" in text,
    )
    # Note the precise claim: the qualified-name SPELLINGS beyond the cap are
    # gone from the relation lines. Tokenized forms of capped callees may
    # still appear in `Terms:` by design — _keyword_terms reads the FULL
    # lists (bounded by its own 24-term cap), and called_by never feeds it.
    check(
        "beyond-cap qualified names are absent from the relation lines (calls)",
        all(item not in text for item in calls[16:]),
    )
    check(
        "beyond-cap qualified names are absent from the relation lines (called_by)",
        all(item not in text for item in called_by[16:]),
    )
    check("items within the cap ARE present (calls)", all(item in text for item in calls[:16]))
    check("items within the cap ARE present (called_by)", all(item in text for item in called_by[:16]))


def test_relation_no_cap_when_under_limit() -> None:
    print("\n-- relation cap: lists at/under the cap render unchanged, no marker --")
    methods = [f"method_{i}" for i in range(5)]
    inherits = ["Base1", "Base2"]
    implements_exact_cap = [f"Iface{i}" for i in range(_MAX_RELATION_ITEMS)]  # exactly 16
    chunk = _base_chunk(
        entity_type="Class", name="Widget", qualified_name="pkg.Widget",
        defined_methods=methods, inherits=inherits, implements=implements_exact_cap,
        source_code="class Widget: pass",
    )
    text = chunk.build_text()
    print(text)

    check("Methods: line lists all 5 methods, unchanged", f"Methods: {', '.join(methods)}" in text)
    check("Inherits: line lists both bases, unchanged", f"Inherits: {', '.join(inherits)}" in text)
    check(
        "Implements: line lists all 16 (exactly at the cap), unchanged",
        f"Implements: {', '.join(implements_exact_cap)}" in text,
    )
    check("no '(+N more)' marker present for any list at/under the cap", "more)" not in text)


def test_no_relations_unaffected() -> None:
    print("\n-- no relations at all: no relation line emitted for any kind --")
    chunk = _base_chunk(
        entity_type="Function", name="lonely_fn", qualified_name="pkg.lonely_fn",
    )
    text = chunk.build_text()
    print(text)
    for label in ("Methods:", "Inherits:", "Implements:", "Overrides:", "Calls:", "Called by:"):
        check(f"no {label!r} line when the chunk has no such relation", label not in text)


def test_full_lists_preserved_on_object() -> None:
    print("\n-- cap is render-time only: full lists remain on the CodeChunk object --")
    calls = [f"pkg.call_{i}" for i in range(20)]
    chunk = _base_chunk(
        entity_type="Function", name="hub_fn2", qualified_name="pkg.hub_fn2", calls=calls,
    )
    chunk.build_text()
    check(
        "CodeChunk.calls retains all 20 items after build_text (render-time cap only)",
        chunk.calls == calls and len(chunk.calls) == 20,
    )


# --- LIVE tests ----------------------------------------------------------


def test_live_imports_roundtrip() -> None:
    print("\n-- LIVE: imports property on the real collection + upsert round-trip --")
    client = get_client()
    try:
        ensure_collection(client)  # idempotent; adds `imports` if this deployment predates it
        schema_names = {
            p.name for p in client.collections.get(COLLECTION_NAME).config.get().properties
        }
        check("real collection schema has the 'imports' property", "imports" in schema_names)

        probe = CodeChunk(
            node_id=1, entity_type="Function", name="probe_fn",
            qualified_name=f"{PROJECT}.probe_fn", imports=["pkg.a", "pkg.b"],
            file_path="probe.py", absolute_path="/repo/probe.py",
            start_line=1, end_line=2, project_name=PROJECT, module_name="probe",
            parent_class=None, source_code="def probe_fn(): pass",
        )
        probe.build_text()
        inserted = upsert_chunks(client, [probe], ingestion_id="chunker-caps-probe-1")
        check("probe chunk inserted", inserted == 1)

        fetched = Retriever(client).get_by_qualified_name(probe.qualified_name, PROJECT)
        check("probe chunk fetched back", fetched is not None)
        check(
            "imports round-trip intact",
            fetched is not None and fetched.imports == ["pkg.a", "pkg.b"],
        )
    finally:
        removed = delete_project_chunks(client, PROJECT)
        print(f"cleanup: removed {removed} probe chunk(s) from project={PROJECT!r}")
        client.close()


def _run_subprocess(script: str, collection_name: str) -> dict:
    import os

    env = os.environ.copy()
    env["COLLECTION_NAME"] = collection_name
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"subprocess failed: {result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


_FRESH_CREATE_SCRIPT = f"""
import sys, json
sys.path.insert(0, {str(SRC)!r})
from dotenv import load_dotenv
load_dotenv()
import services.weaviate_client as wc

client = wc.get_client()
try:
    if client.collections.exists(wc.COLLECTION_NAME):
        client.collections.delete(wc.COLLECTION_NAME)
    wc.ensure_collection(client)  # collection does not exist yet -> the create() branch
    names = {{p.name for p in client.collections.get(wc.COLLECTION_NAME).config.get().properties}}
    print(json.dumps({{"has_imports": "imports" in names}}))
finally:
    if client.collections.exists(wc.COLLECTION_NAME):
        client.collections.delete(wc.COLLECTION_NAME)
    client.close()
"""


def test_live_fresh_create_path() -> None:
    print("\n-- LIVE: fresh-create path (throwaway collection, does not exist yet) --")
    result = _run_subprocess(_FRESH_CREATE_SCRIPT, "ChunkerCapsProbeFresh")
    print(f"fresh-create result: {result}")
    check("freshly-created collection HAS the 'imports' property", result["has_imports"])


_MIGRATION_SCRIPT = f"""
import sys, json
sys.path.insert(0, {str(SRC)!r})
from dotenv import load_dotenv
load_dotenv()
import services.weaviate_client as wc
from services.chunker import CodeChunk

client = wc.get_client()
try:
    if client.collections.exists(wc.COLLECTION_NAME):
        client.collections.delete(wc.COLLECTION_NAME)

    # Simulate a pre-existing collection that predates the `imports` property.
    props_without_imports = [p for p in wc._PROPERTIES if p.name != "imports"]
    client.collections.create(
        name=wc.COLLECTION_NAME,
        vectorizer_config=wc._named_vectors(),
        properties=props_without_imports,
    )
    pre_names = {{p.name for p in client.collections.get(wc.COLLECTION_NAME).config.get().properties}}

    # An object written BEFORE the migration (raw insert, no `imports` key —
    # upsert_chunks would now always write one), with the deterministic UUID
    # so Retriever.get_by_qualified_name can find it afterwards. This is the
    # population the migration serves: post-migration, Weaviate returns its
    # `imports` as key-present-with-None, which _to_chunk must tolerate.
    client.collections.get(wc.COLLECTION_NAME).data.insert(
        properties={{
            "entity_type": "Function", "name": "old_fn",
            "qualified_name": "migrate_probe.old_fn",
            "file_path": "old.py", "absolute_path": "/repo/old.py",
            "start_line": 1, "end_line": 2,
            "project_name": "_migration_probe", "module_name": "probe",
            "parent_class": "", "chunk_text": "[Function] migrate_probe.old_fn",
            "calls": [], "called_by": [], "inherits": [], "implements": [],
            "overrides": [], "defined_methods": [], "description": "",
            "ingestion_id": "pre-migrate-1",
        }},
        uuid=wc.uuid_for_entity("_migration_probe", "migrate_probe.old_fn"),
    )

    wc.ensure_collection(client)  # collection exists, lacks `imports` -> additive migration

    post_names = {{p.name for p in client.collections.get(wc.COLLECTION_NAME).config.get().properties}}

    # Finding-1 regression: the pre-migration object must read back through
    # Retriever._to_chunk without raising, imports as an empty list.
    from services.retriever import Retriever
    old = Retriever(client).get_by_qualified_name("migrate_probe.old_fn", "_migration_probe")
    pre_object_reads_back_clean = old is not None and old.imports == []

    chunk = CodeChunk(
        node_id=1, entity_type="Function", name="probe_fn",
        qualified_name="migrate_probe.probe_fn", imports=["pkg.a", "pkg.b"],
        file_path="probe.py", absolute_path="/repo/probe.py",
        start_line=1, end_line=2, project_name="_migration_probe", module_name="probe",
        parent_class=None, source_code="def probe_fn(): pass",
    )
    chunk.build_text()
    inserted = wc.upsert_chunks(client, [chunk], ingestion_id="migrate-probe-1")

    # Fetch by deterministic UUID, not a qualified_name filter: `equal` on a
    # word-tokenized TEXT prop matches by TOKEN, so "migrate_probe.probe_fn"
    # also matches old_fn (shared tokens) — limit=1 can return the WRONG
    # object. Same reason production get_by_qualified_name is UUID-based.
    coll = client.collections.get(wc.COLLECTION_NAME)
    obj = coll.query.fetch_object_by_id(
        wc.uuid_for_entity("_migration_probe", "migrate_probe.probe_fn"),
        return_properties=["imports"],
    )
    fetched_imports = list(obj.properties.get("imports") or []) if obj else None

    print(json.dumps({{
        "pre_had_imports": "imports" in pre_names,
        "post_has_imports": "imports" in post_names,
        "inserted": inserted,
        "fetched_imports": fetched_imports,
        "pre_object_reads_back_clean": pre_object_reads_back_clean,
    }}))
finally:
    if client.collections.exists(wc.COLLECTION_NAME):
        client.collections.delete(wc.COLLECTION_NAME)
    client.close()
"""


def test_live_migration_path() -> None:
    print("\n-- LIVE: migration path (existing collection lacking `imports`) --")
    result = _run_subprocess(_MIGRATION_SCRIPT, "ChunkerCapsProbeMigrate")
    print(f"migration result: {result}")
    check("pre-migration collection did NOT have 'imports'", not result["pre_had_imports"])
    check("post-migration collection HAS 'imports'", result["post_has_imports"])
    check("upsert with imports succeeded after migration", result["inserted"] == 1)
    check(
        "imports round-trip intact after migration",
        result["fetched_imports"] == ["pkg.a", "pkg.b"],
    )
    check(
        "PRE-migration object reads back through _to_chunk without raising "
        "(imports None -> [])",
        result["pre_object_reads_back_clean"],
    )


def main() -> None:
    test_relation_cap_over_limit()
    test_relation_no_cap_when_under_limit()
    test_no_relations_unaffected()
    test_full_lists_preserved_on_object()

    test_live_imports_roundtrip()
    test_live_fresh_create_path()
    test_live_migration_path()

    print(f"\n{_passed} checks PASSED")


if __name__ == "__main__":
    main()
