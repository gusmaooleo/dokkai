#!/usr/bin/env python
"""
Smoke test for the C3 context-assembly fixes in
``services.retriever.Retriever.build_graph_context`` — this repo has no test
framework yet, so this is a plain assert-and-print script, mirroring
``scripts/test_retriever_seeding.py``. PURE Python: no Weaviate needed —
``RetrievedChunk`` objects are constructed directly (some via
``services.chunker.CodeChunk.build_text()`` so the header-strip/Doc-prelude
fixes are exercised against the REAL chunk_text prelude format, not a
hand-rolled stand-in).

Covers:
  1. Packing (defect 1 + 3): three sections where #2 overflows the budget
     but smaller #3 still fits — #3 is included, #2 is skipped, "Source N"
     stays CONTIGUOUS for the kept sections (no gap left by the skip), and
     the final string's length matches the packing arithmetic EXACTLY
     (separators counted).
  2. Oversized first section alone (defect 2): included, truncated at a
     line boundary with the ``… [truncated]`` marker, result <= max_chars —
     the context is never empty just because the top section is huge. Also
     covers max_chars smaller than the marker itself (plain hard cut).
  3. Header dedup (defect 4) + entity identity: the rendered section
     contains the fenced source block exactly once, no second ``File:``
     line, no ``Terms:``/``Calls:`` prelude line, no fence-less fallback
     regression — but DOES keep the formatter's own ``Summary:`` line and
     states ``[entity_type] qualified_name`` exactly once in its header.
  4. Doc: prelude (defect 3, review round 2): an entity with an EMPTY
     description still surfaces its chunk_text's ``Doc:`` paragraph, and a
     ``Doc:`` paragraph that itself embeds a fenced example does not fool
     the fence detector into rendering from the wrong (inner) fence.
  5. Class/method containment (defect 5): (a) higher-scored container drops
     the contained chunk, (a2) UNLESS the container's own source was
     chunker-truncated (>_MAX_SOURCE_CHARS) — then both are kept so the
     contained chunk's code isn't silently lost, (b) higher-scored
     contained chunk drops the container, (c) identical ranges are NOT
     containment (both kept), (d) different files with overlapping line
     numbers are unrelated (both kept), (e) a chunk with no line range is
     exempt from containment entirely (both kept).
  6. Non-mutation: the caller's chunks list and each chunk's chunk_text are
     unchanged after build_graph_context (chat.py reuses the same list for
     the sources SSE payload).
  7. Empty chunks list: unchanged "(No relevant code found ...)" message.

Usage
-----
    uv run python scripts/test_context_assembly.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from services.chunker import CodeChunk, _MAX_SOURCE_CHARS  # noqa: E402
from services.retriever import Retriever, RetrievedChunk  # noqa: E402

_passed = 0


def check(label: str, condition: bool) -> None:
    global _passed
    if not condition:
        raise AssertionError(f"FAIL: {label}")
    _passed += 1
    print(f"PASS: {label}")


def _chunk(**kw) -> RetrievedChunk:
    return RetrievedChunk(**kw)


def test_packing() -> None:
    print("\n-- packing: skip-and-continue, contiguous numbering, separator-aware arithmetic --")
    chunk_a = _chunk(
        entity_type="Function", qualified_name="pkg.a", file_path="a.py",
        start_line=1, end_line=2, chunk_text="def a(): pass", hop=0,
        via="seed (matched query)",
    )
    chunk_b = _chunk(  # big — will overflow the remaining budget
        entity_type="Function", qualified_name="pkg.b", file_path="b.py",
        start_line=1, end_line=2, chunk_text="X" * 500, hop=1, via="calls ← a",
    )
    chunk_c = _chunk(  # small — still fits after b is skipped
        entity_type="Function", qualified_name="pkg.c", file_path="c.py",
        start_line=1, end_line=2, chunk_text="Y" * 50, hop=1, via="calls ← a",
    )

    section_a = Retriever._format_graph_section(1, chunk_a)
    section_b = Retriever._format_graph_section(2, chunk_b)
    # c ends up SECOND among kept sections (b is skipped), so its expected
    # rendering is numbered 2, not 3 — contiguous numbering (defect 6).
    section_c = Retriever._format_graph_section(2, chunk_c)
    check("fixture: section_b (oversized) is bigger than section_c (small)",
          len(section_b) > len(section_c))

    max_chars = len(section_a) + 2 + len(section_c)  # exactly a + separator + c

    original_list = [chunk_a, chunk_b, chunk_c]
    chunks_arg = list(original_list)
    retriever = Retriever.__new__(Retriever)  # no client needed for pure formatting
    result = retriever.build_graph_context(chunks_arg, max_chars=max_chars)

    check("section b (overflowing) is skipped", "b.py" not in result)
    check("section c (smaller, still fits) is included despite coming after b",
          "c.py" in result)
    check("section a (top score) is included", "a.py" in result)
    check("kept sections are numbered contiguously (Source 1, Source 2 — no gap)",
          "Source 1 " in result and "Source 2 " in result and "Source 3" not in result)
    expected = section_a + "\n\n" + section_c
    check("assembled context matches exact expected packing (a + separator + c)",
          result == expected)
    check("assembled length matches max_chars exactly (separators counted)",
          len(result) == max_chars)


def test_oversized_first_section() -> None:
    print("\n-- oversized first section: truncated at a line boundary --")
    lines = [f"line {i}" for i in range(200)]
    chunk = _chunk(
        entity_type="Function", qualified_name="pkg.big", file_path="big.py",
        start_line=1, end_line=200, chunk_text="\n".join(lines), hop=0,
        via="seed (matched query)",
    )
    full_section = Retriever._format_graph_section(1, chunk)
    max_chars = 300
    check("fixture: full section actually overflows max_chars", len(full_section) > max_chars)

    retriever = Retriever.__new__(Retriever)
    result = retriever.build_graph_context([chunk], max_chars=max_chars)

    check("result respects max_chars", len(result) <= max_chars)
    marker = "\n… [truncated]"
    check("result carries the truncation marker", result.endswith(marker))
    body = result[: -len(marker)]
    check("truncated body is an exact prefix of the untruncated section",
          full_section.startswith(body))
    check("cut lands exactly on a line boundary (next original char is a newline)",
          full_section[len(body)] == "\n")
    check("context is non-empty despite the oversized section (never dropped)",
          result != "")


def test_oversized_first_section_budget_below_marker() -> None:
    print("\n-- oversized first section: max_chars smaller than the marker itself --")
    chunk = _chunk(
        entity_type="Function", qualified_name="pkg.big", file_path="big.py",
        start_line=1, end_line=5, chunk_text="line one\nline two\nline three", hop=0,
        via="seed (matched query)",
    )
    retriever = Retriever.__new__(Retriever)
    result = retriever.build_graph_context([chunk], max_chars=5)
    check("max_chars=5 (below marker length): result still respects the budget",
          len(result) <= 5)


def test_header_dedup() -> None:
    print("\n-- header dedup + entity identity: chunk_text's own prelude is stripped --")

    foo = CodeChunk(
        node_id=1, entity_type="Function", name="foo", qualified_name="mod.foo",
        imports=[], file_path="mod.py", absolute_path="/repo/mod.py",
        start_line=5, end_line=10, project_name="p", module_name="mod",
        parent_class=None, calls=["mod.bar"], source_code="def foo():\n    return bar()\n",
    )
    foo_text = foo.build_text()
    foo_chunk = _chunk(
        entity_type="Function", qualified_name="mod.foo", file_path="mod.py",
        absolute_path="/repo/mod.py", start_line=5, end_line=10,
        chunk_text=foo_text, description="A helper that does X.", hop=0,
        via="seed (matched query)",
    )

    bar = CodeChunk(
        node_id=2, entity_type="Method", name="bar", qualified_name="mod.Bar.bar",
        imports=[], file_path="mod.py", absolute_path="/repo/mod.py",
        start_line=20, end_line=22, project_name="p", module_name="mod",
        parent_class="mod.Bar", decorators=["staticmethod"],
        source_code="@staticmethod\ndef bar():\n    return 1\n",
    )
    bar_text = bar.build_text()
    bar_chunk = _chunk(
        entity_type="Method", qualified_name="mod.Bar.bar", file_path="mod.py",
        absolute_path="/repo/mod.py", start_line=20, end_line=22,
        chunk_text=bar_text, description="Returns a constant.", hop=1, via="calls ← foo",
    )

    for label, chunk, summary in [
        ("foo", foo_chunk, "Summary: A helper that does X."),
        ("bar", bar_chunk, "Summary: Returns a constant."),
    ]:
        section = Retriever._format_graph_section(1, chunk)
        print(f"--- rendered section ({label}) ---\n{section}\n")
        check(f"[{label}] fenced source block appears exactly once",
              section.count("```") == 2)
        check(f"[{label}] no duplicated 'File:' line from chunk_text's own prelude",
              section.count("File:") == 1)
        check(f"[{label}] chunk_text's 'Terms:' prelude line is stripped",
              "Terms:" not in section)
        check(f"[{label}] chunk_text's 'Calls:'/relations prelude line is stripped",
              "Calls:" not in section)
        check(f"[{label}] formatter's own Summary line is kept", summary in section)
        check(f"[{label}] entity identity ([type] qualified_name) appears exactly once",
              section.count(chunk.qualified_name) == 1)
        check(f"[{label}] header states the entity_type",
              f"[{chunk.entity_type}]" in section.splitlines()[0])


def test_fence_less_fallback() -> None:
    print("\n-- defensive fallback: chunk_text with no fence at all --")
    chunk = _chunk(
        entity_type="Function", qualified_name="pkg.raw", file_path="raw.py",
        start_line=1, end_line=1, chunk_text="just some raw text, no code fence here",
        hop=0, via="seed (matched query)",
    )
    section = Retriever._format_graph_section(1, chunk)
    check("fence-less chunk_text is rendered as-is (no crash, no data loss)",
          "just some raw text, no code fence here" in section)
    check("no spurious Doc: line is invented for fence-less text",
          "Doc:" not in section)


def test_doc_prelude() -> None:
    print("\n-- Doc: prelude surfaced when description is empty --")

    ts_source = CodeChunk(
        node_id=3, entity_type="Function", name="formatCurrency",
        qualified_name="lib.money.formatCurrency", imports=[],
        file_path="lib/money.ts", absolute_path="/repo/lib/money.ts",
        start_line=1, end_line=3, project_name="p", module_name="money",
        parent_class=None, doc="Formats a currency amount for display.",
        source_code="function formatCurrency(x) {\n  return `$${x.toFixed(2)}`;\n}\n",
    )
    ts_text = ts_source.build_text()
    ts_chunk = _chunk(
        entity_type="Function", qualified_name="lib.money.formatCurrency",
        file_path="lib/money.ts", absolute_path="/repo/lib/money.ts",
        start_line=1, end_line=3, chunk_text=ts_text, description="", hop=0,
        via="seed (matched query)",
    )
    section = Retriever._format_graph_section(1, ts_chunk)
    print(f"--- rendered section (formatCurrency) ---\n{section}\n")
    check("Doc: text is present when description is empty",
          "Formats a currency amount for display." in section)
    check("Doc: line is explicitly labeled", "Doc: " in section)
    check("no Summary: line is emitted when description is empty",
          "Summary:" not in section)

    # A Doc paragraph that embeds its OWN fenced example must not fool the
    # fence detector into treating the example's fence as the real source
    # fence (defect 4).
    doc_with_example = "Use it like:\n```js\nformatCurrency(5)\n```"
    ts_with_example = CodeChunk(
        node_id=4, entity_type="Function", name="formatCurrency",
        qualified_name="lib.money.formatCurrency", imports=[],
        file_path="lib/money.ts", absolute_path="/repo/lib/money.ts",
        start_line=1, end_line=3, project_name="p", module_name="money",
        parent_class=None, doc=doc_with_example,
        source_code="function formatCurrency(x) { return x; }",
    )
    text_with_example = ts_with_example.build_text()
    chunk_with_example = _chunk(
        entity_type="Function", qualified_name="lib.money.formatCurrency",
        file_path="lib/money.ts", absolute_path="/repo/lib/money.ts",
        start_line=1, end_line=3, chunk_text=text_with_example, description="",
        hop=0, via="seed (matched query)",
    )
    section2 = Retriever._format_graph_section(1, chunk_with_example)
    print(f"--- rendered section (formatCurrency w/ fenced example in Doc) ---\n{section2}\n")
    check("Doc paragraph (including its embedded example) is kept intact",
          doc_with_example in section2)
    check("real source is rendered from the REAL fence, not the inner example's",
          "```\nfunction formatCurrency(x) { return x; }\n```" in section2)
    check("the inner example fence is not duplicated into a second source block",
          section2.count("```") == 4)  # 2 for the inner example, 2 for the real source


def test_containment() -> None:
    print("\n-- class/method containment de-duplication --")

    def cls(score: float, start=1, end=50, file_path="c.py", chunk_text="class C: ...") -> RetrievedChunk:
        return _chunk(
            entity_type="Class", qualified_name="pkg.C", file_path=file_path,
            start_line=start, end_line=end, chunk_text=chunk_text, score=score,
        )

    def meth(score: float, start=10, end=20, file_path="c.py") -> RetrievedChunk:
        return _chunk(
            entity_type="Method", qualified_name="pkg.C.m", file_path=file_path,
            start_line=start, end_line=end, chunk_text="def m(self): ...", score=score,
        )

    # (a) higher-scored container, NOT truncated -> contained method is dropped
    c_a, m_a = cls(0.9), meth(0.5)
    out_a = Retriever._drop_contained([c_a, m_a])
    check("(a) higher-scored, non-truncated class keeps its slot", c_a in out_a)
    check("(a) lower-scored contained method is dropped", m_a not in out_a)

    # (a2) higher-scored container, but its source WAS chunker-truncated ->
    # the contained method's code is not guaranteed present in the class's
    # chunk_text, so BOTH must be kept (the reviewer's 'def pay_invoice
    # vanished' repro).
    big_source = "\n".join(f"    # line {n}" for n in range(1, 3000))
    check("fixture: synthetic class source exceeds chunker's _MAX_SOURCE_CHARS",
          len(big_source) > _MAX_SOURCE_CHARS)
    truncated_source = big_source[:_MAX_SOURCE_CHARS] + "\n… [truncated]"
    truncated_class_src = CodeChunk(
        node_id=10, entity_type="Class", name="Invoicer", qualified_name="pkg.C",
        imports=[], file_path="c.py", absolute_path="/repo/c.py",
        start_line=1, end_line=704, project_name="p", module_name="pkg",
        parent_class=None, source_code=truncated_source,
    )
    truncated_class_text = truncated_class_src.build_text()
    c_a2 = cls(0.9, start=1, end=704, chunk_text=truncated_class_text)
    m_a2 = meth(0.5, start=650, end=660)
    check("fixture: real build_text() output is detected as truncated",
          Retriever._is_source_truncated(c_a2.chunk_text))
    out_a2 = Retriever._drop_contained([c_a2, m_a2])
    check("(a2) truncated container is kept", c_a2 in out_a2)
    check("(a2) contained method is KEPT despite lower score — container's "
          "source was truncated, so its code isn't guaranteed present",
          m_a2 in out_a2)

    # (b) higher-scored contained method -> container class is dropped
    # (unconditional — this direction never risks losing code)
    c_b, m_b = cls(0.4), meth(0.8)
    out_b = Retriever._drop_contained([c_b, m_b])
    check("(b) higher-scored contained method keeps its slot", m_b in out_b)
    check("(b) lower-scored container class is dropped", c_b not in out_b)

    # (c) identical ranges -> NOT containment, both kept
    c_c, m_c = cls(0.9, start=1, end=50), meth(0.5, start=1, end=50)
    out_c = Retriever._drop_contained([c_c, m_c])
    check("(c) identical ranges: container kept", c_c in out_c)
    check("(c) identical ranges: 'contained' kept too (not real containment)", m_c in out_c)

    # (d) different files, overlapping line numbers -> unrelated, both kept
    c_d, m_d = cls(0.9, file_path="c.py"), meth(0.5, file_path="other.py")
    out_d = Retriever._drop_contained([c_d, m_d])
    check("(d) different files: class kept", c_d in out_d)
    check("(d) different files: method kept (no cross-file containment)", m_d in out_d)

    # (e) a chunk with no line range at all is exempt from containment,
    # even if it nominally "overlaps" a fully-ranged chunk on the same file.
    c_e = cls(0.9)
    rangeless = _chunk(
        entity_type="Function", qualified_name="pkg.mystery", file_path="c.py",
        start_line=None, end_line=None, chunk_text="???", score=0.99,
    )
    out_e = Retriever._drop_contained([c_e, rangeless])
    check("(e) fully-ranged chunk kept", c_e in out_e)
    check("(e) range-less chunk kept (guard: containment needs both ranges)",
          rangeless in out_e)


def test_no_mutation() -> None:
    print("\n-- non-mutation: caller's chunks list and chunk_text survive untouched --")
    chunk_a = _chunk(
        entity_type="Function", qualified_name="pkg.a", file_path="a.py",
        start_line=1, end_line=2, chunk_text="def a(): pass", hop=0,
        via="seed (matched query)", score=0.9,
    )
    chunk_b = _chunk(
        entity_type="Method", qualified_name="pkg.C.b", file_path="a.py",
        start_line=1, end_line=2, chunk_text="def b(self): pass", hop=1,
        via="calls ← a", score=0.4,
    )
    chunks = [chunk_a, chunk_b]
    original_order = list(chunks)
    original_ids = [id(c) for c in chunks]
    original_texts = [c.chunk_text for c in chunks]

    retriever = Retriever.__new__(Retriever)
    retriever.build_graph_context(chunks, max_chars=10_000)

    check("caller's list object still holds the same chunk objects, in the same order",
          [id(c) for c in chunks] == original_ids and chunks == original_order)
    check("chunk_text on every original chunk is byte-for-byte unchanged",
          [c.chunk_text for c in chunks] == original_texts)


def test_empty() -> None:
    print("\n-- empty chunks list --")
    retriever = Retriever.__new__(Retriever)
    result = retriever.build_graph_context([])
    check("empty chunks preserve the existing placeholder message",
          result == "(No relevant code found in the codebase.)")


def main() -> None:
    test_packing()
    test_oversized_first_section()
    test_oversized_first_section_budget_below_marker()
    test_header_dedup()
    test_fence_less_fallback()
    test_doc_prelude()
    test_containment()
    test_no_mutation()
    test_empty()
    print(f"\n{_passed} checks PASSED")


if __name__ == "__main__":
    main()
