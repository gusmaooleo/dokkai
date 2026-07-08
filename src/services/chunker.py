"""
Smart chunking engine for code-graph-rag output.

Reads the graph JSON (nodes + relationships) produced by code-graph-rag and
creates one rich-text chunk per code entity (Class, Function, Method,
Interface, Enum, Type).  Structural nodes (Project, Folder, File, Module,
Package, ExternalPackage) are used as contextual metadata but are NOT
chunked themselves.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Node labels that become individual chunks
# ---------------------------------------------------------------------------
CODE_ENTITY_LABELS = {"Class", "Function", "Method", "Interface", "Enum", "Type"}

# Hard cap on the source snippet embedded per chunk, so a huge class body
# cannot dominate the vector or blow up the context budget downstream.
_MAX_SOURCE_CHARS = 6_000

# Hard cap on the extracted docstring / leading comment surfaced per chunk.
_MAX_DOC_CHARS = 600

_CALLS = "CALLS"
_DEFINES = "DEFINES"
_DEFINES_METHOD = "DEFINES_METHOD"
_INHERITS = "INHERITS"
_IMPLEMENTS = "IMPLEMENTS"
_OVERRIDES = "OVERRIDES"
_IMPORTS = "IMPORTS"
_CONTAINS_FILE = "CONTAINS_FILE"
_CONTAINS_MODULE = "CONTAINS_MODULE"


# ---------------------------------------------------------------------------
# Zero-cost text enrichment (Tier 1): identifier tokenization + doc extraction
# ---------------------------------------------------------------------------
# Natural-language queries ("how does the alarm flow work?") embed far from raw
# code identifiers. Splitting `sendDetectionNotification` into "send detection
# notification" and surfacing any hand-written docstring/JSDoc gives the
# embedder real NL tokens to match against — with no extra LLM calls.

_CAMEL_BOUNDARY_1 = re.compile(r"([A-Z]+)([A-Z][a-z])")   # HTTPServer  -> HTTP Server
_CAMEL_BOUNDARY_2 = re.compile(r"([a-z\d])([A-Z])")        # sendAlarm   -> send Alarm
_ID_LEAF_SPLIT = re.compile(r"[.:#/\\]")
_COMMENT_MARKER = re.compile(r"^(//+|#+|\*+|/\*+)\s?")


def _split_identifier(identifier: str) -> list[str]:
    """Decompose a (possibly qualified) identifier into lowercase words."""
    leaf = _ID_LEAF_SPLIT.split(identifier)[-1].split("(")[0]
    leaf = leaf.replace("_", " ").replace("-", " ")
    leaf = _CAMEL_BOUNDARY_1.sub(r"\1 \2", leaf)
    leaf = _CAMEL_BOUNDARY_2.sub(r"\1 \2", leaf)
    return [w.lower() for w in leaf.split() if w]


def _keyword_terms(chunk: "CodeChunk", *, max_terms: int = 24) -> str:
    """
    Build a comma-separated list of natural-language phrases from the entity's
    own name, its parent class, its methods, and (multi-word) callees — so the
    behaviour described by those identifiers becomes searchable in plain words.
    """
    seen: set[str] = set()
    out: list[str] = []

    def add(identifier: str, *, require_phrase: bool = False) -> None:
        words = _split_identifier(identifier)
        if not words or (require_phrase and len(words) < 2):
            return
        phrase = " ".join(words)
        if phrase not in seen:
            seen.add(phrase)
            out.append(phrase)

    add(chunk.name)
    if chunk.parent_class:
        add(chunk.parent_class)
    for method in chunk.defined_methods:
        add(method)
    for callee in chunk.calls:
        if len(out) >= max_terms:
            break
        add(callee, require_phrase=True)  # callees only add NL signal when they decompose
    return ", ".join(out[:max_terms])


def _is_comment_line(line: str) -> bool:
    return bool(line) and (line.startswith(("//", "#", "*", "/*")) or line.endswith("*/"))


def _leading_comment(
    file_lines: list[str] | None, start_line: int | None, max_scan: int = 40
) -> str:
    """Collect the contiguous comment block directly above a declaration."""
    if not file_lines or not start_line:
        return ""
    # line directly above the declaration (0-indexed), clamped in case the
    # recorded start_line exceeds the file on disk (stale path / changed file)
    i = min(start_line - 2, len(file_lines) - 1)
    # bridge over decorators / blank lines sitting between the doc and the decl
    while i >= 0 and (not file_lines[i].strip() or file_lines[i].strip().startswith("@")):
        i -= 1
    block: list[str] = []
    scanned = 0
    while i >= 0 and scanned < max_scan:
        stripped = file_lines[i].strip()
        if _is_comment_line(stripped):
            block.append(stripped)
            i -= 1
            scanned += 1
        else:
            break
    block.reverse()
    return "\n".join(block)


def _clean_comment(text: str) -> str:
    """Strip comment fences/markers, leaving plain prose on a single line."""
    text = text.replace("/**", "").replace("/*", "").replace("*/", "")
    out: list[str] = []
    for line in text.splitlines():
        line = _COMMENT_MARKER.sub("", line.strip()).strip()
        if line:
            out.append(line)
    return " ".join(out).strip()


def _python_docstring(source_code: str) -> str:
    """
    Extract a docstring only when it is the entity's *own* first statement, so a
    class with no docstring isn't mis-assigned its first method's docstring.
    """
    lines = source_code.splitlines()
    i = 0
    while i < len(lines) and lines[i].lstrip().startswith("@"):  # skip decorators
        i += 1
    while i < len(lines) and not lines[i].rstrip().endswith(":"):  # past the signature
        i += 1
    i += 1  # first body line
    while i < len(lines) and not lines[i].strip():  # skip blanks
        i += 1
    if i >= len(lines):
        return ""
    body = "\n".join(lines[i:]).lstrip()
    for quote in ('"""', "'''"):
        if body.startswith(quote):
            end = body[len(quote):].find(quote)
            if end != -1:
                return " ".join(body[len(quote):len(quote) + end].split())
    return ""


def _extract_doc(
    file_lines: list[str] | None,
    start_line: int | None,
    source_code: str,
    file_key: str,
) -> str:
    """
    Best-effort natural-language description of an entity, pulled from source:
    a leading comment/JSDoc block (any language) and a Python docstring.
    """
    parts: list[str] = []
    lead = _clean_comment(_leading_comment(file_lines, start_line))
    if lead:
        parts.append(lead)
    if file_key.endswith(".py"):
        docstring = _python_docstring(source_code)
        if docstring:
            parts.append(" ".join(docstring.split()))
    doc = " ".join(parts).strip()
    if len(doc) > _MAX_DOC_CHARS:
        doc = doc[:_MAX_DOC_CHARS].rstrip() + " …"
    return doc


@dataclass
class CodeChunk:
    node_id: int
    entity_type: str
    name: str
    qualified_name: str
    imports: list[str]
    file_path: str
    absolute_path: str
    start_line: int | None
    end_line: int | None
    project_name: str
    module_name: str
    parent_class: str | None
    decorators: list[str] = field(default_factory=list)
    is_exported: bool = False
    calls: list[str] = field(default_factory=list)
    called_by: list[str] = field(default_factory=list)
    inherits: list[str] = field(default_factory=list)
    implements: list[str] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)
    defined_methods: list[str] = field(default_factory=list)
    source_code: str = ""
    doc: str = ""
    description: str = ""  # Tier 2: LLM-generated natural-language summary
    chunk_text: str = ""

    def build_text(self) -> str:
        """
        Build the text that gets embedded. A compact metadata header (signature,
        location, graph relations) followed by the *actual source code* — the
        source is what carries the real semantic signal for retrieval.
        """
        header: list[str] = [f"[{self.entity_type}] {self.qualified_name}"]
        if self.file_path:
            loc = f"File: {self.file_path}"
            if self.start_line is not None and self.end_line is not None:
                loc += f" (lines {self.start_line}-{self.end_line})"
            header.append(loc)
        if self.parent_class:
            header.append(f"Class: {self.parent_class}")
        if self.decorators:
            header.append(f"Decorators: {', '.join(self.decorators)}")

        rels: list[str] = []
        if self.defined_methods:
            rels.append(f"Methods: {', '.join(self.defined_methods)}")
        if self.inherits:
            rels.append(f"Inherits: {', '.join(self.inherits)}")
        if self.implements:
            rels.append(f"Implements: {', '.join(self.implements)}")
        if self.overrides:
            rels.append(f"Overrides: {', '.join(self.overrides)}")
        if self.calls:
            rels.append(f"Calls: {', '.join(self.calls)}")
        if self.called_by:
            rels.append(f"Called by: {', '.join(self.called_by)}")
        if rels:
            header.append(" | ".join(rels))

        terms = _keyword_terms(self)
        if terms:
            header.append(f"Terms: {terms}")

        parts = ["\n".join(header)]
        if self.doc:
            parts.append(f"Doc: {self.doc}")
        if self.source_code:
            parts.append(f"```\n{self.source_code}\n```")
        self.chunk_text = "\n\n".join(parts)
        return self.chunk_text


def chunk_graph(
    json_path: str | Path,
    source_root: str | Path | None = None,
) -> list[CodeChunk]:
    """
    Turn a code-graph-rag JSON into rich chunks.

    Parameters
    ----------
    json_path:
        Path to the code-graph-rag output JSON.
    source_root:
        Repository root used to read the actual source code for each entity.
        When given, files are read from ``source_root / <relative path>``;
        otherwise the absolute path recorded in the graph is used. Files are
        read once and cached.
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    nodes: list[dict] = data.get("nodes", [])
    relationships: list[dict] = data.get("relationships", [])

    node_by_id: dict[int, dict] = {n["node_id"]: n for n in nodes}

    # --- source code reading (cached per file) ----------------------------
    file_cache: dict[str, list[str] | None] = {}

    def _source_file(props: dict) -> str:
        rel_path = props.get("path")
        if source_root and rel_path:
            return str(Path(source_root) / rel_path)
        return props.get("absolute_path", "") or ""

    def _read_source(file_key: str, start: int | None, end: int | None) -> str:
        if not file_key or start is None or end is None:
            return ""
        if file_key not in file_cache:
            try:
                with open(file_key, "r", encoding="utf-8", errors="replace") as sfh:
                    file_cache[file_key] = sfh.read().splitlines()
            except OSError:
                file_cache[file_key] = None
        lines_ = file_cache[file_key]
        if not lines_:
            return ""
        snippet = "\n".join(lines_[max(start - 1, 0):end])
        if len(snippet) > _MAX_SOURCE_CHARS:
            snippet = snippet[:_MAX_SOURCE_CHARS] + "\n… [truncated]"
        return snippet

    project_name = ""
    for n in nodes:
        if "Project" in n["labels"]:
            project_name = n["properties"].get("name", "")
            break

    outgoing: dict[int, list[tuple[str, int]]] = {}
    incoming: dict[int, list[tuple[str, int]]] = {}

    for rel in relationships:
        fid = rel["from_id"]
        tid = rel["to_id"]
        rtype = rel["type"]
        outgoing.setdefault(fid, []).append((rtype, tid))
        incoming.setdefault(tid, []).append((rtype, fid))

    def _qname(nid: int) -> str:
        n = node_by_id.get(nid)
        if n is None:
            return f"<unknown:{nid}>"
        props = n.get("properties", {})
        return props.get("qualified_name", props.get("name", str(nid)))

    def _name(nid: int) -> str:
        n = node_by_id.get(nid)
        if n is None:
            return f"<unknown:{nid}>"
        return n.get("properties", {}).get("name", str(nid))

    chunks: list[CodeChunk] = []

    for node in nodes:
        labels = set(node["labels"])
        if not labels & CODE_ENTITY_LABELS:
            continue

        props = node.get("properties", {})
        nid = node["node_id"]
        entity_type = (labels & CODE_ENTITY_LABELS).pop()

        module_name = ""
        for rtype, from_id in incoming.get(nid, []):
            if rtype in (_DEFINES, _CONTAINS_MODULE, _CONTAINS_FILE):
                from_node = node_by_id.get(from_id)
                if from_node and {"Module", "File"} & set(from_node["labels"]):
                    module_name = _qname(from_id)
                    break

        parent_class: str | None = None
        if entity_type == "Method":
            for rtype, from_id in incoming.get(nid, []):
                if rtype == _DEFINES_METHOD:
                    parent_class = _qname(from_id)
                    break

        calls = [_qname(tid) for rt, tid in outgoing.get(nid, []) if rt == _CALLS]
        inherits = [_qname(tid) for rt, tid in outgoing.get(nid, []) if rt == _INHERITS]
        implements = [_qname(tid) for rt, tid in outgoing.get(nid, []) if rt == _IMPLEMENTS]
        overrides = [_qname(tid) for rt, tid in outgoing.get(nid, []) if rt == _OVERRIDES]
        imports = [_qname(tid) for rt, tid in outgoing.get(nid, []) if rt == _IMPORTS]
        defined_methods = [_name(tid) for rt, tid in outgoing.get(nid, []) if rt == _DEFINES_METHOD]

        called_by = [_qname(fid) for rt, fid in incoming.get(nid, []) if rt == _CALLS]

        file_key = _source_file(props)
        start_line = props.get("start_line")
        source_code = _read_source(file_key, start_line, props.get("end_line"))
        doc = _extract_doc(file_cache.get(file_key), start_line, source_code, file_key)

        chunk = CodeChunk(
            node_id=nid,
            entity_type=entity_type,
            name=props.get("name", ""),
            qualified_name=props.get("qualified_name", props.get("name", "")),
            file_path=props.get("path", ""),
            absolute_path=props.get("absolute_path", ""),
            start_line=props.get("start_line"),
            end_line=props.get("end_line"),
            project_name=project_name,
            module_name=module_name,
            parent_class=parent_class,
            decorators=props.get("decorators", []),
            is_exported=props.get("is_exported", False),
            calls=calls,
            called_by=called_by,
            inherits=inherits,
            implements=implements,
            overrides=overrides,
            imports=imports,
            defined_methods=defined_methods,
            source_code=source_code,
            doc=doc,
        )
        chunk.build_text()
        chunks.append(chunk)

    return chunks
