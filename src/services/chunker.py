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

_CALLS = "CALLS"
_DEFINES = "DEFINES"
_DEFINES_METHOD = "DEFINES_METHOD"
_INHERITS = "INHERITS"
_IMPLEMENTS = "IMPLEMENTS"
_OVERRIDES = "OVERRIDES"
_IMPORTS = "IMPORTS"
_CONTAINS_FILE = "CONTAINS_FILE"
_CONTAINS_MODULE = "CONTAINS_MODULE"


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

        parts = ["\n".join(header)]
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

        source_code = _read_source(
            _source_file(props), props.get("start_line"), props.get("end_line")
        )

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
        )
        chunk.build_text()
        chunks.append(chunk)

    return chunks
