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
    chunk_text: str = ""

    def build_text(self) -> str:
        lines: list[str] = []
        lines.append(f"[{self.entity_type}] {self.qualified_name}")
        lines.append(f"Name: {self.name}")
        if self.file_path:
            loc = f"File: {self.file_path}"
            if self.start_line is not None and self.end_line is not None:
                loc += f" (lines {self.start_line}-{self.end_line})"
            lines.append(loc)
        if self.module_name:
            lines.append(f"Module: {self.module_name}")
        if self.parent_class:
            lines.append(f"Class: {self.parent_class}")
        if self.decorators:
            lines.append(f"Decorators: {', '.join(self.decorators)}")
        lines.append(f"Exported: {'yes' if self.is_exported else 'no'}")
        if self.defined_methods:
            lines.append(f"Methods: {', '.join(self.defined_methods)}")
        if self.inherits:
            lines.append(f"Inherits: {', '.join(self.inherits)}")
        if self.imports:
            lines.append(f"Imports: {', '.join(self.imports)}")
        if self.implements:
            lines.append(f"Implements: {', '.join(self.implements)}")
        if self.overrides:
            lines.append(f"Overrides: {', '.join(self.overrides)}")
        if self.calls:
            lines.append(f"Calls: {', '.join(self.calls)}")
        if self.called_by:
            lines.append(f"Called by: {', '.join(self.called_by)}")
        self.chunk_text = "\n".join(lines)
        return self.chunk_text


def chunk_graph(json_path: str | Path) -> list[CodeChunk]:
    with open(json_path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    nodes: list[dict] = data.get("nodes", [])
    relationships: list[dict] = data.get("relationships", [])

    node_by_id: dict[int, dict] = {n["node_id"]: n for n in nodes}

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
        )
        chunk.build_text()
        chunks.append(chunk)

    return chunks
