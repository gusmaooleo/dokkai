"use client";

/**
 * Top overlay controls, faithful to the prototype's graph toolbar
 * (`dokkai-ui-prototype.html` ~L271-291): search box with a results
 * dropdown top-left, zoom in/out/fit buttons + node/edge count top-right.
 * The Entities|Files segmented control (decision 12q — not in the
 * prototype, which only ever showed the entity graph) sits above the
 * search box in the same top-left cluster.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { Minus, Plus, Scan, Search } from "lucide-react";

import { cn } from "@/lib/utils";
import { kindColor } from "@/components/chat/kinds";
import type { GraphMode, NormalizedNode } from "./types";

const MAX_RESULTS = 12;

function fileBaseName(path: string | null): string {
  if (!path) return "";
  const idx = path.lastIndexOf("/");
  return idx === -1 ? path : path.slice(idx + 1);
}

function SearchBox({
  nodes,
  onSelectResult,
}: {
  nodes: NormalizedNode[];
  onSelectResult: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    const matches: NormalizedNode[] = [];
    for (const node of nodes) {
      if (node.name.toLowerCase().includes(q) || node.qualified_name?.toLowerCase().includes(q)) {
        matches.push(node);
        if (matches.length >= MAX_RESULTS) break;
      }
    }
    return matches;
  }, [nodes, query]);

  useEffect(() => {
    function onPointerDown(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, []);

  return (
    <div ref={containerRef} className="w-[min(300px,calc(100%-130px))]">
      <div className="relative">
        <Search
          size={16}
          strokeWidth={1.8}
          className="pointer-events-none absolute top-3 left-3 text-[color:var(--text-faint)]"
        />
        <input
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          placeholder="Search entities…"
          className="h-10 w-full rounded-[11px] border border-border-strong bg-surface pr-3 pl-9 text-sm text-foreground shadow-[var(--shadow-sm)] outline-none placeholder:text-[color:var(--text-faint)]"
        />
      </div>
      {open && results.length > 0 && (
        <div className="mt-1.5 overflow-hidden rounded-[11px] border border-border bg-surface p-[5px] shadow-[var(--shadow-lg)]">
          {results.map((node) => (
            <button
              key={node.id}
              type="button"
              onClick={() => {
                onSelectResult(node.id);
                setOpen(false);
              }}
              className="flex w-full items-center gap-[9px] rounded-lg px-[9px] py-2 text-left hover:bg-surface-2"
            >
              <span
                className="size-2 shrink-0 rounded-full"
                style={{ backgroundColor: kindColor(node.kind) }}
              />
              <span className="font-mono text-[12.5px] font-semibold text-foreground">{node.name}</span>
              <span className="ml-auto overflow-hidden font-mono text-[10.5px] text-ellipsis whitespace-nowrap text-[color:var(--text-faint)]">
                {fileBaseName(node.path)}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function ModeToggle({ mode, onModeChange }: { mode: GraphMode; onModeChange: (mode: GraphMode) => void }) {
  return (
    <div className="mb-2 inline-flex gap-0.5 rounded-[10px] border border-border bg-surface p-[3px] shadow-[var(--shadow-sm)]">
      {(["entities", "files"] as const).map((m) => (
        <button
          key={m}
          type="button"
          onClick={() => onModeChange(m)}
          className={cn(
            "h-7 rounded-lg px-3 text-[12px] font-semibold capitalize",
            mode === m ? "bg-[color:var(--accent-weak)] text-accent" : "text-muted-foreground",
          )}
        >
          {m}
        </button>
      ))}
    </div>
  );
}

export function GraphToolbar({
  mode,
  onModeChange,
  nodes,
  onSelectResult,
  onZoomIn,
  onZoomOut,
  onFit,
  nodeCount,
  edgeCount,
}: {
  mode: GraphMode;
  onModeChange: (mode: GraphMode) => void;
  nodes: NormalizedNode[];
  onSelectResult: (id: string) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFit: () => void;
  nodeCount: number;
  edgeCount: number;
}) {
  return (
    <>
      <div className="absolute top-4 left-4 z-[6]">
        <ModeToggle mode={mode} onModeChange={onModeChange} />
        <SearchBox nodes={nodes} onSelectResult={onSelectResult} />
      </div>
      <div className="absolute top-4 right-4 z-[6] flex flex-col items-end gap-2">
        <div className="flex flex-col overflow-hidden rounded-[11px] border border-border bg-surface shadow-[var(--shadow-sm)]">
          <button
            type="button"
            onClick={onZoomIn}
            title="Zoom in"
            className="grid size-[38px] place-items-center border-b border-border text-muted-foreground hover:bg-surface-2 hover:text-foreground"
          >
            <Plus size={17} strokeWidth={2} />
          </button>
          <button
            type="button"
            onClick={onZoomOut}
            title="Zoom out"
            className="grid size-[38px] place-items-center border-b border-border text-muted-foreground hover:bg-surface-2 hover:text-foreground"
          >
            <Minus size={17} strokeWidth={2} />
          </button>
          <button
            type="button"
            onClick={onFit}
            title="Fit to view"
            className="grid size-[38px] place-items-center text-muted-foreground hover:bg-surface-2 hover:text-foreground"
          >
            <Scan size={16} strokeWidth={2} />
          </button>
        </div>
        <div className="rounded-lg border border-border bg-surface px-[9px] py-[5px] font-mono text-[11px] text-muted-foreground shadow-[var(--shadow-sm)]">
          {nodeCount} nodes · {edgeCount} edges
        </div>
      </div>
    </>
  );
}
