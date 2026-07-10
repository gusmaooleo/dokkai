"use client";

/**
 * One retrieved chunk in the sources panel, faithful to the prototype's
 * source card (`dokkai-ui-prototype.html` ~L238-260): kind + hop/via
 * badges, score, qualified name, split path, and an expandable panel with
 * the description and the raw chunk code.
 */

import Link from "next/link";
import { ChevronRight } from "lucide-react";

import { cn } from "@/lib/utils";
import type { SourceChunk } from "@/lib/types";
import { kindColor } from "./kinds";

function splitPath(path: string): { dir: string; file: string } {
  const idx = path.lastIndexOf("/");
  if (idx === -1) return { dir: "", file: path };
  return { dir: path.slice(0, idx + 1), file: path.slice(idx + 1) };
}

function hopLabel(hop: number, via: string): string {
  // hop 0 is a seed match — the backend's `via` for those is just
  // "seed (matched query)" (`src/services/retriever.py:266`), redundant
  // with the "seed" pill itself, so only append `via` for hop >= 1.
  if (hop === 0) return "seed";
  return via ? `hop ${hop} · via ${via}` : `hop ${hop}`;
}

export function SourceCard({
  source,
  open,
  onToggle,
}: {
  source: SourceChunk;
  open: boolean;
  onToggle: () => void;
}) {
  const color = kindColor(source.entity_type);
  const { dir, file } = splitPath(source.file_path);
  const isSeed = source.hop === 0;
  const scorePct = source.score != null ? `${Math.round(source.score * 100)}%` : "—";
  const lines =
    source.start_line != null && source.end_line != null
      ? `lines ${source.start_line}–${source.end_line}`
      : "";

  return (
    <div
      className={cn(
        "mb-2.5 overflow-hidden rounded-xl border bg-surface shadow-[var(--shadow-sm)] transition-colors",
        open ? "border-border-strong" : "border-border",
      )}
    >
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full flex-col gap-2 px-[13px] py-3 text-left"
      >
        <div className="flex flex-wrap items-center gap-[7px]">
          <span
            className="rounded-[5px] px-1.5 py-0.5 font-mono text-[10px] font-semibold tracking-[0.04em] uppercase"
            style={{ color, backgroundColor: `${color}22` }}
          >
            {source.entity_type}
          </span>
          <span
            className={cn(
              "rounded-[5px] px-1.5 py-0.5 font-mono text-[10px]",
              isSeed
                ? "bg-[color:var(--accent-weak)] font-semibold text-accent"
                : "border border-border bg-surface-2 font-medium text-[color:var(--text-faint)]",
            )}
          >
            {hopLabel(source.hop, source.via)}
          </span>
          <span className="flex-1" />
          <span className="font-mono text-[10.5px] text-[color:var(--text-faint)]">{scorePct}</span>
        </div>
        <div className="font-mono text-[13.5px] font-semibold break-words text-foreground">
          {source.qualified_name}
        </div>
        <div className="font-mono text-[11px] break-all text-[color:var(--text-faint)]">
          <span>{dir}</span>
          <span className="text-muted-foreground">{file}</span>
        </div>
      </button>

      {open && (
        <div className="animate-[dk-fade_0.2s_ease] px-[13px] pb-[13px]">
          {source.description && (
            <p className="mb-[11px] text-[12.5px] leading-[1.55] text-muted-foreground">
              {source.description}
            </p>
          )}
          <div className="overflow-hidden rounded-[9px] border border-border bg-surface-2">
            <div className="flex items-center justify-between border-b border-border bg-surface-3 px-2.5 py-1.5">
              <span className="font-mono text-[10.5px] text-[color:var(--text-faint)]">{lines}</span>
              <Link
                href={`/graph?focus=${encodeURIComponent(source.qualified_name)}`}
                className="inline-flex items-center gap-1 font-mono text-[10.5px] text-accent hover:text-[color:var(--accent-press)]"
              >
                view in graph
                <ChevronRight size={11} strokeWidth={2} />
              </Link>
            </div>
            <pre className="overflow-x-auto p-3 font-mono text-[12px] leading-[1.6] text-foreground">
              {source.chunk_text}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
