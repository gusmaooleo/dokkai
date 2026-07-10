"use client";

/**
 * Right-hand sources panel, faithful to the prototype's `<aside>`
 * (~L231-263): header with chunk count + retrieval note, close button, and
 * a scrollable list of `SourceCard`s. Always shows the *latest* assistant
 * message's sources — see `chat-screen.tsx` for why (the prototype's
 * `currentSources()` walks the message list from the end and binds the one
 * global panel to whatever it finds first; there's no per-message focus).
 */

import { useState } from "react";
import { X } from "lucide-react";

import type { SourceChunk } from "@/lib/types";
import { SourceCard } from "./source-card";

function estimateTokens(sources: SourceChunk[]): string {
  const chars = sources.reduce((sum, s) => sum + s.chunk_text.length, 0);
  const tokens = chars / 4;
  return `~${(tokens / 1000).toFixed(1)}k tokens retrieved`;
}

export function SourcesPanel({
  sources,
  onClose,
}: {
  sources: SourceChunk[];
  onClose: () => void;
}) {
  const [openId, setOpenId] = useState<string | null>(null);

  return (
    <aside className="flex h-full w-[340px] shrink-0 flex-col border-l border-border bg-background lg:w-[378px]">
      <div className="flex items-center justify-between border-b border-border px-[18px] pt-4 pb-3">
        <div>
          <div className="font-display text-[15px] font-bold text-foreground">Sources</div>
          <div className="mt-px font-mono text-[11px] text-[color:var(--text-faint)]">
            {sources.length} chunks · BFS depth 2
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="grid size-[30px] place-items-center rounded-lg text-[color:var(--text-faint)] hover:bg-surface-2 hover:text-foreground"
        >
          <X size={16} strokeWidth={1.9} />
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-3.5">
        {sources.length === 0 ? (
          <p className="px-1 py-4 text-[12.5px] text-muted-foreground">No sources yet.</p>
        ) : (
          <>
            {sources.map((source, i) => {
              const id = `${source.qualified_name}-${i}`;
              return (
                <SourceCard
                  key={id}
                  source={source}
                  open={openId === id}
                  onToggle={() => setOpenId((prev) => (prev === id ? null : id))}
                />
              );
            })}
            <p className="mt-1 px-1 font-mono text-[10.5px] text-[color:var(--text-faint)]">
              {estimateTokens(sources)}
            </p>
          </>
        )}
      </div>
    </aside>
  );
}
