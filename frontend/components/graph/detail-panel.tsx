"use client";

/**
 * Right-side node detail panel, faithful to the prototype's `<aside>`
 * (`dokkai-ui-prototype.html` ~L300-318): kind badge + name, path (click to
 * copy), description, relations list (direction-aware labels, clicking one
 * moves the selection) and the source chunk, with an "Ask about this in
 * chat" footer button. Entity-only sections (description, source chunk)
 * are omitted in files mode — files have no Weaviate chunk.
 */

import { Clipboard, MessageCircle, X } from "lucide-react";

import { useToast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { kindColor } from "@/components/chat/kinds";
import type { GraphMode, NormalizedNode } from "./types";

export interface RelationRow {
  id: string;
  name: string;
  kind: string;
  label: string;
}

function splitPath(path: string): { dir: string; file: string } {
  const idx = path.lastIndexOf("/");
  if (idx === -1) return { dir: "", file: path };
  return { dir: path.slice(0, idx + 1), file: path.slice(idx + 1) };
}

export function DetailPanel({
  mode,
  node,
  relations,
  loadingDetail,
  description,
  chunkText,
  onClose,
  onSelectRelation,
  onAskInChat,
}: {
  mode: GraphMode;
  node: NormalizedNode;
  relations: RelationRow[];
  loadingDetail: boolean;
  description: string | null;
  chunkText: string | null;
  onClose: () => void;
  onSelectRelation: (id: string) => void;
  onAskInChat: () => void;
}) {
  const { toast } = useToast();
  const color = kindColor(node.kind);
  const path = node.path ?? "";
  const { dir, file } = splitPath(path);
  const lines =
    node.start_line != null && node.end_line != null ? `lines ${node.start_line}–${node.end_line}` : null;
  const showEntitySections = mode === "entities";

  return (
    <aside className="flex h-full w-[340px] shrink-0 flex-col border-l border-border bg-background lg:w-[378px]">
      <div className="flex items-start justify-between gap-2.5 border-b border-border px-[18px] py-4">
        <div className="min-w-0">
          <span
            className="rounded-[5px] px-1.5 py-0.5 font-mono text-[10px] font-semibold tracking-[0.04em] uppercase"
            style={{ color, backgroundColor: `${color}22` }}
          >
            {node.kind}
          </span>
          <div className="mt-[9px] leading-[1.25] font-mono text-[17px] font-bold break-words text-foreground">
            {node.name}
          </div>
          {node.qualified_name && node.qualified_name !== node.name && (
            <div className="mt-1 font-mono text-[11.5px] break-words text-muted-foreground">
              {node.qualified_name}
            </div>
          )}
        </div>
        <button
          type="button"
          onClick={onClose}
          className="grid size-[30px] shrink-0 place-items-center rounded-lg text-[color:var(--text-faint)] hover:bg-surface-2 hover:text-foreground"
        >
          <X size={16} strokeWidth={1.9} />
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-[18px] py-4">
        {path && (
          <button
            type="button"
            onClick={() => {
              navigator.clipboard?.writeText(path).then(() => toast("Path copied"));
            }}
            title="Copy path"
            className="mb-4 flex w-full items-center gap-2 rounded-[9px] border border-border bg-surface-2 px-[11px] py-[9px] text-left hover:border-border-strong"
          >
            <span className="flex-1 font-mono text-[11px] leading-[1.45] break-all text-muted-foreground">
              <span>{dir}</span>
              <span className="text-foreground">{file}</span>
            </span>
            <Clipboard size={14} strokeWidth={1.7} className="shrink-0 text-[color:var(--text-faint)]" />
          </button>
        )}
        {lines && <p className="mb-4 -mt-2 font-mono text-[11px] text-[color:var(--text-faint)]">{lines}</p>}

        {showEntitySections && (
          <>
            <div className="mb-1.5 font-mono text-[10.5px] font-semibold tracking-[0.06em] text-[color:var(--text-faint)] uppercase">
              Description
            </div>
            {loadingDetail ? (
              <div className="mb-[18px] space-y-1.5">
                <div className="h-3 w-full animate-pulse rounded bg-surface-2" />
                <div className="h-3 w-4/5 animate-pulse rounded bg-surface-2" />
              </div>
            ) : description ? (
              <p className="mb-[18px] text-[13px] leading-[1.6] text-muted-foreground">{description}</p>
            ) : (
              <p className="mb-[18px] text-[13px] text-[color:var(--text-faint)]">No description available.</p>
            )}
          </>
        )}

        <div className="mb-2 font-mono text-[10.5px] font-semibold tracking-[0.06em] text-[color:var(--text-faint)] uppercase">
          Relations · {relations.length}
        </div>
        <div className="mb-[18px] flex flex-col gap-[3px]">
          {relations.length === 0 && (
            <p className="px-[9px] py-1 text-[12.5px] text-[color:var(--text-faint)]">No relations.</p>
          )}
          {relations.map((r, i) => (
            <button
              key={`${r.id}-${r.label}-${i}`}
              type="button"
              onClick={() => onSelectRelation(r.id)}
              className="flex w-full items-center gap-[9px] rounded-lg border border-transparent px-[9px] py-[7px] text-left hover:border-border hover:bg-surface-2"
            >
              <span className="size-2 shrink-0 rounded-full" style={{ backgroundColor: kindColor(r.kind) }} />
              <span className="overflow-hidden font-mono text-[12.5px] font-semibold text-ellipsis whitespace-nowrap text-foreground">
                {r.name}
              </span>
              <span
                className={cn(
                  "ml-auto shrink-0 rounded-[5px] border border-border bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] text-[color:var(--text-faint)]",
                )}
              >
                {r.label}
              </span>
            </button>
          ))}
        </div>

        {showEntitySections && chunkText && (
          <>
            <div className="mb-2 font-mono text-[10.5px] font-semibold tracking-[0.06em] text-[color:var(--text-faint)] uppercase">
              Source chunk
            </div>
            <div className="overflow-hidden rounded-[9px] border border-border bg-surface-2">
              <pre className="overflow-x-auto p-3 font-mono text-[12px] leading-[1.6] text-foreground">
                {chunkText}
              </pre>
            </div>
          </>
        )}
      </div>

      <div className="border-t border-border p-4">
        <button
          type="button"
          onClick={onAskInChat}
          className="flex h-10 w-full items-center justify-center gap-2 rounded-[10px] border border-accent bg-accent text-[13.5px] font-semibold text-white"
        >
          <MessageCircle size={15} strokeWidth={1.9} />
          Ask about this in chat
        </button>
      </div>
    </aside>
  );
}
