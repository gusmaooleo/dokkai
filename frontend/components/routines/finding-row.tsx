"use client";

/**
 * One finding row in the run-detail Findings tab (grouped by file in the
 * parent — see `run-detail.tsx`): a collapsed glance row (severity chip,
 * category, title, `file:line`) that expands to the finding's body,
 * suggestion, evidence entities, and — for review findings that carry a
 * `hunk_excerpt` — the diff-view of the cited hunk. `body`/`suggestion` are
 * plain model text (no markdown contract on that field, unlike the run
 * summary), rendered with `whitespace-pre-wrap` rather than through
 * `react-markdown`. Unanchored findings (decision 9a/9h — the model's cited
 * location couldn't be confirmed against the diff/graph) are dimmed and
 * carry an "unverified anchor" tag.
 */

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

import type { FindingDTO } from "@/lib/types";
import { cn } from "@/lib/utils";
import { DiffView } from "./diff-view";
import { SEVERITY_CLASSES } from "./format";

const CATEGORY_LABEL: Record<string, string> = {
  bug: "Bug",
  security: "Security",
  performance: "Performance",
  maintainability: "Maintainability",
  style: "Style",
  other: "Other",
};

function lineRange(finding: FindingDTO): string {
  if (finding.start_line == null) return "";
  if (finding.end_line == null || finding.end_line === finding.start_line) return `:${finding.start_line}`;
  return `:${finding.start_line}-${finding.end_line}`;
}

export function FindingRow({ finding }: { finding: FindingDTO }) {
  const [open, setOpen] = useState(false);
  const evidence = finding.evidence;

  return (
    <div className={cn("border-b border-border last:border-b-0", !finding.anchored && "opacity-60")}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full flex-wrap items-center gap-2 px-3.5 py-2.5 text-left hover:bg-surface-2 sm:flex-nowrap"
      >
        {open ? (
          <ChevronDown size={14} strokeWidth={2} className="shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight size={14} strokeWidth={2} className="shrink-0 text-muted-foreground" />
        )}
        <span
          className={cn(
            "shrink-0 rounded-md px-1.5 py-0.5 font-mono text-[10.5px]",
            SEVERITY_CLASSES[finding.severity] ?? SEVERITY_CLASSES.info,
          )}
        >
          {finding.severity}
        </span>
        <span className="shrink-0 font-mono text-[10.5px] tracking-[0.04em] text-muted-foreground uppercase">
          {CATEGORY_LABEL[finding.category] ?? finding.category}
        </span>
        <span className="min-w-0 flex-1 truncate text-[13px] text-foreground">{finding.title}</span>
        {!finding.anchored && (
          <span className="shrink-0 rounded-[4px] border border-current/30 px-1 py-px font-mono text-[9px] font-semibold tracking-[0.06em] text-[color:var(--text-faint)] uppercase">
            unverified anchor
          </span>
        )}
        <span className="shrink-0 font-mono text-[11px] text-[color:var(--text-faint)]">
          {finding.file_path}
          {lineRange(finding)}
        </span>
      </button>

      {open && (
        <div className="px-3.5 pb-3.5 pl-[38px]">
          <p className="mb-2.5 whitespace-pre-wrap text-[13px] leading-[1.55] text-foreground">{finding.body}</p>

          {finding.suggestion && (
            <div className="mb-2.5 rounded-[9px] border border-border bg-surface-2 p-2.5">
              <div className="mb-1 font-mono text-[10.5px] font-semibold tracking-[0.06em] text-[color:var(--text-faint)] uppercase">
                Suggested fix
              </div>
              <p className="whitespace-pre-wrap text-[13px] leading-[1.5] text-foreground">{finding.suggestion}</p>
            </div>
          )}

          {evidence?.entities && evidence.entities.length > 0 && (
            <div className="mb-2.5 flex flex-wrap gap-1.5">
              {evidence.entities.map((e) => (
                <span
                  key={e}
                  className="rounded-md border border-border bg-surface-2 px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground"
                >
                  {e}
                </span>
              ))}
            </div>
          )}

          {evidence?.hunk_excerpt && <DiffView text={evidence.hunk_excerpt} />}

          {(evidence?.model || evidence?.provider) && (
            <p className="mt-2 font-mono text-[10.5px] text-[color:var(--text-faint)]">
              {evidence.model}
              {evidence.model && evidence.provider ? " · " : ""}
              {evidence.provider}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
