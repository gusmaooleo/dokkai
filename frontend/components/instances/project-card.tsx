/**
 * One ingested project card, faithful to the prototype's project card
 * (`dokkai-ui-prototype.html` ~L377-382) — name, repo path, chunks/nodes/
 * edges counts, generated-at, "open graph" action. No language dot
 * (decision 12l — the prototype had no data source for it).
 */

import { ArrowRight } from "lucide-react";

import type { ProjectGraphDTO } from "@/lib/types";
import { relativeTime, truncateMiddle } from "./format";

export function ProjectCard({ project, onOpen }: { project: ProjectGraphDTO; onOpen: () => void }) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className="cursor-pointer rounded-[14px] border border-border bg-surface p-4 text-left shadow-[var(--shadow-sm)] hover:border-border-strong sm:px-[17px] sm:py-4"
    >
      <div className="mb-2 font-mono text-[15px] font-semibold text-foreground">{project.project}</div>
      <div className="mb-3.5 font-mono text-[10.5px] leading-[1.45] break-all text-[color:var(--text-faint)]">
        {project.repo_path ? truncateMiddle(project.repo_path) : "—"}
      </div>
      <div className="mb-3.5 flex gap-[18px]">
        {project.chunks !== null && (
          <span className="flex flex-col">
            <span className="text-base font-semibold text-foreground">{project.chunks.toLocaleString()}</span>
            <span className="font-mono text-[10px] text-[color:var(--text-faint)]">chunks</span>
          </span>
        )}
        <span className="flex flex-col">
          <span className="text-base font-semibold text-foreground">{project.nodes.toLocaleString()}</span>
          <span className="font-mono text-[10px] text-[color:var(--text-faint)]">nodes</span>
        </span>
        <span className="flex flex-col">
          <span className="text-base font-semibold text-foreground">{project.edges.toLocaleString()}</span>
          <span className="font-mono text-[10px] text-[color:var(--text-faint)]">edges</span>
        </span>
      </div>
      <div className="flex items-center justify-between border-t border-border pt-3">
        <span className="font-mono text-[11px] text-[color:var(--text-faint)]">
          {relativeTime(project.generated_at)}
        </span>
        <span className="inline-flex items-center gap-1 font-mono text-[11px] text-accent">
          open graph
          <ArrowRight size={11} strokeWidth={2.2} />
        </span>
      </div>
    </button>
  );
}
