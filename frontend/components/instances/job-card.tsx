/**
 * One job row on the Instances screen, faithful to the prototype's job
 * card (`dokkai-ui-prototype.html` ~L361-371): project + kind badge +
 * relative start time, then either a live stage-pill row + progress bar
 * (running/queued) or a single completed/failed summary line.
 *
 * Stage pills use the REAL vocabulary per job kind (decision 12r) — see
 * `format.ts`'s `STAGES_BY_KIND` — not the prototype's invented stages.
 */

import { CircleAlert, CircleCheck } from "lucide-react";

import type { Job } from "@/lib/types";
import { cn } from "@/lib/utils";
import { JOB_KIND_LABEL, relativeTime, resultSummary, stageLabel, stagesFor } from "./format";

type PillState = "done" | "active" | "todo";

function pillState(job: Job, stage: string, index: number): PillState {
  if (job.status === "succeeded" || job.stage === "done") return "done";
  const stages = stagesFor(job.kind);
  const currentIndex = stages.indexOf(job.stage);
  if (currentIndex === -1) return "todo";
  if (index < currentIndex) return "done";
  if (index === currentIndex) return "active";
  return "todo";
}

const PILL_CLASSES: Record<PillState, string> = {
  done: "text-[#2f8f82] bg-[rgba(62,156,147,.14)]",
  active: "border border-[color:var(--accent-ring)] bg-[color:var(--accent-weak)] text-accent",
  todo: "bg-surface-2 text-[color:var(--text-faint)]",
};

export function JobCard({ job }: { job: Job }) {
  const isLive = job.status === "running" || job.status === "queued";
  const stages = stagesFor(job.kind);
  const pct = job.total > 0 ? Math.min(100, Math.round((job.done / job.total) * 100)) : null;

  return (
    <div className="rounded-[14px] border border-border bg-surface p-4 shadow-[var(--shadow-sm)] sm:px-[18px] sm:py-4">
      <div className="mb-3.5 flex flex-wrap items-center gap-2.5">
        <span className="font-mono text-[15px] font-semibold text-foreground">{job.project}</span>
        <span className="rounded-[6px] border border-border bg-surface-2 px-2 py-0.5 font-mono text-[10.5px] text-muted-foreground">
          {JOB_KIND_LABEL[job.kind] ?? job.kind}
        </span>
        <span className="flex-1" />
        <span className="font-mono text-[11px] text-[color:var(--text-faint)]">
          {job.status === "queued" ? "queued" : relativeTime(job.created_at)}
        </span>
      </div>

      {isLive && (
        <>
          <div className="mb-3.5 flex flex-wrap gap-1.5">
            {stages.map((stage, i) => (
              <span
                key={stage}
                className={cn(
                  "rounded-lg px-2.5 py-1 font-mono text-[11px] whitespace-nowrap",
                  PILL_CLASSES[pillState(job, stage, i)],
                )}
              >
                {stageLabel(stage)}
              </span>
            ))}
          </div>
          <div className="flex items-center gap-3">
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-3">
              {pct === null ? (
                <div className="h-full w-full animate-pulse rounded-full bg-accent/50" />
              ) : (
                <div
                  className="h-full rounded-full bg-accent transition-[width] duration-500 ease-out"
                  style={{ width: `${pct}%` }}
                />
              )}
            </div>
            <span className="font-mono text-xs whitespace-nowrap text-muted-foreground">
              {job.done.toLocaleString()} / {job.total.toLocaleString()}
            </span>
            <span className="w-[38px] text-right font-mono text-xs font-semibold whitespace-nowrap text-accent">
              {pct === null ? "…" : `${pct}%`}
            </span>
          </div>
        </>
      )}

      {!isLive && job.status === "succeeded" && (
        <div className="flex items-center gap-2 text-[13px] text-[#2f8f82]">
          <CircleCheck size={16} strokeWidth={2} className="shrink-0" />
          Completed{resultSummary(job) ? ` · ${resultSummary(job)}` : ""}
        </div>
      )}

      {!isLive && job.status === "failed" && (
        <div className="flex items-start gap-2 text-[13px] text-accent">
          <CircleAlert size={16} strokeWidth={2} className="mt-px shrink-0" />
          <span>{job.error ?? "Job failed"}</span>
        </div>
      )}
    </div>
  );
}
