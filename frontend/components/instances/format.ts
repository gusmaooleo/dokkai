/**
 * Small formatting + stage-vocabulary helpers for the Instances screen.
 * Stage lists mirror the REAL vocabulary emitted by `services/jobs.py`'s
 * `report()` callback per job kind (decision 12r) — not the prototype's
 * invented `clone/graph/chunk/describe/embed` stages.
 */

import type { Job, JobEventEntry, JobKind } from "@/lib/types";

/** Mirrors `ConversationCard`'s `rel()`-derived helper — no date lib. */
export function relativeTime(iso: string): string {
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

export function truncateMiddle(text: string, maxLength = 46): string {
  if (text.length <= maxLength) return text;
  const keep = Math.floor((maxLength - 1) / 2);
  return `${text.slice(0, keep)}…${text.slice(text.length - keep)}`;
}

/**
 * In-progress stages for each job kind, in order — excludes the terminal
 * `done`. review/bughunt mirror the exact `emit(stage, ...)` call sites in
 * `services/routines/{review,bughunt}.py` (decision 9d/B — sub-part B added
 * the conditional `playbooks`/`skills` stages on top of sub-part A's
 * diff/context/analyze/summarize).
 *
 * NOT mirrored here: bughunt's `hunt` stage delegates to the agentic tool
 * loop (`services/routines/agent_loop.py`), whose OWN `emit(...)` calls use
 * `stage="agent"` (every tool call / skill load during the loop) — a label
 * this list deliberately does NOT include. `components/routines/
 * step-timeline.tsx`'s `STAGE_ALIASES` folds `"agent"` into `"hunt"` at
 * render time instead, since the loop is conceptually part of the `hunt`
 * step, not a step of its own.
 */
const STAGES_BY_KIND: Record<JobKind, string[]> = {
  pipeline: ["cgr", "chunk", "describe", "upsert"],
  refresh: ["chunk", "describe", "update"],
  graph: ["cgr"],
  review: ["diff", "context", "playbooks", "skills", "analyze", "summarize"],
  bughunt: ["resolve", "playbooks", "hunt", "findings", "summarize"],
};

/**
 * Stages within `STAGES_BY_KIND` that only run conditionally — `playbooks`
 * fires only when the run's launch payload named at least one playbook,
 * `skills` only when the skills catalog is non-empty (`_stage_playbooks`/
 * `_stage_skills` in `review.py`, `_stage_playbooks` in `bughunt.py`). A run
 * with none of these never emits an event for that stage — callers that
 * render a live/expandable timeline (`components/routines/step-timeline.tsx`)
 * should only show it once at least one event names it, never as a
 * perpetually-pending step.
 */
export const OPTIONAL_STAGES_BY_KIND: Partial<Record<JobKind, string[]>> = {
  review: ["playbooks", "skills"],
  bughunt: ["playbooks"],
};

const STAGE_LABELS: Record<string, string> = {
  cgr: "Graph (cgr)",
  chunk: "Chunk",
  describe: "Describe",
  upsert: "Upsert",
  update: "Update",
  diff: "Diff",
  context: "Context",
  playbooks: "Playbooks",
  skills: "Skills",
  analyze: "Analyze",
  resolve: "Resolve",
  hunt: "Hunt",
  findings: "Findings",
  summarize: "Summarize",
  done: "Done",
};

export function stagesFor(kind: JobKind): string[] {
  return STAGES_BY_KIND[kind] ?? [];
}

export function stageLabel(stage: string): string {
  return STAGE_LABELS[stage] ?? stage;
}

/**
 * Raw `event.stage` values that don't correspond 1:1 to a `STAGES_BY_KIND`
 * entry — bughunt's agentic tool loop (`services/routines/agent_loop.py`)
 * emits every tool-call/skill-load event under `stage="agent"`, which folds
 * into `hunt` here. Shared by `step-timeline.tsx` (live/historical timeline)
 * and `job-card.tsx` (Instances list pills) so both derive "current stage"
 * the same way.
 */
export const STAGE_ALIASES: Record<string, string> = {
  agent: "hunt",
};

export function resolveStage(stage: string): string {
  return STAGE_ALIASES[stage] ?? stage;
}

/**
 * The most recent event whose (aliased) stage is actually a known step in
 * *order* — not just the very last event's stage, so a stray/unrecognized
 * stage name doesn't blank out the whole "current stage" signal when an
 * earlier event already gave us a good one.
 */
export function lastKnownStage(events: JobEventEntry[], order: string[]): string | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const resolved = resolveStage(events[i].stage);
    if (order.includes(resolved)) return resolved;
  }
  return null;
}

export const JOB_KIND_LABEL: Record<JobKind, string> = {
  pipeline: "Full ingestion",
  refresh: "Describe refresh",
  graph: "Graph only",
  review: "Code review",
  bughunt: "Bug hunt",
};

/** Short result summary for a terminal job's completed row. */
export function resultSummary(job: Job): string | null {
  if (!job.result) return null;
  if (job.kind === "pipeline" && "vectorize" in job.result) {
    return `${job.result.vectorize.chunks_created.toLocaleString()} chunks`;
  }
  if (job.kind === "refresh" && "chunks_indexed" in job.result) {
    return `${job.result.chunks_indexed.toLocaleString()} chunks indexed`;
  }
  if (job.kind === "graph") {
    return "graph regenerated";
  }
  return null;
}
