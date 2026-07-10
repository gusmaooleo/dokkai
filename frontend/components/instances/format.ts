/**
 * Small formatting + stage-vocabulary helpers for the Instances screen.
 * Stage lists mirror the REAL vocabulary emitted by `services/jobs.py`'s
 * `report()` callback per job kind (decision 12r) — not the prototype's
 * invented `clone/graph/chunk/describe/embed` stages.
 */

import type { Job, JobKind } from "@/lib/types";

/** Mirrors `ConversationCard`'s `rel()`-derived helper — no date lib. */
export function relativeTime(iso: string): string {
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

/** `/Users/leonardo/projects/foo/bar` -> `/Users/leonardo/…/bar` around a target length. */
export function truncateMiddle(text: string, maxLength = 46): string {
  if (text.length <= maxLength) return text;
  const keep = Math.floor((maxLength - 1) / 2);
  return `${text.slice(0, keep)}…${text.slice(text.length - keep)}`;
}

/** In-progress stages for each job kind, in order — excludes the terminal `done`. */
const STAGES_BY_KIND: Record<JobKind, string[]> = {
  pipeline: ["cgr", "chunk", "describe", "upsert"],
  refresh: ["chunk", "describe", "update"],
  graph: ["cgr"],
};

const STAGE_LABELS: Record<string, string> = {
  cgr: "Graph (cgr)",
  chunk: "Chunk",
  describe: "Describe",
  upsert: "Upsert",
  update: "Update",
  done: "Done",
};

export function stagesFor(kind: JobKind): string[] {
  return STAGES_BY_KIND[kind] ?? [];
}

export function stageLabel(stage: string): string {
  return STAGE_LABELS[stage] ?? stage;
}

export const JOB_KIND_LABEL: Record<JobKind, string> = {
  pipeline: "Full ingestion",
  refresh: "Describe refresh",
  graph: "Graph only",
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
