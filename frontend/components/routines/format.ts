/**
 * Small formatting + vocabulary helpers for the Routines screens — mirrors
 * `components/instances/format.ts`'s idiom (relative time, kind/status
 * label maps) for the `routine_runs` domain (decision 9e/9b), which has its
 * own status vocabulary (`queued|running|done|failed` — no `succeeded`).
 */

import type { RoutineKind, RoutineRunSummary, RoutineStatus } from "@/lib/types";

/** Mirrors `ConversationCard`'s/`JobCard`'s `rel()`-derived helper — no date lib. */
export function relativeTime(iso: string): string {
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

export const KIND_LABEL: Record<RoutineKind, string> = {
  review: "Review",
  bughunt: "Bug hunt",
};

export const STATUS_LABEL: Record<RoutineStatus, string> = {
  queued: "queued",
  running: "running",
  done: "done",
  failed: "failed",
};

/** Status chip color classes, echoing `JobCard`'s pill/summary color idiom. */
export const STATUS_CLASSES: Record<RoutineStatus, string> = {
  queued: "bg-surface-2 text-[color:var(--text-faint)]",
  running: "border border-[color:var(--accent-ring)] bg-[color:var(--accent-weak)] text-accent",
  done: "text-[#2f8f82] bg-[rgba(62,156,147,.14)]",
  failed: "bg-[color:var(--accent-weak)] text-accent",
};

const SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"] as const;

const SEVERITY_CLASSES: Record<string, string> = {
  critical: "text-accent bg-[color:var(--accent-weak)]",
  high: "text-[#b8631f] bg-[rgba(216,127,42,.14)]",
  medium: "text-[#a9821f] bg-[rgba(196,155,42,.15)]",
  low: "text-[#3574c9] bg-[rgba(53,116,201,.12)]",
  info: "text-[color:var(--text-faint)] bg-surface-2",
};

/** Non-zero `severity_counts` entries, in critical→info order. */
export function severityEntries(counts: Record<string, number>): { severity: string; count: number; classes: string }[] {
  return SEVERITY_ORDER.filter((s) => (counts[s] ?? 0) > 0).map((s) => ({
    severity: s,
    count: counts[s],
    classes: SEVERITY_CLASSES[s],
  }));
}

/** `target_ref` for a review run, or a truncated `scope` for a bug hunt run. */
export function targetSummary(run: RoutineRunSummary): string {
  const params = run.params ?? {};
  if (run.kind === "review") {
    const target = typeof params.target_ref === "string" ? params.target_ref : null;
    const base = typeof params.base_ref === "string" ? params.base_ref : null;
    if (!target) return "—";
    return base ? `${target} vs ${base}` : target;
  }
  const scope = typeof params.scope === "string" ? params.scope : null;
  if (!scope) return "—";
  return scope.length > 90 ? `${scope.slice(0, 90)}…` : scope;
}
