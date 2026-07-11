"use client";

/**
 * Live/historical step timeline for a routine run — stages in the kind's
 * static order (`components/instances/format.ts`'s `STAGES_BY_KIND`,
 * decision 12r), each row glanceable (done/active/pending/error) and
 * expandable to that stage's own event messages (mono, small — e.g.
 * "analyzing X.ts (3/23)"). Conditional stages (`playbooks`/`skills` for
 * review, `playbooks` for bughunt — `OPTIONAL_STAGES_BY_KIND`) only render
 * once at least one event names them: a run that never used playbooks never
 * shows a pending "Playbooks" row, live or after the fact.
 *
 * IMPORTANT: `job.stage` is NOT a reliable live signal for routine jobs —
 * `services.jobs._run_external_sync`'s `emit(stage, message)` closure only
 * ever appends to `job.events` (`add_job_event`); it never assigns
 * `job.stage` the way the built-in pipeline/refresh/graph kinds' `report()`
 * callback does (`job.stage` stays `""` for a running review/bughunt job,
 * flipping straight to `"done"` on success). The only live per-stage signal
 * is therefore the LAST event's own `.stage` — this component derives
 * "current stage" from that, not from `job.stage`.
 *
 * STAGE ALIASING: bug hunt's tool loop (`agent_loop.run_agent_loop`) emits
 * every tool-call/skill-load event under `stage="agent"` — a label that
 * isn't in `STAGES_BY_KIND.bughunt` (that list mirrors the routine-level
 * `emit()` call sites in `bughunt.py`, not the loop's own internal ones).
 * `STAGE_ALIASES` (shared from `components/instances/format.ts`, also used
 * by `job-card.tsx`'s Instances pills) folds `"agent"` into `"hunt"` for
 * both current-stage derivation and event bucketing, so those
 * `"→ search(...)"`-style messages render inside the Hunt row's expandable
 * list (exactly the 9e wireframe's live tool-call detail) instead of being
 * silently dropped by the `order`-membership filters below. `lastKnownStage`
 * additionally walks events backwards past ANY stage that's still
 * unrecognized after aliasing (belt and braces for a future stage name
 * neither `STAGES_BY_KIND` nor `STAGE_ALIASES` knows about) rather than only
 * ever looking at the very last event.
 *
 * Fed by the job's own bounded (200-entry) event log — either freshly via
 * the job SSE stream (`GET /instances/jobs/{job_id}/events`, while
 * queued/running) or from one `GET /instances/jobs/{job_id}` snapshot for a
 * run that's already terminal (see `run-detail.tsx`). `events[0].seq > 0`
 * means the log's start was trimmed — an "earlier events trimmed" note
 * surfaces that rather than silently rendering a partial history as if it
 * were complete.
 */

import { useState } from "react";
import { ChevronDown, ChevronRight, CircleAlert, CircleCheck, LoaderCircle } from "lucide-react";

import { lastKnownStage, OPTIONAL_STAGES_BY_KIND, resolveStage, stageLabel, stagesFor } from "@/components/instances/format";
import type { Job, JobEventEntry, RoutineKind, RoutineStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

type StageState = "done" | "active" | "pending" | "error";

function groupByStage(events: JobEventEntry[]): Map<string, JobEventEntry[]> {
  const map = new Map<string, JobEventEntry[]>();
  for (const e of events) {
    const key = resolveStage(e.stage);
    const list = map.get(key);
    if (list) list.push(e);
    else map.set(key, [e]);
  }
  return map;
}

function StageRow({ stage, state, events }: { stage: string; state: StageState; events: JobEventEntry[] }) {
  const [open, setOpen] = useState(false);
  const canExpand = events.length > 0;

  return (
    <div className="border-b border-border last:border-b-0">
      <button
        type="button"
        onClick={() => canExpand && setOpen((v) => !v)}
        disabled={!canExpand}
        className={cn("flex w-full items-center gap-2 px-3 py-2 text-left", canExpand && "hover:bg-surface-2")}
      >
        {canExpand ? (
          open ? (
            <ChevronDown size={13} strokeWidth={2} className="shrink-0 text-muted-foreground" />
          ) : (
            <ChevronRight size={13} strokeWidth={2} className="shrink-0 text-muted-foreground" />
          )
        ) : (
          <span className="w-[13px] shrink-0" />
        )}

        {state === "done" && <CircleCheck size={14} strokeWidth={2} className="shrink-0 text-[#2f8f82]" />}
        {state === "active" && (
          <LoaderCircle
            size={14}
            strokeWidth={2.4}
            className="shrink-0 animate-[dk-spin_0.7s_linear_infinite] text-accent"
          />
        )}
        {state === "error" && <CircleAlert size={14} strokeWidth={2} className="shrink-0 text-accent" />}
        {state === "pending" && <span className="size-[14px] shrink-0 rounded-full border border-border" />}

        <span
          className={cn(
            "text-[12.5px] font-semibold",
            state === "pending" ? "text-[color:var(--text-faint)]" : "text-foreground",
          )}
        >
          {stageLabel(stage)}
        </span>
        {events.length > 0 && (
          <span className="font-mono text-[10.5px] text-[color:var(--text-faint)]">{events.length}</span>
        )}
      </button>

      {open && events.length > 0 && (
        <div className="px-3 pb-2.5 pl-[33px]">
          {events.map((e) => (
            <div key={e.seq} className="font-mono text-[11.5px] leading-[1.6] text-muted-foreground">
              → {e.message}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function StepTimeline({
  kind,
  job,
  runStatus,
}: {
  kind: RoutineKind;
  job: Job | null;
  runStatus: RoutineStatus;
}) {
  const [collapsed, setCollapsed] = useState(runStatus === "done" || runStatus === "failed");

  const order = stagesFor(kind);
  const optional = OPTIONAL_STAGES_BY_KIND[kind] ?? [];
  const events = job?.events ?? [];
  const byStage = groupByStage(events);
  const terminal = runStatus === "done" || runStatus === "failed";
  const lastEventStage = lastKnownStage(events, order);
  const lastEventIndex = lastEventStage ? order.indexOf(lastEventStage) : -1;

  const visible = order.filter((stage) => {
    if (terminal) return byStage.has(stage);
    if (optional.includes(stage)) return byStage.has(stage);
    return true;
  });

  function stateFor(stage: string): StageState {
    if (terminal) {
      if (runStatus === "failed" && stage === lastEventStage) return "error";
      return "done";
    }
    if (lastEventIndex === -1) return "pending";
    const idx = order.indexOf(stage);
    if (idx < lastEventIndex) return "done";
    if (idx === lastEventIndex) return "active";
    return "pending";
  }

  const trimmed = events.length > 0 && events[0].seq > 0;

  if (!job) return null;

  return (
    <div className="rounded-[14px] border border-border bg-surface shadow-[var(--shadow-sm)]">
      <button type="button" onClick={() => setCollapsed((v) => !v)} className="flex w-full items-center gap-2 px-4 py-3 text-left">
        {collapsed ? (
          <ChevronRight size={14} strokeWidth={2} className="text-muted-foreground" />
        ) : (
          <ChevronDown size={14} strokeWidth={2} className="text-muted-foreground" />
        )}
        <span className="font-mono text-[11px] font-semibold tracking-[0.07em] text-[color:var(--text-faint)] uppercase">
          Step timeline
        </span>
      </button>

      {!collapsed && (
        <div className="border-t border-border">
          {trimmed && (
            <p className="px-3 pt-2 pb-1 font-mono text-[10.5px] text-[color:var(--text-faint)]">
              earlier events trimmed (log capped at 200 entries)
            </p>
          )}
          {visible.length === 0 ? (
            <p className="px-3 py-2.5 text-[12.5px] text-[color:var(--text-faint)]">
              {terminal ? "no step events were recorded for this run." : "waiting for the run to start…"}
            </p>
          ) : (
            visible.map((stage) => (
              <StageRow key={stage} stage={stage} state={stateFor(stage)} events={byStage.get(stage) ?? []} />
            ))
          )}
        </div>
      )}
    </div>
  );
}
