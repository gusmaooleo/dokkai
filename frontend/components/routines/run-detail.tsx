"use client";

/**
 * Routine run detail screen (decision 9e): header (kind/target/model/status/
 * duration/stats) + step timeline + Findings/Summary tabs.
 *
 * Live progress design: the run's own `job_id` (added to `RoutineRunDetail`
 * alongside this step) drives the EXISTING job machinery —
 *   - one `GET /instances/jobs/{job_id}` snapshot on load, seeding the step
 *     timeline immediately (works for a terminal run too, no SSE needed);
 *   - while the run is queued/running, the job SSE stream
 *     (`GET /instances/jobs/{job_id}/events`) patches that snapshot live,
 *     same `consumeSSE` idiom as `instances-screen.tsx`;
 *   - the SSE stream's terminal `done` event triggers exactly ONE
 *     `GET /routines/runs/{id}` refetch — that's what actually picks up the
 *     run's findings/summary/stats/status, since the job snapshot alone
 *     doesn't carry them;
 *   - a light 10s fallback poll of the run detail while queued/running
 *     hedges against a dropped/failed SSE connection (mirrors
 *     `routines-screen.tsx`'s own history poll) — belt and suspenders, not
 *     the primary mechanism.
 *
 * A 404 on the initial load toasts and redirects to /routines (no dedicated
 * not-found UI for a routine run, mirroring `chat-screen.tsx`'s resume
 * path).
 */

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { LoaderCircle } from "lucide-react";

import { ApiError, instancesApi, routinesApi } from "@/lib/api";
import { consumeSSE } from "@/lib/sse";
import { useToast } from "@/lib/toast";
import type { FindingDTO, Job, JobSSEEvent, RoutineRunDetail } from "@/lib/types";
import { cn } from "@/lib/utils";
import { FindingRow } from "./finding-row";
import {
  formatDuration,
  KIND_LABEL,
  SEVERITY_CLASSES,
  SEVERITY_ORDER,
  STATUS_CLASSES,
  STATUS_LABEL,
  targetSummary,
} from "./format";
import { StepTimeline } from "./step-timeline";

const FALLBACK_POLL_INTERVAL_MS = 10_000;

const demoChipClasses =
  "rounded-[4px] border border-current/30 px-1 py-px font-mono text-[9px] font-semibold tracking-[0.06em] uppercase";

const markdownComponents: Components = {
  p: ({ children }) => <p className="mb-2.5 last:mb-0">{children}</p>,
  ul: ({ children }) => <ul className="mb-2.5 list-disc space-y-1 pl-5 last:mb-0">{children}</ul>,
  ol: ({ children }) => <ol className="mb-2.5 list-decimal space-y-1 pl-5 last:mb-0">{children}</ol>,
  li: ({ children }) => <li className="leading-[1.55]">{children}</li>,
  a: ({ children, href }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-accent underline hover:text-[color:var(--accent-press)]">
      {children}
    </a>
  ),
  strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
  code: ({ className, children, ...props }) => (
    <code
      className={
        className ??
        "rounded-md border border-border bg-surface-2 px-1.5 py-0.5 font-mono text-[0.9em] whitespace-nowrap text-foreground"
      }
      {...props}
    >
      {children}
    </code>
  ),
};

function statLine(run: RoutineRunDetail): string[] {
  const parts: string[] = [];
  const anchored = run.findings.filter((f) => f.anchored).length;
  const s = run.stats ?? {};

  const llmCalls = typeof s.llm_calls === "number" ? s.llm_calls : null;
  if (llmCalls != null) parts.push(`${llmCalls} LLM call${llmCalls === 1 ? "" : "s"}`);
  parts.push(`${run.findings.length} finding${run.findings.length === 1 ? "" : "s"} (${anchored} anchored)`);
  const promptChars = typeof s.prompt_chars_total === "number" ? s.prompt_chars_total : null;
  if (promptChars != null) parts.push(`~${Math.max(1, Math.round(promptChars / 4 / 1000))}k tokens`);

  if (run.kind === "bughunt") {
    const rounds = typeof s.rounds === "number" ? s.rounds : null;
    if (rounds != null) parts.push(`${rounds} round${rounds === 1 ? "" : "s"}`);
    const toolCalls = typeof s.tool_calls_made === "number" ? s.tool_calls_made : null;
    if (toolCalls != null) parts.push(`${toolCalls} tool call${toolCalls === 1 ? "" : "s"}`);
    if (s.pushback_used === true) parts.push("pushback used");
  }
  return parts;
}

function chipGroups(run: RoutineRunDetail): { label: string; items: string[] }[] {
  const s = run.stats ?? {};
  const groups: { label: string; items: string[] }[] = [];
  const playbooks = Array.isArray(s.playbooks_used) ? (s.playbooks_used as string[]) : [];
  if (playbooks.length > 0) groups.push({ label: "Playbooks", items: playbooks });
  const skills = Array.isArray(s.skills_used) ? (s.skills_used as string[]) : [];
  if (skills.length > 0) groups.push({ label: "Skills", items: skills });
  const loadedSkills = Array.isArray(s.loaded_skills) ? (s.loaded_skills as string[]) : [];
  if (loadedSkills.length > 0) groups.push({ label: "Loaded skills", items: loadedSkills });
  return groups;
}

function groupFindingsByFile(findings: FindingDTO[]): { file: string; findings: FindingDTO[] }[] {
  const order: string[] = [];
  const byFile = new Map<string, FindingDTO[]>();
  for (const f of findings) {
    if (!byFile.has(f.file_path)) {
      byFile.set(f.file_path, []);
      order.push(f.file_path);
    }
    byFile.get(f.file_path)!.push(f);
  }
  return order.map((file) => ({ file, findings: byFile.get(file)! }));
}

function maxSeverity(findings: FindingDTO[]): string {
  for (const s of SEVERITY_ORDER) {
    if (findings.some((f) => f.severity === s)) return s;
  }
  return "info";
}

export function RunDetail({ runId }: { runId: string }) {
  const router = useRouter();
  const { toast } = useToast();

  const [run, setRun] = useState<RoutineRunDetail | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [tab, setTab] = useState<"findings" | "summary">("findings");

  const loadRun = useCallback(async () => {
    try {
      const r = await routinesApi.getRun(runId);
      setRun(r);
      setLoadError(null);
      return r;
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        toast("Routine run not found", "error");
        router.replace("/routines");
        return null;
      }
      setLoadError(err instanceof ApiError ? err.detail : "Failed to load routine run");
      return null;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  useEffect(() => {
    setLoading(true);
    loadRun().finally(() => setLoading(false));
  }, [loadRun]);

  // Job snapshot (seeds the step timeline, works for a terminal run too) +
  // the live SSE stream while queued/running.
  useEffect(() => {
    if (!run) return;
    let cancelled = false;

    instancesApi
      .getJob(run.job_id)
      .then((j) => {
        if (!cancelled) setJob(j);
      })
      .catch(() => {});

    const isLive = run.status === "queued" || run.status === "running";
    if (!isLive) return;

    const controller = new AbortController();
    consumeSSE<JobSSEEvent>(
      `/instances/jobs/${run.job_id}/events`,
      (event) => {
        if (cancelled) return;
        setJob(event.data);
        if (event.type === "done") loadRun();
      },
      { signal: controller.signal },
    ).catch(() => {});

    return () => {
      cancelled = true;
      controller.abort();
    };
    // Deliberately narrowed to job_id/status (not the whole `run` object) —
    // this effect only needs to restart when the job to stream changes or
    // the run leaves the live queued/running window, not on every findings/
    // summary/stats update `loadRun()` produces.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run?.job_id, run?.status, loadRun]);

  // Light fallback poll — hedges a dropped/failed SSE connection.
  useEffect(() => {
    if (!run) return;
    if (run.status !== "queued" && run.status !== "running") return;
    const intervalId = setInterval(loadRun, FALLBACK_POLL_INTERVAL_MS);
    return () => clearInterval(intervalId);
    // Narrowed to `run?.status` — see the effect above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run?.status, loadRun]);

  if (loading) {
    return (
      <div className="mx-auto max-w-[860px] px-3.5 py-[18px] sm:px-6.5 sm:py-[30px]">
        <div className="flex justify-center py-10">
          <LoaderCircle size={22} strokeWidth={2.4} className="animate-[dk-spin_0.7s_linear_infinite] text-muted-foreground" />
        </div>
      </div>
    );
  }

  if (!run) {
    return (
      <div className="mx-auto max-w-[860px] px-3.5 py-[18px] sm:px-6.5 sm:py-[30px]">
        <Link href="/routines" className="text-[13px] text-accent hover:underline">
          ← Back to Routines
        </Link>
        <p className="mt-4 text-[13.5px] text-muted-foreground">{loadError ?? "Failed to load routine run"}</p>
      </div>
    );
  }

  const groups = groupFindingsByFile(run.findings);
  const stats = statLine(run);
  const chips = chipGroups(run);

  return (
    <div className="mx-auto max-w-[860px] px-3.5 py-[18px] sm:px-6.5 sm:py-[30px]">
      <Link href="/routines" className="text-[13px] text-accent hover:underline">
        ← Back to Routines
      </Link>

      <div className="mt-4 mb-5 rounded-[14px] border border-border bg-surface p-5 shadow-[var(--shadow-sm)]">
        <div className="mb-2.5 flex flex-wrap items-center gap-2">
          <span className="rounded-[6px] border border-border bg-surface-2 px-2 py-0.5 font-mono text-[10.5px] text-muted-foreground">
            {KIND_LABEL[run.kind] ?? run.kind}
          </span>
          {run.kind === "bughunt" && <span className={cn(demoChipClasses, "text-[color:var(--text-faint)]")}>demo</span>}
          <span className={cn("rounded-lg px-2 py-0.5 font-mono text-[10.5px]", STATUS_CLASSES[run.status])}>
            {STATUS_LABEL[run.status] ?? run.status}
          </span>
          <span className="flex-1" />
          <span className="font-mono text-[11px] text-[color:var(--text-faint)]">
            {formatDuration(run.created_at, run.updated_at)}
          </span>
        </div>

        <div className="mb-1 font-mono text-[15px] text-foreground">{targetSummary(run)}</div>
        <p className="text-[12.5px] text-muted-foreground">
          {run.project}
          {" · "}
          {typeof run.stats?.model === "string"
            ? run.stats.model + (typeof run.stats?.provider === "string" ? ` · ${run.stats.provider}` : "")
            : typeof run.params.model === "string"
              ? String(run.params.model)
              : "active LLM config"}
        </p>

        {run.status === "done" && (
          <p className="mt-2.5 text-[12.5px] text-muted-foreground">{stats.join(" · ")}</p>
        )}

        {chips.length > 0 && (
          <div className="mt-2.5 flex flex-col gap-1.5">
            {chips.map((g) => (
              <div key={g.label} className="flex flex-wrap items-center gap-1.5">
                <span className="font-mono text-[10.5px] text-[color:var(--text-faint)]">{g.label}:</span>
                {g.items.map((item) => (
                  <span
                    key={item}
                    className="rounded-md border border-border bg-surface-2 px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground"
                  >
                    {item}
                  </span>
                ))}
              </div>
            ))}
          </div>
        )}
      </div>

      {run.status === "failed" && run.error && (
        <div className="mb-5 rounded-[14px] border border-[color:var(--accent-ring)] bg-[color:var(--accent-weak)] p-4 text-[13px] whitespace-pre-wrap text-accent">
          {run.error}
        </div>
      )}

      <div className="mb-5">
        <StepTimeline kind={run.kind} job={job} runStatus={run.status} />
      </div>

      <div className="mb-4 inline-flex gap-0.5 rounded-[11px] border border-border bg-surface-2 p-[3px]">
        {(["findings", "summary"] as const).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={cn(
              "h-[30px] rounded-lg px-[13px] text-[12.5px] font-semibold capitalize",
              tab === t ? "bg-surface text-foreground shadow-[var(--shadow-sm)]" : "text-muted-foreground",
            )}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === "findings" && (
        <div>
          {groups.length === 0 ? (
            <p className="py-6 text-center text-[13.5px] text-[color:var(--text-faint)]">
              {run.status === "done" ? "No findings." : "Findings will appear here once the run analyzes some code."}
            </p>
          ) : (
            groups.map((g) => (
              <div
                key={g.file}
                className="mb-3.5 overflow-hidden rounded-[14px] border border-border bg-surface shadow-[var(--shadow-sm)]"
              >
                <div className="flex items-center gap-2.5 border-b border-border bg-surface-2 px-3.5 py-2.5">
                  <span className="min-w-0 flex-1 truncate font-mono text-[12.5px] text-foreground">{g.file}</span>
                  <span
                    className={cn(
                      "shrink-0 rounded-md px-1.5 py-0.5 font-mono text-[10.5px]",
                      SEVERITY_CLASSES[maxSeverity(g.findings)],
                    )}
                  >
                    {g.findings.length} finding{g.findings.length === 1 ? "" : "s"}
                  </span>
                </div>
                {g.findings.map((f) => (
                  <FindingRow key={f.id} finding={f} />
                ))}
              </div>
            ))
          )}
        </div>
      )}

      {tab === "summary" && (
        <div className="rounded-[14px] border border-border bg-surface p-5 text-[14px] leading-[1.6] text-foreground shadow-[var(--shadow-sm)]">
          {run.summary ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {run.summary}
            </ReactMarkdown>
          ) : (
            <p className="text-[13.5px] text-[color:var(--text-faint)]">
              {run.status === "done" || run.status === "failed" ? "No summary." : "The summary will appear once the run finishes."}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
