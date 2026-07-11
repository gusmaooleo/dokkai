"use client";

/**
 * Routines home screen (decision 9e): a persistent discreet demo note
 * (decision 14c), the launch card (hidden for the viewer role — 12p, view
 * -only) and the run history, scoped to the active project (`useProject`).
 *
 * History polls `GET /routines/runs` every ~10s while any listed run is
 * queued/running — the per-run SSE detail stream arrives in a later step
 * (C4); this list poll is enough to reflect progress here.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { LoaderCircle } from "lucide-react";

import { ApiError, routinesApi } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useProject } from "@/lib/project";
import { useToast } from "@/lib/toast";
import type { RoutineRunSummary } from "@/lib/types";
import { LaunchCard } from "./launch-card";
import { RunCard } from "./run-card";
import { RoutineSectionTabs } from "./section-tabs";

const RUN_POLL_INTERVAL_MS = 10_000;

export function RoutinesScreen() {
  const router = useRouter();
  const { user } = useAuth();
  const { active } = useProject();
  const { toast } = useToast();

  const [runs, setRuns] = useState<RoutineRunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const canLaunch = user?.role === "admin" || user?.role === "user";
  const canDelete = user?.role !== "viewer";
  const project = active?.project ?? null;

  const projectRef = useRef(project);
  projectRef.current = project;

  const loadRuns = useCallback(async () => {
    if (!projectRef.current) {
      setRuns([]);
      setLoading(false);
      return;
    }
    try {
      const list = await routinesApi.listRuns({ project: projectRef.current });
      setRuns(list);
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Failed to load routine runs");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setLoading(true);
    loadRuns();
  }, [project, loadRuns]);

  const hasLiveRun = runs.some((r) => r.status === "queued" || r.status === "running");

  useEffect(() => {
    if (!hasLiveRun) return;
    const id = setInterval(loadRuns, RUN_POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [hasLiveRun, loadRuns]);

  async function handleDelete(runId: string) {
    try {
      await routinesApi.deleteRun(runId);
      toast("Routine run deleted");
      loadRuns();
    } catch (err) {
      toast(err instanceof ApiError ? err.detail : "Failed to delete routine run", "error");
    }
  }

  function handleLaunched(runId: string) {
    router.push(`/routines/runs/${runId}`);
  }

  return (
    <div className="mx-auto max-w-[860px] px-3.5 py-[18px] sm:px-6.5 sm:py-[30px]">
      <RoutineSectionTabs active="runs" />

      <div className="mb-6 rounded-[10px] border border-border bg-surface-2 px-3.5 py-2.5 text-[12px] leading-[1.5] text-[color:var(--text-faint)]">
        Routines are an experimental demo — not part of the dokkai core. They may be removed or promoted depending on
        adoption.
      </div>

      {canLaunch && (
        <div className="mb-8">
          {project ? (
            <LaunchCard project={project} onLaunched={handleLaunched} />
          ) : (
            <div className="rounded-[14px] border border-border bg-surface p-5 text-[13.5px] text-muted-foreground shadow-[var(--shadow-sm)]">
              Select an ingested project to launch a routine.
            </div>
          )}
        </div>
      )}

      <div className="mb-3.5 font-mono text-[11px] font-semibold tracking-[0.07em] text-[color:var(--text-faint)] uppercase">
        Run history
      </div>

      {loading && (
        <div className="flex justify-center py-10">
          <LoaderCircle size={22} strokeWidth={2.4} className="animate-[dk-spin_0.7s_linear_infinite] text-muted-foreground" />
        </div>
      )}
      {!loading && error && <p className="py-6 text-center text-[13.5px] text-muted-foreground">{error}</p>}
      {!loading && !error && runs.length === 0 && (
        <p className="py-6 text-center text-[13.5px] text-[color:var(--text-faint)]">
          No routine runs yet — launch one above.
        </p>
      )}
      {!loading && !error && runs.length > 0 && (
        <div className="flex flex-col gap-3">
          {runs.map((run) => (
            <RunCard
              key={run.id}
              run={run}
              canDelete={canDelete}
              onOpen={() => router.push(`/routines/runs/${run.id}`)}
              onDelete={() => handleDelete(run.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
