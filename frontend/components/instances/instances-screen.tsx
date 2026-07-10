"use client";

/**
 * Instances screen, faithful to the prototype's Instances screen
 * (`dokkai-ui-prototype.html` ~L353-387): a "Pipeline jobs" section with a
 * live card per running/queued job and a one-line summary per completed
 * job, a "New ingestion" dialog, and an "Ingested projects" grid.
 *
 * Jobs: `GET /instances/jobs` on mount + a light poll (catches jobs started
 * elsewhere — CLI, another tab). Each running/queued job gets its own SSE
 * stream (`GET /instances/jobs/{id}/events`) that patches that job in place
 * as `job` events arrive; the terminal `done` event triggers exactly one
 * list refetch + projects-context refresh + toast (guarded by
 * `processedRef` so a stream that somehow re-fires `done` can't double it).
 *
 * "New ingestion" is hidden for the viewer role (decision 12p) — write
 * access to `/instances/pipeline` is admin+user per the backend's
 * `require_role("admin", "user")`.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { LoaderCircle, Plus } from "lucide-react";

import { ApiError, instancesApi } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useProject } from "@/lib/project";
import { useToast } from "@/lib/toast";
import { consumeSSE } from "@/lib/sse";
import type { Job, JobSSEEvent } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { JobCard } from "./job-card";
import { NewIngestionDialog } from "./new-ingestion-dialog";
import { ProjectCard } from "./project-card";

const JOB_POLL_INTERVAL_MS = 15_000;

export function InstancesScreen() {
  const router = useRouter();
  const { user } = useAuth();
  const { projects, loading: projectsLoading, setActive, refresh: refreshProjects } = useProject();
  const { toast } = useToast();

  const [jobs, setJobs] = useState<Job[]>([]);
  const [jobsLoading, setJobsLoading] = useState(true);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);

  const streamsRef = useRef<Map<string, AbortController>>(new Map());
  const processedRef = useRef<Set<string>>(new Set());

  const canIngest = user?.role === "admin" || user?.role === "user";

  const loadJobs = useCallback(async () => {
    try {
      const list = await instancesApi.listJobs();
      setJobs(list);
      setJobsError(null);
    } catch (err) {
      setJobsError(err instanceof ApiError ? err.detail : "Failed to load jobs");
    } finally {
      setJobsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadJobs();
    const id = setInterval(loadJobs, JOB_POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [loadJobs]);

  const liveIdsKey = useMemo(
    () =>
      jobs
        .filter((j) => j.status === "running" || j.status === "queued")
        .map((j) => j.id)
        .sort()
        .join(","),
    [jobs],
  );

  useEffect(() => {
    const liveIds = liveIdsKey ? liveIdsKey.split(",") : [];
    for (const id of liveIds) {
      if (streamsRef.current.has(id)) continue;
      const controller = new AbortController();
      streamsRef.current.set(id, controller);
      consumeSSE<JobSSEEvent>(
        `/instances/jobs/${id}/events`,
        (event) => {
          setJobs((prev) => prev.map((j) => (j.id === id ? event.data : j)));
          if (event.type !== "done") return;
          streamsRef.current.delete(id);
          if (processedRef.current.has(id)) return;
          processedRef.current.add(id);
          const ok = event.data.status === "succeeded";
          toast(`Ingestion for ${event.data.project} ${ok ? "finished" : "failed"}`, ok ? "default" : "error");
          loadJobs();
          refreshProjects();
        },
        { signal: controller.signal },
      ).catch(() => {
        streamsRef.current.delete(id);
      });
    }
  }, [liveIdsKey, loadJobs, refreshProjects, toast]);

  // Abort every in-flight stream on unmount.
  useEffect(
    () => () => {
      for (const controller of streamsRef.current.values()) controller.abort();
      streamsRef.current.clear();
    },
    [],
  );

  function openGraph(project: string) {
    setActive(project);
    router.push("/graph");
  }

  return (
    <div className="mx-auto max-w-[940px] px-3.5 py-[18px] sm:px-6.5 sm:py-[30px]">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="font-mono text-[11px] font-semibold tracking-[0.07em] text-[color:var(--text-faint)] uppercase">
          Pipeline jobs
        </div>
        {canIngest && (
          <Button type="button" onClick={() => setDialogOpen(true)}>
            <Plus size={16} strokeWidth={2} />
            New ingestion
          </Button>
        )}
      </div>

      <div className="mb-8 flex flex-col gap-3">
        {jobsLoading && (
          <div className="flex justify-center py-10">
            <LoaderCircle size={22} strokeWidth={2.4} className="animate-[dk-spin_0.7s_linear_infinite] text-muted-foreground" />
          </div>
        )}
        {!jobsLoading && jobsError && (
          <p className="py-6 text-center text-[13.5px] text-muted-foreground">{jobsError}</p>
        )}
        {!jobsLoading && !jobsError && jobs.length === 0 && (
          <p className="py-6 text-center text-[13.5px] text-[color:var(--text-faint)]">No ingestion jobs yet.</p>
        )}
        {!jobsLoading && !jobsError && jobs.map((job) => <JobCard key={job.id} job={job} />)}
      </div>

      <div className="mb-3.5 font-mono text-[11px] font-semibold tracking-[0.07em] text-[color:var(--text-faint)] uppercase">
        Ingested projects
      </div>
      {projectsLoading ? (
        <div className="flex justify-center py-10">
          <LoaderCircle size={22} strokeWidth={2.4} className="animate-[dk-spin_0.7s_linear_infinite] text-muted-foreground" />
        </div>
      ) : projects.length === 0 ? (
        <p className="py-6 text-center text-[13.5px] text-[color:var(--text-faint)]">
          No projects ingested yet{canIngest ? " — start one above." : "."}
        </p>
      ) : (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(268px,1fr))] gap-3.5">
          {projects.map((p) => (
            <ProjectCard key={p.project} project={p} onOpen={() => openGraph(p.project)} />
          ))}
        </div>
      )}

      {canIngest && (
        <NewIngestionDialog
          open={dialogOpen}
          onOpenChange={setDialogOpen}
          onSubmitted={loadJobs}
        />
      )}
    </div>
  );
}
