/**
 * Follow a dokkai ingestion job to completion, preferring the SSE endpoint
 * with a polling fallback. Mirrors `src/services/jobs.py`'s `Job.to_dict()`.
 */

import { authHeaders } from "./auth.js";
import { getJson } from "./http.js";
import { SseNotFoundError, streamSse } from "./sse.js";

export interface JobStageProgress {
  stage: string;
  done: number;
  total: number;
}

export interface DescriptionStats {
  enabled: boolean;
  reason?: string;
  model?: string;
  describable?: number;
  cached?: number;
  generated?: number;
  failed?: number;
  pending?: number;
  skipped?: number;
  templated?: number;
  /** Cache hits whose recorded provider/model is known and differs from the
   * currently configured one (feature 22) — served as-is, warned about;
   * re-describe with `force: true` to refresh them. */
  stale_model_hits?: number;
  /** Cache hits with no recorded producer at all (pre-feature-22 entry) —
   * may or may not match the currently configured provider/model. */
  unknown_model_hits?: number;
}

export interface JobResult {
  ingest?: { output_json?: string };
  vectorize?: {
    project_name?: string;
    chunks_created?: number;
    ingestion_id?: string;
    descriptions?: DescriptionStats;
  };
}

export interface JobDTO {
  id: string;
  repo_path: string;
  recreate: boolean;
  describe: boolean;
  kind: string;
  project: string;
  force: boolean;
  status: "queued" | "running" | "succeeded" | "failed";
  stage: string;
  done: number;
  total: number;
  stage_progress: JobStageProgress | null;
  result: JobResult | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

// No bytes arrive on the SSE stream between job updates (the API has no
// heartbeat/comment frames) — a long, quiet-but-healthy stage like `cgr` on
// a big repo can legitimately exceed this. That's fine: the fallback just
// switches transport, it isn't a failure.
const SSE_WATCHDOG_MS = 60_000;
const POLL_INTERVAL_MS = 2_000;

export class JobNotFoundError extends Error {
  constructor(jobId: string) {
    super(
      `job '${jobId}' was not found — the dokkai API's in-memory job store ` +
        "was likely reset by a restart; re-run the ingest",
    );
    this.name = "JobNotFoundError";
  }
}

function isTerminal(job: JobDTO): boolean {
  return job.status === "succeeded" || job.status === "failed";
}

function is404(error: unknown): boolean {
  return error instanceof Error && /\(404\)/.test(error.message);
}

async function* followViaSse(
  apiUrl: string,
  jobId: string,
): AsyncGenerator<JobDTO> {
  const controller = new AbortController();
  let watchdog: ReturnType<typeof setTimeout> | undefined;

  const armWatchdog = () => {
    if (watchdog) clearTimeout(watchdog);
    watchdog = setTimeout(() => controller.abort(), SSE_WATCHDOG_MS);
  };

  try {
    armWatchdog();
    const headers = await authHeaders(apiUrl);
    const frames = streamSse(`${apiUrl}/instances/jobs/${jobId}/events`, {
      signal: controller.signal,
      onChunk: armWatchdog,
      headers,
    });
    for await (const frame of frames) {
      if (frame.event !== "job" && frame.event !== "done") continue;
      const job = JSON.parse(frame.data) as JobDTO;
      yield job;
      if (isTerminal(job)) return;
    }
  } finally {
    if (watchdog) clearTimeout(watchdog);
  }
}

async function* followViaPolling(
  apiUrl: string,
  jobId: string,
): AsyncGenerator<JobDTO> {
  while (true) {
    let job: JobDTO;
    try {
      job = await getJson<JobDTO>(`${apiUrl}/instances/jobs/${jobId}`);
    } catch (error) {
      if (is404(error)) throw new JobNotFoundError(jobId);
      throw error;
    }
    yield job;
    if (isTerminal(job)) return;
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
}

/**
 * Follow a job to completion, yielding progressive snapshots — the last one
 * is always terminal (`succeeded` or `failed`).
 *
 * Prefers `GET /instances/jobs/{id}/events` (SSE). Falls back to polling
 * `GET /instances/jobs/{id}` every 2s if the SSE connection fails, closes
 * before a terminal state, or goes quiet for 60s. A 404 (job unknown to the
 * API — e.g. it restarted and lost its in-memory store) throws
 * `JobNotFoundError` regardless of which transport hit it.
 */
export async function* followJob(
  apiUrl: string,
  jobId: string,
): AsyncGenerator<JobDTO> {
  let lastSeen: JobDTO | null = null;
  try {
    for await (const job of followViaSse(apiUrl, jobId)) {
      lastSeen = job;
      yield job;
    }
    if (lastSeen && isTerminal(lastSeen)) return;
  } catch (error) {
    if (error instanceof SseNotFoundError) throw new JobNotFoundError(jobId);
    if (error instanceof JobNotFoundError) throw error;
    // any other SSE failure (connection error, watchdog abort, closed
    // early) — fall back to polling below
  }

  yield* followViaPolling(apiUrl, jobId);
}


export function stageLabel(job: JobDTO): string {
  const stage = job.stage || "queued";
  const progress = job.stage_progress;
  if (!progress) return stage;
  if (progress.total > 0) return `${stage} ${progress.done}/${progress.total}`;
  return `${stage} ${progress.done}`;
}
