import { existsSync, statSync, watch, type FSWatcher } from "node:fs";
import { resolve, sep } from "node:path";
import chalk from "chalk";
import { resolveApiUrl } from "../lib/config.js";
import { postJson } from "../lib/http.js";
import { followJob, type JobDTO } from "../lib/jobs.js";

interface WatchFlags {
  api?: string;
  debounce?: string;
  describe: boolean;
}

const DEFAULT_DEBOUNCE_SECONDS = 3;

const IGNORED_SEGMENTS = new Set([
  ".git",
  "node_modules",
  "dist",
  "build",
  "__pycache__",
  ".venv",
  "ingested",
  "data",
]);

/**
 * Ignore anything under one of the noisy/generated directories, and
 * dotfile-only churn (e.g. `.DS_Store`, editor swap files) that never
 * reflects a real source change.
 */
function isIgnored(relativePath: string): boolean {
  const segments = relativePath.split(sep).filter(Boolean);
  if (segments.length === 0) return true;
  const basename = segments[segments.length - 1];
  if (basename.startsWith(".")) return true;
  return segments.some((segment) => IGNORED_SEGMENTS.has(segment));
}

function isConflict(error: unknown): boolean {
  return error instanceof Error && /\(409\)/.test(error.message);
}

function conflictDetail(error: unknown): string {
  if (error instanceof Error) {
    const match = /^dokkai API error \(409\): (.*)$/s.exec(error.message);
    if (match) return match[1];
    return error.message;
  }
  return String(error);
}

function printLine(message: string): void {
  const timestamp = new Date().toLocaleTimeString();
  console.log(`[${timestamp}] ${message}`);
}

export async function runWatch(
  repoPath: string,
  flags: WatchFlags,
): Promise<void> {
  const apiUrl = resolveApiUrl(flags);
  const resolvedPath = resolve(repoPath);

  if (!existsSync(resolvedPath) || !statSync(resolvedPath).isDirectory()) {
    console.error(
      chalk.red(`error: '${resolvedPath}' does not exist or is not a directory`),
    );
    process.exit(1);
  }

  const debounceSeconds = flags.debounce
    ? Number.parseFloat(flags.debounce)
    : DEFAULT_DEBOUNCE_SECONDS;
  if (!Number.isFinite(debounceSeconds) || debounceSeconds < 0) {
    console.error(
      chalk.red(`error: '--debounce ${flags.debounce}' is not a valid number of seconds`),
    );
    process.exit(1);
  }
  const debounceMs = debounceSeconds * 1_000;

  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  let cycleInFlight = false;
  let stopping = false;

  function scheduleCycle(): void {
    if (stopping) return;
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      debounceTimer = null;
      void runCycle();
    }, debounceMs);
  }

  async function runCycle(): Promise<void> {
    if (stopping) return;

    let created: { job_id: string; status: string };
    try {
      created = await postJson<{ job_id: string; status: string }>(
        `${apiUrl}/instances/pipeline`,
        {
          repo_path: resolvedPath,
          recreate: false,
          describe: flags.describe,
        },
      );
    } catch (error) {
      if (isConflict(error)) {
        printLine(chalk.yellow(`cycle skipped: ${conflictDetail(error)}`));
        scheduleCycle();
        return;
      }
      printLine(chalk.red(error instanceof Error ? error.message : String(error)));
      return;
    }

    cycleInFlight = true;
    let finalJob: JobDTO | null = null;
    try {
      for await (const job of followJob(apiUrl, created.job_id)) {
        finalJob = job;
      }
    } catch (error) {
      cycleInFlight = false;
      printLine(chalk.red(error instanceof Error ? error.message : String(error)));
      return;
    }
    cycleInFlight = false;

    if (!finalJob) {
      printLine(chalk.red("cycle failed: job ended with no status"));
      return;
    }

    if (finalJob.status === "succeeded") {
      const vectorize = finalJob.result?.vectorize;
      const chunks = vectorize?.chunks_created ?? 0;
      const descriptions =
        vectorize?.descriptions?.enabled === false
          ? " (no descriptions)"
          : "";
      printLine(
        chalk.green(
          `cycle ok — ${finalJob.project}: ${chunks} chunks${descriptions}`,
        ),
      );
    } else {
      printLine(chalk.red(`cycle failed: ${finalJob.error ?? "unknown error"}`));
    }
  }

  console.log(chalk.bold(`Running initial ingest cycle for ${resolvedPath}...`));
  await runCycle();

  let watcher: FSWatcher;
  try {
    watcher = watch(
      resolvedPath,
      { recursive: true },
      (_eventType, filename) => {
        if (!filename) return;
        if (isIgnored(filename.toString())) return;
        scheduleCycle();
      },
    );
  } catch (error) {
    console.error(
      chalk.red(error instanceof Error ? error.message : String(error)),
    );
    process.exit(1);
  }

  watcher.on("error", (error: Error) => {
    console.error(chalk.red(`watcher error: ${error.message}`));
    process.exit(1);
  });

  console.log(chalk.dim(`Watching ${resolvedPath} (debounce ${debounceSeconds}s)... Ctrl-C to stop.`));

  // `watcher` is an active handle that keeps the event loop (and this
  // function's caller) alive until `.close()` is called below — no need to
  // await anything further here.
  process.on("SIGINT", () => {
    stopping = true;
    if (debounceTimer) clearTimeout(debounceTimer);
    watcher.close();
    if (cycleInFlight) {
      console.log(
        chalk.yellow(
          "\nStopping watcher — a cycle is still running server-side; it will finish in the background.",
        ),
      );
    } else {
      console.log(chalk.yellow("\nStopping watcher."));
    }
    process.exit(0);
  });
}
