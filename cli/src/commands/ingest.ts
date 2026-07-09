import { existsSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { createInterface } from "node:readline/promises";
import chalk from "chalk";
import ora from "ora";
import { resolveApiUrl } from "../lib/config.js";
import { postJson } from "../lib/http.js";
import { followJob, type JobDTO } from "../lib/jobs.js";

interface IngestFlags {
  api?: string;
  recreate?: boolean;
  describe: boolean;
  yes?: boolean;
}

async function confirmRecreate(): Promise<boolean> {
  if (!process.stdin.isTTY || !process.stdout.isTTY) {
    console.error(
      chalk.red(
        "--recreate wipes ALL ingested projects from Weaviate and requires " +
          "confirmation, but this shell is not interactive — pass --yes to " +
          "confirm non-interactively",
      ),
    );
    process.exit(1);
  }
  const rl = createInterface({ input: process.stdin, output: process.stdout });
  try {
    const answer = await rl.question(
      "This wipes ALL ingested projects from Weaviate. Continue? [y/N] ",
    );
    return /^y(es)?$/i.test(answer.trim());
  } finally {
    rl.close();
  }
}

function stageLabel(job: JobDTO): string {
  const stage = job.stage || "queued";
  const progress = job.stage_progress;
  if (!progress) return stage;
  if (progress.total > 0) return `${stage} ${progress.done}/${progress.total}`;
  return `${stage} ${progress.done}`;
}

function printSummary(job: JobDTO): void {
  const ingest = job.result?.ingest;
  const vectorize = job.result?.vectorize;

  console.log(chalk.green(`Ingestion complete: ${job.project}`));
  if (vectorize?.chunks_created !== undefined) {
    console.log(`  chunks created: ${vectorize.chunks_created}`);
  }
  const descriptions = vectorize?.descriptions;
  if (descriptions) {
    if (descriptions.enabled === false) {
      console.log(
        `  descriptions: disabled (${descriptions.reason ?? "opted out"})`,
      );
    } else {
      console.log(
        `  descriptions: ${descriptions.generated ?? 0} generated, ` +
          `${descriptions.cached ?? 0} cached, ${descriptions.templated ?? 0} ` +
          `templated, ${descriptions.failed ?? 0} failed`,
      );
    }
  }
  if (ingest?.output_json) {
    console.log(`  graph JSON: ${ingest.output_json}`);
  }
}

export async function runIngest(
  repoPath: string,
  flags: IngestFlags,
): Promise<void> {
  const apiUrl = resolveApiUrl(flags);
  const resolvedPath = resolve(repoPath);

  if (!existsSync(resolvedPath) || !statSync(resolvedPath).isDirectory()) {
    console.error(
      chalk.red(`error: '${resolvedPath}' does not exist or is not a directory`),
    );
    process.exit(1);
  }

  if (flags.recreate && !flags.yes) {
    const confirmed = await confirmRecreate();
    if (!confirmed) {
      console.log(chalk.yellow("Aborted — no job was created."));
      return;
    }
  }

  let jobId: string;
  try {
    const created = await postJson<{ job_id: string; status: string }>(
      `${apiUrl}/instances/pipeline`,
      {
        repo_path: resolvedPath,
        recreate: Boolean(flags.recreate),
        describe: flags.describe,
      },
    );
    jobId = created.job_id;
  } catch (error) {
    console.error(
      chalk.red(error instanceof Error ? error.message : String(error)),
    );
    process.exit(1);
  }

  const spinner = ora(`Ingesting ${resolvedPath}...`).start();
  let lastLabel: string | null = null;
  let finalJob: JobDTO | null = null;
  try {
    for await (const job of followJob(apiUrl, jobId)) {
      finalJob = job;
      const label = stageLabel(job);
      if (label !== lastLabel) {
        spinner.text = label;
        lastLabel = label;
      }
    }
  } catch (error) {
    spinner.fail(error instanceof Error ? error.message : String(error));
    process.exit(1);
  }

  if (!finalJob) {
    spinner.fail("job ended with no status");
    process.exit(1);
  }

  if (finalJob.status === "succeeded") {
    spinner.succeed(stageLabel(finalJob));
    printSummary(finalJob);
  } else {
    spinner.fail(`Ingestion failed: ${finalJob.error ?? "unknown error"}`);
    process.exit(1);
  }
}
