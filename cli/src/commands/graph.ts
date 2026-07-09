import { existsSync, statSync, writeFileSync } from "node:fs";
import { basename, resolve } from "node:path";
import chalk from "chalk";
import ora from "ora";
import { resolveApiUrl } from "../lib/config.js";
import { getJson, postJson } from "../lib/http.js";
import { followJob, type JobDTO, stageLabel } from "../lib/jobs.js";

interface GraphFlags {
  api?: string;
  out?: string;
}

interface GraphNodeDTO {
  id: number;
  kind: string;
  name?: string | null;
  qualified_name?: string | null;
  path?: string | null;
  absolute_path?: string | null;
  start_line?: number | null;
  end_line?: number | null;
}

interface GraphEdgeDTO {
  source: number;
  target: number;
  type: string;
}

interface GraphStatsDTO {
  total_nodes: number;
  total_edges: number;
  returned_nodes: number;
  returned_edges: number;
  truncated: boolean;
}

interface GraphResponseDTO {
  project: string;
  generated_at: string | null;
  nodes: GraphNodeDTO[];
  edges: GraphEdgeDTO[];
  stats: GraphStatsDTO;
}

async function fetchGraph(
  apiUrl: string,
  project: string,
): Promise<GraphResponseDTO> {
  return getJson<GraphResponseDTO>(
    `${apiUrl}/graph/${encodeURIComponent(project)}?include=structural&limit=1000000`,
  );
}

function writeGraph(graph: GraphResponseDTO, out: string | undefined): void {
  const json = JSON.stringify(graph, null, 2);
  if (out) {
    writeFileSync(out, json);
  } else {
    console.log(json);
  }
}

async function runGraphOnlyGeneration(
  apiUrl: string,
  resolvedPath: string,
  out: string | undefined,
): Promise<void> {
  let jobId: string;
  try {
    const created = await postJson<{ job_id: string; status: string }>(
      `${apiUrl}/instances/graph`,
      { repo_path: resolvedPath },
    );
    jobId = created.job_id;
  } catch (error) {
    console.error(
      chalk.red(error instanceof Error ? error.message : String(error)),
    );
    process.exit(1);
  }

  const spinner = ora(`Generating graph for ${resolvedPath}...`).start();
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

  if (finalJob.status !== "succeeded") {
    spinner.fail(`Graph generation failed: ${finalJob.error ?? "unknown error"}`);
    process.exit(1);
  }

  const outputJson = finalJob.result?.ingest?.output_json;
  spinner.succeed(chalk.green(`Graph generated: ${outputJson ?? "(no path returned)"}`));

  if (out) {
    const project = basename(resolvedPath);
    const exportSpinner = ora({
      text: `Exporting graph for ${project}...`,
      stream: process.stderr,
    }).start();
    try {
      const graph = await fetchGraph(apiUrl, project);
      writeGraph(graph, out);
      exportSpinner.succeed(`Graph exported to ${out}`);
    } catch (error) {
      exportSpinner.fail(error instanceof Error ? error.message : String(error));
      process.exit(1);
    }
  }
}

async function runGraphExport(
  apiUrl: string,
  project: string,
  out: string | undefined,
): Promise<void> {
  const spinner = ora({
    text: `Fetching graph for ${project}...`,
    stream: process.stderr,
  }).start();
  let graph: GraphResponseDTO;
  try {
    graph = await fetchGraph(apiUrl, project);
  } catch (error) {
    spinner.fail(error instanceof Error ? error.message : String(error));
    process.exit(1);
  }
  writeGraph(graph, out);
  spinner.succeed(
    `Graph for ${project}: ${graph.stats.returned_nodes} nodes, ` +
      `${graph.stats.returned_edges} edges`,
  );
}

export async function runGraph(
  target: string,
  flags: GraphFlags,
): Promise<void> {
  const apiUrl = resolveApiUrl(flags);
  const resolvedPath = resolve(target);

  if (existsSync(resolvedPath) && statSync(resolvedPath).isDirectory()) {
    await runGraphOnlyGeneration(apiUrl, resolvedPath, flags.out);
    return;
  }

  await runGraphExport(apiUrl, target, flags.out);
}
