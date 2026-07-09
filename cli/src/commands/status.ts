import chalk from "chalk";
import ora from "ora";
import { envVar, resolveApiUrl, resolveDokkaiHome } from "../lib/config.js";
import { getJson } from "../lib/http.js";
import { modelInstalled, probeOllama } from "../lib/ollama.js";

const API_HEALTH_TIMEOUT_MS = 5_000;

interface ProjectGraphDTO {
  project: string;
  file: string;
  nodes: number;
  edges: number;
  generated_at: string;
}

function tryResolveHome(): string | undefined {
  try {
    return resolveDokkaiHome();
  } catch {
    return undefined;
  }
}

async function fetchWithTimeout(
  url: string,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

export async function runStatus(flags: { api?: string }): Promise<void> {
  const home = tryResolveHome();
  const apiUrl = resolveApiUrl(flags);

  // Plain, unauthenticated probe: `/` is public, and the up/down row must
  // stay true regardless of whether DOKKAI_ROOT_USER/PASSWORD are valid — a
  // credential problem is reported separately below, not as "down".
  const apiSpinner = ora(`Checking dokkai API at ${apiUrl}...`).start();
  let apiUp = false;
  try {
    const response = await fetchWithTimeout(`${apiUrl}/`, API_HEALTH_TIMEOUT_MS);
    apiUp = response.ok;
  } catch {
    apiUp = false;
  }
  if (apiUp) {
    apiSpinner.succeed(chalk.green(`dokkai API: up (${apiUrl})`));
  } else {
    apiSpinner.fail(chalk.red(`dokkai API: down (${apiUrl})`));
    console.log(
      chalk.red(
        `  start it with ./dev.sh${home ? ` (from ${home})` : ""}`,
      ),
    );
  }

  const weaviateHost = envVar("WEAVIATE_HOST", "localhost");
  const weaviatePort = envVar("WEAVIATE_HTTP_PORT", "8080");
  const weaviateUrl = `http://${weaviateHost}:${weaviatePort}`;
  const weaviateSpinner = ora(`Checking Weaviate at ${weaviateUrl}...`).start();
  let weaviateUp = false;
  try {
    const response = await fetchWithTimeout(
      `${weaviateUrl}/v1/.well-known/ready`,
      3_000,
    );
    weaviateUp = response.ok;
  } catch {
    weaviateUp = false;
  }
  if (weaviateUp) {
    weaviateSpinner.succeed(chalk.green(`Weaviate: ready (${weaviateUrl})`));
  } else {
    weaviateSpinner.fail(chalk.red(`Weaviate: not ready (${weaviateUrl})`));
  }

  const ollamaBaseUrl = envVar("OLLAMA_BASE_URL", "http://localhost:11434");
  const embedModel = envVar("EMBED_MODEL", "nomic-embed-text");
  const descModel = envVar("DESC_MODEL", "qwen2.5-coder:3b");
  const ollamaSpinner = ora(`Checking Ollama at ${ollamaBaseUrl}...`).start();
  const ollamaProbe = await probeOllama(ollamaBaseUrl, 5_000);
  if (ollamaProbe.reachable) {
    ollamaSpinner.succeed(chalk.green(`Ollama: up (${ollamaBaseUrl})`));
    for (const [label, model] of [
      ["EMBED_MODEL", embedModel],
      ["DESC_MODEL", descModel],
    ] as const) {
      if (modelInstalled(ollamaProbe.installed, model)) {
        console.log(chalk.green(`  ${label} '${model}': present`));
      } else {
        console.log(
          chalk.yellow(
            `  ${label} '${model}': missing — run \`ollama pull ${model}\``,
          ),
        );
      }
    }
  } else {
    ollamaSpinner.fail(
      chalk.yellow(
        `Ollama: not reachable (${ollamaBaseUrl}) — chat/describe need it; ` +
          "graph-only works without it",
      ),
    );
  }

  console.log();
  if (apiUp) {
    const graphSpinner = ora("Listing ingested projects...").start();
    try {
      const projects = await getJson<ProjectGraphDTO[]>(`${apiUrl}/graph`);
      graphSpinner.stop();
      if (projects.length === 0) {
        console.log(chalk.dim("No ingested projects."));
      } else {
        console.log(chalk.bold("Ingested projects:"));
        for (const p of projects) {
          console.log(
            `  ${chalk.bold(p.project)} — nodes: ${p.nodes}, edges: ${p.edges}, ` +
              `generated: ${p.generated_at} (${p.file})`,
          );
        }
      }
    } catch (error) {
      graphSpinner.fail(
        chalk.red(
          `could not list ingested projects: ${error instanceof Error ? error.message : String(error)}`,
        ),
      );
    }
  } else {
    console.log(chalk.dim("Ingested projects: unavailable (dokkai API is down)"));
  }
}
