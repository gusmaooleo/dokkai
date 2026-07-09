import chalk from "chalk";
import ora from "ora";
import { envVar, resolveApiUrl, resolveDokkaiHome } from "../lib/config.js";
import { getJson } from "../lib/http.js";

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

/** Base-name match, mirroring `src/services/llm_config.py`'s check. */
function modelInstalled(installed: string[], model: string): boolean {
  return installed.some(
    (m) =>
      m === model ||
      m.startsWith(`${model}:`) ||
      model.startsWith(m.split(":")[0]),
  );
}

export async function runStatus(flags: { api?: string }): Promise<void> {
  const home = tryResolveHome();
  const apiUrl = resolveApiUrl(flags);

  const apiSpinner = ora(`Checking dokkai API at ${apiUrl}...`).start();
  let apiUp = false;
  try {
    await getJson(`${apiUrl}/`);
    apiUp = true;
    apiSpinner.succeed(chalk.green(`dokkai API: up (${apiUrl})`));
  } catch {
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
  let ollamaUp = false;
  let installed: string[] = [];
  try {
    const response = await fetchWithTimeout(
      `${ollamaBaseUrl}/api/tags`,
      5_000,
    );
    if (response.ok) {
      ollamaUp = true;
      const data = (await response.json()) as {
        models?: Array<{ name: string }>;
      };
      installed = (data.models ?? []).map((m) => m.name);
    }
  } catch {
    ollamaUp = false;
  }
  if (ollamaUp) {
    ollamaSpinner.succeed(chalk.green(`Ollama: up (${ollamaBaseUrl})`));
    for (const [label, model] of [
      ["EMBED_MODEL", embedModel],
      ["DESC_MODEL", descModel],
    ] as const) {
      if (modelInstalled(installed, model)) {
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
