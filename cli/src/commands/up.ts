import { spawn } from "node:child_process";
import chalk from "chalk";
import ora from "ora";
import { envVar, resolveApiUrl, resolveDokkaiHome } from "../lib/config.js";
import { getJson } from "../lib/http.js";
import { modelInstalled, probeOllama } from "../lib/ollama.js";

const WEAVIATE_READY_TIMEOUT_MS = 60_000;
const WEAVIATE_POLL_INTERVAL_MS = 1_000;
const OLLAMA_PROBE_TIMEOUT_MS = 5_000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function runCommand(
  command: string,
  args: string[],
  cwd: string,
): Promise<number> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, stdio: "inherit" });
    child.on("error", reject);
    child.on("close", (code) => resolve(code ?? 1));
  });
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

async function waitForWeaviateReady(url: string): Promise<boolean> {
  const deadline = Date.now() + WEAVIATE_READY_TIMEOUT_MS;
  while (Date.now() < deadline) {
    try {
      const response = await fetchWithTimeout(
        `${url}/v1/.well-known/ready`,
        3_000,
      );
      if (response.ok) return true;
    } catch {
      // not ready yet — keep polling
    }
    await sleep(WEAVIATE_POLL_INTERVAL_MS);
  }
  return false;
}

async function checkOllama(
  baseUrl: string,
  embedModel: string,
  descModel: string,
): Promise<void> {
  const probe = await probeOllama(baseUrl, OLLAMA_PROBE_TIMEOUT_MS);
  if (!probe.reachable) {
    console.log(
      chalk.yellow(
        `warning: Ollama is not reachable at ${baseUrl} — chat/describe need ` +
          "it; graph-only works without it",
      ),
    );
    return;
  }

  for (const [label, model] of [
    ["EMBED_MODEL", embedModel],
    ["DESC_MODEL", descModel],
  ] as const) {
    if (!modelInstalled(probe.installed, model)) {
      console.log(
        chalk.yellow(
          `warning: ${label} '${model}' is not installed in Ollama — run ` +
            `\`ollama pull ${model}\``,
        ),
      );
    }
  }
}

export async function runUp(flags: { api?: string }): Promise<void> {
  const home = resolveDokkaiHome();

  console.log(chalk.bold(`Starting docker compose services in ${home}...`));
  const exitCode = await runCommand("docker", ["compose", "up", "-d"], home);
  if (exitCode !== 0) {
    console.error(
      chalk.red(`docker compose up failed with exit code ${exitCode}`),
    );
    process.exit(1);
  }

  const weaviateHost = envVar("WEAVIATE_HOST", "localhost");
  const weaviatePort = envVar("WEAVIATE_HTTP_PORT", "8080");
  const weaviateUrl = `http://${weaviateHost}:${weaviatePort}`;

  const spinner = ora("Waiting for Weaviate to be ready...").start();
  const ready = await waitForWeaviateReady(weaviateUrl);
  if (!ready) {
    spinner.fail(`Weaviate did not become ready at ${weaviateUrl} within 60s`);
    process.exit(1);
  }
  spinner.succeed(`Weaviate is ready at ${weaviateUrl}`);

  const ollamaBaseUrl = envVar("OLLAMA_BASE_URL", "http://localhost:11434");
  const embedModel = envVar("EMBED_MODEL", "nomic-embed-text");
  const descModel = envVar("DESC_MODEL", "qwen2.5-coder:3b");
  await checkOllama(ollamaBaseUrl, embedModel, descModel);

  const apiUrl = resolveApiUrl(flags);
  try {
    await getJson(`${apiUrl}/`);
    console.log(chalk.green(`dokkai API is reachable at ${apiUrl}`));
  } catch {
    console.log(
      chalk.cyan(
        `dokkai API is not running — start it with ./dev.sh (from ${home})`,
      ),
    );
  }

  console.log(chalk.green("dokkai is up."));
}
