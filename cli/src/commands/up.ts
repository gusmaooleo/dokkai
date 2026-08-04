import { spawn } from "node:child_process";
import chalk from "chalk";
import ora from "ora";
import { envVar, resolveApiUrl, resolveDokkaiHome } from "../lib/config.js";
import {
  Finding,
  checkDescModel,
  checkDescriptorRemote,
  checkProvidersFile,
  isOllamaDescriptor,
  loadProvidersFile,
  resolveDescriptorProvider,
} from "../lib/descriptor.js";
import { getJson } from "../lib/http.js";
import { modelInstalled, probeOllama } from "../lib/ollama.js";

const WEAVIATE_READY_TIMEOUT_MS = 60_000;
const WEAVIATE_POLL_INTERVAL_MS = 1_000;
const OLLAMA_PROBE_TIMEOUT_MS = 5_000;
const UI_READY_TIMEOUT_MS = 20_000;
const UI_POLL_INTERVAL_MS = 1_000;
const API_READY_TIMEOUT_MS = 30_000;
const API_POLL_INTERVAL_MS = 1_000;

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

async function waitForUiReady(url: string): Promise<boolean> {
  const deadline = Date.now() + UI_READY_TIMEOUT_MS;
  while (Date.now() < deadline) {
    try {
      const response = await fetchWithTimeout(url, 3_000);
      if (response.ok) return true;
    } catch {
      // not ready yet — keep polling
    }
    await sleep(UI_POLL_INTERVAL_MS);
  }
  return false;
}

async function waitForApiReady(url: string): Promise<boolean> {
  const deadline = Date.now() + API_READY_TIMEOUT_MS;
  while (Date.now() < deadline) {
    try {
      const response = await fetchWithTimeout(url, 3_000);
      if (response.ok) return true;
    } catch {
      // not ready yet — keep polling (image build/Postgres migrations take a beat)
    }
    await sleep(API_POLL_INTERVAL_MS);
  }
  return false;
}

/**
 * `descModel` is only passed when `DESC_PROVIDER` is (or defaults to)
 * `ollama` — see `checkDescriptorRemote` (`../lib/descriptor.js`) for the
 * non-Ollama diagnostics, which never check Ollama's installed models.
 */
async function checkOllama(
  baseUrl: string,
  embedModel: string,
  descModel?: string,
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

  const modelsToCheck: Array<[string, string]> = [["EMBED_MODEL", embedModel]];
  if (descModel !== undefined) modelsToCheck.push(["DESC_MODEL", descModel]);
  for (const [label, model] of modelsToCheck) {
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

function printFinding(f: Finding): void {
  if (f.level === "ok") console.log(chalk.green(f.message));
  else if (f.level === "warning") console.log(chalk.yellow(`warning: ${f.message}`));
  else console.log(chalk.cyan(f.message));
}

export async function runUp(
  flags: { api?: string; ui?: boolean; full?: boolean },
): Promise<void> {
  const home = resolveDokkaiHome();

  if (flags.ui) {
    console.log(chalk.yellow("note: --ui is deprecated, use --full"));
  }
  const full = Boolean(flags.full || flags.ui);

  console.log(chalk.bold(`Starting docker compose services in ${home}...`));
  const composeArgs = full
    ? ["compose", "--profile", "full", "up", "-d"]
    : ["compose", "up", "-d"];
  const exitCode = await runCommand("docker", composeArgs, home);
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

  const providersFile = loadProvidersFile(home);
  for (const f of checkProvidersFile(providersFile)) printFinding(f);

  const ollamaBaseUrl = envVar("OLLAMA_BASE_URL", "http://localhost:11434");
  const embedModel = envVar("EMBED_MODEL", "nomic-embed-text");
  const descProvider = resolveDescriptorProvider();
  if (isOllamaDescriptor(descProvider)) {
    // No hardcoded DESC_MODEL default, ever (project rule 9a-3-clar).
    const rawDescModel = envVar("DESC_MODEL", "").trim();
    await checkOllama(ollamaBaseUrl, embedModel, rawDescModel || undefined);
    if (!rawDescModel) printFinding(checkDescModel(descProvider));
  } else {
    await checkOllama(ollamaBaseUrl, embedModel);
    for (const f of checkDescriptorRemote(descProvider, providersFile)) printFinding(f);
  }

  const apiUrl = resolveApiUrl(flags);
  if (full) {
    const apiSpinner = ora(`Waiting for the dokkai API at ${apiUrl}...`).start();
    const apiReady = await waitForApiReady(apiUrl);
    if (apiReady) {
      apiSpinner.succeed(chalk.green(`dokkai API is reachable at ${apiUrl}`));
    } else {
      apiSpinner.fail(
        `dokkai API did not become reachable at ${apiUrl} within 30s`,
      );
    }
  } else {
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
  }

  if (full) {
    const uiUrl = "http://localhost:3000";
    const uiSpinner = ora(`Waiting for the frontend UI at ${uiUrl}...`).start();
    const uiReady = await waitForUiReady(uiUrl);
    if (uiReady) {
      uiSpinner.succeed(chalk.green(`Frontend UI is reachable at ${uiUrl}`));
    } else {
      uiSpinner.fail(
        `Frontend UI did not become reachable at ${uiUrl} within 20s`,
      );
    }
  }

  console.log(chalk.green("dokkai is up."));
}
