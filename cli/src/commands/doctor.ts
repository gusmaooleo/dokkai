import { spawn } from "node:child_process";
import chalk from "chalk";
import { envVar, resolveApiUrl, resolveDokkaiHome } from "../lib/config.js";
import {
  checkDescModel,
  checkDescriptorRemote,
  checkProvidersFile,
  isOllamaDescriptor,
  loadProvidersFile,
  resolveDescriptorProvider,
} from "../lib/descriptor.js";
import { getJson } from "../lib/http.js";
import { modelInstalled, probeOllama } from "../lib/ollama.js";

const MIN_NODE_MAJOR = 22;
const MIN_NODE_MINOR = 12;
const OLLAMA_PROBE_TIMEOUT_MS = 5_000;
const READY_TIMEOUT_MS = 3_000;

type Status = "ok" | "warning" | "missing" | "info";

interface CheckRow {
  status: Status;
  message: string;
}

function row(status: Status, message: string): CheckRow {
  return { status, message };
}

function printRow({ status, message }: CheckRow): void {
  const prefix =
    status === "ok"
      ? chalk.green("[ok]     ")
      : status === "warning"
        ? chalk.yellow("[warn]   ")
        : status === "info"
          ? chalk.cyan("[info]   ")
          : chalk.red("[missing]");
  console.log(`${prefix} ${message}`);
}

function checkNode(): CheckRow {
  const version = process.version;
  const match = /^v(\d+)\.(\d+)\.(\d+)/.exec(version);
  const major = match ? Number(match[1]) : 0;
  const minor = match ? Number(match[2]) : 0;
  const ok = major > MIN_NODE_MAJOR || (major === MIN_NODE_MAJOR && minor >= MIN_NODE_MINOR);
  if (ok) return row("ok", `node ${version} (>= ${MIN_NODE_MAJOR}.${MIN_NODE_MINOR} required)`);
  return row(
    "missing",
    `node ${version} is older than the required >=${MIN_NODE_MAJOR}.${MIN_NODE_MINOR} — ` +
      "install a newer node, e.g. `nvm install 22.12 && nvm use 22.12`",
  );
}

function commandSucceeds(command: string, args: string[]): Promise<boolean> {
  return new Promise((resolvePromise) => {
    const child = spawn(command, args, { stdio: "ignore" });
    child.on("error", () => resolvePromise(false));
    child.on("close", (code) => resolvePromise(code === 0));
  });
}

async function checkDocker(): Promise<CheckRow> {
  const hasDocker = await commandSucceeds("docker", ["--version"]);
  if (!hasDocker) {
    return row(
      "missing",
      "docker is not installed or not on PATH — install Docker Desktop: " +
        "https://docs.docker.com/get-docker/",
    );
  }
  const hasCompose = await commandSucceeds("docker", ["compose", "version"]);
  if (!hasCompose) {
    return row(
      "missing",
      "docker is installed but `docker compose` is unavailable — install/" +
        "upgrade Docker Desktop (compose v2 is bundled)",
    );
  }
  return row("ok", "docker + docker compose available");
}

async function checkUv(): Promise<CheckRow> {
  const hasUv = await commandSucceeds("uv", ["--version"]);
  if (!hasUv) {
    return row(
      "missing",
      "uv is not installed or not on PATH — install it: " +
        "`curl -LsSf https://astral.sh/uv/install.sh | sh`",
    );
  }
  return row("ok", "uv available");
}

function checkDokkaiHome(): { check: CheckRow; home: string | undefined } {
  try {
    const home = resolveDokkaiHome();
    return { check: row("ok", `DOKKAI_HOME resolved: ${home}`), home };
  } catch (error) {
    return {
      check: row(
        "missing",
        "DOKKAI_HOME could not be resolved — set it (`export " +
          "DOKKAI_HOME=/path/to/dokkai`) or run `dokkai doctor` from inside " +
          `the dokkai repo (${error instanceof Error ? error.message : String(error)})`,
      ),
      home: undefined,
    };
  }
}

/**
 * Ollama reachability + installed-model check. `descModel` is only passed
 * when `DESC_PROVIDER` is (or defaults to) `ollama` — embeddings always go
 * through Ollama regardless of the descriptor provider (feature 22 keeps
 * embeddings Ollama-only), so `embedModel` is checked unconditionally, but
 * checking `DESC_MODEL` against Ollama's installed models — and suggesting
 * `ollama pull` — only makes sense when Ollama is actually what will serve
 * the descriptor.
 */
async function checkOllama(
  baseUrl: string,
  embedModel: string,
  descModel?: string,
): Promise<CheckRow[]> {
  const probe = await probeOllama(baseUrl, OLLAMA_PROBE_TIMEOUT_MS);
  if (!probe.reachable) {
    return [
      row(
        "warning",
        `Ollama is not reachable at ${baseUrl} — start it: \`ollama serve\` ` +
          "(or open the Ollama app)",
      ),
    ];
  }

  const rows: CheckRow[] = [row("ok", `Ollama reachable at ${baseUrl}`)];
  const modelsToCheck: Array<[string, string]> = [["EMBED_MODEL", embedModel]];
  if (descModel !== undefined) modelsToCheck.push(["DESC_MODEL", descModel]);
  for (const [label, model] of modelsToCheck) {
    if (modelInstalled(probe.installed, model)) {
      rows.push(row("ok", `${label} '${model}' installed`));
    } else {
      rows.push(
        row("warning", `${label} '${model}' is not installed — run \`ollama pull ${model}\``),
      );
    }
  }
  return rows;
}

async function fetchWithTimeout(url: string, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function checkWeaviate(url: string, home: string | undefined): Promise<CheckRow> {
  try {
    const response = await fetchWithTimeout(`${url}/v1/.well-known/ready`, READY_TIMEOUT_MS);
    if (response.ok) return row("ok", `Weaviate ready at ${url}`);
  } catch {
    // fall through to the warning below
  }
  return row(
    "warning",
    `Weaviate is not ready at ${url} — start it: \`dokkai up\`` +
      (home ? ` (or \`docker compose up -d\` from ${home})` : ""),
  );
}

async function checkApi(apiUrl: string, home: string | undefined): Promise<CheckRow> {
  try {
    await getJson(`${apiUrl}/`);
    return row("ok", `dokkai API reachable at ${apiUrl}`);
  } catch {
    return row(
      "warning",
      `dokkai API is not reachable at ${apiUrl} — start it: \`./dev.sh\`` +
        (home ? ` (from ${home})` : ""),
    );
  }
}

/** Convert a `Finding` (feature 22, `../lib/descriptor.js`) into a `CheckRow`. */
function toCheckRow(f: { level: "ok" | "warning" | "info"; message: string }): CheckRow {
  return row(f.level, f.message);
}

export async function runDoctor(flags: { api?: string }): Promise<void> {
  const apiUrl = resolveApiUrl(flags);

  const nodeRow = checkNode();
  const dockerRow = await checkDocker();
  const uvRow = await checkUv();
  const { check: homeRow, home } = checkDokkaiHome();
  const requiredRows = [nodeRow, dockerRow, uvRow, homeRow];
  for (const check of requiredRows) printRow(check);

  const weaviateHost = envVar("WEAVIATE_HOST", "localhost");
  const weaviatePort = envVar("WEAVIATE_HTTP_PORT", "8080");
  const weaviateUrl = `http://${weaviateHost}:${weaviatePort}`;
  const weaviateRow = await checkWeaviate(weaviateUrl, home);
  printRow(weaviateRow);

  const providersFile = loadProvidersFile(home);
  const providersFileRows = checkProvidersFile(providersFile).map(toCheckRow);
  for (const check of providersFileRows) printRow(check);

  const ollamaBaseUrl = envVar("OLLAMA_BASE_URL", "http://localhost:11434");
  const embedModel = envVar("EMBED_MODEL", "nomic-embed-text");
  const descProvider = resolveDescriptorProvider();

  let ollamaRows: CheckRow[];
  let descriptorRows: CheckRow[];
  if (isOllamaDescriptor(descProvider)) {
    // No hardcoded DESC_MODEL default, ever (project rule 9a-3-clar): only
    // pass a model into the Ollama installed-model check when one is
    // actually configured — otherwise report that plainly instead of
    // silently assuming qwen2.5-coder:3b.
    const rawDescModel = envVar("DESC_MODEL", "").trim();
    ollamaRows = await checkOllama(ollamaBaseUrl, embedModel, rawDescModel || undefined);
    descriptorRows = rawDescModel ? [] : [toCheckRow(checkDescModel(descProvider))];
  } else {
    ollamaRows = await checkOllama(ollamaBaseUrl, embedModel);
    descriptorRows = checkDescriptorRemote(descProvider, providersFile).map(toCheckRow);
  }
  for (const check of ollamaRows) printRow(check);
  for (const check of descriptorRows) printRow(check);

  const apiRow = await checkApi(apiUrl, home);
  printRow(apiRow);

  const allRows = [
    ...requiredRows,
    weaviateRow,
    ...providersFileRows,
    ...ollamaRows,
    ...descriptorRows,
    apiRow,
  ];
  const okCount = allRows.filter((r) => r.status === "ok").length;
  const warningCount = allRows.filter((r) => r.status === "warning").length;
  const infoCount = allRows.filter((r) => r.status === "info").length;
  const missingCount = allRows.filter((r) => r.status === "missing").length;

  // "0 info" for the ordinary case (nothing to report beyond ok/warning/
  // missing) would be a second, gratuitous deviation from the pre-C6
  // summary line beyond the one DESC_MODEL change authorized for this
  // commit — omit the segment entirely when there's nothing to show.
  const summaryParts = [`${okCount} ok`, `${warningCount} warnings`];
  if (infoCount > 0) summaryParts.push(`${infoCount} info`);
  summaryParts.push(`${missingCount} missing`);

  console.log();
  console.log(summaryParts.join(", "));

  if (missingCount > 0) process.exit(1);
}
