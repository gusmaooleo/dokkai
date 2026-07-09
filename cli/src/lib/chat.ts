/**
 * `dokkai srcs --model ollama:<name>` — terminal chat loop backed by the
 * dokkai API's `POST /chat` (SSE), using a local Ollama model configured as
 * the API's global chat model via `POST /config/llm`.
 */

import { createInterface } from "node:readline";
import chalk from "chalk";
import { resolveApiUrl } from "./config.js";
import { getJson, postJson } from "./http.js";

export interface ChatFlags {
  api?: string;
  project?: string;
}

interface ProjectGraphDTO {
  project: string;
  file: string;
  nodes: number;
  edges: number;
  generated_at: string;
}

interface SourceChunkDTO {
  entity_type: string;
  name: string;
  qualified_name: string;
  file_path: string;
  absolute_path: string;
  start_line: number | null;
  end_line: number | null;
  chunk_text: string;
  score: number | null;
}

interface LLMConfigResponse {
  is_local: boolean;
  provider_name: string;
  model: string;
  has_key: boolean;
  base_url: string | null;
}

interface DoneEventDTO {
  conversation_id: string;
  answer: string;
}

interface SseFrame {
  event: string;
  data: string;
}

/**
 * Resolve which project to chat about.
 *
 * `--project` is validated against the ingested projects (`GET /graph`);
 * otherwise exactly one ingested project auto-resolves, multiple projects
 * error listing them, and zero projects error suggesting `dokkai ingest`.
 */
async function resolveProject(
  apiUrl: string,
  flags: ChatFlags,
): Promise<string> {
  const projects = await getJson<ProjectGraphDTO[]>(`${apiUrl}/graph`);

  if (flags.project) {
    if (!projects.some((p) => p.project === flags.project)) {
      const names = projects.map((p) => p.project).join(", ") || "(none)";
      throw new Error(
        `project '${flags.project}' was not found among ingested projects: ${names}`,
      );
    }
    return flags.project;
  }

  if (projects.length === 0) {
    throw new Error(
      "no ingested projects found — run `dokkai ingest <repo-path>` first",
    );
  }
  if (projects.length > 1) {
    const names = projects.map((p) => p.project).join(", ");
    throw new Error(
      `multiple ingested projects found (${names}) — pass --project <name> to pick one`,
    );
  }
  return projects[0].project;
}

function is400(error: unknown): boolean {
  return error instanceof Error && /\(400\)/.test(error.message);
}

/**
 * `POST /config/llm` with the ollama/local shape for `model`. On a 400
 * (e.g. the model isn't pulled), the API's detail is preserved and a
 * `ollama pull` hint is appended.
 */
async function configureOllamaModel(
  apiUrl: string,
  model: string,
): Promise<void> {
  try {
    await postJson<LLMConfigResponse>(`${apiUrl}/config/llm`, {
      is_local: true,
      provider_data: { provider_name: "ollama", model },
    });
  } catch (error) {
    if (is400(error)) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`${message}\nhint: run \`ollama pull ${model}\``);
    }
    throw error;
  }

  console.log(chalk.dim("note: this sets the dokkai API's global chat model"));
}

function parseFrame(raw: string): SseFrame | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trim());
    }
  }
  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}

/**
 * POST to `/chat` and yield each SSE frame (`sources`, `token`, `done`,
 * `error`) as it arrives. `signal` aborts the underlying fetch.
 */
async function* streamChat(
  apiUrl: string,
  body: unknown,
  signal: AbortSignal,
): AsyncGenerator<SseFrame> {
  const response = await fetch(`${apiUrl}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok || !response.body) {
    const detail = await response.text().catch(() => "");
    throw new Error(
      `dokkai API error (${response.status}): ${detail || response.statusText}`,
    );
  }

  const decoder = new TextDecoder();
  let buffer = "";

  for await (const chunk of response.body as AsyncIterable<Uint8Array>) {
    buffer += decoder.decode(chunk, { stream: true });
    buffer = buffer.replace(/\r\n/g, "\n");

    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const rawFrame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const frame = parseFrame(rawFrame);
      if (frame) yield frame;
    }
  }
}

function printSources(sources: SourceChunkDTO[]): void {
  if (sources.length === 0) return;
  console.log(chalk.dim("sources:"));
  for (const s of sources) {
    const lines =
      s.start_line != null && s.end_line != null
        ? `:${s.start_line}-${s.end_line}`
        : "";
    console.log(chalk.dim(`  ${s.file_path}${lines}`));
  }
}

/**
 * Send one message, render sources + streamed tokens, and return the
 * conversation id the API responded with (for continuity on the next
 * turn) — or the previous one if the exchange was aborted/errored.
 */
async function sendMessage(
  apiUrl: string,
  project: string,
  message: string,
  conversationId: string | undefined,
  registerAbort: (controller: AbortController | null) => void,
): Promise<string | undefined> {
  const controller = new AbortController();
  registerAbort(controller);

  let nextConversationId = conversationId;
  let sawToken = false;

  try {
    for await (const frame of streamChat(
      apiUrl,
      {
        message,
        project_name: project,
        conversation_id: conversationId,
      },
      controller.signal,
    )) {
      if (frame.event === "sources") {
        printSources(JSON.parse(frame.data) as SourceChunkDTO[]);
      } else if (frame.event === "token") {
        process.stdout.write(JSON.parse(frame.data) as string);
        sawToken = true;
      } else if (frame.event === "done") {
        const done = JSON.parse(frame.data) as DoneEventDTO;
        nextConversationId = done.conversation_id;
        if (sawToken) process.stdout.write("\n");
      } else if (frame.event === "error") {
        if (sawToken) process.stdout.write("\n");
        console.error(
          chalk.red(`chat error: ${JSON.parse(frame.data) as string}`),
        );
      }
    }
  } catch (error) {
    if (sawToken) process.stdout.write("\n");
    if (error instanceof Error && error.name === "AbortError") {
      console.log(chalk.yellow("(aborted)"));
    } else {
      console.error(
        chalk.red(error instanceof Error ? error.message : String(error)),
      );
    }
  } finally {
    registerAbort(null);
  }

  return nextConversationId;
}

/**
 * Run the `dokkai srcs --model ollama:<name>` REPL: resolve the project,
 * configure the API's chat model, then loop reading lines from stdin and
 * rendering the SSE response for each until `/exit` or EOF.
 */
export async function runOllamaChat(
  model: string,
  flags: ChatFlags,
): Promise<void> {
  const apiUrl = resolveApiUrl(flags);

  let project: string;
  try {
    project = await resolveProject(apiUrl, flags);
  } catch (error) {
    console.error(
      chalk.red(error instanceof Error ? error.message : String(error)),
    );
    process.exit(1);
  }

  try {
    await configureOllamaModel(apiUrl, model);
  } catch (error) {
    console.error(
      chalk.red(error instanceof Error ? error.message : String(error)),
    );
    process.exit(1);
  }

  console.log(
    chalk.green(`Chatting about '${project}' with ${model} — /exit or Ctrl-D to quit.`),
  );
  console.log();

  const rl = createInterface({ input: process.stdin, output: process.stdout });

  // Manually queue `line` events instead of using `rl.question()`/the
  // built-in async iterator: both end up racing the interface's own
  // auto-close on stdin EOF and can throw or silently drop lines that
  // arrived (piped/scripted stdin can deliver several at once) while a
  // chat request is still in flight. `line`/`close` themselves fire
  // synchronously for whatever is already buffered, so queuing them by
  // hand never loses input.
  const pendingLines: string[] = [];
  let closed = false;
  const waiters: Array<(result: { line: string } | { closed: true }) => void> = [];

  rl.on("line", (line) => {
    const waiter = waiters.shift();
    if (waiter) waiter({ line });
    else pendingLines.push(line);
  });
  rl.on("close", () => {
    closed = true;
    let waiter: typeof waiters[number] | undefined;
    while ((waiter = waiters.shift())) waiter({ closed: true });
  });

  function nextLine(): Promise<{ line: string } | { closed: true }> {
    const line = pendingLines.shift();
    if (line !== undefined) return Promise.resolve({ line });
    if (closed) return Promise.resolve({ closed: true });
    return new Promise((resolve) => waiters.push(resolve));
  }

  let conversationId: string | undefined;
  let activeAbort: AbortController | null = null;

  const onSigint = () => {
    if (activeAbort) {
      activeAbort.abort();
      return;
    }
    rl.close();
    console.log();
    console.log(chalk.dim("goodbye"));
    process.exit(0);
  };
  process.on("SIGINT", onSigint);

  try {
    process.stdout.write("dokkai> ");
    while (true) {
      const result = await nextLine();
      if ("closed" in result) break;

      const message = result.line.trim();
      if (!message) {
        process.stdout.write("dokkai> ");
        continue;
      }
      if (message === "/exit") break;

      conversationId = await sendMessage(
        apiUrl,
        project,
        message,
        conversationId,
        (controller) => {
          activeAbort = controller;
        },
      );
      console.log();
      process.stdout.write("dokkai> ");
    }
  } finally {
    process.off("SIGINT", onSigint);
    rl.close();
  }

  console.log(chalk.dim("goodbye"));
  process.exit(0);
}
