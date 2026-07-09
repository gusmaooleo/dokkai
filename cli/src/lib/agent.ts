/**
 * `dokkai srcs --model ollama:<name> --agent` — agentic MCP loop for local
 * models. No FastAPI involved: the CLI spawns the dokkai MCP server itself
 * (stdio, `DOKKAI_MCP_PROFILE=small-model`), mirrors its live tool schemas
 * into Ollama's `/api/chat` tool-calling format, and runs the same
 * search/read/answer loop proven by `scripts/mcp_harness.py` — system
 * prompt, fallback tool-call parser (for models that emit tool JSON as plain
 * content instead of native `tool_calls`), tool-result reminder, and a
 * max-rounds + call-dedupe anti-loop guard — inside the srcs REPL.
 */

import { createInterface } from "node:readline";
import chalk from "chalk";
import { envVar, resolveDokkaiHome } from "./config.js";
import { McpClient, type McpToolSchema } from "./mcp.js";

export interface AgentFlags {
  project?: string;
}

interface OllamaToolCall {
  function: { name: string; arguments: Record<string, unknown> | string };
}

interface OllamaMessage {
  role: string;
  content?: string;
  tool_calls?: OllamaToolCall[];
}

interface OllamaTool {
  type: "function";
  function: { name: string; description: string; parameters: Record<string, unknown> };
}

const MAX_ROUNDS = 8;
const MAX_HISTORY = 20;

const SYSTEM_PROMPT =
  "You are answering questions about a codebase. You have no knowledge of " +
  "this code beyond what the tools return — use them. Call context(query) or " +
  "search(query) ONCE to find relevant entities; if you need more detail on a " +
  "specific entity, call get_entity() or neighbors() on it. Never call the " +
  "same tool with the same arguments twice — you already have that result. " +
  "As soon as the tool results answer the question, STOP calling tools and " +
  "write your final answer as plain English text (not JSON), citing the file " +
  "paths you used. Do not answer without calling at least one tool first. " +
  "When you call a tool, output ONLY the JSON for that one call and nothing " +
  "else — no explanation, no example result. You do not have real results " +
  "until the system gives them back to you; never invent or guess a tool's " +
  "output yourself.";

const TOOL_RESULT_REMINDER =
  "\n\n[If the information above answers the question, reply now with your " +
  "final answer as plain English text — do not call any more tools.]";

function tokensOf(text: string): number {
  return Math.floor(text.length / 4);
}

/** Best-effort fallback for models that emit tool-call JSON as plain content
 * instead of Ollama's structured `tool_calls` field (see mcp_harness.py). */
function extractFallbackToolCall(
  content: string,
): { name: string; arguments: Record<string, unknown> } | null {
  let text = content.trim();
  if (text.includes("<tool_call>")) {
    const start = text.indexOf("<tool_call>") + "<tool_call>".length;
    const end = text.indexOf("</tool_call>");
    text = text.slice(start, end !== -1 ? end : undefined).trim();
  }
  if (text.startsWith("```")) {
    text = text.replace(/^`+/, "").replace(/`+$/, "").trim();
    if (text.startsWith("json")) text = text.slice(4).trim();
  }
  let obj: unknown;
  try {
    obj = JSON.parse(text);
  } catch {
    return null;
  }
  if (
    obj !== null &&
    typeof obj === "object" &&
    typeof (obj as Record<string, unknown>).name === "string" &&
    typeof (obj as Record<string, unknown>).arguments === "object" &&
    (obj as Record<string, unknown>).arguments !== null
  ) {
    return obj as { name: string; arguments: Record<string, unknown> };
  }
  return null;
}

function mcpToolToOllama(tool: McpToolSchema): OllamaTool {
  return {
    type: "function",
    function: { name: tool.name, description: tool.description, parameters: tool.inputSchema },
  };
}

/** First string/number argument, JSON-quoted, for the compact tool-call line. */
function formatArgs(args: Record<string, unknown>): string {
  const value = Object.values(args).find((v) => typeof v === "string" || typeof v === "number");
  return value === undefined ? "" : JSON.stringify(value);
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

/** Base-name match, mirroring `src/services/llm_config.py`'s check. */
function modelInstalled(installed: string[], model: string): boolean {
  return installed.some(
    (m) => m === model || m.startsWith(`${model}:`) || model.startsWith(m.split(":")[0]),
  );
}

async function checkOllamaModel(baseUrl: string, model: string): Promise<void> {
  let response: Response;
  try {
    response = await fetchWithTimeout(`${baseUrl}/api/tags`, 5_000);
  } catch {
    console.error(chalk.red(`could not reach Ollama at ${baseUrl} — is it running?`));
    process.exit(1);
  }
  if (!response.ok) {
    console.error(chalk.red(`could not reach Ollama at ${baseUrl} (HTTP ${response.status})`));
    process.exit(1);
  }
  const data = (await response.json()) as { models?: Array<{ name: string }> };
  const installed = (data.models ?? []).map((m) => m.name);
  if (!modelInstalled(installed, model)) {
    console.error(
      chalk.red(`model '${model}' is not installed in Ollama — run \`ollama pull ${model}\``),
    );
    process.exit(1);
  }
}

async function checkWeaviateReady(): Promise<void> {
  const host = envVar("WEAVIATE_HOST", "localhost");
  const port = envVar("WEAVIATE_HTTP_PORT", "8080");
  const url = `http://${host}:${port}`;
  try {
    const response = await fetchWithTimeout(`${url}/v1/.well-known/ready`, 3_000);
    if (response.ok) return;
  } catch {
    // fall through to error below
  }
  console.error(chalk.red(`Weaviate is not ready at ${url} — run \`dokkai up\` first`));
  process.exit(1);
}

async function chatOllama(
  baseUrl: string,
  model: string,
  messages: OllamaMessage[],
  tools: OllamaTool[],
  signal: AbortSignal,
): Promise<OllamaMessage> {
  let response: Response;
  try {
    response = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, messages, tools, stream: false }),
      signal,
    });
  } catch (cause) {
    if (cause instanceof Error && cause.name === "AbortError") throw cause;
    throw new Error(
      `could not reach Ollama at ${baseUrl} — is it running? ` +
        `(${cause instanceof Error ? cause.message : String(cause)})`,
    );
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(
      `Ollama rejected the request (${response.status}) — is model '${model}' pulled? ` +
        `Try: ollama pull ${model}${detail ? `\n${detail}` : ""}`,
    );
  }
  const data = (await response.json()) as { message: OllamaMessage };
  return data.message;
}

interface SessionTotals {
  tokens: number;
  calls: number;
}

/** Run agent rounds for one user turn: tool calls until a final plain-text
 * answer, up to MAX_ROUNDS. Dedupes identical (name, args) calls across the
 * whole conversation instead of relying on the prompt alone. */
async function runTurn(
  baseUrl: string,
  model: string,
  messages: OllamaMessage[],
  tools: OllamaTool[],
  mcp: McpClient,
  calledSignatures: Set<string>,
  totals: SessionTotals,
  signal: AbortSignal,
): Promise<void> {
  for (let round = 1; round <= MAX_ROUNDS; round++) {
    const message = await chatOllama(baseUrl, model, messages, tools, signal);
    messages.push(message);

    let toolCalls = message.tool_calls ?? [];
    let usedFallback = false;
    if (toolCalls.length === 0) {
      const fallback = extractFallbackToolCall(message.content ?? "");
      if (fallback) {
        toolCalls = [{ function: fallback }];
        usedFallback = true;
      }
    }

    if (toolCalls.length === 0) {
      if (totals.calls > 0) {
        console.log(
          chalk.dim(
            `  (session tool payload: ~${totals.tokens} tokens over ${totals.calls} calls)`,
          ),
        );
      }
      console.log(message.content || "(empty answer)");
      return;
    }

    for (const call of toolCalls) {
      const name = call.function.name;
      let rawArgs = call.function.arguments;
      if (typeof rawArgs === "string") {
        try {
          rawArgs = JSON.parse(rawArgs);
        } catch {
          rawArgs = {};
        }
      }
      const args = (rawArgs ?? {}) as Record<string, unknown>;
      const signature = `${name}:${JSON.stringify(args)}`;
      const note = usedFallback ? " (fallback parse)" : "";

      let text: string;
      if (calledSignatures.has(signature)) {
        text =
          `ERROR: you already called ${name}(${JSON.stringify(args)}) with these exact ` +
          "arguments earlier in this conversation — reuse that result, do not call it again.";
        console.log(chalk.dim(`→ ${name}(${formatArgs(args)}) · skipped (duplicate call)${note}`));
      } else {
        calledSignatures.add(signature);
        const result = await mcp.callTool(name, args);
        text = result.isError ? `ERROR: ${result.text}` : result.text;
        const toks = tokensOf(text);
        totals.tokens += toks;
        totals.calls += 1;
        console.log(chalk.dim(`→ ${name}(${formatArgs(args)}) · ${toks} tokens${note}`));
      }

      messages.push({ role: "tool", content: text + TOOL_RESULT_REMINDER });
    }
  }

  console.log(chalk.yellow(`(max ${MAX_ROUNDS} rounds reached without a final answer)`));
}

/** Keep the system prompt plus the most recent MAX_HISTORY-1 messages. */
function trimHistory(messages: OllamaMessage[]): void {
  const excess = messages.length - MAX_HISTORY;
  if (excess > 0) messages.splice(1, excess);
}

/**
 * Run the `dokkai srcs --model ollama:<name> --agent` REPL: preflight
 * Ollama + Weaviate, spawn the MCP server, mirror its tools into Ollama's
 * tool-calling format, then loop reading lines from stdin — same UX as the
 * non-agent `/chat` mode (`dokkai>` prompt, `/exit`, Ctrl-C).
 */
export async function runOllamaAgent(model: string, flags: AgentFlags): Promise<void> {
  const ollamaBaseUrl = envVar("OLLAMA_BASE_URL", "http://localhost:11434");

  await checkOllamaModel(ollamaBaseUrl, model);
  await checkWeaviateReady();

  const home = resolveDokkaiHome();
  const mcp = new McpClient("uv", ["run", "--directory", home, "python", "src/mcp_server.py"], {
    cwd: home,
    env: { ...process.env, DOKKAI_MCP_PROFILE: "small-model" },
    onStderr: (chunk) => process.stderr.write(chalk.dim(chunk)),
  });

  let tools: OllamaTool[];
  try {
    await mcp.initialize();
    tools = (await mcp.listTools()).map(mcpToolToOllama);
  } catch (error) {
    await mcp.close();
    console.error(
      chalk.red(
        `could not start the dokkai MCP server: ${error instanceof Error ? error.message : String(error)}`,
      ),
    );
    process.exit(1);
  }

  console.log(
    chalk.dim(`MCP server ready — ${tools.length} tools: ${tools.map((t) => t.function.name).join(", ")}`),
  );

  let systemPrompt = SYSTEM_PROMPT;
  if (flags.project) {
    systemPrompt += ` The project is '${flags.project}'; pass project='${flags.project}' to tools when they accept it.`;
  }
  const messages: OllamaMessage[] = [{ role: "system", content: systemPrompt }];
  const calledSignatures = new Set<string>();
  const totals: SessionTotals = { tokens: 0, calls: 0 };

  console.log(
    chalk.green(`Agentic chat with ollama:${model} (MCP tools) — /exit or Ctrl-D to quit.`),
  );
  console.log();

  const rl = createInterface({ input: process.stdin, output: process.stdout });

  // Same manual line-queueing as lib/chat.ts's REPL — avoids rl.question()/
  // the async iterator racing stdin EOF and dropping buffered lines.
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
    let waiter: (typeof waiters)[number] | undefined;
    while ((waiter = waiters.shift())) waiter({ closed: true });
  });

  function nextLine(): Promise<{ line: string } | { closed: true }> {
    const line = pendingLines.shift();
    if (line !== undefined) return Promise.resolve({ line });
    if (closed) return Promise.resolve({ closed: true });
    return new Promise((resolve) => waiters.push(resolve));
  }

  let activeAbort: AbortController | null = null;

  const onSigint = () => {
    if (activeAbort) {
      activeAbort.abort();
      return;
    }
    rl.close();
    console.log();
    console.log(chalk.dim("goodbye"));
    void mcp.close().finally(() => process.exit(0));
  };
  process.on("SIGINT", onSigint);

  try {
    process.stdout.write("dokkai> ");
    while (true) {
      const result = await nextLine();
      if ("closed" in result) break;

      const line = result.line.trim();
      if (!line) {
        process.stdout.write("dokkai> ");
        continue;
      }
      if (line === "/exit") break;

      messages.push({ role: "user", content: line });
      const controller = new AbortController();
      activeAbort = controller;
      try {
        await runTurn(
          ollamaBaseUrl,
          model,
          messages,
          tools,
          mcp,
          calledSignatures,
          totals,
          controller.signal,
        );
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") {
          console.log(chalk.yellow("(aborted)"));
        } else {
          console.error(chalk.red(error instanceof Error ? error.message : String(error)));
        }
      } finally {
        activeAbort = null;
      }
      trimHistory(messages);

      console.log();
      process.stdout.write("dokkai> ");
    }
  } finally {
    process.off("SIGINT", onSigint);
    rl.close();
    await mcp.close();
  }

  console.log(chalk.dim("goodbye"));
  process.exit(0);
}
