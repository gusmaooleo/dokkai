/**
 * Minimal JSON-RPC-over-stdio MCP client: only what the agentic srcs loop
 * needs (`initialize`, `tools/list`, `tools/call`) — no MCP SDK dependency.
 *
 * Framing matches the Python MCP SDK's stdio transport (verified against
 * `mcp.client.stdio.stdio_client` in `.venv`): one JSON-RPC 2.0 object per
 * line, newline-delimited, on stdin/stdout. stderr is free for the server's
 * own logging (the dokkai MCP server never writes to stdout).
 */

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";

export interface McpToolSchema {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
}

export interface McpCallResult {
  text: string;
  isError: boolean;
}

interface McpClientOptions {
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  /** Called with each raw stderr chunk from the child (its own log lines). */
  onStderr?: (chunk: string) => void;
}

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
}

// Any of these are accepted by the reference server (mcp.shared.version);
// pin to a recent stable one rather than "latest" so this doesn't silently
// track SDK version bumps.
const PROTOCOL_VERSION = "2025-06-18";
const REQUEST_TIMEOUT_MS = 120_000;
const SHUTDOWN_STEP_MS = 1_000;

/** Resolves after `promise` settles, or `false` after `timeoutMs` elapses first. */
function withDeadline(promise: Promise<void>, timeoutMs: number): Promise<boolean> {
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(false), timeoutMs);
    promise.then(() => {
      clearTimeout(timer);
      resolve(true);
    });
  });
}

export class McpClient {
  private readonly child: ChildProcessWithoutNullStreams;
  private nextId = 1;
  private readonly pending = new Map<number, PendingRequest>();
  private buffer = "";

  constructor(command: string, args: string[], options: McpClientOptions = {}) {
    this.child = spawn(command, args, {
      cwd: options.cwd,
      env: options.env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.child.stdout.setEncoding("utf8");
    this.child.stdout.on("data", (chunk: string) => this.onStdout(chunk));
    this.child.stderr.setEncoding("utf8");
    this.child.stderr.on("data", (chunk: string) => options.onStderr?.(chunk));
    this.child.on("error", (error) => this.rejectAll(error));
    this.child.on("exit", (code, signal) => {
      this.rejectAll(
        new Error(`dokkai MCP server exited (code ${code ?? "null"}, signal ${signal ?? "null"})`),
      );
    });
  }

  private onStdout(chunk: string): void {
    this.buffer += chunk;
    let index: number;
    while ((index = this.buffer.indexOf("\n")) !== -1) {
      const line = this.buffer.slice(0, index);
      this.buffer = this.buffer.slice(index + 1);
      if (line.trim()) this.handleLine(line);
    }
  }

  private handleLine(line: string): void {
    let message: { id?: number; result?: unknown; error?: { message?: string } };
    try {
      message = JSON.parse(line);
    } catch {
      return; // not a JSON-RPC line (shouldn't happen — stdout is protocol-only)
    }
    if (message.id === undefined || message.id === null) return; // server notification — ignore
    const pending = this.pending.get(message.id);
    if (!pending) return;
    this.pending.delete(message.id);
    if (message.error) {
      pending.reject(new Error(message.error.message ?? "MCP request failed"));
    } else {
      pending.resolve(message.result);
    }
  }

  private rejectAll(error: Error): void {
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }

  private request(method: string, params?: unknown): Promise<any> {
    const id = this.nextId++;
    const payload = JSON.stringify({
      jsonrpc: "2.0",
      id,
      method,
      ...(params !== undefined ? { params } : {}),
    });
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`MCP request '${method}' timed out after ${REQUEST_TIMEOUT_MS / 1000}s`));
      }, REQUEST_TIMEOUT_MS);
      this.pending.set(id, {
        resolve: (value) => {
          clearTimeout(timer);
          resolve(value);
        },
        reject: (error) => {
          clearTimeout(timer);
          reject(error);
        },
      });
      this.child.stdin.write(payload + "\n");
    });
  }

  private notify(method: string, params?: unknown): void {
    const payload = JSON.stringify({
      jsonrpc: "2.0",
      method,
      ...(params !== undefined ? { params } : {}),
    });
    this.child.stdin.write(payload + "\n");
  }

  async initialize(): Promise<void> {
    await this.request("initialize", {
      protocolVersion: PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: { name: "dokkai-cli", version: "0.1.0" },
    });
    this.notify("notifications/initialized");
  }

  async listTools(): Promise<McpToolSchema[]> {
    const result = await this.request("tools/list");
    const tools = (result?.tools ?? []) as Array<{
      name: string;
      description?: string;
      inputSchema?: Record<string, unknown>;
    }>;
    return tools.map((t) => ({
      name: t.name,
      description: t.description ?? "",
      inputSchema: t.inputSchema ?? { type: "object", properties: {} },
    }));
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<McpCallResult> {
    const result = await this.request("tools/call", { name, arguments: args });
    const content = Array.isArray(result?.content)
      ? (result.content as Array<{ text?: string }>)
      : [];
    const text = content
      .map((c) => c.text)
      .filter((t): t is string => typeof t === "string")
      .join("\n");
    return { text, isError: Boolean(result?.isError) };
  }

  /**
   * MCP stdio shutdown sequence: close stdin (server sees EOF and exits on
   * its own — this is when its atexit summary prints), escalating to
   * SIGTERM then SIGKILL if it doesn't exit in time.
   */
  async close(): Promise<void> {
    this.rejectAll(new Error("MCP client closed"));
    if (this.child.exitCode !== null || this.child.signalCode !== null) return;

    const exited = new Promise<void>((resolve) => this.child.once("exit", () => resolve()));
    try {
      this.child.stdin.end();
    } catch {
      // stdin may already be closed
    }
    if (await withDeadline(exited, SHUTDOWN_STEP_MS)) return;

    this.child.kill("SIGTERM");
    if (await withDeadline(exited, SHUTDOWN_STEP_MS)) return;

    this.child.kill("SIGKILL");
    await exited;
  }
}
