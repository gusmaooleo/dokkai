import { spawn } from "node:child_process";
import chalk from "chalk";
import { resolveDokkaiHome } from "../lib/config.js";
import { runOllamaChat } from "../lib/chat.js";

export interface SrcsFlags {
  model?: string;
  api?: string;
  project?: string;
}

const SUPPORTED_TOOLS = ["claude", "codex"] as const;
type SupportedTool = (typeof SUPPORTED_TOOLS)[number];

const INSTALL_HINTS: Record<SupportedTool, string> = {
  claude: "install Claude Code — https://docs.claude.com/en/docs/claude-code",
  codex: "install Codex CLI — https://github.com/openai/codex",
};

const README_POINTER =
  "see the README's MCP server § Registration section for a generic " +
  "mcpServers config block";

function runCommand(
  command: string,
  args: string[],
  stdio: "inherit" | "ignore",
): Promise<number> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio });
    child.on("error", reject);
    child.on("close", (code) => resolve(code ?? 1));
  });
}

/** True iff `command` is runnable on PATH (any exit code counts as present). */
function commandExists(command: string): Promise<boolean> {
  return new Promise((resolve) => {
    const child = spawn(command, ["--version"], { stdio: "ignore" });
    child.on("error", () => resolve(false));
    child.on("close", () => resolve(true));
  });
}

/**
 * Idempotently (re-)register the dokkai MCP server with `tool`: remove any
 * existing registration (failures ignored, output suppressed — there may be
 * none yet), then add it with the byte-exact N-05 launch command.
 */
export async function registerMcpServer(tool: SupportedTool): Promise<void> {
  const home = resolveDokkaiHome();

  await runCommand(tool, ["mcp", "remove", "dokkai"], "ignore");

  const addCode = await runCommand(
    tool,
    [
      "mcp",
      "add",
      "dokkai",
      "--",
      "uv",
      "run",
      "--directory",
      home,
      "python",
      "src/mcp_server.py",
    ],
    "inherit",
  );
  if (addCode !== 0) {
    throw new Error(
      `'${tool} mcp add dokkai' failed with exit code ${addCode}`,
    );
  }
}

async function runSupportedTool(tool: SupportedTool): Promise<never> {
  const onPath = await commandExists(tool);
  if (!onPath) {
    console.error(
      chalk.red(`'${tool}' was not found on PATH — ${INSTALL_HINTS[tool]}`),
    );
    process.exit(1);
  }

  try {
    await registerMcpServer(tool);
  } catch (error) {
    console.error(
      chalk.red(error instanceof Error ? error.message : String(error)),
    );
    process.exit(1);
  }

  console.log(chalk.green(`dokkai MCP server registered with ${tool}.`));

  const exitCode = await runCommand(tool, [], "inherit");
  process.exit(exitCode);
}

export async function runSrcs(flags: SrcsFlags): Promise<void> {
  const model = flags.model ?? "";

  if ((SUPPORTED_TOOLS as readonly string[]).includes(model)) {
    await runSupportedTool(model as SupportedTool);
    return;
  }

  if (model.startsWith("ollama:")) {
    const ollamaModel = model.slice("ollama:".length);
    if (!ollamaModel) {
      console.error(
        chalk.red("--model ollama:<name> requires a model name after 'ollama:'"),
      );
      process.exit(1);
    }
    await runOllamaChat(ollamaModel, flags);
    return;
  }

  console.error(
    chalk.red(
      `unsupported --model '${model}' — srcs supports: claude, codex, ` +
        `ollama:<name>; ${README_POINTER}.`,
    ),
  );
  process.exit(1);
}
