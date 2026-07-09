#!/usr/bin/env node

import { createRequire } from "node:module";
import { Command } from "commander";
import chalk from "chalk";
import { runUp } from "./commands/up.js";
import { runStatus } from "./commands/status.js";
import { runIngest } from "./commands/ingest.js";
import { runGraph } from "./commands/graph.js";
import { runSrcs } from "./commands/srcs.js";

const require = createRequire(import.meta.url);
const { version } = require("../package.json") as { version: string };

const program = new Command();

program
  .name("dokkai")
  .description(
    "Feed the dokkai retrieval database and launch SRCS (Semantic Retrieval " +
      "Codebase System) sessions.",
  )
  .version(version, "-v, --version")
  .option("--api <url>", "dokkai API URL (default: http://localhost:8000)");

program
  .command("up")
  .description(
    "check/start local dependencies via docker compose and verify Ollama " +
      "models are present",
  )
  .action(() => runUp(program.opts()));

program
  .command("status")
  .description("show API/Weaviate/Ollama health and ingested projects")
  .action(() => runStatus(program.opts()));

program
  .command("ingest <repo-path>")
  .description("run the full ingestion pipeline (with graphs) via the API")
  .option("--recreate", "recreate the collection before ingesting")
  .option(
    "--no-describe",
    "ingest without LLM descriptions — the summary vector stays empty; " +
      "search quality is reduced",
  )
  .option("--yes", "skip the --recreate confirmation prompt")
  .action(
    (
      repoPath: string,
      options: { recreate?: boolean; describe: boolean; yes?: boolean },
    ) => runIngest(repoPath, { ...program.opts(), ...options }),
  );

program
  .command("graph <repo-path|project>")
  .description(
    "graph-only run: generate/export the dependency graph with no LLM and " +
      "no vectorization",
  )
  .option("--out <file>", "write the graph JSON to this file")
  .action(
    (target: string, options: { out?: string }) =>
      runGraph(target, { ...program.opts(), ...options }),
  );

program
  .command("srcs")
  .description(
    "SRCS mode: launch a coding agent (or local Ollama model) with dokkai " +
      "retrieval available",
  )
  .requiredOption(
    "--model <target>",
    "claude | codex | ollama:<name>",
  )
  .action((options: { model: string }) => runSrcs(options));

program.parseAsync(process.argv).catch((error: unknown) => {
  console.error(
    chalk.red(error instanceof Error ? error.message : String(error)),
  );
  process.exit(1);
});
