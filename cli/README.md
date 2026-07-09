# dokkai

CLI for [dokkai](https://github.com/gusmaooleo/dokkai) — feed the dokkai retrieval database and launch **SRCS** (Semantic Retrieval Codebase System) sessions.

Dokkai turns a codebase into a graph-aware vector database that local AI agents can read, search and document. This package is the command-line front end for it: ingest a repository, watch job progress live, export its dependency graph, and launch Claude Code / Codex / a local Ollama chat with dokkai's retrieval wired in.

> Full project docs (architecture, API reference, MCP server): see the [main README](https://github.com/gusmaooleo/dokkai#readme).

## Install

```bash
npm install -g dokkai   # once published to npm
```

Until then, install from source:

```bash
git clone https://github.com/gusmaooleo/dokkai.git
cd dokkai/cli
npm install
npm run build
npm link
```

## Requirements

`dokkai` is a client for the dokkai backend — it doesn't run the backend itself. You need, running somewhere reachable over HTTP:

- **Weaviate** (`docker compose up -d` from the dokkai repo root — `dokkai up` does this for you)
- **The dokkai API** (`./dev.sh` from the dokkai repo root)
- **[Ollama](https://ollama.com/)**, with the embedding/descriptor models pulled — required for `dokkai ingest`; `dokkai graph` works without it

`dokkai up` and `dokkai status` need to find the dokkai repo (for `docker-compose.yml` and to read its `.env`) — either run CLI commands from inside that repo, or set `DOKKAI_HOME`.

Not sure what's missing? `dokkai doctor` checks node/docker/uv/`DOKKAI_HOME` plus Weaviate/Ollama/API reachability and prints the fix command for each problem it finds.

## Commands

| Command | Description |
| --- | --- |
| `dokkai up` | `docker compose up -d` for Weaviate, wait for it to become ready, probe Ollama and the configured embed/descriptor models (warnings only), report whether the API is reachable |
| `dokkai status` | Read-only health check (API, Weaviate, Ollama + model presence) and a list of ingested projects; always exits 0 |
| `dokkai ingest <repo-path> [--recreate] [--no-describe] [--yes]` | Run the full ingestion pipeline via the API, with live stage progress (SSE, polling fallback) |
| `dokkai graph <repo-path\|project> [--out <file>]` | Graph-only run (no LLM, no vectorization) for a directory, or a normalized graph export for an already-ingested project name |
| `dokkai srcs --model <claude\|codex\|ollama:<name>> [--project <name>] [--agent]` | Register the dokkai MCP server and launch Claude Code / Codex, or start a terminal chat loop against a local Ollama model. With `--agent` (ollama only): agentic mode — the CLI spawns the dokkai MCP server itself and navigates the codebase with MCP tools locally, no API server needed. |
| `dokkai watch <repo-path> [--debounce <seconds>=3] [--no-describe]` | Initial catch-up ingest, then debounced incremental re-ingestion on every file save; skips a cycle if one is already running for the project; Ctrl-C exits cleanly |
| `dokkai doctor` | Diagnose the local environment (node/docker/uv/`DOKKAI_HOME`, Weaviate, Ollama, API) with fix commands for anything missing; exits 1 if a required item is missing |

Flags:

- `--api <url>` (global) — dokkai API URL. Default `http://localhost:8000`.
- `--recreate` (`ingest`) — drop and rebuild the **entire** Weaviate collection (all projects) before inserting. Prompts for confirmation unless `--yes` is also passed.
- `--no-describe` (`ingest`, `watch`) — skip per-entity LLM descriptions; the `summary` vector stays empty and search quality is reduced. Skips the descriptor pre-flight the API otherwise enforces.
- `--yes` (`ingest`) — skip the `--recreate` confirmation prompt.
- `--out <file>` (`graph`) — write the graph JSON to a file instead of stdout.
- `--project <name>` (`srcs`, with `ollama:<name>`) — project to chat about; auto-detected if exactly one project is ingested.
- `--agent` (`srcs`, with `ollama:<name>` only) — agentic MCP loop instead of the `/chat` REPL; requires only Weaviate + Ollama (no FastAPI). Per-call token lines + session total printed in the REPL.
- `--debounce <seconds>` (`watch`) — seconds to wait after the last change before re-ingesting. Default `3`.

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `DOKKAI_API_URL` | `http://localhost:8000` | dokkai API URL. Overridden by `--api`; falls back to the dokkai repo's `.env` if unset. |
| `DOKKAI_HOME` | _(auto-detected)_ | Path to the dokkai repo, used by `up`/`status` to run `docker compose` and read `.env`. Auto-detected by walking up from the current directory for `docker-compose.yml` + `src/mcp_server.py`; set this if you run the CLI from outside the repo. |

Precedence for all resolved values: CLI flag > environment variable > the dokkai repo's `.env` > built-in default.

## SRCS recipes

```bash
# Claude Code, with the dokkai MCP server registered
dokkai srcs --model claude

# Codex, same registration
dokkai srcs --model codex

# Local Ollama model, terminal chat loop over dokkai retrieval
dokkai srcs --model ollama:qwen2.5-coder:latest --project your-repo

# Local Ollama model, agentic MCP loop — no API server needed
dokkai srcs --model ollama:qwen2.5-coder:latest --project your-repo --agent
```

The `claude`/`codex` variants (re-)register the dokkai MCP server idempotently, then launch the tool interactively. The `ollama:<name>` variant sets the dokkai API's global chat model and starts a REPL (`/exit` or Ctrl-D to quit) with conversation continuity across turns. `--agent` (ollama only) skips the API entirely: the CLI spawns the dokkai MCP server itself and the model navigates the codebase locally with MCP tools — measured live, the canonical question ("how does the alarm flow work?") was answered via one `search` call, ≈679 tokens.

## Keeping the index fresh

```bash
dokkai watch /absolute/path/to/your-repo
dokkai watch /absolute/path/to/your-repo --debounce 5 --no-describe
```

Runs an initial catch-up ingest, then re-ingests incrementally on every save.

## License

MIT
