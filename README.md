# Dokkai

**Turn any codebase into a graph‑aware vector database that local AI agents can read, search and document — using a fraction of the tokens.**

Dokkai ingests a repository, builds a full dependency graph of it (calls, inheritance, definitions, modules), slices the real source into rich chunks, and stores everything in a vector database. On top of that it serves a **graph‑augmented RAG**: instead of returning a handful of loosely‑matched snippets, it retrieves the *connected* neighbourhood of the code you asked about, so a local LLM can answer accurately and generate documentation for parts of the system that were never documented.

Everything runs **100% locally** — Weaviate for vectors, Ollama for both embeddings and generation.

> The name *dokkai* (読解) is Japanese for "reading comprehension" — which is exactly what this gives an LLM over your code.

---

## Table of contents

- [Why](#why)
- [MCP server](#mcp-server)
- [CLI](#cli)
- [How it works](#how-it-works)
- [Descriptions (Tier 2)](#descriptions-tier-2)
- [Features](#features)
- [Tech stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Project structure](#project-structure)
- [Current limitations](#current-limitations)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why

- **Token economy.** Agents grep and read whole files to understand code. Dokkai lets them pull the exact connected context semantically, instead of paying tokens to read everything.
- **Automatic documentation.** Generate complete, audience‑specific docs (developer / manager / customer) for systems that were never documented.
- **Semantic search for agents.** Ask "how does the alarm flow work?" and get the relevant subsystem — not keyword hits.
- **Local & private.** Your code never leaves your machine.

---

## MCP server

Dokkai's core deliverable is a **stdio MCP server** (`dokkai`, `src/mcp_server.py`) that exposes the same graph-aware retrieval used by `/chat` as tools any MCP-capable agent — Claude Code, Codex, or any other MCP client — can call directly. Instead of an agent grepping and reading whole files to understand a codebase, it asks dokkai for exact, connected, token-lean context.

- **Transport: stdio only.** There is no HTTP/SSE MCP transport, and no new HTTP endpoints ship with it — see [API reference](#api-reference) for the separate, unrelated FastAPI routes.
- **Requires only Weaviate up and a project ingested** (`docker compose up -d` + `POST /instances/pipeline`). The FastAPI app (`./dev.sh`) is **not** needed to use the MCP server.
- **Launch command:** `uv run --directory <repo root> python src/mcp_server.py`
- Responses are compact plain text (not JSON), with English-only errors (unknown project, entity not found, out-of-range line, etc.). `project` auto-resolves when exactly one project is ingested; pass it explicitly otherwise.

### Tools

| Tool | Parameters | Returns | Typical size |
| --- | --- | --- | --- |
| `list_projects` | – | Ingested projects with chunk counts, node/edge counts, `generated_at` | ~105 chars |
| `search` | `query`, `project?`, `k=8` | Ranked hits — qualified_name, entity type, absolute `path:lines`, score, truncated description | ~2.7k chars (~680 tok) at `k=8` |
| `grep_project` | `pattern`, `project?`, `k=10` | Literal keyword search — BM25 over stored code text (tokenized, **not regex**) for a known identifier; ranked hits with qualified_name, entity type, absolute `path:lines`, BM25 score; clean "no matches" string on a blank pattern or no hits | ~800 chars (~200 tok) at `k=10` |
| `get_entity` | `qualified_name`, `project?` | One entity's relations (calls/called_by/inherits/implements/overrides/methods), summary and full source | ~0.5–1.6k chars |
| `neighbors` | `qualified_name`, `project?`, `direction=both`, `depth=1`, `limit=30` | Graph neighborhood (calls/inherits/implements/overrides/defines) as a node list + edge list, BFS up to `depth` hops | ~0.6k chars |
| `context` | `query`, `project?`, `k=8` | One-shot seed + graph-expanded context bundle, ready to use as LLM context, capped at 10,000 chars | ≤10k chars (~2.5k tok) |
| `get_file` | `path`, `project?`, `start_line?`, `end_line?` | Raw file content, optionally a 1-indexed inclusive line range; capped at 2,000 lines / 100 KB per call (with a truncation note) | file-dependent |

### Instructions profile

The server's `instructions` string (sent to the MCP client, shaping how the model uses the tools) has two variants, selected by the `DOKKAI_MCP_PROFILE` env var — the tool contracts (names, parameters, return shapes) are **identical** in both:

- **Unset (default)** — the original, concise instructions.
- **`small-model`** — a more directive, anti-loop variant aimed at 3B‑class local models, which the harness (`scripts/mcp_harness.py`) showed can otherwise loop or re-call tools with the same arguments; it explicitly points at `grep_project` for known identifiers and tells the model to stop calling tools once it has enough information. Set this when wiring a small local model through the harness or a future SRCS integration (roadmap feature 04).

### Session watchdog

The server always tracks per-tool call counts and response sizes for the running session and prints a summary (per-tool calls/chars/~tokens plus totals) to stderr on shutdown — no configuration needed. Useful to measure real tool usage against a token budget without instrumenting a client.

### Measured token budget

`scripts/mcp_harness.py` spawns the MCP server as a subprocess, drives a local Ollama model through the tool-calling loop for a fixed question, and reports the total tool-response payload (chars/4 token approximation) against a budget:

```bash
uv run python scripts/mcp_harness.py \
  [--question "how does the alarm flow work?"] \
  [--project saffira_back-end] \
  [--model qwen2.5-coder:3b] \
  [--budget 5000]
```

Measured on `saffira_back-end` with the local `qwen2.5-coder:3b` model:

| Question | Tool calls used | Payload | Budget | Result |
| --- | --- | --- | --- | --- |
| Canonical: "how does the alarm flow work?" | one `context` call | ≈2,478 tokens | 5,000 | PASS |
| A generalization question (different topic) | one `search` call | ≈639 tokens | 5,000 | PASS |

### Registration

**Claude Code:**

```bash
claude mcp add dokkai -- uv run --directory /absolute/path/to/dokkai python src/mcp_server.py
claude mcp list      # verify: dokkai ... ✔ Connected
```

Remove: `claude mcp remove dokkai`

> Verified live: `claude mcp list` reports `dokkai ... ✔ Connected`, and a real call to `list_projects` (via `claude -p "Using only dokkai MCP tools, call list_projects and report the raw output" --allowedTools "mcp__dokkai__*"`) returned:
> ```
> saffira_back-end — chunks: 2770, nodes: 4760, edges: 8959, generated_at: 2026-07-08T22:43:50.164048+00:00
> ```

**Codex:**

```bash
codex mcp add dokkai -- uv run --directory /absolute/path/to/dokkai python src/mcp_server.py
codex mcp list       # verify: dokkai ... enabled
```

Remove: `codex mcp remove dokkai`

> Verified live: `codex mcp add`/`codex mcp list` register the server and report it `enabled` (Codex connects to it and lists its tools correctly). A real tool call via `codex exec "call the dokkai list_projects MCP tool and report its raw output"` could **not** be completed non-interactively: every attempt (default `approval: never`, and again under `-s workspace-write`) failed with `user cancelled MCP tool call` — non-interactive `codex exec` appears to auto-decline the approval prompt it needs for a newly-registered MCP server's first tool call rather than auto-approve it under a `never` policy. This is a Codex CLI non-interactive-mode limitation, not a dokkai server defect. Approve the first call once from an interactive `codex` session to unblock subsequent non-interactive use.

**Generic stdio MCP client** (e.g. an `mcpServers` config block):

```json
{
  "mcpServers": {
    "dokkai": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/dokkai", "python", "src/mcp_server.py"]
    }
  }
}
```

---

## CLI

The dokkai CLI (`dokkai`, package [`cli/`](cli/)) is a Node/TypeScript command-line front end for the API and the MCP server above: ingest a repo with live progress, export the graph, or launch a coding agent with dokkai retrieval wired in.

**Install:**

```bash
cd cli
npm install
npm run build
npm link       # `dokkai` is now on PATH
```

> Will be available as `npm install -g dokkai` once published to npm — for now, `npm link` from `cli/`.

### Commands

| Command | Description |
| --- | --- |
| `dokkai up` | `docker compose up -d` for Weaviate, wait for it to become ready, probe Ollama and the configured embed/descriptor models (warnings only, non-fatal), report whether the API is reachable. Exits 1 only on a `docker compose`/Weaviate failure (or when the dokkai repo root cannot be resolved). |
| `dokkai status` | Read-only health check (API, Weaviate, Ollama + model presence) plus a list of ingested projects. Always exits 0. |
| `dokkai ingest <repo-path> [--recreate] [--no-describe] [--yes]` | Validates the path, confirms `--recreate` interactively (or via `--yes`), calls `POST /instances/pipeline`, and streams live stage progress (SSE with a polling fallback) to a result summary. |
| `dokkai graph <repo-path\|project> [--out <file>]` | Graph-only run: a **directory** argument enqueues `POST /instances/graph` (cgr only, no LLM/Weaviate) and prints the canonical graph JSON path (with `--out`, also exports the normalized graph); a **project name** argument fetches and prints/writes its normalized structural graph (`GET /graph/{project}?include=structural`) — stdout output pipes cleanly when `--out` is omitted. |
| `dokkai srcs --model <claude\|codex\|ollama:<name>> [--project <name>]` | `claude`/`codex`: idempotently (re-)registers the dokkai MCP server, then launches the tool interactively. `ollama:<name>`: sets the API's global chat model (`POST /config/llm`) and starts a terminal REPL over `/chat`'s SSE stream, with conversation continuity across turns. |

Global flag: `--api <url>` — dokkai API URL (default `http://localhost:8000`).

`ingest`-only flags:

- `--recreate` — drop and rebuild the **entire** Weaviate collection (all projects) before inserting; prompts `This wipes ALL ingested projects from Weaviate. Continue? [y/N]` unless `--yes` is also passed. In a non-interactive shell without `--yes`, the command exits 1 instead of proceeding.
- `--no-describe` — sets `describe: false` on the pipeline request: skips the descriptor pre-flight (no fail-loud `400` if `DESC_MODEL` is missing/unpulled) and ingests without per-entity LLM descriptions — the `summary` named vector stays empty for this project, so conceptual/summary search over it is weaker (literal/code-vector search still works). CLI prints `descriptions: disabled (descriptions disabled for this ingestion)` in the result summary.
- `--yes` — skip the `--recreate` confirmation prompt.

`graph`-only flag: `--out <file>` — write the graph JSON to a file instead of stdout.

`srcs`-only flag: `--project <name>` — with `ollama:<name>`, the project to chat about; auto-detected when exactly one project is ingested (errors listing the options if zero or multiple are ingested and `--project` is omitted).

### Environment (CLI-side)

`DOKKAI_API_URL` and `DOKKAI_HOME` (see [Configuration](#configuration) below) are resolved by the CLI itself, not the API — set them in your shell, or rely on the dokkai repo's `.env` as a fallback. Precedence: `--api` flag > `DOKKAI_API_URL` env var > repo `.env` > default.

### SRCS recipes

```bash
# Claude Code, with the dokkai MCP server (re-)registered
dokkai srcs --model claude

# Codex, same registration
dokkai srcs --model codex

# Local Ollama model, terminal chat loop over dokkai retrieval
dokkai srcs --model ollama:qwen2.5-coder:latest --project your-repo
```

### Graph-only runs

`dokkai graph` (and `POST /instances/graph`) run **only** `cgr` — no chunking, no descriptions, no embedding, no Weaviate. Stages go straight `cgr → done`, and the job's result carries just `ingest.output_json`. Useful to inspect or export a repo's dependency graph without paying for (or requiring) Ollama/Weaviate at all.

---

## How it works

```
                      ┌─────────────────────────────────────────────┐
                      │                 Ollama (host)               │
                      │   nomic-embed-text   ·   qwen2.5-coder      │
                      └──────▲───────────────────────────▲──────────┘
              embeddings     │                           │  generation
                             │                           │
  repo ──► cgr ──► graph JSON ──► chunker ──► Weaviate ◄──┤
       (ephemeral      (nodes +     (real source     (vectors +     │
        Memgraph)     relations)   + graph edges)    metadata)      │
                                                         │          │
                                              hybrid search +       │
                                              graph expansion       │
                                                         ▼          │
  user ──► FastAPI  /chat  ──── builds connected context ───────────┘──► streams answer (SSE)
```

**Ingestion pipeline** (`POST /instances/pipeline`):

1. **Graph extraction** — [code-graph-rag](https://github.com/vitali87/code-graph-rag) (`cgr`) spins up an ephemeral Memgraph container and emits a JSON graph of the repo (4k+ nodes, 8k+ relationships for a medium backend).
2. **Chunking** — each code entity (Class / Function / Method / Interface / Enum / Type) becomes one chunk containing a compact header, its **graph relations** (calls, called_by, inherits, …) and the **actual source code** sliced from the file.
3. **Vectorization & storage** — chunks are written to Weaviate, which embeds them via Ollama (`nomic-embed-text`). Each object gets a **deterministic UUID** derived from `(project, qualified_name)`, so re‑ingestion upserts in place (no duplicates).

**Retrieval** (`search_graph`):

1. **Seeds** — hybrid search (vector + BM25) finds the entry points.
2. **Graph expansion** — a breadth‑first walk over the stored `qualified_name` edges pulls in callers/callees/definitions, scored with per‑hop decay and edge weights, with hub‑node protection so a popular utility doesn't flood the context.
3. **Context assembly** — results are ranked and formatted with provenance (`SEED` vs `hop 1 · calls ← X`) so the model understands the structure.

**Generation** (`POST /chat`) — the connected context is injected into an audience‑specific prompt and streamed from the local Ollama model over Server‑Sent Events.

---

## Descriptions (Tier 2)

Every describable entity (not a test, has source, not a one‑liner) gets a one‑sentence natural‑language summary, stored in a dedicated `summary` named vector so conceptual queries reach implementation code.

- **Two ways to get a description, no LLM call when avoidable:**
  - **Template (no‑LLM)** — doc‑less, non‑test `Type` aliases and `Enum`s without methods get a deterministic templated sentence (e.g. `Enumeration Status with members ACTIVE, INACTIVE.`). Free and instant; a human docstring always wins over a template.
  - **LLM‑generated** — everything else goes through the configured descriptor model with a **trimmed prompt**: entity type + qualified name, the extracted doc/leading comment (capped), and a capped source excerpt (cut at a line boundary) — instead of the full source. Measured ~21% faster per entity than the original full‑source prompt (0.61 s → 0.48 s/entity, qwen2.5-coder:3b, warm model, 20‑entity sample), with comparable quality.
- **Cached by source hash** — unchanged source reuses its cached description on re‑ingestion, so incremental ingests only pay the LLM for what changed.
- **Pluggable descriptor provider** — the descriptor is routed through the same LLM provider abstraction used for chat: `DESC_PROVIDER=ollama` (default, local) `| openai | anthropic`, selected via environment variables only (no `/config` endpoint for it).

### Refreshing descriptions

`POST /instances/{project}/describe` re-runs **only** the describe pass over a project that's already ingested — no `cgr`, no re-embedding of unchanged code. Useful when you switch descriptor models (`DESC_MODEL`/`DESC_PROVIDER`) or want to fill in gaps without paying for a full re-ingest.

- The cache is still honored by default — unchanged entities are cache hits and cost nothing. Pass `{"force": true}` in the body to bypass the cache and regenerate every eligible description.
- **Anti-drift guard**: the endpoint re-chunks the latest graph JSON in `ingested/` and only describes chunks whose source is byte-identical to what's stored in Weaviate. Entities whose source changed on disk since ingestion are counted as `stale_source` and skipped (with a note to re-ingest) — refresh never describes text that isn't what's actually indexed.
- A transient generation failure never overwrites a good stored description with an empty one — such entities are counted as `preserved`.
- Only the `description` property (and the `summary` vector it feeds) is updated; the `code` vector is untouched.
- Runs as a background job like the pipeline: returns `{job_id, status}`, poll `GET /instances/jobs/{job_id}`.
- **Fail‑loud policy** — `POST /instances/pipeline` always ingests with descriptions on. The descriptor is checked *before* the job is created: if it's missing or unavailable, the request fails immediately with **HTTP 400** and **no job is created** (nothing is ingested):
  - No model configured: `no descriptor model specified — set DESC_MODEL (e.g. qwen2.5-coder:3b) or configure a remote provider via DESC_PROVIDER`
  - Model not pulled: `descriptor model '<m>' is not available in Ollama at <base_url> — pull it with 'ollama pull <m>'`
  - Set `describe: false` in the `POST /instances/pipeline` body (or `dokkai ingest --no-describe`) to opt out explicitly — this skips the descriptor pre-flight and ingests without descriptions.

### Ingestion cost (measured)

Full pipeline run on `saffira_back-end` (medium TS/Python backend), local Ollama, `qwen2.5-coder:3b`, `DESC_CONCURRENCY=4`, cold description cache:

| Metric | Value |
| --- | --- |
| Graph entities → chunks upserted | 2,771 → 2,770 |
| Descriptions | 1,577 LLM‑generated + 162 templated = 1,739 described |
| Skipped (tests / trivial / no source) | 1,032 |
| Failed | 0 |
| Full pipeline wall‑clock (graph + describe + embed/upsert) | **14 min 40 s** |

---

## Features

- ✅ Full repo → dependency graph → vectorized chunks pipeline
- ✅ Real source code embedded (not just metadata)
- ✅ Graph‑augmented retrieval (multi‑hop, decay, hub protection)
- ✅ Deterministic UUIDs → idempotent re‑ingestion / upsert
- ✅ Absolute file paths on every chunk — chat sources and LLM context cite real paths on disk (`/full/path/file.py:42`)
- ✅ Streaming chat over the codebase (SSE) with 3 audiences: `developer`, `manager`, `customer`
- ✅ 100% local: Weaviate + Ollama (embeddings **and** generation)
- ✅ Pluggable providers — LLM: Ollama / OpenAI / Anthropic · embeddings: Ollama / OpenAI / Cohere
- ✅ Auto‑configure + warm the chat model on startup (no cold starts)
- ✅ **MCP server** — 7 tools (search, literal grep, graph navigation, entity/file lookup) for Claude Code, Codex and any stdio MCP client, with a small-model instructions profile and a session usage watchdog (see [MCP server](#mcp-server))
- ✅ **npm CLI** (`dokkai`) — `up`/`status`/`ingest`/`graph`/`srcs` commands with live job progress and one-command SRCS sessions (Claude Code, Codex, or a local Ollama REPL) (see [CLI](#cli))
- ✅ **Graph-only ingestion** (`POST /instances/graph`, `dokkai graph`) — run just `cgr` with no LLM/Weaviate involved

---

## Tech stack

| Layer | Technology |
| --- | --- |
| API | FastAPI (Python ≥ 3.14, managed with `uv`) |
| Vector DB | Weaviate `1.28` (`text2vec-ollama`) |
| Graph extraction | code-graph-rag (`cgr`) + ephemeral Memgraph |
| Embeddings | Ollama · `nomic-embed-text` |
| Generation | Ollama · `qwen2.5-coder` (any Ollama chat model) |
| MCP server | Official Python `mcp` SDK (`FastMCP`), stdio transport |
| CLI | Node.js (≥22.12) / TypeScript, `commander` + `chalk` + `ora` |
| Frontend (WIP) | Next.js 16 · React 19 · Tailwind v4 |

---

## Prerequisites

- **Docker** (runs Weaviate, and the ephemeral Memgraph that `cgr` uses)
- **[uv](https://docs.astral.sh/uv/)** (Python dependency manager)
- **[Ollama](https://ollama.com/)** running on the host

---

## Quickstart

### 1. Start Ollama and pull the models

```bash
ollama serve                      # if not already running
ollama pull nomic-embed-text      # embeddings (8k context) — REQUIRED before ingestion
ollama pull qwen2.5-coder         # chat/generation (or :14b, :32b if your hardware allows)
ollama pull qwen2.5-coder:3b      # descriptor model — REQUIRED before ingestion (see DESC_MODEL below)
```

> ⚠️ Ollama must be up **before** you run the pipeline — Weaviate calls it to embed each chunk at insert time, and `POST /instances/pipeline` rejects the request with HTTP 400 if the descriptor model (`DESC_MODEL`) isn't configured and pulled — see [Descriptions](#descriptions-tier-2).

### 2. Start Weaviate

```bash
docker compose up -d
```

> ⚠️ If you ever change the vectorizer (e.g. `EMBED_MODEL` or `VECTORIZER_PROVIDER`), wipe the volume first so the collection is recreated with the new config: `docker compose down -v && docker compose up -d`.

### 3. Configure the environment

Create a `.env` in the project root:

```dotenv
# --- Weaviate (client side) ---
WEAVIATE_HOST=localhost
WEAVIATE_HTTP_PORT=8080
WEAVIATE_GRPC_PORT=50051
COLLECTION_NAME=CodeEntity

# --- Embeddings (Ollama, local) ---
VECTORIZER_PROVIDER=ollama
EMBED_MODEL=nomic-embed-text
# As seen by the Weaviate *server* (Docker → host):
OLLAMA_EMBED_ENDPOINT=http://host.docker.internal:11434

# --- Chat LLM (Ollama, local) ---
# As seen by the API *process*:
OLLAMA_BASE_URL=http://localhost:11434
# Set this to auto-configure + warm the chat model on startup:
OLLAMA_CHAT_MODEL=qwen2.5-coder:latest
OLLAMA_KEEP_ALIVE=-1
OLLAMA_NUM_CTX=8192
OLLAMA_TIMEOUT=600

# --- Descriptions (Tier 2, required for ingestion — see Descriptions section) ---
DESC_MODEL=qwen2.5-coder:3b
DESC_CONCURRENCY=4
# DESC_PROVIDER=ollama       # ollama (default) | openai | anthropic
```

### 4. Run the API

```bash
./dev.sh            # serves on http://localhost:8000
PORT=9000 ./dev.sh  # custom port
```

`dev.sh` wraps `uv run uvicorn main:app --reload --app-dir src` (the app uses absolute imports rooted at `src/`).

### 5. Ingest a repository

```bash
curl -X POST localhost:8000/instances/pipeline \
  -H 'Content-Type: application/json' \
  -d '{"repo_path":"/absolute/path/to/your/repo"}'
```

This runs the full pipeline: graph extraction (`cgr`) → chunking (with source) → descriptions → embedding/storage — the job reports these as stages `cgr → chunk → describe → upsert → done`. The `project_name` is taken from the repo's top‑level folder name.

### 6. Chat with the codebase

If you set `OLLAMA_CHAT_MODEL`, the model is already configured and warm. Otherwise configure it once:

```bash
curl -X POST localhost:8000/config/llm \
  -H 'Content-Type: application/json' \
  -d '{"is_local":true,"provider_data":{"provider_name":"ollama","model":"qwen2.5-coder:latest"}}'
```

Then ask away (SSE stream — note `-N` and that `project_name` must match the ingested repo):

```bash
curl -N -X POST localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"how does the alarm flow work?","project_name":"your-repo","audience":"developer"}'
```

### 7. (Optional) Frontend

A Next.js scaffold lives in [`frontend/`](frontend/) (not yet wired to the API — see [Roadmap](#roadmap)). The API already allows CORS from any origin.

```bash
cd frontend
npm install
npm run dev         # http://localhost:3000
```

---

## Configuration

All configuration is via environment variables (loaded from `.env` at startup).

| Variable | Default | Description |
| --- | --- | --- |
| `WEAVIATE_HOST` | `localhost` | Weaviate host (client side) |
| `WEAVIATE_HTTP_PORT` | `8080` | Weaviate HTTP port |
| `WEAVIATE_GRPC_PORT` | `50051` | Weaviate gRPC port |
| `COLLECTION_NAME` | `CodeEntity` | Weaviate collection name |
| `VECTORIZER_PROVIDER` | `ollama` | `ollama` \| `openai` \| `cohere` \| `local` |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `OLLAMA_EMBED_ENDPOINT` | `http://host.docker.internal:11434` | Ollama URL **as seen by the Weaviate server** (Docker → host) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama URL **as seen by the API process** (chat) |
| `OLLAMA_CHAT_MODEL` | _(unset)_ | If set: auto‑configure + warm this chat model on boot |
| `OLLAMA_NUM_CTX` | `8192` | Context window for generation |
| `OLLAMA_TIMEOUT` | `600` | httpx read timeout, in seconds (covers cold model loads) |
| `OLLAMA_KEEP_ALIVE` | `-1` | Keep model resident: `-1` = forever, or a duration like `30m` |
| `OPENAI_API_KEY` / `COHERE_API_KEY` | _(unset)_ | Only when using those vectorizer providers, or when `DESC_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | _(unset)_ | Only required when `DESC_PROVIDER=anthropic` |
| `DESC_PROVIDER` | `ollama` | Provider for per-entity descriptions: `ollama` \| `openai` \| `anthropic` |
| `DESC_MODEL` | _(unset)_ | Model used to generate per-entity descriptions (e.g. `qwen2.5-coder:3b`). If unset (or unavailable), `POST /instances/pipeline` fails with HTTP 400 before creating a job — see [Descriptions](#descriptions-tier-2) |
| `DESC_CONCURRENCY` | `4` | Max concurrent description requests |
| `RETRIEVAL_TEST_PENALTY` | `0.35` | Score multiplier applied to test-file results during retrieval |
| `DOKKAI_RECREATE_COLLECTION` | _(unset)_ | If truthy: recreate the Weaviate collection instead of failing on schema mismatch |
| `DOKKAI_MCP_PROFILE` | _(unset)_ | MCP server only. `small-model` swaps the server's instructions for a more directive, anti-loop variant tuned for small local models — see [Instructions profile](#instructions-profile) |
| `MEMGRAPH_IMAGE` | `memgraph/memgraph:latest` | Image used by `cgr` for graph extraction |
| `DOKKAI_API_URL` | `http://localhost:8000` | **CLI-side only** (not read by the API). dokkai API URL for the `dokkai` CLI; overridden by `--api`, falls back to this repo's `.env` — see [CLI](#cli) |
| `DOKKAI_HOME` | _(auto-detected)_ | **CLI-side only.** Path to this repo, used by `dokkai up`/`dokkai status` to run `docker compose` and read `.env`. Auto-detected by walking up from the cwd for `docker-compose.yml` + `src/mcp_server.py` — see [CLI](#cli) |

> **Two Ollama endpoints, on purpose:** `OLLAMA_EMBED_ENDPOINT` is called by the Weaviate **container** (so it needs `host.docker.internal`), while `OLLAMA_BASE_URL` is called by the **API process** on the host (so it's `localhost`).

---

## API reference

> These are the HTTP endpoints exposed by the FastAPI app (`./dev.sh`). The **MCP server is separate and stdio-only** — it adds no HTTP endpoints; see [MCP server](#mcp-server).

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/instances/pipeline` | Run the full ingestion pipeline for a `repo_path`, `describe: true` by default (409 if a job is already running for that project; `describe: false` skips the descriptor pre-flight and ingests without descriptions) |
| `POST` | `/instances/graph` | Graph-only run for a `repo_path` — `cgr` only, no LLM, no vectorization, no Weaviate (stages `cgr → done`, `kind: "graph"`; 409 if a job is already running for that project) |
| `POST` | `/instances/{project}/describe` | Refresh descriptions for an already-ingested project (background job; optional `{"force": true}`; 409 if a job is already running for that project) — see [Refreshing descriptions](#refreshing-descriptions) |
| `GET` | `/instances/jobs` | List ingestion/refresh jobs |
| `GET` | `/instances/jobs/{id}` | Get a job's status, `stage`/`stage_progress` and result |
| `GET` | `/instances/jobs/{id}/events` | Stream a job's progress as Server‑Sent Events (`event: job` on each update, `event: done` when it finishes) |
| `GET` | `/graph` | List projects with an ingested graph |
| `GET` | `/graph/{project}` | Full project graph (code entities by default, `?include=structural` for everything), `?limit=N` |
| `GET` | `/graph/{project}/neighborhood` | BFS neighborhood of an entity — `?entity=<qualified_name>&depth=&direction=&limit=` |
| `GET` | `/graph/{project}/files` | File‑to‑file dependency view (internal files only) |
| `POST` | `/chat` | Chat over the codebase (SSE: `sources` → `token` → `done`) |
| `GET` | `/chat/conversations` | List conversations |
| `GET` | `/chat/conversations/{id}` | Get a conversation's history |
| `DELETE` | `/chat/conversations/{id}` | Delete a conversation |
| `POST` | `/config/llm` | Set the active LLM provider/model (rejects an invalid/retired remote model) |
| `GET` | `/config/llm` | Get the current LLM config |
| `GET` | `/config/llm/models` | List available models for the provider (live catalog, static fallback on failure) |
| `GET` | `/config/llm/health` | Check connectivity — probes the configured model |
| `GET` | `/` | Health check |

Jobs report `stage` through the unified vocabulary `cgr → chunk → describe → upsert → done` (refresh jobs use `chunk → describe → update → done`; graph-only jobs use just `cgr → done`), plus a `stage_progress` object `{"stage", "done", "total"}` mirroring the current stage's `done`/`total` counters, and a `kind` (`"pipeline"` | `"refresh"` | `"graph"`).

**Per‑project job lock** — only one job (pipeline, refresh, or graph-only) may run at a time for a given project; submitting a second one (any of `/instances/pipeline`, `/instances/{project}/describe`, `/instances/graph`) returns HTTP 409 `another job is running for project '<project>'` and creates no job.

**Job progress over SSE** — instead of polling `GET /instances/jobs/{id}`, `GET /instances/jobs/{id}/events` streams the same job payload over Server‑Sent Events (same framing as `/chat`): it emits `event: job` on every `updated_at` change and a terminal `event: done` with the final payload, then closes the stream. An unknown job id is a plain 404, not a stream.

### Graph API

All four `/graph...` endpoints are read‑only and served straight from the ingested graph JSON in `ingested/` — no separate database, restart‑proof, lazily loaded and cached in memory per project.

- **`GET /graph`** — list every project with an ingested graph: `[{project, file, nodes, edges, generated_at}]`.
- **`GET /graph/{project}?include=structural&limit=N`** — the project's graph, normalized to `{id, kind, name, qualified_name, path, absolute_path, start_line, end_line}` nodes and `{source, target, type}` edges, plus a `stats` block (`total_nodes`/`total_edges`/`returned_nodes`/`returned_edges`/`truncated`). By default only code‑entity nodes (Class/Function/Method/Interface/Enum/Type) and edges among them are returned; `include=structural` returns everything (folders, files, modules, packages, the project, external packages). `limit` defaults to 5000 with no upper bound. Edges never dangle — both endpoints are always in the returned node set.
- **`GET /graph/{project}/neighborhood?entity=<qualified_name>&depth=1&direction=both&limit=200`** — BFS from a single entity over `CALLS`/`INHERITS`/`IMPLEMENTS`/`OVERRIDES`/`DEFINES`/`DEFINES_METHOD` edges only (never `IMPORTS` or containment/external edges). Nodes carry a `hop` (0 = the entity itself); the center node is also returned separately as `center`. `direction` picks which adjacency to follow (`in` | `out` | `both`); `depth` bounds hops; `limit` caps the total node count, closest‑first.
- **`GET /graph/{project}/files`** — file‑to‑file dependency view: `{files:[{path,name,absolute_path}], edges:[{source,target,weight,types:{TYPE:count}}], stats:{files,edges}}`. Internal files only (external packages/modules are dropped) and self‑loops are skipped.
- Unknown project → 404 `no graph found for project '<project>' — run POST /instances/pipeline to ingest it`. Unknown entity on the neighborhood endpoint → 404 `entity '<qualified_name>' not found in project '<project>' graph`.

---

## Project structure

```
dokkai/
├── src/
│   ├── main.py                  # FastAPI app + startup (auto-config/warm LLM)
│   ├── controllers/             # instances, graph, chat, config routes
│   ├── services/
│   │   ├── ingest.py            # runs code-graph-rag (cgr), promotes canonical JSON
│   │   ├── chunker.py           # graph JSON → rich code chunks (with source)
│   │   ├── vectorize.py         # chunk + store orchestrator
│   │   ├── weaviate_client.py   # collection schema, upsert, deterministic UUIDs
│   │   ├── graph_store.py       # loads/caches ingested graph JSON; graph query endpoints
│   │   ├── retriever.py         # hybrid search + graph expansion
│   │   ├── chat.py              # RAG chat pipeline (retrieve → prompt → stream)
│   │   ├── jobs.py              # background job store (per-project lock, SSE events)
│   │   ├── llm_provider.py      # Ollama / OpenAI / Anthropic abstraction
│   │   ├── llm_config.py        # provider config store + validation
│   │   └── chat_store.py        # conversation history (in-memory)
│   └── models/dtos/             # pydantic request/response models
├── shell/run_cgr.sh             # code-graph-rag runner (ephemeral Memgraph)
├── cli/                         # npm CLI package (`dokkai`) — see CLI section
├── frontend/                    # Next.js app (WIP)
├── docker-compose.yml           # Weaviate (text2vec-ollama)
├── dev.sh                       # dev server launcher
└── ingested/                    # canonical graph JSON per project — <project>.json,
                                  # promoted from cgr's timestamped output after each
                                  # successful pipeline run; served by the Graph API
```

---

## Current limitations

These are known and tracked in the [Roadmap](#roadmap):

- **In‑memory LLM config** — the active LLM provider/model resets on restart (re‑seedable via `OLLAMA_CHAT_MODEL`). Chat history *is* persisted — as JSON files under `data/conversations/` — but not yet in a database.
- **Single repository at a time** per collection.
- **Retrieval bias toward tests** — for "how does X work?" queries, test files often out‑rank the real implementation (test descriptions read like specs). End‑to‑end tests are also decoupled from implementation by the HTTP boundary, so graph expansion can't bridge them.
- **The API is not yet containerized** — only Weaviate runs in `docker compose`; the API runs via `dev.sh`.
- **Anonymous functions** (e.g. test closures) have weak graph edges and unstable identifiers.
- **Absolute paths are denormalized per chunk at ingest time, not re‑derived.** If a repo moves on disk (or is cloned to a new location), re‑ingest it — there's no project‑root registry that keeps paths correct automatically. Re‑ingesting is cheap: deterministic UUIDs upsert unchanged entities in place, and their descriptions come from the source‑hash cache instead of a fresh LLM call — only the changed paths/content cost anything.

---

## Roadmap

### Feature roadmap

| # | Feature | Status |
| --- | --- | --- |
| 01 | Describe v2 (lighter descriptions, template descriptions, provider selection, fail‑loud policy) + absolute file paths | ✅ done |
| 02 | Graph query API | ✅ done |
| 03 | MCP server (core) | ✅ done (pending merge) |
| 04 | npm CLI (`dokkai`) + SRCS mode | ✅ done (pending merge) |
| 05 | Postgres conversation history | planned |
| 06 | Basic auth (root user via env) | planned |
| 07 | Frontend UI (chat + graph + config) | planned |
| 08 | PDF ingestion — local NotebookLM | planned |
| 09 | Code review & bug‑hunt routines | planned |
| 10 | Documentation site (en/pt/zh/es) | planned |
| 11 | Post‑01 polish (live provider model catalogs, stage‑level job progress, describe refresh endpoint) | ✅ done |
| 12 | MCP polish (`grep_project` tool, small-model instructions profile, session watchdog) | ✅ done (pending merge) |

### Retrieval quality

**Tier 1 — zero‑cost enrichment** (no extra LLM calls; improves the embedded text and ranking):
- [x] **De‑prioritize test files** in seed retrieval (down‑weight `*.test.*` / `*.spec.*` / `/tests/`) so implementation surfaces first. _(highest impact / lowest effort — next up; re‑ranks existing data, no re‑ingest needed)_
- [x] **Identifier tokenization** — split `camelCase` / `snake_case` in the embedded text (`sendDetectionNotification` → "send detection notification") so code matches natural‑language queries.
- [x] **Reuse existing docstrings / leading comments** as the entity description when present, before falling back to LLM generation.

**Tier 2 — LLM micro‑descriptions** (the big unlock for conceptual queries — a one‑time, incremental cost):
- [x] **Per‑entity natural‑language summary**, stored as a dedicated **named vector** (`summary` alongside `code`) so retrieval can target intent or literal code.
- [x] Generated by a **small dedicated model** (e.g. `qwen2.5:3b`), **selectively** — skip tests, getters/setters and trivial one‑liners.
- [x] **Cached by source hash** and **incremental via the deterministic UUID**: describe each entity once; re‑ingestion only re‑describes what changed.
- [x] **Background ingestion job** with progress — the description pass makes ingestion long‑running, so `/instances/pipeline` becomes an async job (status endpoint) instead of a blocking request.

**Later:**
- [ ] **Feature clustering** — group related entities (controller + use‑case + repository + job) into a feature and retrieve the whole subgraph together.

### Frontend (Next.js)
- [ ] Wire the UI to the API: streaming chat, sources panel with provenance (`hop` / `via`), audience switcher.
- [ ] **Interactive system graph** — visualize the dependency graph of the ingested codebase.
- [ ] Environment & model configuration UI.

### One‑command infrastructure
- [ ] Containerize the API (fill in the `Dockerfile`) and add the API + frontend to `docker-compose.yml`, so `docker compose up` brings up the **entire** stack.

### Persistence & multi‑environment
- [ ] Move state to a database — the in‑memory LLM config and the JSON‑file chat history.
- [ ] **Encrypted storage** of environments and API keys.
- [ ] **Remote environment configuration** (manage environments from the UI) — with `.env` kept as a fallback.

### Exploratory (longer‑term)
- [ ] **CI/CD code review** — run Dokkai on pull requests; on merge, incrementally upsert only the changed entities (the deterministic‑UUID groundwork is already in place), enabling versioned indexing without duplication.

---

## License

Open source — license TBD.

Contributions and ideas welcome.
