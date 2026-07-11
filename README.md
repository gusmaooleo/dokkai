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
- [Frontend UI](#frontend-ui)
- [One-command full stack](#one-command-full-stack)
- [Code review routines](#code-review-routines)
  - [Bug hunt](#bug-hunt)
  - [Playbooks & skills](#playbooks--skills)
  - [Routines UI](#routines-ui)
- [How it works](#how-it-works)
- [Descriptions (Tier 2)](#descriptions-tier-2)
- [Features](#features)
- [Tech stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Authentication](#authentication)
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

Something not working? `dokkai doctor` complements `up`/`status`: it checks the whole local toolchain (node, docker, uv, `DOKKAI_HOME`) as well as Weaviate/Ollama/API reachability, and prints the exact command to fix whatever's missing.

### Commands

| Command | Description |
| --- | --- |
| `dokkai up [--full]` | `docker compose up -d` for Weaviate/Postgres/Memgraph, wait for Weaviate to become ready, probe Ollama and the configured embed/descriptor models (warnings only, non-fatal), report whether the API is reachable. With `--full`, brings up the **whole stack** instead (`docker compose --profile full up -d`) — the containerized API and frontend UI — and waits for both to become reachable. Exits 1 only on a `docker compose`/Weaviate failure (or when the dokkai repo root cannot be resolved). |
| `dokkai status` | Read-only health check (API, Weaviate, Ollama + model presence) plus a list of ingested projects. Always exits 0. |
| `dokkai ingest <repo-path> [--recreate] [--no-describe] [--yes]` | Validates the path, confirms `--recreate` interactively (or via `--yes`), calls `POST /instances/pipeline`, and streams live stage progress (SSE with a polling fallback) to a result summary. |
| `dokkai graph <repo-path\|project> [--out <file>]` | Graph-only run: a **directory** argument enqueues `POST /instances/graph` (cgr only, no LLM/Weaviate) and prints the canonical graph JSON path (with `--out`, also exports the normalized graph); a **project name** argument fetches and prints/writes its normalized structural graph (`GET /graph/{project}?include=structural`) — stdout output pipes cleanly when `--out` is omitted. |
| `dokkai srcs --model <claude\|codex\|ollama:<name>> [--project <name>] [--agent]` | `claude`/`codex`: idempotently (re-)registers the dokkai MCP server, then launches the tool interactively. `ollama:<name>`: sets the API's global chat model (`POST /config/llm`) and starts a terminal REPL over `/chat`'s SSE stream, with conversation continuity across turns. With `--agent` (ollama only): agentic mode — the CLI spawns the dokkai MCP server itself and navigates the codebase locally with MCP tools, no API server needed (only Weaviate + Ollama). |
| `dokkai watch <repo-path> [--debounce <seconds>=3] [--no-describe]` | Runs an initial ingest cycle, then watches the repo and debounces file changes into incremental re-ingestions (`POST /instances/pipeline`, `recreate: false`) via the API. Ignores `.git`, `node_modules`, build/venv/data dirs and dotfiles; skips a cycle (and reschedules) on a 409 (job already running for the project); Ctrl-C stops cleanly. |
| `dokkai doctor` | Read-only environment diagnosis: node/docker/uv/`DOKKAI_HOME` (required), Weaviate readiness, Ollama reachability + `EMBED_MODEL`/`DESC_MODEL` presence, and API reachability (warnings). Prints the exact fix command for each missing/warning item. Exits 1 if a required item is missing, 0 otherwise. |

Global flag: `--api <url>` — dokkai API URL (default `http://localhost:8000`).

`up`-only flags: `--full` — bring up the containerized API and frontend UI too (compose `full` profile) and wait for both to become reachable (see [One-command full stack](#one-command-full-stack)); `--ui` — **deprecated alias for `--full`**, kept for backward compatibility.

`ingest`-only flags:

- `--recreate` — drop and rebuild the **entire** Weaviate collection (all projects) before inserting; prompts `This wipes ALL ingested projects from Weaviate. Continue? [y/N]` unless `--yes` is also passed. In a non-interactive shell without `--yes`, the command exits 1 instead of proceeding.
- `--no-describe` — sets `describe: false` on the pipeline request: skips the descriptor pre-flight (no fail-loud `400` if `DESC_MODEL` is missing/unpulled) and ingests without per-entity LLM descriptions — the `summary` named vector stays empty for this project, so conceptual/summary search over it is weaker (literal/code-vector search still works). CLI prints `descriptions: disabled (descriptions disabled for this ingestion)` in the result summary.
- `--yes` — skip the `--recreate` confirmation prompt.

`graph`-only flag: `--out <file>` — write the graph JSON to a file instead of stdout.

`watch`-only flags: `--debounce <seconds>` (default `3`) — seconds to wait after the last change before re-ingesting; `--no-describe` — same as `ingest`'s flag, applied on every cycle.

`srcs`-only flags:

- `--project <name>` — with `ollama:<name>`, the project to chat about; auto-detected when exactly one project is ingested (errors listing the options if zero or multiple are ingested and `--project` is omitted).
- `--agent` — with `ollama:<name>` only (errors otherwise): agentic MCP loop instead of the `/chat` REPL. The CLI spawns the dokkai MCP server itself over a hand-rolled stdio JSON-RPC client (`DOKKAI_MCP_PROFILE=small-model`), mirrors its live tool schemas into Ollama `/api/chat` tool calling, and runs the same search/read/answer loop as `scripts/mcp_harness.py` — fallback tool-call parser, tool-result reminder, 8-round cap, identical-call dedupe — in the same `dokkai>` REPL. Prints a per-call token line and a session total. Requires only Weaviate + Ollama running (no FastAPI). Measured live with the API stopped: the canonical question ("how does the alarm flow work?") answered via one `search` call, ≈679 tokens.

### Environment (CLI-side)

`DOKKAI_API_URL` and `DOKKAI_HOME` (see [Configuration](#configuration) below) are resolved by the CLI itself, not the API — set them in your shell, or rely on the dokkai repo's `.env` as a fallback. Precedence: `--api` flag > `DOKKAI_API_URL` env var > repo `.env` > default.

### What is SRCS?

**SRCS** ("Semantic Retrieval Codebase System") is how you plug dokkai's
retrieval into the coding agent you already use, in one command:
`dokkai srcs`. With `--model claude` or `--model codex` it registers dokkai
as an **MCP server** for Claude Code / Codex, so that agent gets dokkai's
search/graph tools ([MCP server](#mcp-server)) alongside its own, and
launches it interactively. With `--model ollama:<name>` it instead runs a
**terminal chat/agentic loop against a local Ollama model** over that same
retrieval — either a `/chat`-backed REPL (needs the API), or, with
`--agent`, a fully local MCP tool-loop that needs only Weaviate + Ollama
(no FastAPI). Same underlying retrieval, three different front ends —
pick whichever fits how you already work.

| I want to... | Use |
| --- | --- |
| Keep using Claude Code / Codex, with dokkai as one more tool it can call | `dokkai srcs --model claude` (or `codex`) |
| Chat about a codebase from a terminal, no coding-agent CLI installed | `dokkai srcs --model ollama:<name> --project <name>` |
| Run a fully local agentic loop with no API server at all | `dokkai srcs --model ollama:<name> --project <name> --agent` |

### SRCS recipes

```bash
# Claude Code, with the dokkai MCP server (re-)registered
dokkai srcs --model claude

# Codex, same registration
dokkai srcs --model codex

# Local Ollama model, terminal chat loop over dokkai retrieval
dokkai srcs --model ollama:qwen2.5-coder:latest --project your-repo

# Local Ollama model, agentic MCP loop — no API server needed
dokkai srcs --model ollama:qwen2.5-coder:latest --project your-repo --agent
```

### Keeping the index fresh

```bash
dokkai watch /absolute/path/to/your-repo
dokkai watch /absolute/path/to/your-repo --debounce 5 --no-describe
```

Runs an initial catch-up ingest, then re-ingests incrementally on every save (deterministic UUIDs + the description cache keep each cycle cheap). Leave it running alongside your editor while using `srcs` against the same project.

### Graph-only runs

`dokkai graph` (and `POST /instances/graph`) run **only** `cgr` — no chunking, no descriptions, no embedding, no Weaviate. Stages go straight `cgr → done`, and the job's result carries just `ingest.output_json`. Useful to inspect or export a repo's dependency graph without paying for (or requiring) Ollama/Weaviate at all.

---

## Frontend UI

A Next.js web UI (`frontend/`) covers the same surface as the API: chat with
the codebase, browse its dependency graph, manage ingestions, and configure
the LLM — all authenticated against the same `admin`/`user`/`viewer` accounts
as the CLI and MCP server.

**Run it:**

```bash
# API must already be reachable — ./dev.sh + docker compose up -d (see Quickstart)
cd frontend
npm install
npm run dev         # http://localhost:3000
```

**Or, one command for the whole stack** — see
[One-command full stack](#one-command-full-stack) below.

First login is `admin`/`admin` (or your `DOKKAI_ROOT_USER`/`DOKKAI_ROOT_PASSWORD` — see [Authentication](#authentication)).

**Screens:**

- **Login** — username/password against `/auth/login`; shows a hint when the default `admin`/`admin` credentials are still active.
- **Shell** — sidebar project switcher, nav, a live job-pulse indicator, light/dark theme, and a persistent banner for admins while the default admin credentials are active.
- **Chat** — SSE-streamed answers with a sources panel (seed/hop/via provenance badges, code, jump-to-graph), an audience switcher (developer/manager/customer), and markdown rendering.
- **History** — search, group and resume past conversations, or delete them.
- **Graph** — interactive dependency graph (sigma.js): search/focus/zoom, neighborhood highlighting on selection, a detail panel (description, relations), an Entities/Files toggle, `?focus=` deep links, and a "chat about this" prefill.
- **Instances** — live job progress over SSE with the real stage vocabulary, a new-ingestion dialog (including the `--recreate` wipes-all-projects warning), and a projects grid.
- **Settings** — LLM provider/model selection and health check, plus an account card to change your own password.
- **Admin** — user management (create/list/delete, roles `admin`/`user`/`viewer`).

**Configuration:**

| Variable | Default | Description |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Base URL of the Dokkai API, inlined at build time |

If the frontend is served from an origin other than `http://localhost:3000` /
`http://127.0.0.1:3000`, add it to the API's `DOKKAI_CORS_ORIGINS` — see
[Configuration](#configuration).

**Roles in the UI** mirror the API (see [Authentication](#authentication)):
`viewer` is read-only (no chat composer, no destructive buttons); `user` gets
everything except the Admin screen and LLM config writes; `admin` gets
everything, plus the default-admin banner.

See [`frontend/README.md`](frontend/README.md) for the stack, folder layout and dev commands.

---

## One-command full stack

```bash
dokkai up --full
# equivalent: docker compose --profile full up -d
```

`docker compose up -d` (no profile) is unchanged — it stays infra-only
(Weaviate, Postgres, a resident Memgraph). The `full` profile additionally
builds and runs the **API** (`Dockerfile`) and the **frontend UI**
containers, so `dokkai up --full` alone reproduces the whole system with no
`./dev.sh`. `--ui` is a **deprecated alias for `--full`**, kept for
backward compatibility.

**Repo access (`DOKKAI_REPOS_DIR`, required for the containerized API):**
the container needs read/write access to the repos you ingest/review. Set
`DOKKAI_REPOS_DIR` to the **parent directory** of those repos — it's
bind-mounted into the API container at the exact same absolute path, so
`repo_path` values (stored at ingest time) resolve identically whether the
API runs in the container or via `./dev.sh` on the host:

```bash
# .env — repos live under /Users/you/projects/*
DOKKAI_REPOS_DIR=/Users/you/projects
```

Only repos under that mounted root are reachable from the containerized
API. The mount is read-write — routines' git operations and `cgr`'s
`.cgr-hash-cache.json` both write into the repo.

**Ollama config just works.** If you saved an LLM config pointing at
`http://localhost:11434` (e.g. from a host `./dev.sh` run, sharing the same
Postgres), the containerized API rewrites `localhost`/`127.0.0.1` to
`host.docker.internal` **at request time only** — never persisted to
storage — so the same saved config works from both a host run and the
container.

**Ingested data is per-runtime — important caveat.** The containerized API
stores `ingested/` (graph JSON) and its description cache (`data/`) in
**named Docker volumes**, separate from the host's `ingested/`/`data/`
directories. A project ingested via `./dev.sh` on the host is **not**
visible to the containerized API, and vice versa — ingest again against
whichever runtime you're actually using.

**Port 8000 is exclusive.** The containerized API and `./dev.sh` both bind
`:8000` — run one or the other, not both, at a time.

**Memgraph is now a resident compose service** (not an ephemeral
per-ingest container): ingestion is slightly faster (no ~5s container boot
per run), and `shell/run_cgr.sh` connects to it via `MEMGRAPH_HOST` — set
by default in the `full` profile and in `dev.sh`. Unset `MEMGRAPH_HOST` to
fall back to the legacy ephemeral-container flow. Because the resident
instance is shared, ingestion/review/graph runs across **all** projects are
now globally serialized (a process-wide lock around the `cgr` subprocess)
rather than merely colliding on port 7687 as before.

**macOS bind-mount performance.** Docker Desktop's bind-mount filesystem
sharing on macOS is noticeably slower than a native filesystem for
large ingests (many files/large repos) — expect ingestion under `--full` to
take longer than the same repo via `./dev.sh` on macOS. Linux (native bind
mounts) isn't affected.

---

## Code review routines

> **Routines are an experimental demo — not part of the dokkai core.**
> Core = the retrieval pipeline, MCP server, chat and graph. Code review,
> bug hunt and the playbooks/skills library below are all routines; they
> may be removed in a future release, or promoted to core, depending on
> adoption.

Dokkai can review a git branch against a base branch using the same graph-aware
context as chat: it diffs the two refs, pulls in the graph entities and
1-hop call-graph neighbors touched by each changed file, and asks the
configured LLM for per-file findings (severity, category, an anchored line
range when the model's `start_line` falls inside a changed range, and an
optional fix suggestion), then writes a markdown summary of the whole run.

This feature shipped in three sub-parts: sub-part A shipped the review
routine backend; sub-part B added the bug-hunt routine
([Bug hunt](#bug-hunt)) and project-specific playbooks/skills
([Playbooks & skills](#playbooks--skills)); sub-part C added the
[Routines UI](#routines-ui) (launch, live progress, findings/diff browser,
history, playbooks/skills library) and the containerized full-stack compose
profile ([One-command full stack](#one-command-full-stack)).

**Run it** (review runs as a background job on the same shared job system as
ingestion — see [API reference](#api-reference)):

```bash
TOKEN=$(curl -s -X POST localhost:8000/auth/login -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')

# Launch a review of target_ref against the default base branch (origin/HEAD, or main/master)
curl -X POST localhost:8000/routines/runs \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"kind":"review","project":"your-repo","target_ref":"feature/my-branch"}'
# -> 202 {"run_id":"...", "job_id":"..."}

# Poll the run detail until status is "done" or "failed"
curl localhost:8000/routines/runs/<run_id> -H "Authorization: Bearer $TOKEN"
# -> {"status":"done", "summary":"<markdown>", "findings":[...], "stats":{...}, ...}
```

Pass `base_ref` to review against something other than the default base
branch, and `model`/`provider` to override the LLM for just this run (see
below). List branches with `GET /routines/git/branches?project=your-repo`
(also returns the resolved `default_base`). Live per-stage progress (`diff`
→ `context` → `analyze` → `summarize`) streams over the **existing**
`GET /instances/jobs/{job_id}/events` SSE endpoint — no separate routines
SSE route.

**Model resolution — no hardcoded default.** A review run resolves its
model in this order: the launch payload's `model`/`provider` fields, then
the active LLM config (set via `POST /config/llm`). If neither supplies a
usable provider+model, the launch fails synchronously with **HTTP 400**:

```
No LLM provider configured for this review. POST to /config/llm to set one up, or pass model/provider in the launch payload.
```

**Model recommendation.** Review is a harder judgment task than chat.
Measured against real diffs (including a committed merge conflict),
3B-class local models are **not reliable** for review — they reported zero
findings. Recommended local model for review runs:

```bash
ollama pull qwen2.5-coder:14b   # ~9GB
```

Fully-local (Ollama) remains the primary, documented path; a remote
provider (OpenAI/Anthropic) works as an explicit per-run opt-in via the
`model`/`provider` launch overrides.

**Budgets/limits:** at most 60 changed files per run (a larger diff fails
the launch with an actionable message — narrow the range or split the
change); target-file excerpts are capped at ~6,000 characters per file
(shrunk proportionally around the changed ranges when the diff touches a
lot of a file).

**Findings without a reliable line anchor are kept, not dropped** — each
finding carries an `anchored` boolean; `anchored: false` means the model's
`start_line` didn't fall within (±2 lines of) a changed range, so treat its
line numbers as approximate.

**Known limitation — retrieval staleness.** The graph entities and their
Weaviate descriptions used as context reflect the **ingested index** (i.e.
the base branch as of the last `POST /instances/pipeline`), while the diff
itself and the findings' line anchors come straight from `git show` against
the actual `target_ref` (always accurate). If the target branch has drifted
far from what's indexed, the descriptive context may be stale even though
the diff and anchors are not.

**Locking:** review and ingestion of the **same project** are mutually
exclusive — they share the same per-project job lock as
`/instances/pipeline`/`/instances/graph`/`/instances/{project}/describe`.
Launching a review while an ingestion (or another review) is in flight for
that project returns `409 another job is running for project '<project>'`.

**Roles:** launching (`POST /routines/runs`) and deleting
(`DELETE /routines/runs/{id}`) require `admin` or `user`; all `GET` routes
(list, detail, branches) are open to any authenticated role, including
`viewer`.

### Bug hunt

> **Demo.** In addition to the routines-wide disclaimer above, bug hunt
> specifically is marked experimental in the UI — results improve with more
> capable models (we recommend `qwen2.5-coder:14b` or larger; see the
> quality note below).

`kind: "bughunt"` runs a free-text investigation over the **whole** ingested
project (not a diff): a bounded in-process agentic tool loop (≤12 rounds)
gives the model `search`, `grep_project`, `get_entity`, `neighbors`,
`get_file` and `load_skill` tools over the project graph/Weaviate corpus, and
the model decides for itself what to look at, based on the `scope` you give
it. It never scores or ranks anything by graph metrics (centrality, fan-in,
etc.) — every finding must trace back to a tool call the model actually made.

```bash
curl -X POST localhost:8000/routines/runs \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{
    "kind": "bughunt",
    "project": "your-repo",
    "scope": "look for unhandled errors and resource leaks in the ingestion pipeline",
    "path_prefix": "src/services",
    "playbooks": ["security-basics"]
  }'
# -> 202 {"run_id":"...", "job_id":"..."}
```

`scope` is required (422 if omitted) — a free-text description of what to
investigate. `path_prefix` is optional: it restricts *reported* findings to
files under that repo-relative prefix, but the agent may still read files
outside it for context. `playbooks`/`model`/`provider` work the same as for
`kind: "review"` (see [Playbooks & skills](#playbooks--skills) and the
model-resolution note above).

**Ollama-native only in this release.** The agentic tool-calling loop uses
Ollama's native `tools=`/`tool_calls` mechanism — there is no bug-hunt
equivalent of review's provider-agnostic single-shot analyze call yet. If
the resolved provider (from the launch payload or the active config) isn't
`ollama`, the launch fails synchronously with **HTTP 400**:

```
bug hunt requires Ollama-native tool calling in this release — provider '<name>' is not supported yet
```

**Evidence and anchoring.** Every raw finding from the model's JSON verdict
is validated against the project graph/repo before it's stored: a
`file_path` that matches neither a graph node nor a file on disk is dropped
as a hallucination (including absolute paths and `../` traversal attempts,
which are rejected outright); `anchored: true` only when the cited
`start_line` falls inside a real code-entity's line span in the graph —
otherwise the finding is kept with `anchored: false` (never dropped just for
that). A mechanical **read-before-report** guard runs after the model's
first verdict: if it reported findings without ever calling `get_entity` or
`get_file`, the run pushes back once ("you have not read the actual code —
go verify") before accepting the corrected answer; `stats.pushback_used`
records whether that happened. If the model's final answer still isn't
parseable JSON after one reformat retry, the run doesn't fail — it
*degrades* to `status: "done"` with zero findings and the raw agent answer
preserved verbatim in `summary` (prefixed `"agent answer (unparsed): "`) and
`stats.parse_failures: 1`.

**Honest quality limitation.** Bug-hunt finding quality is bounded by the
local model's own code-reading comprehension, not just by the tools it's
given. Measured with `qwen2.5-coder:14b`, roughly **half** of
read-verified findings (i.e. findings backed by an actual `get_entity`/
`get_file` call) can still misinterpret the code they cite — for example,
calling an explicit `try`/`catch` block "no exception handling". The
read-before-report guard trades recall for fewer unverified claims, but it
does not fix misreadings of code the model *did* read. The evidence trail
(tool calls made, cited entities, line anchors) exists precisely so you can
verify a finding against the real code before acting on it — treat bug-hunt
output as a lead, not a verdict. A larger local model, or a remote provider
via the per-run `model`/`provider` override (once a non-Ollama loop ships),
raises this ceiling; for now:

```bash
ollama pull qwen2.5-coder:14b   # same recommendation as review — see above
```

### Playbooks & skills

Two kinds of reusable, markdown document that a routine run can pull in,
stored via `POST/GET/PATCH/DELETE /routines/playbooks` and
`/routines/skills` (16KB content limit each, optional leading YAML
frontmatter stored/returned verbatim but not parsed by the API):

- **Playbooks are PUSHED.** You name them explicitly in a run's `playbooks`
  launch field (selection order = priority order); at most 4 per run. Each
  playbook has a `routines` field (`["review"]`, `["bughunt"]`, or both —
  defaults to both) and is rejected at launch (`400`) if it doesn't apply to
  the kind you're launching, or doesn't exist.
- **Skills are PULLED.** There is no launch field for them — every run sees
  the full `{name, description}` catalog and the model itself picks at most
  3 relevant ones by name (review: one small selection call up front, over a
  diffstat; bug hunt: a `load_skill(name)` tool it can call mid-investigation).
  Injected content is budgeted so neither crowds out the other: playbooks
  get 12k chars in both routines; skills get 8k in review's analyze prompt,
  while bug-hunt skill loads count against the agent's overall 20k tool
  payload budget.

```bash
# Create a playbook (applies to both routine kinds by default)
curl -X POST localhost:8000/routines/playbooks \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"name":"security-basics","content":"# Security basics\n\n- ...","routines":["review","bughunt"]}'

# Create a skill (description is what the model sees when deciding whether to load it)
curl -X POST localhost:8000/routines/skills \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"name":"sql-injection-review","description":"How to spot SQL injection in raw-query code paths.","content":"..."}'
```

| Method | Endpoint | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/routines/playbooks` | any | List playbooks — omits `content`, adds `content_bytes` |
| `GET` | `/routines/playbooks/{name}` | any | A playbook's full detail, `content` included |
| `POST` | `/routines/playbooks` | admin, user | Create a playbook (`name`, `content`, optional `routines`) |
| `PATCH` | `/routines/playbooks/{name}` | admin, user | Update `content`/`routines` — fields left unset are unchanged |
| `DELETE` | `/routines/playbooks/{name}` | admin, user | Delete a playbook |
| `GET` | `/routines/skills` | any | List skills — omits `content`, adds `content_bytes` |
| `GET` | `/routines/skills/{name}` | any | A skill's full detail, `content` included |
| `POST` | `/routines/skills` | admin, user | Create a skill (`name`, `description`, `content`) |
| `PATCH` | `/routines/skills/{name}` | admin, user | Update `description`/`content` — fields left unset are unchanged |
| `DELETE` | `/routines/skills/{name}` | admin, user | Delete a skill |

`name` must be non-empty, ≤128 chars, no leading/trailing whitespace, and
must not contain `/`, `\`, or control characters (it's a REST path segment).
A duplicate `name` on create returns `409`; an unknown `name` on get/patch/
delete returns `404`.

### Routines UI

The [Frontend UI](#frontend-ui) has a dedicated `/routines` screen (nav
item carries a **beta** badge) for everything above — no screenshots here,
see the demo disclaimer at the top of this section:

- **Runs tab** — a launch card (review or bug hunt, project-scoped) and a
  run history list with severity pills.
- **Run detail** — a live step timeline over the same job-events SSE stream
  as ingestion, a findings browser with a hand-rolled diff view per finding,
  and a summary tab with the run's markdown summary.
- **Library tab** — create/edit/upload dialogs for playbooks and skills.
- The [Instances](#frontend-ui) screen also shows event-derived routine
  pills on job cards, with a "View in Routines →" link back here.

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
- ✅ **npm CLI** (`dokkai`) — `up`/`status`/`ingest`/`graph`/`srcs`/`watch`/`doctor` commands: live job progress, one-command SRCS sessions (Claude Code, Codex, a local Ollama REPL, or an agentic local-model MCP loop), debounced incremental re-ingestion on save, and environment diagnostics (see [CLI](#cli))
- ✅ **Graph-only ingestion** (`POST /instances/graph`, `dokkai graph`) — run just `cgr` with no LLM/Weaviate involved
- ✅ **Authentication** — always-on, Grafana-style bearer-token auth with 3 roles (admin/user/viewer), a default `admin`/`admin` seed overridable via env, and CLI auto-login (see [Authentication](#authentication))
- ✅ **Frontend UI** — Next.js web app: streaming chat with sources, interactive dependency graph, conversation history, ingestion/job monitoring, LLM config and user administration, role-gated (see [Frontend UI](#frontend-ui))
- ✅ **Code review & bug-hunt routines** _(experimental demo, not part of dokkai core — see disclaimer)_ — branch-vs-base review and free-text bug-hunt investigations (agentic tool loop, Ollama-native), both with graph-anchored findings, pushable playbooks and model-pulled skills, run as background jobs over the shared job system with a dedicated UI (see [Code review routines](#code-review-routines))
- ✅ **One-command full stack** — `dokkai up --full` / `docker compose --profile full up -d` containerizes the API and frontend UI alongside Weaviate/Postgres/Memgraph (see [One-command full stack](#one-command-full-stack))

---

## Tech stack

| Layer | Technology |
| --- | --- |
| API | FastAPI (Python ≥ 3.14, managed with `uv`) |
| Vector DB | Weaviate `1.28` (`text2vec-ollama`) |
| Relational DB | Postgres `18` (`asyncpg`) — conversation and job history |
| Graph extraction | code-graph-rag (`cgr`) + ephemeral Memgraph |
| Embeddings | Ollama · `nomic-embed-text` |
| Generation | Ollama · `qwen2.5-coder` (any Ollama chat model) |
| MCP server | Official Python `mcp` SDK (`FastMCP`), stdio transport |
| CLI | Node.js (≥22.12) / TypeScript, `commander` + `chalk` + `ora` |
| Frontend | Next.js 16 · React 19 · Tailwind v4 · shadcn · sigma.js |

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

### 2. Start Weaviate and Postgres

```bash
docker compose up -d
```

This brings up both **Weaviate** (vector storage) and **Postgres** (conversation and job history). If Postgres isn't up yet when the API starts, the API still serves — see [Current limitations](#current-limitations) — so it's not strictly required before ingestion, but is required for chat/conversations.

> ⚠️ If you ever change the vectorizer (e.g. `EMBED_MODEL` or `VECTORIZER_PROVIDER`), wipe the Weaviate volume first so the collection is recreated with the new config: `docker compose down -v && docker compose up -d` (this also wipes Postgres data — back it up first if you need to keep conversation/job history).

### 3. Configure the environment

Create a `.env` in the project root:

```dotenv
# --- Postgres (conversation and job history) ---
DATABASE_URL=postgresql://dokkai:dokkai@localhost:5432/dokkai

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

> ⚠️ **First boot seeds an `admin`/`admin` user** (auth is always on — see [Authentication](#authentication)). Log in and change that password (or set `DOKKAI_ROOT_USER`/`DOKKAI_ROOT_PASSWORD` before first boot) before exposing the API beyond your own machine:
> ```bash
> TOKEN=$(curl -s -X POST localhost:8000/auth/login -H 'Content-Type: application/json' \
>   -d '{"username":"admin","password":"admin"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
> curl -X PATCH localhost:8000/auth/users/me/password -H "Authorization: Bearer $TOKEN" \
>   -H 'Content-Type: application/json' -d '{"current_password":"admin","new_password":"<a strong password>"}'
> ```
> The `dokkai` CLI does this login for you automatically on every command (see [Authentication](#authentication)) and prints a warning while the default credentials are still active.

### 5. Ingest a repository

Every route below `/` is authenticated (see [Authentication](#authentication)) — grab a bearer token first:

```bash
TOKEN=$(curl -s -X POST localhost:8000/auth/login -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
```

```bash
curl -X POST localhost:8000/instances/pipeline \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"repo_path":"/absolute/path/to/your/repo"}'
```

This runs the full pipeline: graph extraction (`cgr`) → chunking (with source) → descriptions → embedding/storage — the job reports these as stages `cgr → chunk → describe → upsert → done`. The `project_name` is taken from the repo's top‑level folder name.

### 6. Chat with the codebase

If you set `OLLAMA_CHAT_MODEL`, the model is already configured and warm. Otherwise configure it once (requires the `admin` role):

```bash
curl -X POST localhost:8000/config/llm \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"is_local":true,"provider_data":{"provider_name":"ollama","model":"qwen2.5-coder:latest"}}'
```

Then ask away (SSE stream — note `-N` and that `project_name` must match the ingested repo):

```bash
curl -N -X POST localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"message":"how does the alarm flow work?","project_name":"your-repo","audience":"developer"}'
```

### 7. (Optional) Frontend

A Next.js web UI lives in [`frontend/`](frontend/) — see [Frontend UI](#frontend-ui) for the full walkthrough.

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
| `DATABASE_URL` | `postgresql://dokkai:dokkai@localhost:5432/dokkai` | Postgres connection string for conversation and job history (`docker compose up -d` starts a matching Postgres service). Boot-time SQL migrations run automatically. If Postgres is unreachable at startup, the API still serves — see [Current limitations](#current-limitations) |
| `DOKKAI_ROOT_USER` | _(unset — `admin` seeded)_ | Optional override for the admin username, applied on boot (and re-applied if changed) — see [Authentication](#authentication) |
| `DOKKAI_ROOT_PASSWORD` | _(unset — `admin` seeded)_ | Optional override for the admin password, applied on boot; changing it invalidates that user's existing sessions — see [Authentication](#authentication) |
| `DOKKAI_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated list of origins allowed to call the API from a browser (CORS) |
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
| `MEMGRAPH_HOST` | `localhost` (`dev.sh` default) | When set, `shell/run_cgr.sh` connects to this resident Memgraph instead of starting an ephemeral container per run (the `full` profile sets it to `memgraph`, the compose service). Unset to fall back to the legacy ephemeral-container flow — see [One-command full stack](#one-command-full-stack) |
| `DOKKAI_REPOS_DIR` | _(unset — defaults to a harmless path outside `full`)_ | **Containerized API only** (`full` profile / `dokkai up --full`). Parent directory of the repos you ingest/review, bind-mounted into the API container at the same absolute path — see [One-command full stack](#one-command-full-stack) |
| `DOKKAI_API_URL` | `http://localhost:8000` | **CLI-side only** (not read by the API). dokkai API URL for the `dokkai` CLI; overridden by `--api`, falls back to this repo's `.env` — see [CLI](#cli) |
| `DOKKAI_HOME` | _(auto-detected)_ | **CLI-side only.** Path to this repo, used by `dokkai up`/`dokkai status` to run `docker compose` and read `.env`. Auto-detected by walking up from the cwd for `docker-compose.yml` + `src/mcp_server.py` — see [CLI](#cli) |

> **Two Ollama endpoints, on purpose:** `OLLAMA_EMBED_ENDPOINT` is called by the Weaviate **container** (so it needs `host.docker.internal`), while `OLLAMA_BASE_URL` is called by the **API process** on the host (so it's `localhost`).

> **LLM config persistence:** `POST /config/llm` write‑through persists the active provider/model to Postgres, and it's reloaded from there on every boot — taking precedence over `OLLAMA_CHAT_MODEL`. The env var is only a one‑time seed for a fresh install: it's used when no config has been saved yet (or Postgres is unreachable at boot), and that seed itself is never written back to the DB.

---

## Authentication

Auth is **always on** — Grafana-style: there's no way to disable it, and there's nothing to configure before first boot.

- **First boot seeds `admin`/`admin`** (only when the `users` table is empty). Setting `DOKKAI_ROOT_USER`/`DOKKAI_ROOT_PASSWORD` before that first boot seeds those credentials instead. If you set/change either var on a **later** boot, the API re-applies it onto user id 1 (the seeded admin), overwriting its username/password and invalidating all of its existing sessions — this is the supported way to reset a lost admin password.
- **⚠️ Change the default password.** `GET /auth/status` (public) and `GET /auth/me` both return `default_admin_active: true` for as long as any user named `admin` still verifies against the password `admin` — the CLI prints a one-time warning when it sees this flag, and the [frontend UI](#frontend-ui) shows a persistent banner for admins. Change it with `PATCH /auth/users/me/password`, or set `DOKKAI_ROOT_PASSWORD` and restart.
- **Tokens** are opaque, random 30-day bearer tokens (`Authorization: Bearer <token>`), stored SHA-256-hashed in Postgres — the plaintext token is only ever returned once, at login. `POST /auth/logout` revokes the session used for that request; changing your own password (`PATCH /auth/users/me/password`) revokes every *other* session for your user but keeps the one making the request. Passwords are hashed with stdlib `hashlib.scrypt` (no new dependency).
- **Roles** (assigned per-user, checked per-route):

  | Role | Can | Cannot |
  | --- | --- | --- |
  | `admin` | Everything below, plus user management (`GET`/`POST /auth/users`, `DELETE /auth/users/{id}`) and `POST /config/llm` | — |
  | `user` | Chat (`POST /chat`), ingestion (`POST /instances/...`), deletes (`DELETE /chat/conversations/{id}`), all reads | User management, `POST /config/llm` |
  | `viewer` | Read-only `GET` routes (graph, jobs, conversations, config) | `POST /chat`, ingestion, any delete, user management |

  A role that can't perform an action gets `403` with the exact body `role '<role>' cannot perform this action`. Every authenticated user (any role) can call `GET /auth/me` and `PATCH /auth/users/me/password` for themselves.
- **Public routes** (no token required): `GET /`, `GET /docs`, `GET /redoc`, `GET /openapi.json`, `POST /auth/login`, `GET /auth/status`. Every other route requires a valid bearer token; a missing/invalid/expired one is `401` with a `WWW-Authenticate: Bearer` header.
- **Postgres down** → protected routes return `503` (auth itself lives in Postgres, so it can't be checked); `POST /auth/login` and `GET /auth/status` also `503` in that case since they need to read the `users` table.
- **CLI behavior**: every `dokkai` command logs in automatically, using `DOKKAI_ROOT_USER`/`DOKKAI_ROOT_PASSWORD` from its environment (falling back to `admin`/`admin`), caches the token for the process, and self-heals a `401` by re-logging in — no separate `dokkai login` command. It prints the same one-time default-admin warning as the API. The MCP server (stdio) is untouched by any of this — it has no HTTP auth.

### Managing users

```bash
# Create a user (admin only)
curl -X POST localhost:8000/auth/users -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"username":"alice","password":"...","role":"user"}'

# List users (admin only)
curl localhost:8000/auth/users -H "Authorization: Bearer $TOKEN"

# Delete a user (admin only; you cannot delete yourself or the last admin)
curl -X DELETE localhost:8000/auth/users/2 -H "Authorization: Bearer $TOKEN"
```

---

## API reference

> These are the HTTP endpoints exposed by the FastAPI app (`./dev.sh`). The **MCP server is separate and stdio-only** — it adds no HTTP endpoints; see [MCP server](#mcp-server).

> **Auth column key** (see [Authentication](#authentication) for the full model): `public` = no token needed · `any` = any authenticated role · `admin, user` = those two roles only · `admin` = admin only. Everything not `public` returns `401` without a valid bearer token, and `403 role '<role>' cannot perform this action` for a role not listed.

| Method | Endpoint | Auth | Description |
| --- | --- | --- | --- |
| `POST` | `/auth/login` | public | Authenticate with `username`/`password`, get a bearer token, role, and `default_admin_active` |
| `GET` | `/auth/status` | public | Whether auth is enabled (always `true`) and whether the default `admin`/`admin` credentials are still active |
| `POST` | `/auth/logout` | any | Invalidate the session used for this request |
| `GET` | `/auth/me` | any | Current user's username, role, token expiry, `default_admin_active` |
| `GET` | `/auth/users` | admin | List all users |
| `POST` | `/auth/users` | admin | Create a user with a role (`admin` \| `user` \| `viewer`) |
| `DELETE` | `/auth/users/{id}` | admin | Delete a user (400 on self-delete or deleting the last admin) |
| `PATCH` | `/auth/users/me/password` | any | Change your own password (401 if `current_password` is wrong); keeps the current session, revokes all others |
| `POST` | `/instances/pipeline` | admin, user | Run the full ingestion pipeline for a `repo_path`, `describe: true` by default (409 if a job is already running for that project; `describe: false` skips the descriptor pre-flight and ingests without descriptions) |
| `POST` | `/instances/graph` | admin, user | Graph-only run for a `repo_path` — `cgr` only, no LLM, no vectorization, no Weaviate (stages `cgr → done`, `kind: "graph"`; 409 if a job is already running for that project) |
| `POST` | `/instances/{project}/describe` | admin, user | Refresh descriptions for an already-ingested project (background job; optional `{"force": true}`; 409 if a job is already running for that project) — see [Refreshing descriptions](#refreshing-descriptions) |
| `GET` | `/instances/jobs` | any | List ingestion/refresh jobs — merges live (in-memory) jobs with history persisted in Postgres, so results survive a restart |
| `GET` | `/instances/jobs/{id}` | any | Get a job's status, `stage`/`stage_progress` and result |
| `GET` | `/instances/jobs/{id}/events` | any | Stream a job's progress as Server‑Sent Events (`event: job` on each update, `event: done` when it finishes) |
| `GET` | `/graph` | any | List projects with an ingested graph |
| `GET` | `/graph/{project}` | any | Full project graph (code entities by default, `?include=structural` for everything), `?limit=N` |
| `GET` | `/graph/{project}/neighborhood` | any | BFS neighborhood of an entity — `?entity=<qualified_name>&depth=&direction=&limit=` |
| `GET` | `/graph/{project}/files` | any | File‑to‑file dependency view (internal files only) |
| `POST` | `/routines/runs` | admin, user | Launch a code review (`kind: "review"`) or bug-hunt (`kind: "bughunt"`) run as a background job on the shared job system; bug hunt requires an Ollama-resolved LLM (400 otherwise). See [Code review routines](#code-review-routines) |
| `GET` | `/routines/runs` | any | List routine runs, most recent first, each with `severity_counts` (optional `?project=&kind=`) |
| `GET` | `/routines/runs/{id}` | any | A run's full detail, including its findings |
| `DELETE` | `/routines/runs/{id}` | admin, user | Delete a run (cascades to its findings) |
| `GET` | `/routines/git/branches` | any | A project's local git branches plus its resolved default base branch |
| `GET` | `/routines/playbooks` | any | List playbooks (summary — `content_bytes`, no `content`) |
| `GET` | `/routines/playbooks/{name}` | any | A playbook's full detail, `content` included |
| `POST` | `/routines/playbooks` | admin, user | Create a playbook |
| `PATCH` | `/routines/playbooks/{name}` | admin, user | Update a playbook's `content`/`routines` |
| `DELETE` | `/routines/playbooks/{name}` | admin, user | Delete a playbook |
| `GET` | `/routines/skills` | any | List skills (summary — `content_bytes`, no `content`) |
| `GET` | `/routines/skills/{name}` | any | A skill's full detail, `content` included |
| `POST` | `/routines/skills` | admin, user | Create a skill |
| `PATCH` | `/routines/skills/{name}` | admin, user | Update a skill's `description`/`content` |
| `DELETE` | `/routines/skills/{name}` | admin, user | Delete a skill |
| `POST` | `/chat` | admin, user | Chat over the codebase (SSE: `sources` → `token` → `done`) |
| `GET` | `/chat/conversations` | any | List conversations (persisted in Postgres; 503 if Postgres is unreachable) |
| `GET` | `/chat/conversations/{id}` | any | Get a conversation's history (503 if Postgres is unreachable) |
| `DELETE` | `/chat/conversations/{id}` | admin, user | Delete a conversation (503 if Postgres is unreachable) |
| `POST` | `/config/llm` | admin | Set the active LLM provider/model (rejects an invalid/retired remote model) |
| `GET` | `/config/llm` | any | Get the current LLM config |
| `GET` | `/config/llm/models` | any | List available models for the provider (live catalog, static fallback on failure) |
| `GET` | `/config/llm/health` | any | Check connectivity — probes the configured model |
| `GET` | `/` | public | Health check |

Jobs report `stage` through the unified vocabulary `cgr → chunk → describe → upsert → done` (refresh jobs use `chunk → describe → update → done`; graph-only jobs use just `cgr → done`), plus a `stage_progress` object `{"stage", "done", "total"}` mirroring the current stage's `done`/`total` counters, and a `kind` (`"pipeline"` | `"refresh"` | `"graph"`).

**Per‑project job lock** — only one job (pipeline, refresh, graph-only, or a review routine) may run at a time for a given project; submitting a second one (any of `/instances/pipeline`, `/instances/{project}/describe`, `/instances/graph`, `POST /routines/runs`) returns HTTP 409 `another job is running for project '<project>'` and creates no job.

**Job progress over SSE** — instead of polling `GET /instances/jobs/{id}`, `GET /instances/jobs/{id}/events` streams the same job payload over Server‑Sent Events (same framing as `/chat`): it emits `event: job` on every `updated_at` change and a terminal `event: done` with the final payload, then closes the stream. An unknown job id is a plain 404, not a stream.

**Job history across restarts** — jobs live in memory while running (the source of truth for live progress) and are write‑through persisted to Postgres on lifecycle transitions (queued/running/terminal), so results survive a server restart. On boot, any job still `queued`/`running` from a previous run is swept and marked `failed` (`"interrupted by server restart"`), since its worker thread no longer exists. If Postgres is unreachable, job history is silently unavailable and only in-memory (live) jobs are returned — this graceful-degradation path only applies while Postgres is reachable enough to authenticate the request; since every route (including this one) now requires resolving a bearer token against Postgres (see [Authentication](#authentication)), a fully unreachable Postgres 503s at the auth layer before this endpoint's own handler runs.

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
│   │   ├── chat_store.py        # conversation history (Postgres)
│   │   └── db.py                # asyncpg connection pool + boot-time migrations
│   ├── db/migrations/           # boot-time SQL migrations, applied in order
│   └── models/dtos/             # pydantic request/response models
├── shell/run_cgr.sh             # code-graph-rag runner (ephemeral Memgraph)
├── cli/                         # npm CLI package (`dokkai`) — see CLI section
├── frontend/                    # Next.js web UI — see Frontend UI section
├── docker-compose.yml           # Weaviate (text2vec-ollama) + Postgres (conversation/job history)
├── dev.sh                       # dev server launcher
└── ingested/                    # canonical graph JSON per project — <project>.json,
                                  # promoted from cgr's timestamped output after each
                                  # successful pipeline run; served by the Graph API
```

---

## Current limitations

These are known and tracked in the [Roadmap](#roadmap):

- **LLM config persistence needs Postgres** — `POST /config/llm` always applies the change in memory immediately, but if Postgres is down at the time, the change won't survive a restart (logged server-side as a warning) until Postgres is back and you re‑POST. See [Configuration](#configuration) for the full persistence semantics.
- Chat history is persisted in Postgres (see [Configuration](#configuration)); if Postgres is unreachable, the conversation endpoints and `/chat` degrade gracefully (503 / SSE `error` event) instead of crashing the API — though in practice, since [auth](#authentication) also requires Postgres to resolve a session, a fully unreachable Postgres now 503s every protected route at the auth layer regardless.
- **Single repository at a time** per collection.
- **Retrieval bias toward tests** — for "how does X work?" queries, test files often out‑rank the real implementation (test descriptions read like specs). End‑to‑end tests are also decoupled from implementation by the HTTP boundary, so graph expansion can't bridge them.
- **Containerized (`--full`) and host (`./dev.sh`) runs don't share ingested data** — the API container stores `ingested/`/`data/` in named Docker volumes, separate from the host directories; a project ingested one way isn't visible from the other. See [One-command full stack](#one-command-full-stack).
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
| 05 | Postgres conversation history | ✅ done (pending merge) |
| 06 | Basic auth (root user via env) | ✅ done (pending merge) |
| 07 | Frontend UI (chat + graph + config) | ✅ done (pending merge) |
| 08 | Code review & bug‑hunt routines _(experimental demo, not core — see [disclaimer](#code-review-routines))_ | ✅ done (2026-07-11, pending merge) |
| 09 | Documentation site (en/pt/zh/es) | planned |
| 10 | Post‑01 polish (live provider model catalogs, stage‑level job progress, describe refresh endpoint) | ✅ done |
| 11 | MCP polish (`grep_project` tool, small-model instructions profile, session watchdog) | ✅ done (pending merge) |
| 12 | CLI polish (agentic `srcs --agent` MCP loop, `watch` incremental re-ingestion, `doctor` diagnostics) | ✅ done (pending merge) |

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
- [x] Wire the UI to the API: streaming chat, sources panel with provenance (`hop` / `via`), audience switcher — see [feature 07](#feature-roadmap).
- [x] **Interactive system graph** — visualize the dependency graph of the ingested codebase.
- [x] Model configuration UI (LLM provider/model + health check). Full environment/API-key management from the UI is a separate, still-planned item — see "Remote environment configuration" below.

### One‑command infrastructure
- [x] Containerize the API (fill in the `Dockerfile`) and add the API + frontend to `docker-compose.yml`, gated behind the `full` profile so `docker compose --profile full up -d` (or `dokkai up --full`) brings up the **entire** stack — see [One-command full stack](#one-command-full-stack).

### Persistence & multi‑environment
- [x] Move chat/conversation history to a database (Postgres) — see [feature 05](#feature-roadmap).
- [ ] Move the remaining in‑memory state to a database — the LLM config (live job progress is expected to stay in-memory; job history is already persisted, see [feature 05](#feature-roadmap)).
- [ ] **Encrypted storage** of environments and API keys.
- [ ] **Remote environment configuration** (manage environments from the UI) — with `.env` kept as a fallback.

### Exploratory (longer‑term)
- [ ] **CI/CD code review** — run Dokkai on pull requests; on merge, incrementally upsert only the changed entities (the deterministic‑UUID groundwork is already in place), enabling versioned indexing without duplication.

---

## License

Open source — license TBD.

Contributions and ideas welcome.
