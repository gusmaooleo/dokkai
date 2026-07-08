# Dokkai

**Turn any codebase into a graph‑aware vector database that local AI agents can read, search and document — using a fraction of the tokens.**

Dokkai ingests a repository, builds a full dependency graph of it (calls, inheritance, definitions, modules), slices the real source into rich chunks, and stores everything in a vector database. On top of that it serves a **graph‑augmented RAG**: instead of returning a handful of loosely‑matched snippets, it retrieves the *connected* neighbourhood of the code you asked about, so a local LLM can answer accurately and generate documentation for parts of the system that were never documented.

Everything runs **100% locally** — Weaviate for vectors, Ollama for both embeddings and generation.

> The name *dokkai* (読解) is Japanese for "reading comprehension" — which is exactly what this gives an LLM over your code.

---

## Table of contents

- [Why](#why)
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
- **Fail‑loud policy** — `POST /instances/pipeline` always ingests with descriptions on. The descriptor is checked *before* the job is created: if it's missing or unavailable, the request fails immediately with **HTTP 400** and **no job is created** (nothing is ingested):
  - No model configured: `no descriptor model specified — set DESC_MODEL (e.g. qwen2.5-coder:3b) or configure a remote provider via DESC_PROVIDER`
  - Model not pulled: `descriptor model '<m>' is not available in Ollama at <base_url> — pull it with 'ollama pull <m>'`
  - Ingesting without descriptions exists internally (`process_and_store(..., describe=False)`) but isn't exposed through the public API yet — that UX lands with the CLI (roadmap feature 04).

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

---

## Tech stack

| Layer | Technology |
| --- | --- |
| API | FastAPI (Python ≥ 3.14, managed with `uv`) |
| Vector DB | Weaviate `1.28` (`text2vec-ollama`) |
| Graph extraction | code-graph-rag (`cgr`) + ephemeral Memgraph |
| Embeddings | Ollama · `nomic-embed-text` |
| Generation | Ollama · `qwen2.5-coder` (any Ollama chat model) |
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

This runs the full pipeline: graph extraction → chunking (with source) → embedding → storage. The `project_name` is taken from the repo's top‑level folder name.

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
| `MEMGRAPH_IMAGE` | `memgraph/memgraph:latest` | Image used by `cgr` for graph extraction |

> **Two Ollama endpoints, on purpose:** `OLLAMA_EMBED_ENDPOINT` is called by the Weaviate **container** (so it needs `host.docker.internal`), while `OLLAMA_BASE_URL` is called by the **API process** on the host (so it's `localhost`).

---

## API reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/instances/pipeline` | Run the full ingestion pipeline for a `repo_path` |
| `POST` | `/chat` | Chat over the codebase (SSE: `sources` → `token` → `done`) |
| `GET` | `/chat/conversations` | List conversations |
| `GET` | `/chat/conversations/{id}` | Get a conversation's history |
| `DELETE` | `/chat/conversations/{id}` | Delete a conversation |
| `POST` | `/config/llm` | Set the active LLM provider/model |
| `GET` | `/config/llm` | Get the current LLM config |
| `GET` | `/config/llm/models` | List available models for the provider |
| `GET` | `/config/llm/health` | Check provider connectivity |
| `GET` | `/` | Health check |

---

## Project structure

```
dokkai/
├── src/
│   ├── main.py                  # FastAPI app + startup (auto-config/warm LLM)
│   ├── controllers/             # instances, chat, config routes
│   ├── services/
│   │   ├── ingest.py            # runs code-graph-rag (cgr)
│   │   ├── chunker.py           # graph JSON → rich code chunks (with source)
│   │   ├── vectorize.py         # chunk + store orchestrator
│   │   ├── weaviate_client.py   # collection schema, upsert, deterministic UUIDs
│   │   ├── retriever.py         # hybrid search + graph expansion
│   │   ├── chat.py              # RAG chat pipeline (retrieve → prompt → stream)
│   │   ├── llm_provider.py      # Ollama / OpenAI / Anthropic abstraction
│   │   ├── llm_config.py        # provider config store + validation
│   │   └── chat_store.py        # conversation history (in-memory)
│   └── models/dtos/             # pydantic request/response models
├── shell/run_cgr.sh             # code-graph-rag runner (ephemeral Memgraph)
├── frontend/                    # Next.js app (WIP)
├── docker-compose.yml           # Weaviate (text2vec-ollama)
├── dev.sh                       # dev server launcher
└── ingested/                    # generated graph JSON outputs
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
| 02 | Graph query API | planned |
| 03 | MCP server (core) | planned |
| 04 | npm CLI (`dokkai`) + SRCS mode | planned |
| 05 | Postgres conversation history | planned |
| 06 | Basic auth (root user via env) | planned |
| 07 | Frontend UI (chat + graph + config) | planned |
| 08 | PDF ingestion — local NotebookLM | planned |
| 09 | Code review & bug‑hunt routines | planned |
| 10 | Documentation site (en/pt/zh/es) | planned |

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
