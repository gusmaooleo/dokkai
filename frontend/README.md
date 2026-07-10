# Dokkai frontend

The web UI for Dokkai: login, chat with streaming answers and sources, an
interactive dependency graph, conversation history, ingestion/job monitoring,
and LLM/user administration. See the root [`README.md`](../README.md) for the
full stack (API, Weaviate, Postgres, Ollama) and how they fit together — this
file only covers the frontend.

## Stack

- [Next.js](https://nextjs.org) 16 (App Router) + React 19
- Tailwind CSS v4
- [shadcn](https://ui.shadcn.com/) components (Radix primitives)
- [sigma.js](https://www.sigmajs.org/) + [graphology](https://graphology.github.io/) for the graph screen
- `react-markdown` + `remark-gfm` for chat message rendering

## Prerequisites

- Node.js ≥ 20
- The Dokkai API running (`./dev.sh` from the repo root) and Weaviate/Postgres
  up (`docker compose up -d`) — see the root README's
  [Quickstart](../README.md#quickstart)

## Environment

Copy `.env.example` to `.env.local` (or `.env`) and adjust as needed:

| Variable | Default | Description |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Base URL of the Dokkai API. Inlined at build time (must be set before `next build` in each environment). |

If you serve the frontend from a different origin than
`http://localhost:3000` / `http://127.0.0.1:3000`, add it to the API's
`DOKKAI_CORS_ORIGINS` — see the root README's
[Configuration](../README.md#configuration).

## Development

```bash
npm install
npm run dev      # http://localhost:3000
```

First login: `admin`/`admin` (or the `DOKKAI_ROOT_USER`/`DOKKAI_ROOT_PASSWORD`
you configured on the API) — see the root README's
[Authentication](../README.md#authentication).

## Other commands

```bash
npm run build    # production build
npm run start    # serve the production build
npm run lint      # eslint
```

## Folder layout

```
frontend/
├── app/
│   ├── login/                # public login page
│   ├── (app)/                # authenticated route group — sidebar shell + screens
│   │   ├── chat/
│   │   ├── history/
│   │   ├── graph/
│   │   ├── instances/
│   │   ├── settings/
│   │   └── admin/
│   ├── layout.tsx            # root layout (fonts, providers)
│   └── globals.css           # design tokens (Tailwind v4 @theme) + shadcn base styles
├── components/
│   ├── shell/                # sidebar, header, nav, mobile nav, default-admin banner
│   ├── chat/                 # composer, message list/markdown, sources panel, audience switch
│   ├── graph/                # sigma canvas, toolbar, legend, detail panel
│   ├── history/               # conversation cards
│   ├── instances/             # job cards, new-ingestion dialog, project cards
│   ├── settings/               # LLM provider/model, health, account
│   ├── admin/                   # user management
│   └── ui/                      # shadcn primitives (button, dialog, select, ...)
└── lib/
    ├── api.ts                # typed fetch wrapper for the API (mirrors docs/openapi.yaml)
    ├── auth.tsx               # auth provider/context, token storage
    ├── project.tsx            # active-project context (sidebar switcher)
    ├── sse.ts                 # Server-Sent Events parser (chat + job progress)
    ├── theme.tsx              # light/dark theme provider
    ├── toast.tsx              # toast notifications
    ├── types.ts               # DTO types mirroring the API's Pydantic models
    └── utils.ts               # cn() class-merge helper (shadcn)
```

shadcn components live in `components/ui/`; the design tokens (colors, fonts,
radii) are defined in `app/globals.css` under `@theme`.

## Role behavior

The UI mirrors the API's roles (see the root README's
[Authentication](../README.md#authentication)): `viewer` is read-only (no chat
composer, no destructive actions); `user` can chat, ingest and delete but
can't reach admin/user-management or LLM config writes; `admin` sees
everything, including the Admin screen and a banner while the default
`admin`/`admin` credentials are still active.
