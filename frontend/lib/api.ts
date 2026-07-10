/**
 * Typed fetch wrapper for the Dokkai API. Thin — no caching layer, one
 * exported method per endpoint. See `docs/openapi.yaml` for the full
 * contract and `lib/types.ts` for the DTO mirrors used here.
 */

import type {
  AuthStatus,
  BranchesResponse,
  Conversation,
  ConversationSummary,
  CreateUserRequest,
  CreateUserResponse,
  DescribeRefreshRequest,
  EntityDetail,
  FilesGraph,
  GraphOnlyRequest,
  GraphPayload,
  HealthResponse,
  IngestRequest,
  Job,
  LaunchRoutineRequest,
  LaunchRoutineResponse,
  LLMConfig,
  LLMConfigRequest,
  LoginResponse,
  Me,
  ModelsListResponse,
  NeighborhoodPayload,
  PlaybookSummary,
  ProjectGraphDTO,
  ProviderHealth,
  RoutineRunDetail,
  RoutineRunSummary,
  RunJobResponse,
  User,
} from "./types";

const TOKEN_KEY = "dokkai.token";

/** Base URL for the API. Bundled at build time (NEXT_PUBLIC_*). */
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// -----------------------------------------------------------------------
// Token storage (localStorage, decision 12e) — SSR-safe no-ops on the server.
// -----------------------------------------------------------------------

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(TOKEN_KEY);
}

// -----------------------------------------------------------------------
// Error type + request plumbing
// -----------------------------------------------------------------------

/** Thrown by every request helper below. `status` is 0 for network failures. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

interface RequestOptions {
  method?: "GET" | "POST" | "PATCH" | "DELETE";
  body?: unknown;
  query?: Record<string, string | number | boolean | undefined>;
  /** Set on the login call itself — a 401 there is "bad credentials", not "session expired". */
  skipAuthRedirect?: boolean;
}

function buildQuery(query: RequestOptions["query"]): string {
  if (!query) return "";
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined) params.set(key, String(value));
  }
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

/** Extracts FastAPI's `{"detail": "..."}` (or validation-error array) from an error response. */
async function parseErrorDetail(res: Response): Promise<string> {
  try {
    const data: unknown = await res.json();
    if (data && typeof data === "object" && "detail" in data) {
      const detail = (data as { detail: unknown }).detail;
      if (typeof detail === "string") return detail;
      if (Array.isArray(detail)) {
        const messages = detail
          .map((item) => (item && typeof item === "object" && "msg" in item ? String((item as { msg: unknown }).msg) : null))
          .filter((msg): msg is string => Boolean(msg));
        if (messages.length > 0) return messages.join("; ");
      }
    }
  } catch {
    // Non-JSON body — fall through to a generic message.
  }
  return res.statusText || `HTTP ${res.status}`;
}

function redirectToLogin(): void {
  if (typeof window === "undefined") return;
  clearToken();
  window.location.href = "/login";
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, query, skipAuthRedirect = false } = options;
  const token = getToken();
  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (token) headers["Authorization"] = `Bearer ${token}`;

  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}${buildQuery(query)}`, {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new ApiError(0, `could not reach the API at ${API_BASE_URL} — is the server running?`);
  }

  if (!res.ok) {
    const detail = await parseErrorDetail(res);
    if (res.status === 401 && !skipAuthRedirect) {
      redirectToLogin();
    }
    throw new ApiError(res.status, detail);
  }

  const text = await res.text();
  if (!text) return undefined as T;
  return JSON.parse(text) as T;
}

// -----------------------------------------------------------------------
// Auth
// -----------------------------------------------------------------------

export const authApi = {
  login: (username: string, password: string) =>
    request<LoginResponse>("/auth/login", {
      method: "POST",
      body: { username, password },
      skipAuthRedirect: true,
    }),
  status: () => request<AuthStatus>("/auth/status", { skipAuthRedirect: true }),
  logout: () => request<{ message: string }>("/auth/logout", { method: "POST" }),
  me: () => request<Me>("/auth/me"),
  listUsers: () => request<User[]>("/auth/users"),
  createUser: (data: CreateUserRequest) =>
    request<CreateUserResponse>("/auth/users", { method: "POST", body: data }),
  deleteUser: (userId: number) => request<{ message: string }>(`/auth/users/${userId}`, { method: "DELETE" }),
  changePassword: (currentPassword: string, newPassword: string) =>
    request<{ message: string }>("/auth/users/me/password", {
      method: "PATCH",
      body: { current_password: currentPassword, new_password: newPassword },
    }),
};

// -----------------------------------------------------------------------
// Chat (non-streaming endpoints — the SSE POST /chat itself lives in lib/sse.ts)
// -----------------------------------------------------------------------

export const chatApi = {
  listConversations: () => request<ConversationSummary[]>("/chat/conversations"),
  getConversation: (conversationId: string) =>
    request<Conversation>(`/chat/conversations/${encodeURIComponent(conversationId)}`),
  deleteConversation: (conversationId: string) =>
    request<{ message: string; conversation_id: string }>(
      `/chat/conversations/${encodeURIComponent(conversationId)}`,
      { method: "DELETE" },
    ),
};

// -----------------------------------------------------------------------
// Graph
// -----------------------------------------------------------------------

export const graphApi = {
  list: () => request<ProjectGraphDTO[]>("/graph"),
  get: (project: string, opts?: { include?: "structural"; limit?: number }) =>
    request<GraphPayload>(`/graph/${encodeURIComponent(project)}`, {
      query: { include: opts?.include, limit: opts?.limit },
    }),
  entity: (project: string, name: string) =>
    request<EntityDetail>(`/graph/${encodeURIComponent(project)}/entity`, { query: { name } }),
  neighborhood: (
    project: string,
    entity: string,
    opts?: { depth?: number; direction?: "in" | "out" | "both"; limit?: number },
  ) =>
    request<NeighborhoodPayload>(`/graph/${encodeURIComponent(project)}/neighborhood`, {
      query: { entity, depth: opts?.depth, direction: opts?.direction, limit: opts?.limit },
    }),
  files: (project: string) => request<FilesGraph>(`/graph/${encodeURIComponent(project)}/files`),
};

// -----------------------------------------------------------------------
// Instances (ingestion + jobs)
// -----------------------------------------------------------------------

export const instancesApi = {
  runPipeline: (data: IngestRequest) =>
    request<RunJobResponse>("/instances/pipeline", { method: "POST", body: data }),
  runGraphOnly: (data: GraphOnlyRequest) =>
    request<RunJobResponse>("/instances/graph", { method: "POST", body: data }),
  refreshDescriptions: (project: string, data?: DescribeRefreshRequest) =>
    request<RunJobResponse>(`/instances/${encodeURIComponent(project)}/describe`, {
      method: "POST",
      body: data ?? {},
    }),
  listJobs: () => request<Job[]>("/instances/jobs"),
  getJob: (jobId: string) => request<Job>(`/instances/jobs/${jobId}`),
};

// -----------------------------------------------------------------------
// Routines (code review / bug hunt)
// -----------------------------------------------------------------------

export const routinesApi = {
  launch: (data: LaunchRoutineRequest) =>
    request<LaunchRoutineResponse>("/routines/runs", { method: "POST", body: data }),
  listRuns: (opts?: { project?: string; kind?: string; limit?: number }) =>
    request<RoutineRunSummary[]>("/routines/runs", {
      query: { project: opts?.project, kind: opts?.kind, limit: opts?.limit },
    }),
  getRun: (runId: string) => request<RoutineRunDetail>(`/routines/runs/${encodeURIComponent(runId)}`),
  deleteRun: (runId: string) =>
    request<{ message: string; run_id: string }>(`/routines/runs/${encodeURIComponent(runId)}`, {
      method: "DELETE",
    }),
  branches: (project: string) =>
    request<BranchesResponse>("/routines/git/branches", { query: { project } }),
  listPlaybooks: () => request<PlaybookSummary[]>("/routines/playbooks"),
};

// -----------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------

export const configApi = {
  setLlm: (data: LLMConfigRequest) => request<LLMConfig>("/config/llm", { method: "POST", body: data }),
  getLlm: () => request<LLMConfig>("/config/llm"),
  listModels: () => request<ModelsListResponse>("/config/llm/models"),
  health: () => request<ProviderHealth>("/config/llm/health"),
};

// -----------------------------------------------------------------------
// Health
// -----------------------------------------------------------------------

export const healthApi = {
  root: () => request<HealthResponse>("/", { skipAuthRedirect: true }),
};
