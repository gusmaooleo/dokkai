/**
 * TypeScript mirrors of the Dokkai API DTOs consumed by the frontend.
 *
 * Kept close to `docs/openapi.yaml` / the backend's Pydantic models
 * (`src/models/dtos/*.py`) — field names and nullability should match
 * exactly. See that file for the full contract, including endpoints not
 * yet used by the UI.
 */

// -----------------------------------------------------------------------
// Auth
// -----------------------------------------------------------------------

export type Role = "admin" | "user" | "viewer";

/** POST /auth/login response. */
export interface LoginResponse {
  token: string;
  expires_at: string;
  role: Role;
  default_admin_active: boolean;
}

/** GET /auth/me response. */
export interface Me {
  username: string;
  role: Role;
  expires_at: string;
  default_admin_active: boolean;
}

/** GET /auth/status response (public). */
export interface AuthStatus {
  enabled: boolean;
  default_admin_active: boolean;
}

/** GET /auth/users entry. */
export interface User {
  id: number;
  username: string;
  role: Role;
  created_at: string;
}

export interface CreateUserRequest {
  username: string;
  password: string;
  role: Role;
}

export interface CreateUserResponse {
  id: number;
  username: string;
  role: Role;
}

// -----------------------------------------------------------------------
// Chat
// -----------------------------------------------------------------------

export type Audience = "developer" | "manager" | "customer";

export interface ChatRequest {
  message: string;
  project_name: string;
  audience?: Audience;
  conversation_id?: string | null;
  top_k?: number;
}

/** A retrieved code chunk, as emitted in the `sources` SSE event. */
export interface SourceChunk {
  entity_type: string;
  name: string;
  qualified_name: string;
  file_path: string;
  absolute_path: string;
  start_line: number | null;
  end_line: number | null;
  chunk_text: string;
  description: string;
  score: number | null;
  hop: number;
  via: string;
}

export interface ConversationSummary {
  conversation_id: string;
  project_name: string;
  audience: string;
  title: string | null;
  message_count: number;
  last_message_preview: string;
  created_at: string;
  updated_at: string;
}

export interface ConversationMessage {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string;
  sources: SourceChunk[];
}

export interface Conversation {
  conversation_id: string;
  project_name: string;
  audience: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  messages: ConversationMessage[];
}

/**
 * `POST /chat`'s SSE event stream, in order: one `sources`, many `token`,
 * then either one `done` or one `error` (terminal). Exception: if no LLM
 * provider is configured, the stream instead emits a single `error` event
 * first, with no `sources` event at all.
 */
export type ChatSSEEvent =
  | { type: "sources"; data: SourceChunk[] }
  | { type: "token"; data: string }
  | { type: "done"; data: { conversation_id: string; answer: string } }
  | { type: "error"; data: string };

// -----------------------------------------------------------------------
// Graph
// -----------------------------------------------------------------------

/** One entry of `GET /graph`'s listing. */
export interface ProjectGraphDTO {
  project: string;
  file: string;
  nodes: number;
  edges: number;
  generated_at: string;
  chunks: number | null;
  repo_path: string | null;
}

export interface GraphNode {
  id: number;
  kind: string;
  name: string | null;
  qualified_name: string | null;
  path: string | null;
  absolute_path: string | null;
  start_line: number | null;
  end_line: number | null;
}

export interface GraphEdge {
  source: number;
  target: number;
  type: string;
}

export interface GraphStats {
  total_nodes: number;
  total_edges: number;
  returned_nodes: number;
  returned_edges: number;
  truncated: boolean;
}

/** GET /graph/{project} response. */
export interface GraphPayload {
  project: string;
  generated_at: string | null;
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: GraphStats;
}

export interface NeighborhoodNode extends GraphNode {
  hop: number;
}

/** GET /graph/{project}/entity response. */
export interface EntityDetail extends GraphNode {
  description: string | null;
  chunk_text: string | null;
}

export interface NeighborhoodStats {
  depth: number;
  direction: "in" | "out" | "both";
  limit: number;
  returned_nodes: number;
  returned_edges: number;
  truncated: boolean;
}

/** GET /graph/{project}/neighborhood response. */
export interface NeighborhoodPayload {
  project: string;
  entity: string;
  center: GraphNode;
  nodes: NeighborhoodNode[];
  edges: GraphEdge[];
  stats: NeighborhoodStats;
}

export interface FileNode {
  path: string;
  name: string;
  absolute_path: string | null;
}

export interface FileEdge {
  source: string;
  target: string;
  weight: number;
  types: Record<string, number>;
}

/** GET /graph/{project}/files response. */
export interface FilesGraph {
  project: string;
  files: FileNode[];
  edges: FileEdge[];
  stats: { files: number; edges: number };
}

// -----------------------------------------------------------------------
// Instances / jobs
// -----------------------------------------------------------------------

export interface IngestRequest {
  repo_path: string;
  recreate?: boolean;
  describe?: boolean;
}

export interface GraphOnlyRequest {
  repo_path: string;
}

export interface DescribeRefreshRequest {
  force?: boolean;
}

export interface RunJobResponse {
  job_id: string;
  status: string;
}

export type JobKind = "pipeline" | "refresh" | "graph" | "review" | "bughunt";
export type JobStatus = "queued" | "running" | "succeeded" | "failed";
export type JobStage =
  | ""
  | "cgr"
  | "chunk"
  | "describe"
  | "upsert"
  | "update"
  | "diff"
  | "context"
  | "playbooks"
  | "skills"
  | "analyze"
  | "summarize"
  | "resolve"
  | "hunt"
  | "findings"
  | "done";

export interface StageProgress {
  stage: string;
  done: number;
  total: number;
}

/** One entry of a job's bounded step-event log (routine jobs only — see
 * `services.jobs.add_job_event`). `seq` is monotonic even across the
 * 200-entry trim, so a client can detect dropped history by a seq gap. */
export interface JobEventEntry {
  seq: number;
  stage: string;
  message: string;
  ts: string;
}

export interface DescriptionStats {
  enabled: boolean;
  reason?: string | null;
  model?: string | null;
  describable: number;
  cached: number;
  generated: number;
  failed: number;
  pending: number;
  skipped: number;
  templated: number;
}

export interface VectorizeResult {
  project_name: string;
  chunks_created: number;
  ingestion_id: string;
  descriptions: DescriptionStats;
  /** Chunks removed because they belonged to an earlier ingestion_id of this
   * same project (deleted at the source, or filtered out by gitignore-aware
   * ingestion on this re-ingest). */
  stale_chunks_purged: number;
}

/** Gitignore-aware ingestion filtering stats, nested under `ingest.filter`
 * on pipeline/graph-only job results. */
export interface IngestFilterStats {
  files_filtered_gitignore: number;
  files_filtered_defaults: number;
  nodes_removed: number;
  edges_removed: number;
  nodes_outside_repo: number;
  /** "error" means a real git repo's gitignore rules were silently skipped
   * this run (e.g. a container's "dubious ownership" guard) — distinct from
   * "not-a-repo" (expected, harmless). */
  git_status: "ok" | "not-a-repo" | "error";
}

export interface RefreshResult {
  project_name: string;
  chunks_indexed: number;
  eligible: number;
  stale_source: number;
  not_indexed: number;
  updated: number;
  unchanged: number;
  preserved: number;
  descriptions: DescriptionStats;
  note?: string;
}

export interface PipelineResult {
  ingest: { output_json: string; filter: IngestFilterStats };
  vectorize: VectorizeResult;
}

/** Result of a `kind: "graph"` job — cgr only, no vectorize stage. */
export interface GraphOnlyResult {
  ingest: { output_json: string; filter: IngestFilterStats };
}

/** The `Job` object returned by `GET /instances/jobs/{id}` and its SSE stream. */
export interface Job {
  id: string;
  repo_path: string;
  recreate: boolean;
  describe: boolean;
  kind: JobKind;
  project: string;
  force: boolean;
  status: JobStatus;
  stage: JobStage;
  done: number;
  total: number;
  stage_progress: StageProgress | null;
  result: PipelineResult | RefreshResult | GraphOnlyResult | null;
  error: string | null;
  /** Routine (review/bughunt) step log only — null for pipeline/refresh/graph
   * jobs, which never call `add_job_event`. */
  events: JobEventEntry[] | null;
  created_at: string;
  updated_at: string;
}

/**
 * `GET /instances/jobs/{id}/events`'s SSE event stream: `job` for
 * in-progress updates, `done` for the terminal state (stream then closes).
 */
export type JobSSEEvent = { type: "job"; data: Job } | { type: "done"; data: Job };

// -----------------------------------------------------------------------
// Routines (code review / bug hunt) — decision 9e
// -----------------------------------------------------------------------

export type RoutineKind = "review" | "bughunt";
export type RoutineStatus = "queued" | "running" | "done" | "failed";

export interface LaunchRoutineRequest {
  kind: RoutineKind;
  project: string;
  target_ref?: string | null;
  base_ref?: string | null;
  scope?: string | null;
  path_prefix?: string | null;
  model?: string | null;
  provider?: string | null;
  playbooks?: string[] | null;
}

export interface LaunchRoutineResponse {
  run_id: string;
  job_id: string;
}

/** One entry of GET /routines/runs's listing. */
export interface RoutineRunSummary {
  id: string;
  kind: RoutineKind;
  project: string;
  status: RoutineStatus;
  params: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  severity_counts: Record<string, number>;
}

/**
 * A finding's supporting evidence — see `services.routines.review`'s /
 * `services.routines.bughunt`'s `_validate_finding`. `hunk_excerpt` is
 * review-only (bughunt has no diff to excerpt from); every other field is
 * shared.
 */
export interface FindingEvidence {
  hunk_excerpt?: string;
  entities?: string[];
  model?: string;
  provider?: string;
}

export interface FindingDTO {
  id: number;
  file_path: string;
  start_line: number | null;
  end_line: number | null;
  severity: string;
  category: string;
  title: string;
  body: string;
  suggestion: string | null;
  evidence: FindingEvidence | null;
  anchored: boolean;
  created_at: string;
}

/** GET /routines/runs/{id} response — a run's full detail, findings included. */
export interface RoutineRunDetail {
  id: string;
  /** The underlying job this run executed as — stream/poll it the same way
   * as any other job (GET /instances/jobs/{job_id} and its /events route). */
  job_id: string;
  kind: RoutineKind;
  project: string;
  status: RoutineStatus;
  params: Record<string, unknown>;
  summary: string | null;
  stats: Record<string, unknown> | null;
  error: string | null;
  created_at: string;
  updated_at: string;
  findings: FindingDTO[];
}

export interface BranchInfo {
  name: string;
  is_current: boolean;
  last_commit_date: string;
}

/** GET /routines/git/branches response. */
export interface BranchesResponse {
  branches: BranchInfo[];
  default_base: string;
}

/** One entry of GET /routines/playbooks's listing — omits content in favor of content_bytes. */
export interface PlaybookSummary {
  id: number;
  name: string;
  routines: RoutineKind[];
  content_bytes: number;
  created_at: string;
  updated_at: string;
}

/** GET /routines/playbooks/{name} response — full detail, content included. */
export interface PlaybookDTO {
  id: number;
  name: string;
  routines: RoutineKind[];
  content: string;
  created_at: string;
  updated_at: string;
}

export interface CreatePlaybookRequest {
  name: string;
  content: string;
  routines?: RoutineKind[] | null;
}

export interface UpdatePlaybookRequest {
  content?: string | null;
  routines?: RoutineKind[] | null;
}

/** One entry of GET /routines/skills's listing — omits content in favor of content_bytes. */
export interface SkillSummary {
  id: number;
  name: string;
  description: string;
  content_bytes: number;
  created_at: string;
  updated_at: string;
}

/** GET /routines/skills/{name} response — full detail, content included. */
export interface SkillDTO {
  id: number;
  name: string;
  description: string;
  content: string;
  created_at: string;
  updated_at: string;
}

export interface CreateSkillRequest {
  name: string;
  description: string;
  content: string;
}

export interface UpdateSkillRequest {
  description?: string | null;
  content?: string | null;
}

// -----------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------

export interface ProviderData {
  provider_name: "openai" | "anthropic" | "ollama";
  model: string;
  key?: string;
  base_url?: string | null;
}

export interface LLMConfigRequest {
  is_local: boolean;
  provider_data: ProviderData;
}

/** GET/POST /config/llm response. */
export interface LLMConfig {
  is_local: boolean;
  provider_name: string;
  model: string;
  has_key: boolean;
  base_url: string | null;
}

export interface AvailableModel {
  name: string;
  provider: string;
}

export interface ModelsListResponse {
  provider_name: string;
  models: AvailableModel[];
}

export interface ProviderHealth {
  provider_name: string;
  model: string;
  is_local: boolean;
  status: "healthy" | "unhealthy";
  message: string;
}

// -----------------------------------------------------------------------
// Health
// -----------------------------------------------------------------------

export interface HealthResponse {
  status: string;
}
