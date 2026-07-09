/**
 * Per-process CLI authentication against the dokkai API (auth is always
 * enabled — decision 6b). `authHeaders()` lazily logs in once using
 * `DOKKAI_ROOT_USER`/`DOKKAI_ROOT_PASSWORD` (falling back to the server's
 * own admin/admin seed), caches the bearer token for the rest of the
 * process, and prints a one-time warning if the default admin/admin
 * credentials are still active.
 */

import chalk from "chalk";
import { envVar } from "./config.js";

const REQUEST_TIMEOUT_MS = 10_000;

interface LoginResponse {
  token: string;
  expires_at: string;
  role: string;
  default_admin_active: boolean;
}

interface StatusResponse {
  enabled: boolean;
  default_admin_active: boolean;
}

/** Raised only for a network-level failure (unreachable/timeout) — never for
 * an HTTP response the server actually sent back (e.g. 401/503). Lets
 * callers tell "server is down" apart from "server answered, but no". */
class ConnectionError extends Error {}

async function fetchWithTimeout(url: string, init: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } catch (cause) {
    throw new ConnectionError(cause instanceof Error ? cause.message : String(cause));
  } finally {
    clearTimeout(timer);
  }
}

function resolveCredentials(): { username: string; password: string } {
  return {
    username: envVar("DOKKAI_ROOT_USER", "admin"),
    password: envVar("DOKKAI_ROOT_PASSWORD", "admin"),
  };
}

function authFailureMessage(username: string): string {
  return (
    `authentication failed for user '${username}' — set DOKKAI_ROOT_USER and ` +
    "DOKKAI_ROOT_PASSWORD (or update the admin password) and retry"
  );
}

/** Error to raise when a request is still unauthorized after a re-login retry. */
export function describeAuthFailure(): Error {
  const { username } = resolveCredentials();
  return new Error(authFailureMessage(username));
}

let warnedDefaultAdmin = false;

function warnDefaultAdminOnce(): void {
  if (warnedDefaultAdmin) return;
  warnedDefaultAdmin = true;
  console.error(
    chalk.yellow(
      "warning: the default admin/admin credentials are active — create your " +
        "own admin user and delete the default one",
    ),
  );
}

let statusChecked = false;

/**
 * GET /auth/status once per process, best-effort, only to surface the
 * default-admin warning as early as possible. Any failure (connection error,
 * non-ok response) is ignored — it never disables later auth attempts, since
 * a down/booting API at process start must not permanently latch every
 * subsequent `authHeaders()` call to `{}` (a still-running long-lived
 * command like `dokkai watch`/the chat REPL must self-heal once the API
 * comes up, with no CLI restart). The default-admin warning still fires from
 * `login()`'s own response as a fallback when this check missed it.
 */
async function checkStatusOnce(apiUrl: string): Promise<void> {
  if (statusChecked) return;
  statusChecked = true;
  try {
    const response = await fetchWithTimeout(`${apiUrl}/auth/status`, {});
    if (!response.ok) return;
    const data = (await response.json()) as StatusResponse;
    if (data.default_admin_active) warnDefaultAdminOnce();
  } catch {
    // ignored — see comment above
  }
}

async function login(apiUrl: string): Promise<string> {
  const { username, password } = resolveCredentials();
  const response = await fetchWithTimeout(`${apiUrl}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });

  if (response.status === 401) {
    throw new Error(authFailureMessage(username));
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(
      `dokkai API login error (${response.status}): ${detail || response.statusText}`,
    );
  }

  const data = (await response.json()) as LoginResponse;
  if (data.default_admin_active) warnDefaultAdminOnce();
  return data.token;
}

let cachedApiUrl: string | null = null;
let tokenPromise: Promise<string> | null = null;

function getToken(apiUrl: string): Promise<string> {
  if (cachedApiUrl !== apiUrl) {
    cachedApiUrl = apiUrl;
    tokenPromise = null;
  }
  if (!tokenPromise) {
    tokenPromise = login(apiUrl).catch((error: unknown) => {
      tokenPromise = null;
      throw error;
    });
  }
  return tokenPromise;
}

/** Drop the cached token — the next `authHeaders()` call re-logs in. */
export function invalidateAuthCache(): void {
  tokenPromise = null;
}

/**
 * Resolve the `Authorization` header for requests to `apiUrl`, logging in
 * (and caching the token) on first use.
 *
 * A connection-level failure (API unreachable/booting) resolves to `{}` for
 * THIS call only — it isn't cached — so the caller's own request fails on
 * its own and produces the usual "could not reach the dokkai API" hint, and
 * the very next call tries a fresh login (self-healing once the API comes
 * up, without a CLI restart).
 */
export async function authHeaders(apiUrl: string): Promise<Record<string, string>> {
  await checkStatusOnce(apiUrl);
  try {
    const token = await getToken(apiUrl);
    return { Authorization: `Bearer ${token}` };
  } catch (error) {
    if (error instanceof ConnectionError) return {};
    throw error;
  }
}
