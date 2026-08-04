import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";

const DEFAULT_API_URL = "http://localhost:8000";

/**
 * Minimal `.env` parser: KEY=VALUE lines, blank lines and lines starting
 * with `#` are ignored, values are trimmed. An optional leading `export `
 * (as `python-dotenv` — what the API actually loads `.env` with — also
 * accepts) is stripped before splitting on `=`, so `export FOO=bar` reads
 * back as `FOO=bar` here too instead of a silently-dropped line (which
 * this module's callers would otherwise read as "not set" — a false
 * negative, e.g. reporting a genuinely-configured API key as missing). No
 * quote handling beyond that.
 */
function parseEnvFile(path: string): Record<string, string> {
  const result: Record<string, string> = {};
  const contents = readFileSync(path, "utf8");
  for (const rawLine of contents.split("\n")) {
    let line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    line = line.replace(/^export\s+/, "");
    const eq = line.indexOf("=");
    if (eq === -1) continue;
    const key = line.slice(0, eq).trim();
    const value = line.slice(eq + 1).trim();
    if (key) result[key] = value;
  }
  return result;
}

/**
 * Walk up from cwd looking for the dokkai repo root, identified by the
 * presence of both `docker-compose.yml` and `src/mcp_server.py`.
 */
export function resolveDokkaiHome(): string {
  const envHome = process.env.DOKKAI_HOME;
  if (envHome) return envHome;

  let dir = process.cwd();
  while (true) {
    const hasCompose = existsSync(join(dir, "docker-compose.yml"));
    const hasMcpServer = existsSync(join(dir, "src", "mcp_server.py"));
    if (hasCompose && hasMcpServer) return dir;

    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }

  throw new Error(
    "could not locate the dokkai repo root — set DOKKAI_HOME to point at it, " +
      "or run this command from inside the dokkai repo",
  );
}

let cachedRepoEnv: Record<string, string> | null = null;

/**
 * Read the repo's `.env` file (KEY=VALUE pairs), relative to
 * `resolveDokkaiHome()`. Returns `{}` if the repo root can't be found or
 * has no `.env`. Cached after the first read.
 */
export function readRepoEnv(): Record<string, string> {
  if (cachedRepoEnv) return cachedRepoEnv;
  try {
    const home = resolveDokkaiHome();
    const envPath = join(home, ".env");
    cachedRepoEnv = existsSync(envPath) ? parseEnvFile(envPath) : {};
  } catch {
    cachedRepoEnv = {};
  }
  return cachedRepoEnv;
}

/**
 * Resolve a config value, in precedence order: process env var > repo
 * `.env` > `fallback`.
 */
export function envVar(name: string, fallback: string): string {
  return process.env[name] ?? readRepoEnv()[name] ?? fallback;
}

/**
 * Resolve the dokkai API URL, in precedence order:
 * `--api` flag > `DOKKAI_API_URL` env var > repo `.env` > default.
 */
export function resolveApiUrl(flags: { api?: string }): string {
  if (flags.api) return flags.api;
  return envVar("DOKKAI_API_URL", DEFAULT_API_URL);
}
