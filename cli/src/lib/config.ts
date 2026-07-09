import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";

const DEFAULT_API_URL = "http://localhost:8000";

/**
 * Minimal `.env` parser: KEY=VALUE lines, blank lines and lines starting
 * with `#` are ignored, values are trimmed. No quote handling beyond that.
 */
function parseEnvFile(path: string): Record<string, string> {
  const result: Record<string, string> = {};
  const contents = readFileSync(path, "utf8");
  for (const rawLine of contents.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
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

function readRepoEnv(): Record<string, string> {
  try {
    const home = resolveDokkaiHome();
    const envPath = join(home, ".env");
    if (!existsSync(envPath)) return {};
    return parseEnvFile(envPath);
  } catch {
    return {};
  }
}

/**
 * Resolve the dokkai API URL, in precedence order:
 * `--api` flag > `DOKKAI_API_URL` env var > repo `.env` > default.
 */
export function resolveApiUrl(flags: { api?: string }): string {
  if (flags.api) return flags.api;
  if (process.env.DOKKAI_API_URL) return process.env.DOKKAI_API_URL;

  const repoEnv = readRepoEnv();
  if (repoEnv.DOKKAI_API_URL) return repoEnv.DOKKAI_API_URL;

  return DEFAULT_API_URL;
}
