/**
 * Shared Ollama reachability + installed-model probe, used by `up`,
 * `status`, and `doctor` so the `/api/tags` fetch + model-name matching
 * logic lives in one place.
 */

export interface OllamaProbe {
  reachable: boolean;
  installed: string[];
}

async function fetchWithTimeout(url: string, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

/** Base-name match, mirroring `src/services/llm_config.py`'s check. */
export function modelInstalled(installed: string[], model: string): boolean {
  return installed.some(
    (m) =>
      m === model ||
      m.startsWith(`${model}:`) ||
      model.startsWith(m.split(":")[0]),
  );
}

/**
 * Probe `${baseUrl}/api/tags`. Any failure (unreachable, non-2xx, timeout)
 * collapses to `{ reachable: false, installed: [] }` — callers don't need
 * to distinguish the cause.
 */
export async function probeOllama(
  baseUrl: string,
  timeoutMs: number,
): Promise<OllamaProbe> {
  try {
    const response = await fetchWithTimeout(`${baseUrl}/api/tags`, timeoutMs);
    if (!response.ok) return { reachable: false, installed: [] };
    const data = (await response.json()) as {
      models?: Array<{ name: string }>;
    };
    return { reachable: true, installed: (data.models ?? []).map((m) => m.name) };
  } catch {
    return { reachable: false, installed: [] };
  }
}
