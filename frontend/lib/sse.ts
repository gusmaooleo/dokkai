/**
 * Fetch-based Server-Sent Events consumer for `POST /chat` and
 * `GET /instances/jobs/{id}/events` — both stream `event: <type>\ndata:
 * <json>\n\n` frames over `text/event-stream` (see `src/controllers/chat.py`
 * and `src/controllers/instances.py`). The `fetch`+`ReadableStream` reader
 * is used instead of `EventSource` because `EventSource` can't send a
 * bearer `Authorization` header or a POST body.
 */

import { ApiError, API_BASE_URL, getToken } from "./api";

interface ConsumeSSEOptions {
  method?: "GET" | "POST";
  body?: unknown;
  signal?: AbortSignal;
}

/** One raw frame parsed into its event type + still-JSON-encoded data string. */
interface RawFrame {
  event: string;
  data: string;
}

/**
 * Parses a single frame (already split off on the blank-line boundary).
 * Lines are `event: <type>` and `data: <json>`; per the SSE spec, multiple
 * `data:` lines are joined with `\n` (this backend always sends exactly
 * one). Returns `null` for a frame with no `data:` line (e.g. a stray
 * blank chunk) — nothing to dispatch.
 */
function parseFrame(frame: string): RawFrame | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trim());
    }
  }
  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}

/**
 * Consumes an SSE stream at `path`, invoking `onEvent` once per frame with
 * its `data:` payload JSON-parsed. `E` should be the discriminated union of
 * events the endpoint can emit (e.g. `ChatSSEEvent`, `JobSSEEvent`) — the
 * frame's `event:` line becomes `type` and `data:` becomes `data`.
 *
 * Resolves cleanly when the server closes the stream; rejects with
 * `ApiError` on a non-2xx response or a network failure, or with a plain
 * `Error` if a frame's `data:` isn't valid JSON. Pass `signal` to cancel
 * early (e.g. `AbortController`) — an abort resolves cleanly rather than
 * rejecting.
 */
export async function consumeSSE<E extends { type: string; data: unknown }>(
  path: string,
  onEvent: (event: E) => void,
  { method = "GET", body, signal }: ConsumeSSEOptions = {},
): Promise<void> {
  const token = getToken();
  const headers: Record<string, string> = { Accept: "text/event-stream" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (token) headers["Authorization"] = `Bearer ${token}`;

  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") return;
    throw new ApiError(0, `could not reach the API at ${API_BASE_URL} — is the server running?`);
  }

  if (!res.ok || !res.body) {
    let detail = res.statusText || `HTTP ${res.status}`;
    try {
      const data: unknown = await res.json();
      if (data && typeof data === "object" && "detail" in data) {
        const parsed = (data as { detail: unknown }).detail;
        if (typeof parsed === "string") detail = parsed;
      }
    } catch {
      // Non-JSON (or empty) error body — keep the generic message.
    }
    throw new ApiError(res.status, detail);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  // Normalize CRLF/CR line endings to LF as chunks arrive, so both frame
  // boundaries ("\n\n") and per-line splitting are CRLF-agnostic; a frame
  // split across two `read()` calls is handled by buffering rather than
  // parsing until a full "\n\n" boundary is seen.
  let buffer = "";

  const dispatch = (frame: string) => {
    const parsed = parseFrame(frame);
    if (parsed === null) return;
    const data = JSON.parse(parsed.data) as E["data"];
    onEvent({ type: parsed.event, data } as E);
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n|\r/g, "\n");

      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        dispatch(frame);
      }
    }
    // Flush a trailing frame that wasn't terminated by a final blank line.
    const trailing = buffer.trim();
    if (trailing) dispatch(trailing);
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") return;
    throw err;
  } finally {
    reader.releaseLock();
  }
}
