/**
 * Minimal Server-Sent Events client over `fetch`. Mirrors the framing used
 * by `POST /chat` and `GET /instances/jobs/{id}/events`
 * (`event: <type>\ndata: <json>\n\n`), tolerating frames split across
 * network chunks and CRLF line endings.
 */

export interface SseFrame {
  event: string;
  data: string;
}

export interface StreamSseOptions {
  signal?: AbortSignal;
  /** Called once per raw chunk received from the network — useful for a
   * connection-liveness watchdog upstream (a full frame may not have
   * arrived yet, but bytes did). */
  onChunk?: () => void;
  /** Extra headers to send with the request (e.g. `Authorization`). */
  headers?: Record<string, string>;
}

/** Thrown when the SSE endpoint itself responds 404 (unknown resource). */
export class SseNotFoundError extends Error {
  constructor(url: string) {
    super(`SSE endpoint not found (404): ${url}`);
    this.name = "SseNotFoundError";
  }
}

function parseFrame(raw: string): SseFrame | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of raw.split("\n")) {
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
 * Connect to `url` and yield each SSE frame as it arrives. The connection
 * stays open for as long as the server keeps it open (or `options.signal`
 * is aborted) — callers drive termination.
 */
export async function* streamSse(
  url: string,
  options: StreamSseOptions = {},
): AsyncGenerator<SseFrame> {
  const response = await fetch(url, {
    signal: options.signal,
    headers: { Accept: "text/event-stream", ...options.headers },
  });

  if (response.status === 404) {
    throw new SseNotFoundError(url);
  }
  if (!response.ok || !response.body) {
    throw new Error(`SSE connection to ${url} failed: HTTP ${response.status}`);
  }

  const decoder = new TextDecoder();
  let buffer = "";

  for await (const chunk of response.body as AsyncIterable<Uint8Array>) {
    options.onChunk?.();
    buffer += decoder.decode(chunk, { stream: true });
    // Normalize CRLF -> LF. Idempotent across calls, so re-running it on
    // the accumulated buffer after every chunk is safe even when a \r and
    // its \n land in separate network chunks.
    buffer = buffer.replace(/\r\n/g, "\n");

    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const rawFrame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const frame = parseFrame(rawFrame);
      if (frame) yield frame;
    }
  }
}
