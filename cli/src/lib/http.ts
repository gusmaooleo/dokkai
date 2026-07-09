/**
 * Thin typed helpers over global `fetch` for talking to the dokkai API.
 */

function connectionHint(apiUrl: string, cause: unknown): Error {
  return new Error(
    `could not reach the dokkai API at ${apiUrl} — is it running? Try ./dev.sh\n` +
      `(${cause instanceof Error ? cause.message : String(cause)})`,
  );
}

async function extractErrorDetail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string") return detail;
      return JSON.stringify(detail);
    }
  } catch {
    // response body wasn't JSON — fall through to the status text
  }
  return response.statusText || `HTTP ${response.status}`;
}

async function request(url: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (cause) {
    throw connectionHint(url, cause);
  }

  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(`dokkai API error (${response.status}): ${detail}`);
  }

  return response;
}

export async function getJson<T>(url: string): Promise<T> {
  const response = await request(url);
  return (await response.json()) as T;
}

export async function postJson<T>(url: string, body?: unknown): Promise<T> {
  const response = await request(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  return (await response.json()) as T;
}
