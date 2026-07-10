/**
 * Local (client-only) chat message shape used while a conversation is on
 * screen — a superset of `lib/types.ts`'s `ConversationMessage` that also
 * tracks in-flight streaming state (`streaming`, `error`) which never
 * reaches the API.
 */

import type { Audience, SourceChunk } from "@/lib/types";

export interface ChatMessageState {
  id: string;
  role: "user" | "assistant";
  content: string;
  /** True while tokens are still streaming in for this message. */
  streaming?: boolean;
  /** Present once the `sources` SSE event for this exchange has arrived. */
  sources?: SourceChunk[];
  /** Set if the `error` SSE event fired for this exchange. */
  error?: string;
  /**
   * The audience active when this (assistant) message was requested. Known
   * exactly for messages sent this session; for history loaded from the
   * API (which has no per-message audience) this falls back to the
   * conversation's overall `audience` field.
   */
  audience?: Audience;
}
