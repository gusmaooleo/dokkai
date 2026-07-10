"use client";

/**
 * Shared chat screen used by both `/chat` (new conversation) and
 * `/chat/[id]` (resume) — faithful to the prototype's chat section
 * (`dokkai-ui-prototype.html` ~L177-264): audience switch + sources
 * toggle bar, scrollable message list, composer, and the sources panel.
 *
 * Streaming: POST /chat via `consumeSSE` — `sources` populates the
 * in-flight assistant message's source list, `token`s append to its
 * content, `done` finalizes it and (for a brand-new conversation) syncs
 * the URL to `/chat/<id>` via `history.replaceState` — deliberately *not*
 * `router.replace`, which would remount this screen mid-stream and drop
 * the in-progress message. `error` surfaces both as a toast and inline on
 * the message.
 */

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { List } from "lucide-react";

import { ApiError } from "@/lib/api";
import { chatApi, configApi } from "@/lib/api";
import { consumeSSE } from "@/lib/sse";
import { useAuth } from "@/lib/auth";
import { useProject } from "@/lib/project";
import { useToast } from "@/lib/toast";
import type { Audience, ChatSSEEvent } from "@/lib/types";
import { AudienceSwitch } from "./audience-switch";
import { Composer } from "./composer";
import { MessageList } from "./message-list";
import { SourcesPanel } from "./sources-panel";
import type { ChatMessageState } from "./types";

function isAudience(value: string): value is Audience {
  return value === "developer" || value === "manager" || value === "customer";
}

function latestSources(messages: ChatMessageState[]) {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role === "assistant" && m.sources) return m.sources;
  }
  return [];
}

export function ChatScreen({ conversationId: initialConversationId }: { conversationId?: string }) {
  const router = useRouter();
  const { user } = useAuth();
  const { active, loading: projectLoading } = useProject();
  const { toast } = useToast();

  const [messages, setMessages] = useState<ChatMessageState[]>([]);
  const [composerText, setComposerText] = useState("");
  const [audience, setAudience] = useState<Audience>("developer");
  const [conversationId, setConversationId] = useState<string | undefined>(initialConversationId);
  const [streaming, setStreaming] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(true);
  const [modelNote, setModelNote] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  const viewer = user?.role === "viewer";
  // `loading` guards the initial `GET /graph` round-trip so the note
  // doesn't flash before the project list has had a chance to resolve.
  const noProject = !projectLoading && !active;

  // Resume an existing conversation.
  useEffect(() => {
    if (!initialConversationId) return;
    let cancelled = false;

    chatApi
      .getConversation(initialConversationId)
      .then((conv) => {
        if (cancelled) return;
        const convAudience: Audience = isAudience(conv.audience) ? conv.audience : "developer";
        setAudience(convAudience);
        setConversationId(conv.conversation_id);
        setMessages(
          conv.messages
            .filter((m) => m.role === "user" || m.role === "assistant")
            .map((m) => ({
              id: crypto.randomUUID(),
              role: m.role as "user" | "assistant",
              content: m.content,
              sources: m.role === "assistant" ? m.sources : undefined,
              audience: m.role === "assistant" ? convAudience : undefined,
            })),
        );
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 404) {
          toast("Conversation not found", "error");
          router.replace("/history");
        } else {
          toast(err instanceof Error ? err.message : "Failed to load conversation", "error");
        }
      });

    return () => {
      cancelled = true;
    };
    // Only re-runs if the route's [id] segment itself changes — never
    // called with a different id from this same mounted instance.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialConversationId]);

  // Active model footnote (12r default; 404 → "not configured").
  useEffect(() => {
    let cancelled = false;
    configApi
      .getLlm()
      .then((cfg) => {
        if (cancelled) return;
        const location = cfg.is_local && cfg.base_url ? cfg.base_url : cfg.provider_name;
        setModelNote(`${cfg.model} · ${location}`);
      })
      .catch(() => {
        if (!cancelled) setModelNote("no LLM configured — see Settings");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Abort any in-flight stream when this screen unmounts (e.g. "New chat").
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  async function handleSend() {
    const text = composerText.trim();
    if (!text || streaming || viewer || !active) return;

    setComposerText("");
    const assistantId = crypto.randomUUID();
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "user", content: text },
      { id: assistantId, role: "assistant", content: "", streaming: true, audience },
    ]);
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await consumeSSE<ChatSSEEvent>(
        "/chat",
        (event) => {
          if (event.type === "sources") {
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, sources: event.data } : m)),
            );
          } else if (event.type === "token") {
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + event.data } : m)),
            );
          } else if (event.type === "done") {
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, streaming: false } : m)),
            );
            // `conversationId` here is the value captured when this stream
            // started (this closure runs for the lifetime of one `consumeSSE`
            // call) — safe to read directly rather than through a setState
            // updater, which must stay pure (no DOM side effects).
            if (!conversationId) {
              window.history.replaceState(null, "", `/chat/${event.data.conversation_id}`);
              setConversationId(event.data.conversation_id);
            }
          } else if (event.type === "error") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, streaming: false, error: event.data } : m,
              ),
            );
            toast(event.data, "error");
          }
        },
        {
          method: "POST",
          body: {
            message: text,
            project_name: active.project,
            audience,
            conversation_id: conversationId,
          },
          signal: controller.signal,
        },
      );
    } catch (err) {
      const detail = err instanceof Error ? err.message : "Request failed";
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, streaming: false, error: detail } : m)),
      );
      toast(detail, "error");
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  const lastMessage = messages[messages.length - 1];
  const thinking =
    streaming &&
    !!lastMessage &&
    lastMessage.role === "assistant" &&
    lastMessage.content.length === 0 &&
    !lastMessage.error;
  const currentSources = latestSources(messages);

  return (
    <div className="flex h-full min-h-0">
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex flex-wrap items-center gap-3 border-b border-border px-3.5 py-3 sm:px-6.5">
          <AudienceSwitch value={audience} onChange={setAudience} />
          <div className="flex-1" />
          {!sourcesOpen && (
            <button
              type="button"
              onClick={() => setSourcesOpen(true)}
              className="inline-flex h-[34px] items-center gap-[7px] rounded-[9px] border border-border-strong bg-surface px-3 text-[12.5px] font-semibold text-muted-foreground"
            >
              <List size={15} strokeWidth={1.7} />
              Sources · {currentSources.length}
            </button>
          )}
        </div>

        <MessageList
          messages={messages}
          thinking={thinking}
          onOpenSources={() => setSourcesOpen(true)}
        />

        <Composer
          value={composerText}
          onChange={setComposerText}
          onSend={handleSend}
          streaming={streaming}
          viewer={viewer}
          noProject={noProject}
          modelNote={modelNote}
        />
      </div>

      {sourcesOpen && (
        <div className="hidden lg:flex lg:h-full">
          <SourcesPanel sources={currentSources} onClose={() => setSourcesOpen(false)} />
        </div>
      )}
    </div>
  );
}
