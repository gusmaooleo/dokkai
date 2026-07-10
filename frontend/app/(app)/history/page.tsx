"use client";

/**
 * Conversation history, faithful to the prototype's History screen
 * (`dokkai-ui-prototype.html` ~L322-352): a search box over the
 * conversation list, grouped Today / Yesterday / Earlier by `updated_at`
 * (mirrors the prototype's `bucket()` ~L999 — <24h / <48h / else), most
 * recent first within each group. Delete asks for confirmation (see
 * `ConversationCard`), then refetches. Delete buttons are hidden entirely
 * for the viewer role (decision 12p) — resuming/reading is still allowed.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { LoaderCircle, Search } from "lucide-react";

import { ApiError, chatApi } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useToast } from "@/lib/toast";
import type { ConversationSummary } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { ConversationCard, conversationTitle } from "@/components/history/conversation-card";

type GroupLabel = "Today" | "Yesterday" | "Earlier";
const GROUP_ORDER: GroupLabel[] = ["Today", "Yesterday", "Earlier"];

/** Mirrors the prototype's `bucket()` (~L999). */
function bucketOf(iso: string): GroupLabel {
  const hours = (Date.now() - new Date(iso).getTime()) / 3_600_000;
  if (hours < 24) return "Today";
  if (hours < 48) return "Yesterday";
  return "Earlier";
}

export default function HistoryPage() {
  const router = useRouter();
  const { user } = useAuth();
  const { toast } = useToast();

  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  const canDelete = user?.role !== "viewer";

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const list = await chatApi.listConversations();
      setConversations(list);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Failed to load conversations");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const groups = useMemo(() => {
    const q = search.trim().toLowerCase();
    const filtered = conversations.filter((c) => {
      if (!q) return true;
      return (
        conversationTitle(c).toLowerCase().includes(q) ||
        c.last_message_preview.toLowerCase().includes(q)
      );
    });
    const sorted = [...filtered].sort(
      (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
    );
    const byLabel = new Map<GroupLabel, ConversationSummary[]>();
    for (const c of sorted) {
      const label = bucketOf(c.updated_at);
      const bucket = byLabel.get(label) ?? [];
      bucket.push(c);
      byLabel.set(label, bucket);
    }
    return GROUP_ORDER.filter((label) => byLabel.has(label)).map((label) => ({
      label,
      items: byLabel.get(label)!,
    }));
  }, [conversations, search]);

  async function handleDelete(id: string) {
    try {
      await chatApi.deleteConversation(id);
      toast("Conversation deleted");
      load();
    } catch (err) {
      toast(err instanceof ApiError ? err.detail : "Failed to delete conversation", "error");
    }
  }

  return (
    <div className="mx-auto max-w-[860px] px-3.5 py-[18px] sm:px-6.5 sm:py-[30px]">
      <div className="relative mb-6">
        <Search
          size={16}
          strokeWidth={1.8}
          className="pointer-events-none absolute top-1/2 left-3.5 -translate-y-1/2 text-[color:var(--text-faint)]"
        />
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search conversations…"
          className="h-11 w-full rounded-xl border border-border-strong bg-surface pr-3.5 pl-10 text-[14.5px] text-foreground outline-none"
        />
      </div>

      {loading && (
        <div className="flex justify-center py-20">
          <LoaderCircle
            size={26}
            strokeWidth={2.4}
            className="animate-[dk-spin_0.7s_linear_infinite] text-muted-foreground"
          />
        </div>
      )}

      {!loading && error && (
        <div className="flex flex-col items-center gap-3 py-20 text-center">
          <p className="text-[14px] text-muted-foreground">{error}</p>
          <Button variant="outline" onClick={load}>
            Retry
          </Button>
        </div>
      )}

      {!loading && !error && groups.length === 0 && (
        <div className="py-[70px] text-center text-[14px] text-[color:var(--text-faint)]">
          {conversations.length === 0
            ? "No conversations yet — start one from the Chat tab."
            : "No conversations match your search."}
        </div>
      )}

      {!loading &&
        !error &&
        groups.map((g) => (
          <div key={g.label} className="mb-[26px]">
            <div className="mb-3 font-mono text-[11px] font-semibold tracking-[0.07em] text-[color:var(--text-faint)] uppercase">
              {g.label}
            </div>
            <div className="flex flex-col gap-2.5">
              {g.items.map((c) => (
                <ConversationCard
                  key={c.conversation_id}
                  conversation={c}
                  canDelete={canDelete}
                  onOpen={() => router.push(`/chat/${c.conversation_id}`)}
                  onDelete={() => handleDelete(c.conversation_id)}
                />
              ))}
            </div>
          </div>
        ))}
    </div>
  );
}
