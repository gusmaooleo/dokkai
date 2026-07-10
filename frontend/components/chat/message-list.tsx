"use client";

/**
 * Scrollable message column, faithful to the prototype's chat scroll area
 * (~L192-215): a centered 820px column of `Message`s, the "retrieving
 * graph context…" thinking indicator while the latest assistant turn has
 * no content yet, and auto-scroll to the bottom on new content.
 *
 * The prototype has no dedicated empty-chat state for the chat screen
 * itself (only History does, ~L349) — this hero is a reasonable
 * from-scratch composition using the same design tokens, not lifted from
 * the reference file. See the implementer's report for details.
 */

import { useEffect, useRef } from "react";

import { Message } from "./message";
import type { ChatMessageState } from "./types";

function EmptyState() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 py-20 text-center">
      <img src="/dokkai-logo.svg" alt="" aria-hidden="true" className="size-9 opacity-90" />
      <h2 className="font-display text-[19px] font-bold tracking-[-0.02em] text-foreground">
        Ask about the codebase
      </h2>
      <p className="max-w-[380px] text-[13.5px] leading-[1.55] text-muted-foreground">
        Questions are answered from the retrieved graph context — implementation, callers and
        dependencies, cited with the exact source chunks.
      </p>
    </div>
  );
}

function ThinkingIndicator() {
  return (
    <div className="flex items-center gap-[9px] py-2.5 text-[13px] text-[color:var(--text-faint)]">
      <span className="inline-flex gap-[3px]">
        <span className="size-1.5 animate-[dk-pulse_1s_ease-in-out_infinite] rounded-full bg-accent" />
        <span className="size-1.5 animate-[dk-pulse_1s_ease-in-out_0.2s_infinite] rounded-full bg-accent" />
        <span className="size-1.5 animate-[dk-pulse_1s_ease-in-out_0.4s_infinite] rounded-full bg-accent" />
      </span>
      retrieving graph context…
    </div>
  );
}

export function MessageList({
  messages,
  thinking,
  onOpenSources,
}: {
  messages: ChatMessageState[];
  thinking: boolean;
  onOpenSources: () => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, thinking]);

  if (messages.length === 0) {
    return (
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-3.5 sm:px-6.5">
        <EmptyState />
      </div>
    );
  }

  return (
    <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-3.5 py-[18px] sm:px-6.5 sm:py-[30px]">
      <div className="mx-auto max-w-[820px]">
        {messages.map((m) => (
          <Message key={m.id} message={m} onOpenSources={onOpenSources} />
        ))}
        {thinking && <ThinkingIndicator />}
      </div>
    </div>
  );
}
