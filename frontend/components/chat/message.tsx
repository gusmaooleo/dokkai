"use client";

/**
 * One chat turn, faithful to the prototype's message row (~L194-209): user
 * bubbles right-aligned in an accent tint, assistant bubbles left-aligned
 * with a logo avatar, a blinking caret while streaming, and (once settled)
 * a "N sources" chip + token estimate footer.
 */

import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { Layers } from "lucide-react";

import { cn } from "@/lib/utils";
import type { Audience } from "@/lib/types";
import type { ChatMessageState } from "./types";

const AUDIENCE_LABEL: Record<Audience, string> = {
  developer: "DEVELOPER",
  manager: "MANAGER",
  customer: "CUSTOMER",
};

function estimateTokenNote(message: ChatMessageState): string {
  const chars = (message.sources ?? []).reduce((sum, s) => sum + s.chunk_text.length, 0);
  return `~${(chars / 4 / 1000).toFixed(1)}k tokens retrieved`;
}

const markdownComponents: Components = {
  p: ({ children }) => <p className="mb-2.5 last:mb-0">{children}</p>,
  ul: ({ children }) => <ul className="mb-2.5 list-disc space-y-1 pl-5 last:mb-0">{children}</ul>,
  ol: ({ children }) => <ol className="mb-2.5 list-decimal space-y-1 pl-5 last:mb-0">{children}</ol>,
  li: ({ children }) => <li className="leading-[1.55]">{children}</li>,
  a: ({ children, href }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-accent underline hover:text-[color:var(--accent-press)]">
      {children}
    </a>
  ),
  strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
  blockquote: ({ children }) => (
    <blockquote className="mb-2.5 border-l-2 border-border-strong pl-3 text-muted-foreground last:mb-0">
      {children}
    </blockquote>
  ),
  pre: ({ children }) => (
    <pre className="mb-2.5 overflow-x-auto rounded-[9px] border border-border bg-surface-2 p-3 font-mono text-[12.5px] leading-[1.6] text-foreground last:mb-0">
      {children}
    </pre>
  ),
  code: ({ className, children, ...props }) => {
    const isBlock = /language-/.test(className ?? "");
    if (isBlock) {
      return (
        <code className={className} {...props}>
          {children}
        </code>
      );
    }
    return (
      <code
        className="rounded-md border border-border bg-surface-2 px-1.5 py-0.5 font-mono text-[0.9em] whitespace-nowrap text-foreground"
        {...props}
      >
        {children}
      </code>
    );
  },
};

export function Message({
  message,
  onOpenSources,
}: {
  message: ChatMessageState;
  onOpenSources: () => void;
}) {
  const isAssistant = message.role === "assistant";
  const hasSources = isAssistant && !message.streaming && !!message.sources?.length;

  return (
    <div className={cn("mb-6 flex animate-[dk-fadeup_0.25s_ease] flex-col", isAssistant ? "items-start" : "items-end")}>
      <div className={cn("mb-[7px] flex items-center gap-2", !isAssistant && "flex-row-reverse")}>
        {isAssistant && (
          <span className="grid size-6 shrink-0 place-items-center rounded-[7px] border border-border bg-surface-2">
            <img src="/dokkai-logo.svg" alt="" aria-hidden="true" className="size-3.5" />
          </span>
        )}
        <span className="text-[12.5px] font-bold text-foreground">{isAssistant ? "dokkai" : "You"}</span>
        {isAssistant && message.audience && (
          <span className="font-mono text-[10.5px] tracking-[0.05em] text-[color:var(--text-faint)] uppercase">
            {AUDIENCE_LABEL[message.audience]}
          </span>
        )}
      </div>

      <div
        className={cn(
          "text-[15px]",
          isAssistant
            ? "max-w-[min(760px,100%)] rounded-[4px_15px_15px_15px] border border-border bg-surface px-[17px] py-[15px] leading-[1.62] text-foreground shadow-[var(--shadow-sm)]"
            : "max-w-[min(680px,100%)] rounded-[15px_4px_15px_15px] border border-[color:var(--accent-ring)] bg-[color:var(--accent-weak)] px-4 py-3 leading-[1.55] text-foreground",
        )}
      >
        {isAssistant ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
            {message.content}
          </ReactMarkdown>
        ) : (
          <span className="whitespace-pre-wrap">{message.content}</span>
        )}
        {message.streaming && (
          <span className="ml-px inline-block h-4 w-2 translate-y-[2px] animate-[dk-blink_1s_step-end_infinite] rounded-[2px] bg-accent align-middle" />
        )}
        {message.error && (
          <p className="mt-2 text-[13px] text-[color:var(--accent-press)]">{message.error}</p>
        )}

        {hasSources && (
          <div className="mt-[13px] flex flex-wrap items-center gap-2 border-t border-border pt-3">
            <button
              type="button"
              onClick={onOpenSources}
              className="inline-flex h-7 items-center gap-1.5 rounded-lg border border-border bg-surface-2 px-2.5 text-[11.5px] font-semibold text-muted-foreground hover:border-accent hover:text-accent"
            >
              <Layers size={13} strokeWidth={1.8} />
              {message.sources!.length} sources
            </button>
            <span className="font-mono text-[10.5px] text-[color:var(--text-faint)]">
              {estimateTokenNote(message)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
