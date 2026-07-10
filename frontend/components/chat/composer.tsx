"use client";

/**
 * Bottom composer bar, faithful to the prototype's textarea + send button
 * (~L217-228): autosizing textarea (Enter sends, Shift+Enter newlines),
 * a send button that lights up once there's text, and a footer note with
 * the active model + the "Enter to send" hint. Disabled (with an inline
 * note) for the viewer role — decision 12p — and, the same way, when no
 * project is selected (there's nothing to query against).
 */

import { useEffect, useRef } from "react";
import { ArrowUp } from "lucide-react";

import { cn } from "@/lib/utils";

const MAX_HEIGHT_PX = 150;

export function Composer({
  value,
  onChange,
  onSend,
  streaming,
  viewer,
  noProject,
  modelNote,
}: {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  streaming: boolean;
  viewer: boolean;
  noProject: boolean;
  modelNote: string;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const disabled = streaming || viewer || noProject;
  const canSend = value.trim().length > 0 && !disabled;

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT_PX)}px`;
  }, [value]);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    // Don't send on the Enter that confirms an IME composition (e.g.
    // finishing a Japanese/Chinese/Korean candidate) — only a "real" Enter.
    if (e.nativeEvent.isComposing) return;
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (canSend) onSend();
    }
  }

  return (
    <div className="shrink-0 border-t border-border bg-background px-3.5 pt-3 pb-4 sm:px-6.5">
      <div className="mx-auto max-w-[820px]">
        {viewer && (
          <p className="mb-2 text-[12.5px] text-muted-foreground">
            Viewers can&apos;t send messages.
          </p>
        )}
        {!viewer && noProject && (
          <p className="mb-2 text-[12.5px] text-muted-foreground">
            Select a project from the sidebar to start chatting.
          </p>
        )}
        <div className="flex items-end gap-2.5 rounded-2xl border border-border-strong bg-surface py-2 pr-2 pl-4 shadow-[var(--shadow-sm)]">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={disabled}
            placeholder="Ask about the codebase…"
            className="min-h-6 flex-1 resize-none border-none bg-transparent py-[9px] text-[15px] leading-[1.5] text-foreground outline-none placeholder:text-[color:var(--text-faint)] disabled:cursor-not-allowed"
          />
          <button
            type="button"
            onClick={onSend}
            disabled={!canSend}
            className={cn(
              "grid size-[38px] shrink-0 place-items-center rounded-[11px] transition-colors",
              canSend ? "cursor-pointer bg-accent text-white" : "cursor-default bg-surface-2 text-[color:var(--text-faint)]",
            )}
          >
            <ArrowUp size={18} strokeWidth={2} />
          </button>
        </div>
        <div className="mt-[9px] flex items-center justify-between gap-2.5 px-1">
          <span className="font-mono text-[11px] text-[color:var(--text-faint)]">{modelNote}</span>
          <span className="font-mono text-[11px] text-[color:var(--text-faint)]">
            Enter to send · Shift+Enter for newline
          </span>
        </div>
      </div>
    </div>
  );
}
