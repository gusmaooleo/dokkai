"use client";

/**
 * One conversation row in /history, faithful to the prototype's history
 * card (`dokkai-ui-prototype.html` ~L334-344): title + audience badge,
 * snippet, a project/message-count/relative-time meta row, and a resume +
 * delete action column. The prototype deletes on a single click with no
 * confirmation — this adds an AlertDialog confirm since it's a real
 * destructive call against Postgres; hidden entirely for the viewer role
 * (decision 12p — the caller passes `canDelete={false}`).
 */

import { ChevronRight, Trash2 } from "lucide-react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import type { ConversationSummary } from "@/lib/types";

const AUDIENCE_LABEL: Record<string, string> = {
  developer: "Developer",
  manager: "Manager",
  customer: "Customer",
};

/** Mirrors the prototype's `rel()` (~L998) — no date lib. */
export function relativeTime(iso: string): string {
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

/** Decision 12j: `title` is nullable — fall back to the preview, then a generic label. */
export function conversationTitle(c: ConversationSummary): string {
  return c.title || c.last_message_preview || "Untitled conversation";
}

export function ConversationCard({
  conversation,
  canDelete,
  onOpen,
  onDelete,
}: {
  conversation: ConversationSummary;
  canDelete: boolean;
  onOpen: () => void;
  onDelete: () => void;
}) {
  const title = conversationTitle(conversation);

  return (
    <div className="flex items-stretch overflow-hidden rounded-[13px] border border-border bg-surface shadow-[var(--shadow-sm)] hover:border-border-strong">
      <button
        type="button"
        onClick={onOpen}
        className="min-w-0 flex-1 px-[17px] py-[15px] text-left"
      >
        <div className="mb-1.5 flex flex-wrap items-center gap-[9px]">
          <span className="text-[14.5px] font-semibold text-foreground">{title}</span>
          <span className="rounded-[5px] border border-border px-1.5 py-px font-mono text-[10px] tracking-[0.04em] text-[color:var(--text-faint)] uppercase">
            {AUDIENCE_LABEL[conversation.audience] ?? conversation.audience}
          </span>
        </div>
        <div className="mb-[9px] overflow-hidden text-ellipsis whitespace-nowrap text-[13px] leading-[1.5] text-muted-foreground">
          {conversation.last_message_preview || "Empty conversation"}
        </div>
        <div className="flex items-center gap-3 font-mono text-[11px] text-[color:var(--text-faint)]">
          <span>{conversation.project_name}</span>
          <span>{conversation.message_count} messages</span>
          <span>{relativeTime(conversation.updated_at)}</span>
        </div>
      </button>

      <div className="flex flex-col justify-center gap-1 border-l border-border px-3 py-2.5">
        <button
          type="button"
          onClick={onOpen}
          title="Resume"
          className="grid size-8 place-items-center rounded-lg text-[color:var(--text-faint)] hover:bg-surface-2 hover:text-accent"
        >
          <ChevronRight size={16} strokeWidth={1.8} />
        </button>

        {canDelete && (
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <button
                type="button"
                title="Delete"
                className="grid size-8 place-items-center rounded-lg text-[color:var(--text-faint)] hover:bg-[color:var(--accent-weak)] hover:text-accent"
              >
                <Trash2 size={15} strokeWidth={1.8} />
              </button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Delete conversation?</AlertDialogTitle>
                <AlertDialogDescription>
                  &ldquo;{title}&rdquo; will be permanently deleted. This can&apos;t be undone.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction variant="destructive" onClick={onDelete}>
                  Delete
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        )}
      </div>
    </div>
  );
}
