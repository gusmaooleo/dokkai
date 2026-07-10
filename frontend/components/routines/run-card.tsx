"use client";

/**
 * One routine run row on the Routines home screen — kind badge, status
 * chip, target/scope summary, severity pills and relative time on a
 * clickable card, plus a delete action column, faithful to
 * `ConversationCard`'s split-button layout (`components/history/
 * conversation-card.tsx`): a delete `<button>` can't nest inside the
 * open `<button>`, so the two live in sibling columns instead.
 */

import { Trash2 } from "lucide-react";

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
import type { RoutineRunSummary } from "@/lib/types";
import { cn } from "@/lib/utils";
import { KIND_LABEL, relativeTime, severityEntries, STATUS_CLASSES, STATUS_LABEL, targetSummary } from "./format";

export function RunCard({
  run,
  canDelete,
  onOpen,
  onDelete,
}: {
  run: RoutineRunSummary;
  canDelete: boolean;
  onOpen: () => void;
  onDelete: () => void;
}) {
  const pills = severityEntries(run.severity_counts);

  return (
    <div className="flex items-stretch overflow-hidden rounded-[14px] border border-border bg-surface shadow-[var(--shadow-sm)] hover:border-border-strong">
      <button type="button" onClick={onOpen} className="min-w-0 flex-1 px-4 py-3.5 text-left sm:px-[18px]">
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <span className="rounded-[6px] border border-border bg-surface-2 px-2 py-0.5 font-mono text-[10.5px] text-muted-foreground">
            {KIND_LABEL[run.kind] ?? run.kind}
          </span>
          <span className={cn("rounded-lg px-2 py-0.5 font-mono text-[10.5px]", STATUS_CLASSES[run.status])}>
            {STATUS_LABEL[run.status] ?? run.status}
          </span>
          <span className="flex-1" />
          <span className="font-mono text-[11px] text-[color:var(--text-faint)]">{relativeTime(run.created_at)}</span>
        </div>

        <div className="mb-2 truncate font-mono text-[13px] text-foreground">{targetSummary(run)}</div>

        {pills.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {pills.map((p) => (
              <span key={p.severity} className={cn("rounded-md px-1.5 py-0.5 font-mono text-[10.5px]", p.classes)}>
                {p.count} {p.severity}
              </span>
            ))}
          </div>
        )}
      </button>

      {canDelete && (
        <div className="flex items-center border-l border-border px-3">
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
                <AlertDialogTitle>Delete this routine run?</AlertDialogTitle>
                <AlertDialogDescription>
                  {KIND_LABEL[run.kind]} run for &ldquo;{run.project}&rdquo; and its findings will be permanently
                  deleted. This can&apos;t be undone.
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
        </div>
      )}
    </div>
  );
}
