"use client";

/**
 * One playbook or skill row on the Library screen (C5) — faithful to
 * `RunCard`'s split-button layout (open button + a sibling delete column, so
 * the delete `<button>` never nests inside the open `<button>`). Playbooks
 * show their applicable-routines chips; skills show their (required)
 * description instead — the two document types don't share those fields.
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
import type { PlaybookSummary, SkillSummary } from "@/lib/types";
import { formatBytes, relativeTime } from "./format";

export type DocumentEntry =
  | { docType: "playbook"; doc: PlaybookSummary }
  | { docType: "skill"; doc: SkillSummary };

export function DocumentCard({
  entry,
  canDelete,
  onOpen,
  onDelete,
}: {
  entry: DocumentEntry;
  canDelete: boolean;
  onOpen: () => void;
  onDelete: () => void;
}) {
  const { docType, doc } = entry;

  return (
    <div className="flex items-stretch overflow-hidden rounded-[14px] border border-border bg-surface shadow-[var(--shadow-sm)] hover:border-border-strong">
      <button type="button" onClick={onOpen} className="min-w-0 flex-1 px-4 py-3.5 text-left sm:px-[18px]">
        <div className="mb-1.5 flex flex-wrap items-center gap-2">
          <span className="truncate text-[14.5px] font-semibold text-foreground">{doc.name}</span>
        </div>

        {docType === "skill" ? (
          <div className="mb-2 overflow-hidden text-ellipsis whitespace-nowrap text-[13px] leading-[1.5] text-muted-foreground">
            {doc.description}
          </div>
        ) : doc.routines.length > 0 ? (
          <div className="mb-2 flex flex-wrap gap-1.5">
            {doc.routines.map((r) => (
              <span
                key={r}
                className="rounded-[6px] border border-border bg-surface-2 px-2 py-0.5 font-mono text-[10.5px] text-muted-foreground"
              >
                {r}
              </span>
            ))}
          </div>
        ) : (
          <div className="mb-2 text-[12.5px] text-[color:var(--text-faint)]">applies to no routines yet</div>
        )}

        <div className="flex items-center gap-3 font-mono text-[11px] text-[color:var(--text-faint)]">
          <span>{formatBytes(doc.content_bytes)}</span>
          <span>updated {relativeTime(doc.updated_at)}</span>
        </div>
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
                <AlertDialogTitle>
                  Delete {docType} &lsquo;{doc.name}&rsquo;?
                </AlertDialogTitle>
                <AlertDialogDescription>
                  It will be permanently deleted. This can&apos;t be undone.
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
