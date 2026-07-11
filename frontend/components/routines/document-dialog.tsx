"use client";

/**
 * Create/edit dialog for a playbook or skill, faithful to `CreateUserDialog`'s
 * dialog shape (`components/admin/create-user-dialog.tsx`). Shared by both
 * document types (`docType`) — `mode` is derived from `name` (`null` means
 * create; a string means edit, fetching that document's full content on
 * open).
 *
 * `name` is immutable after creation (decision 9d — it's the routine
 * engine's lookup key), so it's a locked field once in edit mode. Content is
 * capped at 16KB server-side (`services.routines.documents._MAX_CONTENT_BYTES`)
 * — this dialog pre-checks that limit client-side at save time (typed,
 * pasted or uploaded content all funnel through the same textarea) and
 * blocks the request with the SAME message shape the backend's 400 would use
 * (`services.routines.documents._validate_content`), rather than a generic
 * "too large" toast.
 *
 * `readOnly` (the viewer role, 12p) renders every field disabled and drops
 * the Save button — the dialog becomes a read-only viewer for that
 * document's content instead of an editor.
 */

import { useEffect, useRef, useState } from "react";
import { Upload } from "lucide-react";

import { ApiError, routinesApi } from "@/lib/api";
import { useToast } from "@/lib/toast";
import type { RoutineKind } from "@/lib/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

const MAX_CONTENT_BYTES = 16_384;

const ROUTINE_OPTIONS: { value: RoutineKind; label: string }[] = [
  { value: "review", label: "Review" },
  { value: "bughunt", label: "Bug hunt" },
];

const DOC_LABEL: Record<"playbook" | "skill", string> = {
  playbook: "playbook",
  skill: "skill",
};

const DOC_DESCRIPTION: Record<"playbook" | "skill", string> = {
  playbook: "Playbooks are pushed into every matching routine run — house rules the model must follow.",
  skill: "Skills are pulled by the model at its own discretion during a run, chosen by relevance.",
};

function byteLength(text: string): number {
  return new TextEncoder().encode(text).length;
}

export function DocumentDialog({
  open,
  onOpenChange,
  docType,
  name,
  readOnly,
  onSaved,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  docType: "playbook" | "skill";
  /** `null` = create a new document; a string = edit/view that document. */
  name: string | null;
  readOnly: boolean;
  onSaved: () => void;
}) {
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mode: "create" | "edit" = name === null ? "create" : "edit";

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formName, setFormName] = useState("");
  const [description, setDescription] = useState("");
  const [routines, setRoutines] = useState<RoutineKind[]>(["review", "bughunt"]);
  const [content, setContent] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!open) return;

    if (name === null) {
      setFormName("");
      setDescription("");
      setRoutines(["review", "bughunt"]);
      setContent("");
      setError(null);
      setLoading(false);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);
    const fetcher = docType === "playbook" ? routinesApi.getPlaybook(name) : routinesApi.getSkill(name);
    fetcher
      .then((doc) => {
        if (cancelled) return;
        setFormName(doc.name);
        setContent(doc.content);
        if (docType === "skill" && "description" in doc) setDescription(doc.description);
        if (docType === "playbook" && "routines" in doc) setRoutines(doc.routines as RoutineKind[]);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof ApiError ? err.detail : `Failed to load ${docType}`);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, docType, name]);

  function toggleRoutine(kind: RoutineKind) {
    setRoutines((prev) => (prev.includes(kind) ? prev.filter((r) => r !== kind) : [...prev, kind]));
  }

  function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setContent(String(reader.result ?? ""));
    reader.onerror = () => toast("Could not read the selected file.", "error");
    reader.readAsText(file);
  }

  async function handleSave() {
    if (mode === "create" && !formName.trim()) {
      toast("Enter a name.", "error");
      return;
    }
    if (docType === "skill" && !description.trim()) {
      toast("Enter a description.", "error");
      return;
    }
    const bytes = byteLength(content);
    if (bytes > MAX_CONTENT_BYTES) {
      toast(`${DOC_LABEL[docType]} content exceeds the 16KB limit (${bytes} bytes)`, "error");
      return;
    }

    const label = docType === "playbook" ? "Playbook" : "Skill";
    setSubmitting(true);
    try {
      if (mode === "create") {
        const created =
          docType === "playbook"
            ? await routinesApi.createPlaybook({ name: formName.trim(), content, routines })
            : await routinesApi.createSkill({ name: formName.trim(), description: description.trim(), content });
        toast(`${label} '${created.name}' created`);
      } else if (name) {
        if (docType === "playbook") {
          await routinesApi.updatePlaybook(name, { content, routines });
        } else {
          await routinesApi.updateSkill(name, { description: description.trim(), content });
        }
        toast(`${label} '${name}' updated`);
      }
      onSaved();
      onOpenChange(false);
    } catch (err) {
      toast(err instanceof ApiError ? err.detail : `Failed to save ${docType}`, "error");
    } finally {
      setSubmitting(false);
    }
  }

  const contentBytes = byteLength(content);
  const overLimit = contentBytes > MAX_CONTENT_BYTES;
  const disabled = readOnly || loading;

  const title =
    mode === "create"
      ? `New ${DOC_LABEL[docType]}`
      : readOnly
        ? `View ${DOC_LABEL[docType]}`
        : `Edit ${DOC_LABEL[docType]}`;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle className="font-display text-[17px] font-bold">{title}</DialogTitle>
          <DialogDescription className="text-[12.5px] text-muted-foreground">
            {DOC_DESCRIPTION[docType]}
            {readOnly && " You have view-only access — sign in as an admin or user account to make changes."}
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <p className="py-6 text-center text-[13.5px] text-muted-foreground">Loading…</p>
        ) : error ? (
          <p className="py-6 text-center text-[13.5px] text-muted-foreground">{error}</p>
        ) : (
          <div className="flex flex-col gap-[18px]">
            <div>
              <Label htmlFor="doc-name" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
                Name
              </Label>
              <Input
                id="doc-name"
                value={formName}
                onChange={(e) => setFormName(e.target.value)}
                disabled={disabled || mode === "edit"}
                className="h-[42px] rounded-[10px] border-border-strong bg-background px-3 font-mono text-[13.5px]"
                autoComplete="off"
              />
              {mode === "edit" && (
                <p className="mt-1.5 text-[11.5px] text-muted-foreground">Immutable after creation.</p>
              )}
            </div>

            {docType === "skill" && (
              <div>
                <Label
                  htmlFor="doc-description"
                  className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground"
                >
                  Description
                </Label>
                <Input
                  id="doc-description"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  disabled={disabled}
                  placeholder="What this skill covers, so the model can decide when it's relevant"
                  className="h-[42px] rounded-[10px] border-border-strong bg-background px-3 text-[13.5px]"
                />
              </div>
            )}

            {docType === "playbook" && (
              <div>
                <Label className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">Routines</Label>
                <div className="flex gap-4">
                  {ROUTINE_OPTIONS.map((opt) => (
                    <label
                      key={opt.value}
                      className={cn(
                        "flex items-center gap-2 text-[13px] text-foreground",
                        disabled && "opacity-50",
                      )}
                    >
                      <input
                        type="checkbox"
                        checked={routines.includes(opt.value)}
                        onChange={() => toggleRoutine(opt.value)}
                        disabled={disabled}
                        className="size-4 rounded border-border-strong accent-[var(--accent)]"
                      />
                      {opt.label}
                    </label>
                  ))}
                </div>
              </div>
            )}

            <div>
              <div className="mb-[7px] flex items-center justify-between gap-3">
                <Label htmlFor="doc-content" className="text-[12.5px] font-semibold text-muted-foreground">
                  Content
                </Label>
                {!disabled && (
                  <>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".md,text/markdown,text/plain"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <Upload size={13} strokeWidth={2} />
                      Upload .md
                    </Button>
                  </>
                )}
              </div>
              <textarea
                id="doc-content"
                value={content}
                onChange={(e) => setContent(e.target.value)}
                disabled={disabled}
                rows={14}
                className="w-full resize-y rounded-[10px] border border-border-strong bg-background px-3 py-2.5 font-mono text-[12.5px] text-foreground outline-none placeholder:text-muted-foreground disabled:opacity-60"
              />
              <div className="mt-1.5 flex items-center justify-between gap-3">
                <p className="text-[11px] text-[color:var(--text-faint)]">
                  Optional YAML frontmatter is stored as-is; only the body is injected into prompts.
                </p>
                <span
                  className={cn(
                    "shrink-0 font-mono text-[11px]",
                    overLimit ? "font-semibold text-accent" : "text-[color:var(--text-faint)]",
                  )}
                >
                  {contentBytes.toLocaleString()} / {MAX_CONTENT_BYTES.toLocaleString()} bytes
                </span>
              </div>
            </div>
          </div>
        )}

        <DialogFooter>
          <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
            {readOnly ? "Close" : "Cancel"}
          </Button>
          {!readOnly && !loading && !error && (
            <Button type="button" onClick={handleSave} disabled={submitting}>
              {submitting ? "Saving…" : mode === "create" ? `Create ${DOC_LABEL[docType]}` : "Save changes"}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
