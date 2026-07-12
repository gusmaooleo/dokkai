"use client";

/**
 * "New ingestion" dialog, faithful to the prototype's ingest dialog
 * (`dokkai-ui-prototype.html` ~L464-473): repository path input, a
 * "Generate LLM descriptions" toggle (default on) and a "Recreate
 * collection" toggle (default off).
 *
 * The recreate warning text is CORRECTED per decision 12o — the prototype's
 * text was factually wrong (it claimed the delete was scoped to "this
 * project"; recreate actually drops and rebuilds the ENTIRE Weaviate
 * collection, wiping every ingested project's chunks). Turning the toggle
 * on shows an inline warning with the corrected text; submitting with it on
 * requires a second, explicit confirmation step (AlertDialog) before the
 * request is sent.
 */

import { useState } from "react";
import { TriangleAlert } from "lucide-react";

import { ApiError, instancesApi } from "@/lib/api";
import { useToast } from "@/lib/toast";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
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
import { Switch } from "@/components/ui/switch";

const RECREATE_WARNING =
  "Recreating drops and rebuilds the vector index for ALL ingested projects, not just this one — " +
  "the collection is dropped and rebuilt from scratch. Every other ingested project will need to be " +
  "re-ingested before it's searchable again.";

export function NewIngestionDialog({
  open,
  onOpenChange,
  onSubmitted,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSubmitted: () => void;
}) {
  const { toast } = useToast();

  const [repoPath, setRepoPath] = useState("");
  const [describe, setDescribe] = useState(true);
  const [recreate, setRecreate] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);

  function reset() {
    setRepoPath("");
    setDescribe(true);
    setRecreate(false);
    setSubmitting(false);
    setConfirmOpen(false);
  }

  function handleOpenChange(next: boolean) {
    if (!next) reset();
    onOpenChange(next);
  }

  async function submit() {
    setSubmitting(true);
    try {
      await instancesApi.runPipeline({ repo_path: repoPath.trim(), recreate, describe });
      toast(`Ingestion started for ${repoPath.trim()}`);
      onSubmitted();
      handleOpenChange(false);
    } catch (err) {
      toast(err instanceof ApiError ? err.detail : "Failed to start ingestion", "error");
      setSubmitting(false);
    }
  }

  function handleStart() {
    if (!repoPath.trim()) {
      toast("Enter a repository path.", "error");
      return;
    }
    if (recreate) {
      setConfirmOpen(true);
      return;
    }
    submit();
  }

  return (
    <>
      <Dialog open={open} onOpenChange={handleOpenChange}>
        <DialogContent className="sm:max-w-[460px]">
          <DialogHeader>
            <DialogTitle className="font-display text-[17px] font-bold">New ingestion</DialogTitle>
            <DialogDescription className="text-[12.5px] text-muted-foreground">
              Run the pipeline on a repository to make it searchable.
            </DialogDescription>
          </DialogHeader>

          <div className="flex flex-col gap-[18px]">
            <div>
              <Label htmlFor="repo-path" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
                Repository path
              </Label>
              <Input
                id="repo-path"
                value={repoPath}
                onChange={(e) => setRepoPath(e.target.value)}
                placeholder="/Users/username/projects/…"
                className="h-[42px] rounded-[10px] border-border-strong bg-background px-3 font-mono text-[13.5px]"
              />
              <p className="mt-1.5 text-[11.5px] text-muted-foreground">Absolute path, on the server running dokkai.</p>
            </div>

            <div className="flex items-center gap-3">
              <Switch id="ingest-describe" checked={describe} onCheckedChange={setDescribe} />
              <label htmlFor="ingest-describe" className="flex cursor-pointer flex-col text-left">
                <span className="text-[13.5px] font-semibold text-foreground">Generate LLM descriptions</span>
                <span className="text-xs text-muted-foreground">
                  Tier-2 micro-descriptions (uses the describe model)
                </span>
              </label>
            </div>

            <div className="flex items-center gap-3">
              <Switch id="ingest-recreate" checked={recreate} onCheckedChange={setRecreate} />
              <label htmlFor="ingest-recreate" className="flex cursor-pointer flex-col text-left">
                <span className="text-[13.5px] font-semibold text-accent">Recreate collection</span>
                <span className="text-xs text-muted-foreground">Drops and rebuilds the vector index for ALL projects</span>
              </label>
            </div>

            {recreate && (
              <div className="flex gap-2.5 rounded-[10px] border border-[color:var(--accent-ring)] bg-[color:var(--accent-weak)] p-3">
                <TriangleAlert size={16} strokeWidth={2} className="mt-px shrink-0 text-accent" />
                <span className="text-[12.5px] leading-[1.5] text-foreground">{RECREATE_WARNING}</span>
              </div>
            )}
          </div>

          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => handleOpenChange(false)}>
              Cancel
            </Button>
            <Button type="button" onClick={handleStart} disabled={submitting}>
              {submitting ? "Starting…" : "Start ingestion"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete the index for ALL projects?</AlertDialogTitle>
            <AlertDialogDescription>{RECREATE_WARNING}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                setConfirmOpen(false);
                submit();
              }}
            >
              Yes, delete and rebuild
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
