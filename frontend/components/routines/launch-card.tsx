"use client";

/**
 * Routine launch card — kind toggle (Review | Bug hunt, faithful to
 * `AudienceSwitch`'s segmented idiom), the per-kind inputs (branch select
 * for review, scope for bug hunt), a shared playbook multiselect and an
 * optional model override, then the Launch button.
 *
 * No hardcoded default model anywhere (decision 9a-3-clar) — an empty
 * override means "use the active LLM config"; if nothing is configured the
 * backend 400s synchronously at launch and that verbatim detail surfaces
 * via toast.
 */

import { useCallback, useEffect, useState } from "react";
import { Info } from "lucide-react";

import { ApiError, routinesApi } from "@/lib/api";
import { useToast } from "@/lib/toast";
import type { BranchInfo, LaunchRoutineRequest, PlaybookSummary, RoutineKind } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { KIND_LABEL } from "./format";

const MAX_PLAYBOOKS = 4;

const KIND_OPTIONS: { value: RoutineKind; label: string }[] = [
  { value: "review", label: "Review" },
  { value: "bughunt", label: "Bug hunt" },
];

const chipClasses =
  "rounded-[4px] border border-current/30 px-1 py-px font-mono text-[9px] font-semibold tracking-[0.06em] uppercase";

export function LaunchCard({ project, onLaunched }: { project: string; onLaunched: (runId: string) => void }) {
  const { toast } = useToast();

  const [kind, setKind] = useState<RoutineKind>("review");

  // Review-only state
  const [branches, setBranches] = useState<BranchInfo[]>([]);
  const [defaultBase, setDefaultBase] = useState<string | null>(null);
  const [branchesLoading, setBranchesLoading] = useState(true);
  const [branchesError, setBranchesError] = useState<string | null>(null);
  const [targetRef, setTargetRef] = useState("");
  const [showBaseOverride, setShowBaseOverride] = useState(false);
  const [baseOverride, setBaseOverride] = useState("");

  // Bug hunt-only state
  const [scope, setScope] = useState("");
  const [pathPrefix, setPathPrefix] = useState("");

  // Shared state
  const [playbooks, setPlaybooks] = useState<PlaybookSummary[]>([]);
  const [playbooksLoading, setPlaybooksLoading] = useState(true);
  const [selectedPlaybooks, setSelectedPlaybooks] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const loadBranches = useCallback(async () => {
    setBranchesLoading(true);
    setBranchesError(null);
    setTargetRef("");
    // A base override typed for the previous project is meaningless here.
    setBaseOverride("");
    setShowBaseOverride(false);
    try {
      const res = await routinesApi.branches(project);
      setBranches(res.branches);
      setDefaultBase(res.default_base);
    } catch (err) {
      setBranchesError(err instanceof ApiError ? err.detail : "Failed to load branches");
    } finally {
      setBranchesLoading(false);
    }
  }, [project]);

  useEffect(() => {
    loadBranches();
  }, [loadBranches]);

  useEffect(() => {
    let cancelled = false;
    setPlaybooksLoading(true);
    routinesApi
      .listPlaybooks()
      .then((res) => {
        if (!cancelled) setPlaybooks(res);
      })
      .catch(() => {
        if (!cancelled) setPlaybooks([]);
      })
      .finally(() => {
        if (!cancelled) setPlaybooksLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const applicablePlaybooks = playbooks.filter((p) => p.routines.includes(kind));

  // Drop selections that no longer apply after a kind switch.
  useEffect(() => {
    setSelectedPlaybooks((prev) => prev.filter((name) => applicablePlaybooks.some((p) => p.name === name)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind, playbooks]);

  function togglePlaybook(name: string) {
    setSelectedPlaybooks((prev) => {
      if (prev.includes(name)) return prev.filter((n) => n !== name);
      if (prev.length >= MAX_PLAYBOOKS) return prev;
      return [...prev, name];
    });
  }

  async function handleLaunch() {
    if (kind === "review" && !targetRef) {
      toast("Select a target branch.", "error");
      return;
    }
    if (kind === "bughunt" && !scope.trim()) {
      toast("Describe what to investigate.", "error");
      return;
    }

    const base: LaunchRoutineRequest =
      kind === "review"
        ? {
            kind,
            project,
            target_ref: targetRef,
            base_ref: baseOverride.trim() || undefined,
            model: model.trim() || undefined,
            playbooks: selectedPlaybooks.length > 0 ? selectedPlaybooks : undefined,
          }
        : {
            kind,
            project,
            scope: scope.trim(),
            path_prefix: pathPrefix.trim() || undefined,
            model: model.trim() || undefined,
            playbooks: selectedPlaybooks.length > 0 ? selectedPlaybooks : undefined,
          };

    setSubmitting(true);
    try {
      const res = await routinesApi.launch(base);
      toast(`${KIND_LABEL[kind]} launched for ${project}`);
      onLaunched(res.run_id);
    } catch (err) {
      toast(err instanceof ApiError ? err.detail : `Failed to launch ${KIND_LABEL[kind].toLowerCase()}`, "error");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="rounded-[14px] border border-border bg-surface p-5 shadow-[var(--shadow-sm)]">
      <div className="mb-4 inline-flex gap-0.5 rounded-[11px] border border-border bg-surface-2 p-[3px]">
        {KIND_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            type="button"
            onClick={() => setKind(opt.value)}
            className={cn(
              "flex h-[30px] items-center gap-1.5 rounded-lg px-[13px] text-[12.5px] font-semibold",
              kind === opt.value ? "bg-surface text-foreground shadow-[var(--shadow-sm)]" : "text-muted-foreground",
            )}
          >
            {opt.label}
            {opt.value === "bughunt" && (
              <span className={cn(chipClasses, "text-[color:var(--text-faint)]")}>demo</span>
            )}
          </button>
        ))}
      </div>

      {kind === "bughunt" && (
        <div className="mb-[18px] flex gap-2.5 rounded-[10px] border border-border bg-surface-2 p-3">
          <Info size={16} strokeWidth={2} className="mt-px shrink-0 text-[color:var(--text-faint)]" />
          <span className="text-[12.5px] leading-[1.5] text-muted-foreground">
            Experimental — results improve with more capable models (we recommend qwen2.5-coder:14b or larger).
          </span>
        </div>
      )}

      {kind === "review" ? (
        <div className="mb-[18px] flex flex-col gap-3">
          <div>
            <Label htmlFor="routine-target" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
              Target branch
            </Label>
            {branchesLoading ? (
              <p className="text-[13px] text-muted-foreground">Loading branches…</p>
            ) : branchesError ? (
              <div className="flex items-center gap-2.5">
                <p className="text-[13px] text-muted-foreground">{branchesError}</p>
                <Button type="button" variant="outline" size="sm" onClick={loadBranches}>
                  Retry
                </Button>
              </div>
            ) : (
              <Select value={targetRef} onValueChange={setTargetRef}>
                <SelectTrigger
                  id="routine-target"
                  className="h-[42px] w-full rounded-[10px] border-border-strong bg-background px-3 font-mono text-[13.5px]"
                >
                  <SelectValue placeholder="Select a branch" />
                </SelectTrigger>
                <SelectContent>
                  {branches.map((b) => (
                    <SelectItem key={b.name} value={b.name}>
                      {b.name}
                      {b.is_current ? " (current)" : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            {!branchesLoading && !branchesError && defaultBase && (
              <p className="mt-1.5 text-[11.5px] text-muted-foreground">
                vs {baseOverride.trim() || defaultBase}
                {!showBaseOverride && (
                  <button
                    type="button"
                    onClick={() => setShowBaseOverride(true)}
                    className="ml-2 text-accent hover:underline"
                  >
                    override base branch
                  </button>
                )}
              </p>
            )}
            {showBaseOverride && (
              <Input
                value={baseOverride}
                onChange={(e) => setBaseOverride(e.target.value)}
                placeholder={defaultBase ?? "e.g. develop"}
                className="mt-2 h-[38px] rounded-[10px] border-border-strong bg-background px-3 font-mono text-[13px]"
              />
            )}
          </div>
        </div>
      ) : (
        <div className="mb-[18px] flex flex-col gap-3">
          <div>
            <Label htmlFor="routine-scope" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
              Scope
            </Label>
            <textarea
              id="routine-scope"
              value={scope}
              onChange={(e) => setScope(e.target.value)}
              placeholder="e.g. look for unhandled errors and resource leaks in the ingestion pipeline"
              rows={3}
              className="w-full resize-y rounded-[10px] border border-border-strong bg-background px-3 py-2.5 text-[13.5px] text-foreground outline-none placeholder:text-muted-foreground"
            />
          </div>
          <div>
            <Label htmlFor="routine-path-prefix" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
              Path prefix (optional)
            </Label>
            <Input
              id="routine-path-prefix"
              value={pathPrefix}
              onChange={(e) => setPathPrefix(e.target.value)}
              placeholder="e.g. src/services"
              className="h-[42px] rounded-[10px] border-border-strong bg-background px-3 font-mono text-[13.5px]"
            />
            <p className="mt-1.5 text-[11.5px] text-muted-foreground">
              Restricts reported findings to files under this repo-relative prefix.
            </p>
          </div>
        </div>
      )}

      <div className="mb-[18px]">
        <Label className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
          Playbooks (optional, max {MAX_PLAYBOOKS})
        </Label>
        {playbooksLoading ? (
          <p className="text-[13px] text-muted-foreground">Loading playbooks…</p>
        ) : applicablePlaybooks.length === 0 ? (
          <p className="text-[12.5px] text-[color:var(--text-faint)]">No playbooks apply to this routine.</p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {applicablePlaybooks.map((p) => {
              const order = selectedPlaybooks.indexOf(p.name);
              const selected = order !== -1;
              return (
                <button
                  key={p.name}
                  type="button"
                  onClick={() => togglePlaybook(p.name)}
                  className={cn(
                    "flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-[12.5px] font-medium",
                    selected
                      ? "border-[color:var(--accent-ring)] bg-[color:var(--accent-weak)] text-accent"
                      : "border-border bg-surface-2 text-muted-foreground",
                  )}
                >
                  {selected && (
                    <span className="grid size-4 place-items-center rounded-full bg-accent font-mono text-[9.5px] font-bold text-white">
                      {order + 1}
                    </span>
                  )}
                  {p.name}
                </button>
              );
            })}
          </div>
        )}
        {selectedPlaybooks.length >= MAX_PLAYBOOKS && (
          <p className="mt-1.5 text-[11.5px] text-[color:var(--text-faint)]">
            Maximum {MAX_PLAYBOOKS} playbooks selected — deselect one to add another.
          </p>
        )}
      </div>

      <div className="mb-[18px]">
        <Label htmlFor="routine-model" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
          Model override (optional)
        </Label>
        <Input
          id="routine-model"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          placeholder="e.g. qwen2.5-coder:14b"
          className="h-[42px] rounded-[10px] border-border-strong bg-background px-3 font-mono text-[13.5px]"
        />
        <p className="mt-1.5 text-[11.5px] text-muted-foreground">
          Defaults to the active LLM config; for routines we recommend qwen2.5-coder:14b or larger.
        </p>
      </div>

      <div className="flex justify-end border-t border-border pt-[14px]">
        <Button type="button" onClick={handleLaunch} disabled={submitting}>
          {submitting ? "Launching…" : `Launch ${KIND_LABEL[kind].toLowerCase()}`}
        </Button>
      </div>
    </div>
  );
}
