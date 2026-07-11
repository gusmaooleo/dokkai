"use client";

/**
 * Library screen (decision 9e screen 3, step C5): the "Runs | Library" tabs
 * (`section-tabs.tsx`), then two sections — Playbooks and Skills — each a
 * header row (count + "New …" button, hidden for the viewer role, 12p) and a
 * list of `DocumentCard`s. Clicking a card opens `DocumentDialog` in edit
 * mode (fetching that document's full content); for the viewer role it opens
 * read-only instead (`readOnly={!canWrite}` — the dialog itself renders the
 * disabled/no-save presentation).
 */

import { useCallback, useEffect, useState } from "react";
import { LoaderCircle, Plus } from "lucide-react";

import { ApiError, routinesApi } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useToast } from "@/lib/toast";
import type { PlaybookSummary, SkillSummary } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { DocumentCard } from "./document-card";
import { DocumentDialog } from "./document-dialog";
import { RoutineSectionTabs } from "./section-tabs";

export function LibraryScreen() {
  const { user } = useAuth();
  const { toast } = useToast();
  const canWrite = user?.role === "admin" || user?.role === "user";

  const [playbooks, setPlaybooks] = useState<PlaybookSummary[]>([]);
  const [playbooksLoading, setPlaybooksLoading] = useState(true);
  const [playbooksError, setPlaybooksError] = useState<string | null>(null);

  const [skills, setSkills] = useState<SkillSummary[]>([]);
  const [skillsLoading, setSkillsLoading] = useState(true);
  const [skillsError, setSkillsError] = useState<string | null>(null);

  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogDocType, setDialogDocType] = useState<"playbook" | "skill">("playbook");
  const [dialogName, setDialogName] = useState<string | null>(null);

  const loadPlaybooks = useCallback(async () => {
    setPlaybooksLoading(true);
    setPlaybooksError(null);
    try {
      setPlaybooks(await routinesApi.listPlaybooks());
    } catch (err) {
      setPlaybooksError(err instanceof ApiError ? err.detail : "Failed to load playbooks");
    } finally {
      setPlaybooksLoading(false);
    }
  }, []);

  const loadSkills = useCallback(async () => {
    setSkillsLoading(true);
    setSkillsError(null);
    try {
      setSkills(await routinesApi.listSkills());
    } catch (err) {
      setSkillsError(err instanceof ApiError ? err.detail : "Failed to load skills");
    } finally {
      setSkillsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPlaybooks();
    loadSkills();
  }, [loadPlaybooks, loadSkills]);

  function openCreate(docType: "playbook" | "skill") {
    setDialogDocType(docType);
    setDialogName(null);
    setDialogOpen(true);
  }

  function openDocument(docType: "playbook" | "skill", name: string) {
    setDialogDocType(docType);
    setDialogName(name);
    setDialogOpen(true);
  }

  function handleSaved() {
    if (dialogDocType === "playbook") loadPlaybooks();
    else loadSkills();
  }

  async function handleDeletePlaybook(name: string) {
    try {
      await routinesApi.deletePlaybook(name);
      toast(`Playbook '${name}' deleted`);
      loadPlaybooks();
    } catch (err) {
      toast(err instanceof ApiError ? err.detail : "Failed to delete playbook", "error");
    }
  }

  async function handleDeleteSkill(name: string) {
    try {
      await routinesApi.deleteSkill(name);
      toast(`Skill '${name}' deleted`);
      loadSkills();
    } catch (err) {
      toast(err instanceof ApiError ? err.detail : "Failed to delete skill", "error");
    }
  }

  return (
    <div className="mx-auto max-w-[860px] px-3.5 py-[18px] sm:px-6.5 sm:py-[30px]">
      <RoutineSectionTabs active="library" />

      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="flex items-baseline gap-[9px]">
          <span className="font-display text-[17px] font-bold text-foreground">Playbooks</span>
          <span className="font-mono text-xs text-[color:var(--text-faint)]">{playbooks.length} total</span>
        </div>
        {canWrite && (
          <Button type="button" onClick={() => openCreate("playbook")}>
            <Plus size={16} strokeWidth={2} />
            New playbook
          </Button>
        )}
      </div>

      {playbooksLoading && (
        <div className="flex justify-center py-8">
          <LoaderCircle size={20} strokeWidth={2.4} className="animate-[dk-spin_0.7s_linear_infinite] text-muted-foreground" />
        </div>
      )}
      {!playbooksLoading && playbooksError && (
        <p className="py-6 text-center text-[13.5px] text-muted-foreground">{playbooksError}</p>
      )}
      {!playbooksLoading && !playbooksError && playbooks.length === 0 && (
        <p className="py-6 text-center text-[13.5px] text-[color:var(--text-faint)]">
          No playbooks yet — they inject house rules into runs.
        </p>
      )}
      {!playbooksLoading && !playbooksError && playbooks.length > 0 && (
        <div className="mb-10 flex flex-col gap-3">
          {playbooks.map((p) => (
            <DocumentCard
              key={p.id}
              entry={{ docType: "playbook", doc: p }}
              canDelete={canWrite}
              onOpen={() => openDocument("playbook", p.name)}
              onDelete={() => handleDeletePlaybook(p.name)}
            />
          ))}
        </div>
      )}

      <div className="mb-4 mt-10 flex items-center justify-between gap-3">
        <div className="flex items-baseline gap-[9px]">
          <span className="font-display text-[17px] font-bold text-foreground">Skills</span>
          <span className="font-mono text-xs text-[color:var(--text-faint)]">{skills.length} total</span>
        </div>
        {canWrite && (
          <Button type="button" onClick={() => openCreate("skill")}>
            <Plus size={16} strokeWidth={2} />
            New skill
          </Button>
        )}
      </div>

      {skillsLoading && (
        <div className="flex justify-center py-8">
          <LoaderCircle size={20} strokeWidth={2.4} className="animate-[dk-spin_0.7s_linear_infinite] text-muted-foreground" />
        </div>
      )}
      {!skillsLoading && skillsError && (
        <p className="py-6 text-center text-[13.5px] text-muted-foreground">{skillsError}</p>
      )}
      {!skillsLoading && !skillsError && skills.length === 0 && (
        <p className="py-6 text-center text-[13.5px] text-[color:var(--text-faint)]">
          No skills yet — the model loads relevant skills during runs.
        </p>
      )}
      {!skillsLoading && !skillsError && skills.length > 0 && (
        <div className="flex flex-col gap-3">
          {skills.map((s) => (
            <DocumentCard
              key={s.id}
              entry={{ docType: "skill", doc: s }}
              canDelete={canWrite}
              onOpen={() => openDocument("skill", s.name)}
              onDelete={() => handleDeleteSkill(s.name)}
            />
          ))}
        </div>
      )}

      <DocumentDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        docType={dialogDocType}
        name={dialogName}
        readOnly={!canWrite}
        onSaved={handleSaved}
      />
    </div>
  );
}
