"use client";

/**
 * Active-project context for the authenticated shell: the project switcher
 * (sidebar) reads/writes it, screens (chat, graph, history…) read `active`
 * to scope their requests. Persists the selection in localStorage under
 * `dokkai.project` (decision 12e's pattern) and re-selects it once `GET
 * /graph` resolves, falling back to the first ingested project.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

import { graphApi } from "./api";
import type { ProjectGraphDTO } from "./types";

const STORAGE_KEY = "dokkai.project";

interface ProjectContextValue {
  projects: ProjectGraphDTO[];
  active: ProjectGraphDTO | null;
  setActive: (project: string) => void;
  /** True until the first `GET /graph` resolves. */
  loading: boolean;
  refresh: () => Promise<void>;
}

const ProjectContext = createContext<ProjectContextValue | undefined>(undefined);

export function ProjectProvider({ children }: { children: ReactNode }) {
  const [projects, setProjects] = useState<ProjectGraphDTO[]>([]);
  const [activeName, setActiveName] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const list = await graphApi.list();
      setProjects(list);
      setActiveName((prev) => {
        if (prev && list.some((p) => p.project === prev)) return prev;
        const stored = typeof window !== "undefined" ? window.localStorage.getItem(STORAGE_KEY) : null;
        if (stored && list.some((p) => p.project === stored)) return stored;
        return list[0]?.project ?? null;
      });
    } catch {
      setProjects([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const setActive = useCallback((project: string) => {
    setActiveName(project);
    if (typeof window !== "undefined") window.localStorage.setItem(STORAGE_KEY, project);
  }, []);

  const active = projects.find((p) => p.project === activeName) ?? null;

  return (
    <ProjectContext.Provider value={{ projects, active, setActive, loading, refresh }}>
      {children}
    </ProjectContext.Provider>
  );
}

export function useProject(): ProjectContextValue {
  const ctx = useContext(ProjectContext);
  if (!ctx) throw new Error("useProject must be used within a ProjectProvider");
  return ctx;
}
