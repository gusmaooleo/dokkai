"use client";

/**
 * Graph screen orchestrator: loads `GET /graph/{project}` (entities) or
 * `GET /graph/{project}/files` (files, decision 12q), normalizes either
 * into the shared `NormalizedNode`/`NormalizedEdge` shape, and wires the
 * canvas, toolbar, legend and detail panel together — selection, search,
 * relation navigation, the `?focus=<qualified_name>` deep link (used by
 * chat's source cards) and "Ask about this in chat".
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { LoaderCircle, TriangleAlert } from "lucide-react";

import { ApiError, graphApi } from "@/lib/api";
import { useProject } from "@/lib/project";
import { useTheme } from "@/lib/theme";
import { useToast } from "@/lib/toast";
import type { FileEdge, FileNode, FilesGraph, GraphEdge, GraphNode, GraphPayload } from "@/lib/types";
import { DetailPanel, type RelationRow } from "./detail-panel";
import { GraphCanvas, type GraphCanvasHandle } from "./graph-canvas";
import { GraphLegend } from "./legend";
import { dominantType, relationLabel } from "./relations";
import { GraphToolbar } from "./toolbar";
import type { GraphMode, NormalizedEdge, NormalizedNode } from "./types";

function normalizeEntityNodes(nodes: GraphNode[]): NormalizedNode[] {
  return nodes.map((n) => ({
    id: String(n.id),
    kind: n.kind,
    name: n.name ?? n.qualified_name ?? `#${n.id}`,
    qualified_name: n.qualified_name,
    path: n.path,
    absolute_path: n.absolute_path,
    start_line: n.start_line,
    end_line: n.end_line,
  }));
}

function normalizeEntityEdges(edges: GraphEdge[]): NormalizedEdge[] {
  return edges.map((e, i) => ({
    id: `e${i}:${e.source}->${e.target}:${e.type}`,
    source: String(e.source),
    target: String(e.target),
    type: e.type,
  }));
}

function normalizeFileNodes(files: FileNode[]): NormalizedNode[] {
  return files.map((f) => ({
    id: f.path,
    kind: "File",
    name: f.name,
    qualified_name: f.path,
    path: f.path,
    absolute_path: f.absolute_path,
    start_line: null,
    end_line: null,
  }));
}

function normalizeFileEdges(edges: FileEdge[]): NormalizedEdge[] {
  return edges.map((e, i) => ({
    id: `f${i}:${e.source}->${e.target}`,
    source: e.source,
    target: e.target,
    type: dominantType(e.types),
    weight: e.weight,
    types: e.types,
  }));
}

export function GraphScreen() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { active, loading: projectLoading } = useProject();
  const { isDark } = useTheme();
  const { toast } = useToast();

  const [mode, setMode] = useState<GraphMode>("entities");
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "empty" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState("");
  const [entityData, setEntityData] = useState<GraphPayload | null>(null);
  const [fileData, setFileData] = useState<FilesGraph | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [layoutComputing, setLayoutComputing] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [description, setDescription] = useState<string | null>(null);
  const [chunkText, setChunkText] = useState<string | null>(null);

  const canvasRef = useRef<GraphCanvasHandle>(null);
  const appliedFocusRef = useRef<string | null>(null);
  const focusParam = searchParams.get("focus");

  // Load the active project's graph whenever the project or entities/files
  // mode changes.
  useEffect(() => {
    if (!active) {
      setStatus("idle");
      return;
    }
    let cancelled = false;
    setStatus("loading");
    setSelectedId(null);
    const project = active.project;

    async function run() {
      try {
        if (mode === "entities") {
          const data = await graphApi.get(project);
          if (cancelled) return;
          setEntityData(data);
        } else {
          const data = await graphApi.files(project);
          if (cancelled) return;
          setFileData(data);
        }
        if (!cancelled) setStatus("ready");
      } catch (err) {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 404) {
          setErrorMessage(err.detail);
          setStatus("empty");
        } else {
          const detail = err instanceof Error ? err.message : "Failed to load the graph";
          setErrorMessage(detail);
          setStatus("error");
          toast(detail, "error");
        }
      }
    }
    run();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active?.project, mode]);

  const nodes = useMemo<NormalizedNode[]>(() => {
    if (mode === "entities") return entityData ? normalizeEntityNodes(entityData.nodes) : [];
    return fileData ? normalizeFileNodes(fileData.files) : [];
  }, [mode, entityData, fileData]);

  const edges = useMemo<NormalizedEdge[]>(() => {
    if (mode === "entities") return entityData ? normalizeEntityEdges(entityData.edges) : [];
    return fileData ? normalizeFileEdges(fileData.edges) : [];
  }, [mode, entityData, fileData]);

  const nodesById = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);
  const kinds = useMemo(() => Array.from(new Set(nodes.map((n) => n.kind))).sort(), [nodes]);

  const edgesBySource = useMemo(() => {
    const m = new Map<string, NormalizedEdge[]>();
    for (const e of edges) {
      const arr = m.get(e.source);
      if (arr) arr.push(e);
      else m.set(e.source, [e]);
    }
    return m;
  }, [edges]);
  const edgesByTarget = useMemo(() => {
    const m = new Map<string, NormalizedEdge[]>();
    for (const e of edges) {
      const arr = m.get(e.target);
      if (arr) arr.push(e);
      else m.set(e.target, [e]);
    }
    return m;
  }, [edges]);

  const selectedNode = selectedId ? (nodesById.get(selectedId) ?? null) : null;

  const relations = useMemo<RelationRow[]>(() => {
    if (!selectedId) return [];
    const rows: RelationRow[] = [];
    for (const e of edgesBySource.get(selectedId) ?? []) {
      const other = nodesById.get(e.target);
      if (!other) continue;
      const label = e.weight ? `${relationLabel(e.type, "out")} (${e.weight})` : relationLabel(e.type, "out");
      rows.push({ id: other.id, name: other.name, kind: other.kind, label });
    }
    for (const e of edgesByTarget.get(selectedId) ?? []) {
      const other = nodesById.get(e.source);
      if (!other) continue;
      const label = e.weight ? `${relationLabel(e.type, "in")} (${e.weight})` : relationLabel(e.type, "in");
      rows.push({ id: other.id, name: other.name, kind: other.kind, label });
    }
    return rows;
  }, [selectedId, edgesBySource, edgesByTarget, nodesById]);

  // Entity detail (description + chunk) — entities mode only; degrades
  // gracefully (no fetch, sections hidden) for files and for nodes with no
  // qualified name.
  useEffect(() => {
    if (mode !== "entities" || !selectedId || !active) {
      setDescription(null);
      setChunkText(null);
      setDetailLoading(false);
      return;
    }
    const node = nodesById.get(selectedId);
    if (!node?.qualified_name) {
      setDescription(null);
      setChunkText(null);
      setDetailLoading(false);
      return;
    }
    let cancelled = false;
    setDetailLoading(true);
    setDescription(null);
    setChunkText(null);
    graphApi
      .entity(active.project, node.qualified_name)
      .then((detail) => {
        if (cancelled) return;
        setDescription(detail.description);
        setChunkText(detail.chunk_text);
      })
      .catch(() => {
        if (!cancelled) {
          setDescription(null);
          setChunkText(null);
        }
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [mode, selectedId, active, nodesById]);

  const selectAndFocus = useCallback((id: string) => {
    setSelectedId(id);
    canvasRef.current?.focusNode(id);
  }, []);

  const handleCanvasReady = useCallback(() => {
    if (!focusParam || mode !== "entities") return;
    if (appliedFocusRef.current === focusParam) return;
    const match = nodes.find((n) => n.qualified_name === focusParam);
    if (match) {
      appliedFocusRef.current = focusParam;
      selectAndFocus(match.id);
    }
  }, [focusParam, mode, nodes, selectAndFocus]);

  function handleAskInChat() {
    if (!selectedNode) return;
    const subject = selectedNode.qualified_name ?? selectedNode.name;
    const question = `What does \`${subject}\` do, and what does it depend on?`;
    router.push(`/chat?prefill=${encodeURIComponent(question)}`);
  }

  const noProject = !projectLoading && !active;
  const cacheKey = active ? `${active.project}:${mode}` : "";
  const truncated = mode === "entities" && !!entityData?.stats.truncated;

  return (
    <div className="flex h-full min-h-0">
      <div className="relative min-w-0 flex-1">
        {noProject && (
          <div className="grid h-full place-items-center p-6 text-center">
            <div className="max-w-[360px]">
              <h2 className="font-display text-[17px] font-bold text-foreground">No project ingested yet</h2>
              <p className="mt-2 text-[13.5px] text-muted-foreground">
                Ingest a repository first — the graph is generated as part of the pipeline.
              </p>
              <Link
                href="/instances"
                className="mt-4 inline-flex h-9 items-center rounded-[10px] border border-border-strong bg-surface px-3.5 text-[13.5px] font-semibold text-foreground hover:border-accent hover:text-accent"
              >
                Go to Instances
              </Link>
            </div>
          </div>
        )}

        {active && status === "loading" && (
          <div className="grid h-full place-items-center">
            <LoaderCircle
              size={26}
              strokeWidth={2.4}
              className="animate-[dk-spin_0.7s_linear_infinite] text-muted-foreground"
            />
          </div>
        )}

        {active && (status === "empty" || status === "error") && (
          <div className="grid h-full place-items-center p-6 text-center">
            <div className="max-w-[420px]">
              <TriangleAlert size={22} strokeWidth={1.8} className="mx-auto text-[color:var(--text-faint)]" />
              <p className="mt-3 text-[13.5px] text-muted-foreground">{errorMessage}</p>
              <Link
                href="/instances"
                className="mt-4 inline-flex h-9 items-center rounded-[10px] border border-border-strong bg-surface px-3.5 text-[13.5px] font-semibold text-foreground hover:border-accent hover:text-accent"
              >
                Go to Instances
              </Link>
            </div>
          </div>
        )}

        {active && status === "ready" && (
          <>
            <GraphCanvas
              key={cacheKey}
              ref={canvasRef}
              cacheKey={cacheKey}
              // `active.generated_at` (from the `/graph` project listing) is
              // used for both modes rather than `entityData.generated_at`
              // (nullable, entities-only) since `GET /graph/{project}/files`
              // doesn't return one — both views come from the same ingested
              // graph, so the project's fingerprint is valid for either.
              generatedAt={active.generated_at}
              nodes={nodes}
              edges={edges}
              selectedId={selectedId}
              onSelectNode={setSelectedId}
              isDark={isDark}
              onLayoutStateChange={setLayoutComputing}
              onReady={handleCanvasReady}
            />

            {layoutComputing && (
              <div className="absolute inset-0 z-[7] grid place-items-center bg-background/70">
                <div className="flex items-center gap-2.5 rounded-[11px] border border-border bg-surface px-4 py-2.5 shadow-[var(--shadow-sm)]">
                  <LoaderCircle
                    size={16}
                    strokeWidth={2.4}
                    className="animate-[dk-spin_0.7s_linear_infinite] text-muted-foreground"
                  />
                  <span className="text-[13px] text-muted-foreground">Computing layout…</span>
                </div>
              </div>
            )}

            {truncated && (
              <div className="absolute inset-x-0 top-0 z-[6] bg-[color:var(--accent-weak)] px-4 py-1.5 text-center font-mono text-[11.5px] text-accent">
                Graph truncated to {entityData?.stats.returned_nodes} of {entityData?.stats.total_nodes} nodes
              </div>
            )}

            <GraphToolbar
              mode={mode}
              onModeChange={setMode}
              nodes={nodes}
              onSelectResult={selectAndFocus}
              onZoomIn={() => canvasRef.current?.zoomIn()}
              onZoomOut={() => canvasRef.current?.zoomOut()}
              onFit={() => canvasRef.current?.fit()}
              nodeCount={nodes.length}
              edgeCount={edges.length}
            />
            <GraphLegend kinds={kinds} />
            <div className="pointer-events-none absolute right-4 bottom-4 z-[5] font-mono text-[10.5px] text-[color:var(--text-faint)]">
              drag node · scroll zoom · click to inspect
            </div>
          </>
        )}
      </div>

      {selectedNode && (
        <div className="hidden lg:flex lg:h-full">
          <DetailPanel
            mode={mode}
            node={selectedNode}
            relations={relations}
            loadingDetail={detailLoading}
            description={description}
            chunkText={chunkText}
            onClose={() => setSelectedId(null)}
            onSelectRelation={selectAndFocus}
            onAskInChat={handleAskInChat}
          />
        </div>
      )}
    </div>
  );
}
