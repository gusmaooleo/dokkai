"use client";

/**
 * Sigma.js renderer for the dependency graph — owns the graphology `Graph`
 * instance, the forceatlas2 layout run, the Sigma renderer, and the
 * selection dim/highlight (nodeReducer/edgeReducer). Presentational data
 * (`nodes`/`edges`) and interaction callbacks are props; layout and camera
 * control are exposed imperatively via `ref` for the toolbar's zoom
 * buttons and the search/relations/deep-link "focus this node" actions.
 *
 * Layout: forceatlas2 runs in a Web Worker (`graphology-layout-forceatlas2/
 * worker`) for a fixed ~3s budget, then the final positions are cached in
 * a module-level `Map` keyed by `cacheKey` (`<project>:<mode>`) so revisiting
 * a project/mode within the session skips the layout entirely. Initial
 * positions are a deterministic circular spread by node index — stable
 * across runs, no `Math.random`.
 */

import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
} from "react";
import Graph from "graphology";
import Sigma from "sigma";
import forceAtlas2 from "graphology-layout-forceatlas2";
import FA2Layout from "graphology-layout-forceatlas2/worker";
import type { Attributes } from "graphology-types";

import { kindColor } from "@/components/chat/kinds";
import type { NormalizedEdge, NormalizedNode } from "./types";

/** Node/edge attributes as stored on the graphology graph — distinct from
 *  Sigma's `NodeDisplayData`/`EdgeDisplayData`, which also carry
 *  render-only fields (`hidden`, `forceLabel`, `zIndex`…) computed by the
 *  reducers below. */
interface NodeAttrs extends Attributes {
  label: string | null;
  kind: string;
  color: string;
  size: number;
  x: number;
  y: number;
}
interface EdgeAttrs extends Attributes {
  // Named `relType`, not `type` — Sigma reserves the `type` node/edge
  // attribute to pick a *rendering program* (e.g. "line", "arrow"); reusing
  // it for our own relationship kind (CALLS/DEFINES/…) makes Sigma try to
  // look up a program named "DEFINES" and throw.
  relType: string;
  size: number;
}

const LAYOUT_BUDGET_MS = 3000;
const MIN_NODE_SIZE = 2;
const MAX_NODE_SIZE = 12;

const EDGE_COLOR = { light: "#e7e7e2", dark: "#34333c" };
const EDGE_COLOR_HIGHLIGHT = { light: "#9c9ca2", dark: "#6e6d78" };
const NODE_COLOR_DIM = { light: "#d8d8d2", dark: "#43424c" };
const LABEL_COLOR = { light: "#1B1B1F", dark: "#F9F9F9" };

/** Session-lifetime cache: `<project>:<mode>` -> node id -> final layout position. */
const positionCache = new Map<string, Record<string, { x: number; y: number }>>();

export interface GraphCanvasHandle {
  zoomIn: () => void;
  zoomOut: () => void;
  fit: () => void;
  focusNode: (id: string) => void;
}

interface GraphCanvasProps {
  cacheKey: string;
  nodes: NormalizedNode[];
  edges: NormalizedEdge[];
  selectedId: string | null;
  onSelectNode: (id: string | null) => void;
  isDark: boolean;
  onLayoutStateChange: (computing: boolean) => void;
  onReady?: () => void;
}

export const GraphCanvas = forwardRef<GraphCanvasHandle, GraphCanvasProps>(function GraphCanvas(
  { cacheKey, nodes, edges, selectedId, onSelectNode, isDark, onLayoutStateChange, onReady },
  ref,
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<Graph<NodeAttrs, EdgeAttrs> | null>(null);
  const sigmaRef = useRef<Sigma<NodeAttrs, EdgeAttrs, Attributes> | null>(null);
  const supervisorRef = useRef<FA2Layout | null>(null);
  // Sigma only re-measures its canvases on `window` resize, not container
  // resize — without this, opening/closing the detail panel (a flex-layout
  // width change, not a window resize) leaves stale, oversized canvases
  // that visually look fine (transparent) but still intercept clicks over
  // the panel.
  const resizeObserverRef = useRef<ResizeObserver | null>(null);

  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  const onSelectNodeRef = useRef(onSelectNode);
  const onLayoutStateChangeRef = useRef(onLayoutStateChange);
  const onReadyRef = useRef(onReady);
  const isDarkRef = useRef(isDark);
  const highlightSetRef = useRef<Set<string> | null>(null);

  // Keep the latest props on refs so the effects below (which intentionally
  // don't depend on every prop — see their own comments) always read
  // current values without re-triggering a graph/layout rebuild.
  useEffect(() => {
    nodesRef.current = nodes;
    edgesRef.current = edges;
    onSelectNodeRef.current = onSelectNode;
    onLayoutStateChangeRef.current = onLayoutStateChange;
    onReadyRef.current = onReady;
  });

  useImperativeHandle(
    ref,
    () => ({
      zoomIn: () => {
        sigmaRef.current?.getCamera().animatedZoom({ duration: 200 });
      },
      zoomOut: () => {
        sigmaRef.current?.getCamera().animatedUnzoom({ duration: 200 });
      },
      fit: () => {
        sigmaRef.current?.getCamera().animatedReset({ duration: 300 });
      },
      focusNode: (id: string) => {
        const sigma = sigmaRef.current;
        if (!sigma) return;
        const data = sigma.getNodeDisplayData(id);
        if (!data) return;
        sigma.getCamera().animate({ x: data.x, y: data.y, ratio: 0.12 }, { duration: 500 });
      },
    }),
    [],
  );

  // Rebuild the graph, run/apply layout, and (re)mount the Sigma renderer
  // whenever the dataset changes (`cacheKey` — project or entities/files
  // mode). Reads `nodesRef`/`edgesRef` rather than depending on `nodes`/
  // `edges` directly so a parent re-render with a fresh-but-equal array
  // doesn't trigger a pointless rebuild.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    onLayoutStateChangeRef.current(true);

    const graph = new Graph<NodeAttrs, EdgeAttrs>({ type: "directed", multi: true, allowSelfLoops: true });
    const nodeList = nodesRef.current;
    const edgeList = edgesRef.current;

    const degree = new Map<string, number>();
    for (const e of edgeList) {
      degree.set(e.source, (degree.get(e.source) ?? 0) + 1);
      degree.set(e.target, (degree.get(e.target) ?? 0) + 1);
    }
    const degrees = nodeList.map((n) => degree.get(n.id) ?? 0);
    const minDeg = degrees.length ? Math.min(...degrees) : 0;
    const maxDeg = degrees.length ? Math.max(...degrees) : 0;

    const n = nodeList.length || 1;
    const radius = 12 * Math.sqrt(n);
    nodeList.forEach((node, i) => {
      const d = degree.get(node.id) ?? 0;
      const size =
        maxDeg > minDeg
          ? MIN_NODE_SIZE + ((d - minDeg) / (maxDeg - minDeg)) * (MAX_NODE_SIZE - MIN_NODE_SIZE)
          : (MIN_NODE_SIZE + MAX_NODE_SIZE) / 2;
      const angle = (i / n) * Math.PI * 2;
      graph.mergeNode(node.id, {
        label: node.name,
        kind: node.kind,
        color: kindColor(node.kind),
        size,
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
      });
    });

    for (const e of edgeList) {
      if (!graph.hasNode(e.source) || !graph.hasNode(e.target)) continue;
      try {
        graph.addEdgeWithKey(e.id, e.source, e.target, { relType: e.type, size: 1 });
      } catch {
        // Duplicate edge key — ignore, keep the first.
      }
    }

    graphRef.current = graph;

    // Sigma's node/edge reducers must return a *total* replacement object —
    // Sigma does not merge the reducer's return value with the graph's own
    // attributes (see `Sigma.addNode`/`addEdge` in its source), so every
    // branch below spreads the incoming `data` rather than returning a bare
    // partial (which would drop `x`/`y` and crash the renderer).
    function nodeReducer(node: string, data: NodeAttrs) {
      const highlight = highlightSetRef.current;
      if (!highlight) return data;
      if (highlight.has(node)) {
        return { ...data, forceLabel: true, zIndex: 1 };
      }
      const dim = isDarkRef.current ? NODE_COLOR_DIM.dark : NODE_COLOR_DIM.light;
      return { ...data, color: dim, label: null, forceLabel: false, zIndex: 0 };
    }

    function edgeReducer(edge: string, data: EdgeAttrs) {
      const highlight = highlightSetRef.current;
      if (!highlight) return data;
      const [source, target] = graph.extremities(edge);
      if (highlight.has(source) && highlight.has(target)) {
        return { ...data, color: isDarkRef.current ? EDGE_COLOR_HIGHLIGHT.dark : EDGE_COLOR_HIGHLIGHT.light };
      }
      return { ...data, hidden: true };
    }

    function killSigma() {
      resizeObserverRef.current?.disconnect();
      resizeObserverRef.current = null;
      sigmaRef.current?.kill();
      sigmaRef.current = null;
    }

    function mountSigma() {
      if (cancelled || !container) return;
      killSigma();
      const sigma = new Sigma<NodeAttrs, EdgeAttrs, Attributes>(graph, container, {
        renderLabels: true,
        labelDensity: 0.6,
        labelGridCellSize: 140,
        labelRenderedSizeThreshold: 9,
        defaultEdgeColor: isDarkRef.current ? EDGE_COLOR.dark : EDGE_COLOR.light,
        labelColor: { color: isDarkRef.current ? LABEL_COLOR.dark : LABEL_COLOR.light },
        minCameraRatio: 0.02,
        maxCameraRatio: 4,
        // The container can be transiently zero-sized (e.g. right as the
        // detail panel mounts/unmounts and the flex row is mid-reflow) —
        // fall back to a 1px size instead of throwing; the ResizeObserver
        // re-measures and corrects it on the next real layout pass.
        allowInvalidContainer: true,
        nodeReducer,
        edgeReducer,
      });
      sigmaRef.current = sigma;
      sigma.on("clickNode", ({ node }) => onSelectNodeRef.current(node));
      sigma.on("clickStage", () => onSelectNodeRef.current(null));
      const observer = new ResizeObserver(() => sigma.resize());
      observer.observe(container);
      resizeObserverRef.current = observer;
      onLayoutStateChangeRef.current(false);
      onReadyRef.current?.();
    }

    function finish() {
      if (cancelled) return;
      const snapshot: Record<string, { x: number; y: number }> = {};
      graph.forEachNode((id, attrs) => {
        snapshot[id] = { x: attrs.x, y: attrs.y };
      });
      positionCache.set(cacheKey, snapshot);
      mountSigma();
    }

    const cached = positionCache.get(cacheKey);
    if (cached) {
      graph.forEachNode((id) => {
        const pos = cached[id];
        if (pos) {
          graph.setNodeAttribute(id, "x", pos.x);
          graph.setNodeAttribute(id, "y", pos.y);
        }
      });
      finish();
    } else {
      const settings = forceAtlas2.inferSettings(graph);
      const supervisor = new FA2Layout(graph, { settings: { ...settings, barnesHutOptimize: true } });
      supervisorRef.current = supervisor;
      supervisor.start();
      timer = setTimeout(() => {
        supervisor.stop();
        supervisor.kill();
        supervisorRef.current = null;
        finish();
      }, LAYOUT_BUDGET_MS);
    }

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
      supervisorRef.current?.kill();
      supervisorRef.current = null;
      killSigma();
      graphRef.current = null;
    };
  }, [cacheKey]);

  // Selection change — recompute the highlight set and ask Sigma to
  // re-run the reducers (cheap: no graph/layout rebuild).
  useEffect(() => {
    const graph = graphRef.current;
    if (selectedId && graph && graph.hasNode(selectedId)) {
      const set = new Set<string>(graph.neighbors(selectedId));
      set.add(selectedId);
      highlightSetRef.current = set;
    } else {
      highlightSetRef.current = null;
    }
    sigmaRef.current?.refresh();
  }, [selectedId]);

  // Theme toggle — swap the few colors Sigma bakes into its settings
  // (default edge color, label color) and re-run the reducers so dimmed
  // node/edge colors follow suit.
  useEffect(() => {
    isDarkRef.current = isDark;
    const sigma = sigmaRef.current;
    if (sigma) {
      sigma.setSetting("defaultEdgeColor", isDark ? EDGE_COLOR.dark : EDGE_COLOR.light);
      sigma.setSetting("labelColor", { color: isDark ? LABEL_COLOR.dark : LABEL_COLOR.light });
      sigma.refresh();
    }
  }, [isDark]);

  return (
    <div
      ref={containerRef}
      className="absolute inset-0"
      style={{
        backgroundImage: "radial-gradient(var(--border-strong) 1px, transparent 1px)",
        backgroundSize: "22px 22px",
      }}
    />
  );
});
