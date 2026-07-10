/**
 * Shared shapes for the graph screen — a normalized view over both
 * `GET /graph/{project}` (entities) and `GET /graph/{project}/files` (files)
 * so `GraphCanvas`, the toolbar, legend and detail panel don't need to know
 * which endpoint the data came from.
 */

export type GraphMode = "entities" | "files";

export interface NormalizedNode {
  /** Sigma/graphology node key — the entity id (stringified) or the file path. */
  id: string;
  /** Entity kind (Class/Function/Method/…) or "File" in files mode. */
  kind: string;
  name: string;
  qualified_name: string | null;
  path: string | null;
  absolute_path: string | null;
  start_line: number | null;
  end_line: number | null;
}

export interface NormalizedEdge {
  id: string;
  source: string;
  target: string;
  /** Dominant relationship type — the edge's own type (entities) or the
   *  highest-count key in `types` (files). */
  type: string;
  /** Files mode only: total weight and the per-type breakdown. */
  weight?: number;
  types?: Record<string, number>;
}
