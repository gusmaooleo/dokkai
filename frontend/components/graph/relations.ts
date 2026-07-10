/**
 * Direction-aware relation phrasing for the detail panel's relations list —
 * "X calls Y" reads as "calls" from X's card and "called by" from Y's.
 * Covers every edge type the backend emits between code entities (CALLS,
 * DEFINES, DEFINES_METHOD, INHERITS, IMPLEMENTS, OVERRIDES) plus IMPORTS,
 * used for the files view's dominant-type label.
 */

const OUT_LABELS: Record<string, string> = {
  CALLS: "calls",
  DEFINES: "defines",
  DEFINES_METHOD: "defines method",
  INHERITS: "inherits",
  IMPLEMENTS: "implements",
  OVERRIDES: "overrides",
  IMPORTS: "imports",
};

const IN_LABELS: Record<string, string> = {
  CALLS: "called by",
  DEFINES: "defined by",
  DEFINES_METHOD: "defined by",
  INHERITS: "inherited by",
  IMPLEMENTS: "implemented by",
  OVERRIDES: "overridden by",
  IMPORTS: "imported by",
};

/** `direction` is "out" when the selected node is the edge's source. */
export function relationLabel(type: string, direction: "out" | "in"): string {
  const map = direction === "out" ? OUT_LABELS : IN_LABELS;
  return map[type] ?? (direction === "out" ? type.toLowerCase() : `${type.toLowerCase()} (incoming)`);
}

/** The highest-count relationship type in a files-view edge's `types` breakdown. */
export function dominantType(types: Record<string, number> | undefined): string {
  if (!types) return "IMPORTS";
  let best = "IMPORTS";
  let bestCount = -1;
  for (const [type, count] of Object.entries(types)) {
    if (count > bestCount) {
      best = type;
      bestCount = count;
    }
  }
  return best;
}
