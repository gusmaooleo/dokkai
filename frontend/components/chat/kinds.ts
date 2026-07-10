/**
 * Badge color per graph entity kind, styled after the prototype's `KMETA`
 * (~L491-505) — colored uppercase mono badges on a low-alpha tint of the
 * same color. The prototype's kinds (controller/usecase/port/…) are
 * fictional, invented for its TypeScript demo project; dokkai's real
 * `entity_type` values come straight off the code graph's node labels
 * (`src/services/chunker.py`'s `CODE_ENTITY_LABELS`), so this map uses
 * those instead.
 */

const KIND_COLORS: Record<string, string> = {
  Class: "#B5638C",
  Interface: "#4B78B0",
  Function: "#FF4A4A",
  Method: "#3E9C93",
  Enum: "#8A8A5E",
  Type: "#7C74C9",
};

const DEFAULT_KIND_COLOR = "#7A7A82";

export function kindColor(kind: string): string {
  return KIND_COLORS[kind] ?? DEFAULT_KIND_COLOR;
}
