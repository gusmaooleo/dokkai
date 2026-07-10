/**
 * Kind → color legend, faithful to the prototype's bottom-left card
 * (~L292-297): dot + label per kind actually present in the loaded graph.
 */

import { kindColor } from "@/components/chat/kinds";

export function GraphLegend({ kinds }: { kinds: string[] }) {
  if (kinds.length === 0) return null;

  return (
    <div className="absolute bottom-4 left-4 z-[6] max-w-[calc(100%-32px)] rounded-[11px] border border-border bg-surface px-[13px] py-[11px] shadow-[var(--shadow-sm)]">
      <div className="mb-2 font-mono text-[10px] font-semibold tracking-[0.08em] text-[color:var(--text-faint)] uppercase">
        Node kinds
      </div>
      <div className="flex max-w-[340px] flex-wrap gap-x-3.5 gap-y-2">
        {kinds.map((kind) => (
          <span key={kind} className="inline-flex items-center gap-1.5 text-[11.5px] text-muted-foreground">
            <span className="size-2.5 rounded-full" style={{ backgroundColor: kindColor(kind) }} />
            {kind}
          </span>
        ))}
      </div>
    </div>
  );
}
