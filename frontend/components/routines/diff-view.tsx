/**
 * Hand-rolled unified-diff renderer for a finding's `evidence.hunk_excerpt`
 * (review runs only — bughunt findings carry no diff to excerpt from, see
 * `services.routines.bughunt._validate_finding`). No new dependency: splits
 * the excerpt into lines and classifies each by its unified-diff prefix —
 * `@@ ... @@` hunk headers muted, `+` lines green-tinted, `-` lines
 * red-tinted, `\` (the "no newline at end of file" marker) muted italic,
 * everything else (context lines, leading space) plain.
 */

import { cn } from "@/lib/utils";

function lineClasses(line: string): string {
  if (line.startsWith("@@")) return "text-[color:var(--text-faint)]";
  if (line.startsWith("+") && !line.startsWith("+++")) return "bg-[rgba(62,156,147,.12)] text-[#2f8f82]";
  if (line.startsWith("-") && !line.startsWith("---")) return "bg-[rgba(216,60,42,.1)] text-[#c0392b]";
  if (line.startsWith("\\")) return "text-[color:var(--text-faint)] italic";
  return "text-foreground";
}

export function DiffView({ text }: { text: string }) {
  const lines = text.split("\n");

  return (
    <div className="overflow-x-auto rounded-[9px] border border-border bg-surface-2 py-2 font-mono text-[12px] leading-[1.55]">
      {lines.map((line, i) => (
        <div key={i} className={cn("whitespace-pre px-2.5", lineClasses(line))}>
          {line.length > 0 ? line : " "}
        </div>
      ))}
    </div>
  );
}
