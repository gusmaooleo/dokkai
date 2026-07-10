"use client";

/**
 * Developer/Manager/Customer segmented control, faithful to the
 * prototype's `aud` buttons (~L181-185, `audStyle` ~L989). Applies to the
 * *next* message sent, not retroactively (decision 12r).
 */

import { cn } from "@/lib/utils";
import type { Audience } from "@/lib/types";

const OPTIONS: { value: Audience; label: string }[] = [
  { value: "developer", label: "Developer" },
  { value: "manager", label: "Manager" },
  { value: "customer", label: "Customer" },
];

export function AudienceSwitch({
  value,
  onChange,
}: {
  value: Audience;
  onChange: (audience: Audience) => void;
}) {
  return (
    <div className="inline-flex gap-0.5 rounded-[11px] border border-border bg-surface-2 p-[3px]">
      {OPTIONS.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => onChange(opt.value)}
          className={cn(
            "h-[30px] rounded-lg px-[13px] text-[12.5px] font-semibold",
            value === opt.value
              ? "bg-surface text-foreground shadow-[var(--shadow-sm)]"
              : "text-muted-foreground",
          )}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
