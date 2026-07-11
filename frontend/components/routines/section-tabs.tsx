"use client";

/**
 * "Runs | Library" tab bar shared by the Routines home (run history) and
 * the Library screen (C5) — same segmented-control visual as `LaunchCard`'s
 * kind toggle, but these are route links (not local state) since Runs and
 * Library are separate pages.
 */

import Link from "next/link";

import { cn } from "@/lib/utils";

const TABS: { href: string; label: string; key: "runs" | "library" }[] = [
  { href: "/routines", label: "Runs", key: "runs" },
  { href: "/routines/library", label: "Library", key: "library" },
];

export function RoutineSectionTabs({ active }: { active: "runs" | "library" }) {
  return (
    <div className="mb-6 inline-flex gap-0.5 rounded-[11px] border border-border bg-surface-2 p-[3px]">
      {TABS.map((tab) => (
        <Link
          key={tab.key}
          href={tab.href}
          className={cn(
            "flex h-[30px] items-center rounded-lg px-[13px] text-[12.5px] font-semibold",
            active === tab.key ? "bg-surface text-foreground shadow-[var(--shadow-sm)]" : "text-muted-foreground",
          )}
        >
          {tab.label}
        </Link>
      ))}
    </div>
  );
}
