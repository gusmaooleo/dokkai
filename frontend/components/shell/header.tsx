"use client";

/**
 * Header, faithful to `dokkai-ui-prototype.html`'s `<header>` (~L159-173):
 * hamburger (mobile only), route title + subtitle + active-project badge,
 * and a "New chat" link shown only on /chat.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, Plus } from "lucide-react";

import { useProject } from "@/lib/project";
import { ROUTE_META } from "./nav-items";

/** Most routes key `ROUTE_META` by their first segment alone, but a few
 * sub-pages (e.g. `/routines/library`) need their own title/subtitle — those
 * are keyed by their first TWO segments and checked first. */
function routeMetaKey(pathname: string): string {
  const segments = pathname.split("/").filter(Boolean);
  if (segments.length === 0) return "chat";
  const twoSegment = segments.slice(0, 2).join("/");
  if (twoSegment in ROUTE_META) return twoSegment;
  return segments[0];
}

function routeSegment(pathname: string): string {
  return pathname.split("/")[1] || "chat";
}

export function Header({ onOpenMobileNav }: { onOpenMobileNav: () => void }) {
  const pathname = usePathname();
  const { active } = useProject();
  const segment = routeSegment(pathname);
  const meta = ROUTE_META[routeMetaKey(pathname)] ?? ROUTE_META.chat;

  return (
    <header className="flex h-[60px] shrink-0 items-center gap-3.5 border-b border-border bg-background px-3.5 sm:px-6">
      <button
        type="button"
        onClick={onOpenMobileNav}
        className="grid size-[38px] shrink-0 place-items-center rounded-[10px] border border-border bg-surface text-foreground min-[760px]:hidden"
      >
        <Menu size={18} strokeWidth={1.8} />
      </button>

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <h1 className="truncate font-display text-[19px] font-bold tracking-[-0.02em] text-foreground">
            {meta.title}
          </h1>
          {active && (
            <span className="shrink-0 rounded-md border border-border bg-surface-2 px-[7px] py-0.5 font-mono text-[11.5px] text-[color:var(--text-faint)]">
              {active.project}
            </span>
          )}
        </div>
        <p className="truncate text-[12.5px] text-muted-foreground">{meta.subtitle}</p>
      </div>

      {segment === "chat" && (
        <Link
          href="/chat"
          className="flex h-[38px] shrink-0 items-center gap-[7px] rounded-[10px] border border-border-strong bg-surface px-3.5 text-[13.5px] font-semibold text-foreground hover:border-accent hover:text-accent"
        >
          <Plus size={16} strokeWidth={1.9} />
          <span className="hidden sm:inline">New chat</span>
        </Link>
      )}
    </header>
  );
}
