"use client";

/**
 * Mobile drawer nav, faithful to `dokkai-ui-prototype.html`'s
 * `mobileNavOpen` block (~L438-450): backdrop + a slide-in `aside` with the
 * logo, text-only nav buttons (no icons — the prototype omits them here)
 * and a "Sign out" button at the bottom. Only rendered below 760px.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";

import { useAuth } from "@/lib/auth";
import { cn } from "@/lib/utils";
import { isNavActive, PRIMARY_NAV, SECONDARY_NAV, type NavItem } from "./nav-items";

export function MobileNav({ open, onClose }: { open: boolean; onClose: () => void }) {
  const pathname = usePathname();
  const { user, logout } = useAuth();

  if (!open || !user) return null;

  const items: NavItem[] = [
    ...PRIMARY_NAV,
    ...SECONDARY_NAV.filter((item) => !item.adminOnly || user.role === "admin"),
  ];

  return (
    <>
      <div
        onClick={onClose}
        className="fixed inset-0 z-40 animate-[dk-fade_0.2s_ease] bg-black/40 min-[760px]:hidden"
      />
      <aside className="fixed inset-y-0 left-0 z-[41] flex w-[250px] animate-[dk-fadeup_0.2s_ease] flex-col gap-[3px] border-r border-border bg-surface p-3 min-[760px]:hidden">
        <div className="flex items-center gap-2.5 px-1.5 pb-3">
          <img src="/dokkai-logo.svg" alt="" aria-hidden="true" className="size-6" />
          <span className="font-display text-[19px] font-extrabold tracking-[-0.03em] text-[color:var(--wordmark)]">
            dokkai
          </span>
        </div>
        {items.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            onClick={onClose}
            className={cn(
              "flex h-[39px] items-center gap-2 rounded-[10px] border border-transparent px-[11px] text-sm",
              isNavActive(pathname, item.href)
                ? "border-[color:var(--accent-ring)] bg-[color:var(--accent-weak)] font-semibold text-accent"
                : "font-medium text-muted-foreground",
            )}
          >
            <span className="flex-1">{item.label}</span>
            {item.beta && (
              <span className="shrink-0 rounded-[4px] border border-border px-[5px] py-px font-mono text-[9px] font-semibold tracking-[0.06em] text-[color:var(--text-faint)] uppercase">
                beta
              </span>
            )}
          </Link>
        ))}
        <button
          type="button"
          onClick={() => {
            onClose();
            logout();
          }}
          className="mt-auto flex h-[39px] items-center rounded-[10px] border border-transparent px-[11px] text-left text-sm font-medium text-muted-foreground"
        >
          Sign out
        </button>
      </aside>
    </>
  );
}
