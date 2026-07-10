"use client";

/**
 * Desktop sidebar, faithful to `dokkai-ui-prototype.html`'s aside
 * (~L114-156): logo, project switcher, Workspace/Manage nav sections,
 * theme toggle and user chip + logout at the bottom. Always expanded on
 * desktop — no collapse control (decision 12r). Hidden below 760px in
 * favor of the header hamburger + `MobileNav` drawer.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { ChevronDown, LogOut, Moon, Sun } from "lucide-react";

import { instancesApi } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useProject } from "@/lib/project";
import { useTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { isNavActive, PRIMARY_NAV, SECONDARY_NAV, type NavItem } from "./nav-items";

const JOB_POLL_INTERVAL_MS = 15_000;

function NavLink({ item, active, showJobBadge }: { item: NavItem; active: boolean; showJobBadge?: boolean }) {
  const Icon = item.icon;
  return (
    <Link
      href={item.href}
      title={item.label}
      className={cn(
        "flex h-[39px] items-center gap-[11px] rounded-[10px] border border-transparent px-[11px] text-sm transition-colors",
        active
          ? "border-[color:var(--accent-ring)] bg-[color:var(--accent-weak)] font-semibold text-accent"
          : "font-medium text-muted-foreground hover:bg-surface-2",
      )}
    >
      <Icon size={19} strokeWidth={1.7} className="shrink-0" />
      <span className="flex-1 text-left">{item.label}</span>
      {showJobBadge && (
        <span className="size-[7px] shrink-0 animate-[dk-pulse_1.4s_ease-in-out_infinite] rounded-full bg-accent" />
      )}
    </Link>
  );
}

function ProjectSwitcher() {
  const { projects, active, setActive, loading } = useProject();

  if (!loading && projects.length === 0) {
    return (
      <div className="mb-3.5 flex h-[46px] w-full items-center gap-2 rounded-[11px] border border-border bg-surface-2 px-2 text-[13px] text-muted-foreground">
        No projects ingested
      </div>
    );
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          disabled={loading}
          className="mb-3.5 flex h-[46px] w-full items-center justify-between gap-2 rounded-[11px] border border-border bg-surface-2 px-2 disabled:opacity-60"
        >
          <span className="flex min-w-0 items-center gap-[9px]">
            <span className="grid size-[26px] shrink-0 place-items-center rounded-[7px] bg-[color:var(--accent-weak)] font-mono text-[13px] font-bold text-accent">
              {(active?.project ?? "?").charAt(0).toLowerCase()}
            </span>
            <span className="flex min-w-0 flex-col text-left">
              <span className="truncate text-[13.5px] font-semibold text-foreground">
                {active?.project ?? (loading ? "Loading…" : "Select project")}
              </span>
              <span className="font-mono text-[11px] text-[color:var(--text-faint)]">
                {active ? `${active.chunks ?? active.nodes} chunks` : " "}
              </span>
            </span>
          </span>
          <ChevronDown size={15} strokeWidth={1.8} className="shrink-0 text-[color:var(--text-faint)]" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-[220px]">
        {projects.map((p) => (
          <DropdownMenuItem
            key={p.project}
            onSelect={() => setActive(p.project)}
            // Override stock shadcn focus:bg-accent — the token map paints it
            // brand red; the prototype's hover idiom is a subtle surface tint.
            className="focus:bg-surface-2 focus:text-[color:var(--text)]"
          >
            <span className="truncate">{p.project}</span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export function Sidebar() {
  const pathname = usePathname();
  const { user, logout } = useAuth();
  const { isDark, toggleTheme } = useTheme();
  const [hasRunningJob, setHasRunningJob] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const jobs = await instancesApi.listJobs();
        if (!cancelled) setHasRunningJob(jobs.some((j) => j.status === "running" || j.status === "queued"));
      } catch {
        // Transient failure — keep the last known badge state.
      }
    }

    poll();
    const id = setInterval(poll, JOB_POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  if (!user) return null;

  const secondaryNav = SECONDARY_NAV.filter((item) => !item.adminOnly || user.role === "admin");

  return (
    <aside className="hidden w-[250px] shrink-0 flex-col overflow-hidden border-r border-border bg-surface p-3 min-[760px]:flex">
      <Link href="/chat" className="mb-2 flex h-[52px] items-center gap-2.5 overflow-hidden px-1.5">
        <img src="/dokkai-logo.svg" alt="" aria-hidden="true" className="size-[26px] shrink-0" />
        <span className="font-display text-xl font-extrabold tracking-[-0.03em] text-[color:var(--wordmark)]">
          dokkai
        </span>
      </Link>

      <ProjectSwitcher />

      <div className="px-2.5 pt-3.5 pb-[7px] font-mono text-[10.5px] font-semibold tracking-[0.08em] text-[color:var(--text-faint)] uppercase">
        Workspace
      </div>
      <nav className="flex flex-col gap-[3px]">
        {PRIMARY_NAV.map((item) => (
          <NavLink key={item.href} item={item} active={isNavActive(pathname, item.href)} />
        ))}
      </nav>

      <div className="px-2.5 pt-3.5 pb-[7px] font-mono text-[10.5px] font-semibold tracking-[0.08em] text-[color:var(--text-faint)] uppercase">
        Manage
      </div>
      <nav className="flex flex-col gap-[3px]">
        {secondaryNav.map((item) => (
          <NavLink
            key={item.href}
            item={item}
            active={isNavActive(pathname, item.href)}
            showJobBadge={item.showJobBadge && hasRunningJob}
          />
        ))}
      </nav>

      <div className="mt-auto flex flex-col gap-1.5">
        <button
          type="button"
          onClick={toggleTheme}
          className="flex h-[39px] w-full items-center gap-[11px] rounded-[10px] border border-transparent px-[11px] text-sm font-medium text-muted-foreground hover:bg-surface-2"
        >
          {isDark ? <Sun size={19} strokeWidth={1.7} /> : <Moon size={19} strokeWidth={1.7} />}
          <span>{isDark ? "Light mode" : "Dark mode"}</span>
        </button>
        <div className="mx-1 h-px bg-border" />
        <div className="flex items-center gap-[9px] px-1.5 py-1.5">
          <span className="grid size-[30px] shrink-0 place-items-center rounded-[9px] bg-[#211F1B] text-[13px] font-bold text-[#F9F9F9]">
            {user.username.charAt(0).toUpperCase()}
          </span>
          <span className="flex min-w-0 flex-1 flex-col">
            <span className="truncate text-[13px] font-semibold text-foreground">{user.username}</span>
            <span className="font-mono text-[11px] text-[color:var(--text-faint)]">{user.role}</span>
          </span>
          <button
            type="button"
            onClick={logout}
            title="Sign out"
            className="grid size-[30px] shrink-0 place-items-center rounded-lg border-none bg-transparent text-[color:var(--text-faint)] hover:bg-surface-2 hover:text-accent"
          >
            <LogOut size={17} strokeWidth={1.7} />
          </button>
        </div>
      </div>
    </aside>
  );
}
