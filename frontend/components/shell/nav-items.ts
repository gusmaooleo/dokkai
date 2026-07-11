/**
 * Shared nav config for the sidebar and mobile drawer, plus the header's
 * route title/subtitle copy — mirrors `dokkai-ui-prototype.html`'s
 * `nav`/`titles`/`subs` locals (~L1103-1105).
 */

import type { LucideIcon } from "lucide-react";
import {
  Boxes,
  History,
  MessageCircle,
  Network,
  SearchCode,
  ShieldUser,
  SlidersHorizontal,
} from "lucide-react";

export interface NavItem {
  href: string;
  label: string;
  icon: LucideIcon;
  /** Admin nav item — hidden for non-admins (decision 12p). */
  adminOnly?: boolean;
  /** Instances — shows the running-job pulse badge. */
  showJobBadge?: boolean;
  /** Experimental/demo feature — shows a subtle "beta" chip (decision 14c). */
  beta?: boolean;
}

export const PRIMARY_NAV: NavItem[] = [
  { href: "/chat", label: "Chat", icon: MessageCircle },
  { href: "/graph", label: "Graph", icon: Network },
  { href: "/routines", label: "Routines", icon: SearchCode, beta: true },
  { href: "/history", label: "History", icon: History },
];

export const SECONDARY_NAV: NavItem[] = [
  { href: "/instances", label: "Instances", icon: Boxes, showJobBadge: true },
  { href: "/admin", label: "Admin", icon: ShieldUser, adminOnly: true },
  { href: "/settings", label: "Settings", icon: SlidersHorizontal },
];

export const ROUTE_META: Record<string, { title: string; subtitle: string }> = {
  chat: {
    title: "Chat",
    subtitle: "Ask a local model about your codebase — answers cite the exact retrieved chunks.",
  },
  graph: {
    title: "Graph",
    subtitle: "Explore the dependency graph. Drag to arrange, scroll to zoom, click a node for its source.",
  },
  routines: {
    title: "Routines",
    subtitle: "Experimental code review and bug hunt.",
  },
  "routines/library": {
    title: "Library",
    subtitle: "Playbooks and skills for routine runs.",
  },
  history: {
    title: "History",
    subtitle: "Every conversation, stored in Postgres. Resume or delete.",
  },
  instances: {
    title: "Instances",
    subtitle: "Ingested projects and live pipeline jobs.",
  },
  admin: {
    title: "Admin",
    subtitle: "Manage the users who can sign in to this dokkai instance.",
  },
  settings: {
    title: "Settings",
    subtitle: "Configure the LLM provider, models and health checks.",
  },
};

export function isNavActive(pathname: string, href: string): boolean {
  return pathname === href || pathname.startsWith(`${href}/`);
}
