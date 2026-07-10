/**
 * Small formatting helpers for the Admin screen. Role descriptions mirror
 * the permission map documented inline in the backend controllers
 * (decision 6k): `viewer` is read-only, `user` can ingest/chat, `admin` can
 * additionally manage users and the LLM configuration.
 */

import type { Role } from "@/lib/types";

/** Mirrors `ConversationCard`'s `rel()`-derived helper — no date lib. */
export function relativeTime(iso: string): string {
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

export const ROLE_DESCRIPTION: Record<Role, string> = {
  admin: "Full access — manage users, LLM configuration, and all data.",
  user: "Can ingest repositories and chat, but not manage users or settings.",
  viewer: "Read-only — can browse, chat, and search, but not change anything.",
};

const ROLE_BADGE_CLASS: Record<Role, string> = {
  admin: "text-[#4B78B0] bg-[rgba(75,120,176,.16)]",
  user: "text-[#2f8f82] bg-[rgba(62,156,147,.14)]",
  viewer: "text-muted-foreground bg-surface-2",
};

export function roleBadgeClass(role: Role): string {
  return ROLE_BADGE_CLASS[role] ?? ROLE_BADGE_CLASS.viewer;
}
