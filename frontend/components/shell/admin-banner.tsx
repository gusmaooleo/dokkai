"use client";

/**
 * Default-admin banner (decision 12n): persistent for the session — no
 * dismiss — shown only when the signed-in user is the admin role and the
 * default admin/admin credentials are still active.
 */

import Link from "next/link";
import { CircleAlert } from "lucide-react";

import { useAuth } from "@/lib/auth";

export function AdminBanner() {
  const { user } = useAuth();

  if (!user || user.role !== "admin" || !user.default_admin_active) return null;

  return (
    <div className="flex items-center gap-2 border-b border-[color:var(--accent-ring)] bg-[color:var(--accent-weak)] px-3.5 py-2 text-[13px] text-[color:var(--accent-press)] sm:px-6">
      <CircleAlert size={15} strokeWidth={2} className="shrink-0" />
      <span>
        Default admin credentials (admin/admin) are active — change the password in{" "}
        <Link href="/settings" className="font-semibold underline">
          Settings
        </Link>
        .
      </span>
    </div>
  );
}
