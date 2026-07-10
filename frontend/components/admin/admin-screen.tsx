"use client";

/**
 * Admin screen, faithful to the prototype's Admin screen
 * (`dokkai-ui-prototype.html` ~L388-411), aligned to reality per decision
 * 12m: roles are admin/user/viewer (no Root role), columns are
 * id/username/role/created_at (no last-login/enable-disable), and the
 * footer note reflects the actual seeding behaviour (`services/auth.py`'s
 * `ensure_seeded`) instead of the prototype's invented root-user story.
 *
 * `GET /auth/users` 403s for non-admins — the nav already hides the Admin
 * link for them, but a deep link still reaches this screen, so a 403 is
 * rendered as a friendly "admins only" state rather than a raw error.
 *
 * Own-row delete is hidden (can't delete yourself, per the backend's 400 on
 * self-delete) — that backend 400 remains a backstop surfaced as a toast in
 * case the delete is ever triggered another way (e.g. a stale row after a
 * role change in another tab).
 */

import { useCallback, useEffect, useState } from "react";
import { Plus, ShieldAlert, Trash2 } from "lucide-react";

import { ApiError, authApi } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useToast } from "@/lib/toast";
import type { User } from "@/lib/types";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { CreateUserDialog } from "./create-user-dialog";
import { relativeTime, roleBadgeClass } from "./format";

const FOOTER_NOTE =
  "Users are stored in Postgres. On first boot dokkai seeds admin/admin; " +
  "DOKKAI_ROOT_USER / DOKKAI_ROOT_PASSWORD environment variables override the admin credentials on boot.";

export function AdminScreen() {
  const { user: me } = useAuth();
  const { toast } = useToast();

  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [forbidden, setForbidden] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setForbidden(false);
    try {
      const list = await authApi.listUsers();
      setUsers(list);
    } catch (err) {
      if (err instanceof ApiError && err.status === 403) {
        setForbidden(true);
      } else {
        setError(err instanceof ApiError ? err.detail : "Failed to load users");
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function handleDelete(user: User) {
    try {
      await authApi.deleteUser(user.id);
      toast(`User '${user.username}' deleted`);
      load();
    } catch (err) {
      toast(err instanceof ApiError ? err.detail : "Failed to delete user", "error");
    }
  }

  if (loading) {
    return <div className="py-20 text-center text-[13.5px] text-muted-foreground">Loading…</div>;
  }

  if (forbidden) {
    return (
      <div className="flex flex-col items-center gap-3 py-24 text-center">
        <ShieldAlert size={28} strokeWidth={1.8} className="text-[color:var(--text-faint)]" />
        <p className="text-[14px] font-semibold text-foreground">Admins only</p>
        <p className="max-w-xs text-[13px] text-muted-foreground">
          You don&apos;t have permission to view or manage users.
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center gap-3 py-24 text-center">
        <p className="text-[14px] text-muted-foreground">{error}</p>
        <Button variant="outline" onClick={load}>
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[840px] px-3.5 py-[18px] sm:px-6.5 sm:py-[30px]">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="flex items-baseline gap-[9px]">
          <span className="font-display text-[17px] font-bold text-foreground">Users</span>
          <span className="font-mono text-xs text-[color:var(--text-faint)]">{users.length} total</span>
        </div>
        <Button type="button" onClick={() => setDialogOpen(true)}>
          <Plus size={16} strokeWidth={2} />
          Create user
        </Button>
      </div>

      <div className="overflow-hidden rounded-[14px] border border-border bg-surface shadow-[var(--shadow-sm)]">
        {users.map((u) => {
          const isSelf = u.username === me?.username;
          return (
            <div
              key={u.id}
              className="flex flex-wrap items-center gap-[13px] border-b border-border px-[18px] py-3.5 last:border-b-0"
            >
              <span className="grid size-9 shrink-0 place-items-center rounded-[10px] bg-surface-2 text-sm font-bold text-muted-foreground">
                {u.username.charAt(0).toUpperCase()}
              </span>
              <div className="min-w-[130px] flex-1">
                <div className="flex items-center gap-[7px]">
                  <span className="text-[14.5px] font-semibold text-foreground">{u.username}</span>
                  {isSelf && (
                    <span className="rounded-[5px] border border-border px-1.5 py-px font-mono text-[10px] text-[color:var(--text-faint)] uppercase">
                      you
                    </span>
                  )}
                </div>
                <div className="mt-px font-mono text-[11px] text-[color:var(--text-faint)]">
                  created {relativeTime(u.created_at)}
                </div>
              </div>
              <span className={`rounded-full px-2.5 py-[3px] text-[11px] font-semibold ${roleBadgeClass(u.role)}`}>
                {u.role}
              </span>
              <div className="flex min-w-[40px] items-center justify-end gap-[5px]">
                {!isSelf && (
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <button
                        type="button"
                        title="Delete user"
                        className="grid size-[30px] place-items-center rounded-lg border border-border-strong bg-surface text-[color:var(--text-faint)] hover:border-[color:var(--accent-ring)] hover:bg-[color:var(--accent-weak)] hover:text-accent"
                      >
                        <Trash2 size={14} strokeWidth={1.8} />
                      </button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>Delete user &lsquo;{u.username}&rsquo;?</AlertDialogTitle>
                        <AlertDialogDescription>
                          This permanently removes their account and ends every active session. This can&apos;t be
                          undone.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction variant="destructive" onClick={() => handleDelete(u)}>
                          Delete
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <p className="mt-3.5 px-0.5 font-mono text-[11.5px] text-[color:var(--text-faint)]">{FOOTER_NOTE}</p>

      <CreateUserDialog open={dialogOpen} onOpenChange={setDialogOpen} onCreated={load} />
    </div>
  );
}
