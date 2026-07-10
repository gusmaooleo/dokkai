"use client";

/**
 * Account card (decision 12n) — not in the prototype. Lets ANY authenticated
 * user (all roles) change their own password via
 * `PATCH /auth/users/me/password`. On success, calls `useAuth().refresh()`
 * so a cleared `default_admin_active` flag drops the admin banner
 * immediately without a re-login.
 */

import { useState, type FormEvent } from "react";

import { ApiError, authApi } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useToast } from "@/lib/toast";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export function AccountCard() {
  const { user, refresh } = useAuth();
  const { toast } = useToast();

  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!currentPassword || !newPassword || !confirmPassword) {
      toast("Fill in every field.", "error");
      return;
    }
    if (newPassword !== confirmPassword) {
      toast("New password and confirmation do not match.", "error");
      return;
    }

    setSubmitting(true);
    try {
      await authApi.changePassword(currentPassword, newPassword);
      toast("Password updated");
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      // Outside the catch below: the password DID change — a transient
      // /auth/me failure must not toast a contradictory error. The banner
      // self-heals on the next successful load.
      refresh().catch(() => {});
    } catch (err) {
      toast(err instanceof ApiError ? err.detail : "Failed to update password", "error");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="rounded-[14px] border border-border bg-surface p-5 shadow-[var(--shadow-sm)]">
      <p className="mb-4 text-[13px] text-muted-foreground">
        Change the password for <span className="font-semibold text-foreground">{user?.username}</span>.
      </p>
      <div className="grid grid-cols-[repeat(auto-fit,minmax(180px,1fr))] gap-[14px]">
        <div>
          <Label htmlFor="current-password" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
            Current password
          </Label>
          <Input
            id="current-password"
            type="password"
            value={currentPassword}
            onChange={(e) => setCurrentPassword(e.target.value)}
            className="h-[42px] rounded-[10px] border-border-strong bg-background px-3 text-[13.5px]"
            autoComplete="current-password"
          />
        </div>
        <div>
          <Label htmlFor="new-password" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
            New password
          </Label>
          <Input
            id="new-password"
            type="password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            className="h-[42px] rounded-[10px] border-border-strong bg-background px-3 text-[13.5px]"
            autoComplete="new-password"
          />
        </div>
        <div>
          <Label htmlFor="confirm-password" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
            Confirm new password
          </Label>
          <Input
            id="confirm-password"
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            className="h-[42px] rounded-[10px] border-border-strong bg-background px-3 text-[13.5px]"
            autoComplete="new-password"
          />
        </div>
      </div>
      <div className="mt-[16px] flex justify-end border-t border-border pt-[14px]">
        <Button type="submit" disabled={submitting}>
          {submitting ? "Updating…" : "Update password"}
        </Button>
      </div>
    </form>
  );
}
