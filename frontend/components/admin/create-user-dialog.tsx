"use client";

/**
 * "Create user" dialog, faithful to the prototype's user dialog
 * (`dokkai-ui-prototype.html` ~L455): username, password, role select.
 * Role options carry a one-line description per decision 6k's permission
 * map. Duplicate usernames surface the backend's verbatim 409 detail
 * (`username '…' already exists`, `controllers/auth.py`).
 */

import { useState } from "react";

import { ApiError, authApi } from "@/lib/api";
import { useToast } from "@/lib/toast";
import type { Role } from "@/lib/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ROLE_DESCRIPTION } from "./format";

const ROLES: Role[] = ["admin", "user", "viewer"];

export function CreateUserDialog({
  open,
  onOpenChange,
  onCreated,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated: () => void;
}) {
  const { toast } = useToast();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<Role>("user");
  const [submitting, setSubmitting] = useState(false);

  function reset() {
    setUsername("");
    setPassword("");
    setRole("user");
    setSubmitting(false);
  }

  function handleOpenChange(next: boolean) {
    if (!next) reset();
    onOpenChange(next);
  }

  async function handleCreate() {
    if (!username.trim() || !password) {
      toast("Enter a username and password.", "error");
      return;
    }
    setSubmitting(true);
    try {
      await authApi.createUser({ username: username.trim(), password, role });
      toast(`User '${username.trim()}' created`);
      onCreated();
      handleOpenChange(false);
    } catch (err) {
      toast(err instanceof ApiError ? err.detail : "Failed to create user", "error");
      setSubmitting(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-[420px]">
        <DialogHeader>
          <DialogTitle className="font-display text-[17px] font-bold">Create user</DialogTitle>
          <DialogDescription className="text-[12.5px] text-muted-foreground">
            Add a new user who can sign in to this dokkai instance.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-[18px]">
          <div>
            <Label htmlFor="new-user-username" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
              Username
            </Label>
            <Input
              id="new-user-username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="h-[42px] rounded-[10px] border-border-strong bg-background px-3 text-[13.5px]"
              autoComplete="off"
            />
          </div>

          <div>
            <Label htmlFor="new-user-password" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
              Password
            </Label>
            <Input
              id="new-user-password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="h-[42px] rounded-[10px] border-border-strong bg-background px-3 text-[13.5px]"
              autoComplete="new-password"
            />
          </div>

          <div>
            <Label htmlFor="new-user-role" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
              Role
            </Label>
            <Select value={role} onValueChange={(v) => setRole(v as Role)}>
              <SelectTrigger id="new-user-role" className="h-[42px] w-full rounded-[10px] border-border-strong bg-background px-3 text-[13.5px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {ROLES.map((r) => (
                  <SelectItem key={r} value={r}>
                    {r}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="mt-1.5 text-[11.5px] text-muted-foreground">{ROLE_DESCRIPTION[role]}</p>
          </div>
        </div>

        <DialogFooter>
          <Button type="button" variant="outline" onClick={() => handleOpenChange(false)}>
            Cancel
          </Button>
          <Button type="button" onClick={handleCreate} disabled={submitting}>
            {submitting ? "Creating…" : "Create user"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
