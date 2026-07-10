"use client";

/**
 * Split-screen login, faithful to `dokkai_agent/reference/dokkai-ui-prototype.html`'s
 * LOGIN SCREEN section: a fixed dark brand panel on the left (independent of
 * the light/dark theme toggle, which only affects the form panel) and the
 * sign-in form on the right.
 */

import { useEffect, useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { CircleAlert, Eye, EyeOff, LoaderCircle, Moon, Sun } from "lucide-react";

import { authApi, ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useTheme } from "@/lib/theme";
import type { AuthStatus } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const FEATURE_CHIPS = ["local-only", "weaviate + ollama", "mcp · cli · ui"] as const;

export default function LoginPage() {
  const router = useRouter();
  const { user, loading: authLoading, login } = useAuth();
  const { toggleTheme } = useTheme();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [authStatus, setAuthStatus] = useState<AuthStatus | null>(null);

  useEffect(() => {
    if (!authLoading && user) {
      router.replace("/chat");
    }
  }, [authLoading, user, router]);

  useEffect(() => {
    let cancelled = false;
    authApi
      .status()
      .then((status) => {
        if (!cancelled) setAuthStatus(status);
      })
      .catch(() => {
        // Public endpoint — if it's unreachable the login submit below will
        // surface the same failure, so there's nothing more to do here.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      await login(username, password);
      router.push("/chat");
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "something went wrong — please try again");
    } finally {
      setSubmitting(false);
    }
  }

  // Hydrating an existing session, or about to redirect one away — avoid
  // flashing the login form in either case.
  if (authLoading || user) {
    return null;
  }

  return (
    <div className="grid h-screen grid-cols-[repeat(auto-fit,minmax(340px,1fr))] overflow-y-auto bg-background">
      <div className="relative flex min-h-[280px] flex-col justify-between overflow-hidden bg-[#211F1B] p-8 text-[#F9F9F9] sm:p-12">
        <img
          src="/dokkai-logo.svg"
          alt=""
          aria-hidden="true"
          className="pointer-events-none absolute top-1/2 right-[-160px] w-[min(640px,80%)] -translate-y-1/2 opacity-[0.92]"
        />
        <div className="relative flex items-center gap-[11px]">
          <img src="/dokkai-logo.svg" alt="" aria-hidden="true" className="h-[30px] w-[30px]" />
          <span className="font-display text-[23px] font-extrabold tracking-[-0.03em] text-[#F9F9F9]">
            dokkai
          </span>
        </div>
        <div className="relative max-w-[420px]">
          <h1 className="mb-4 font-display text-[clamp(28px,3.4vw,42px)] leading-[1.05] font-extrabold tracking-[-0.03em]">
            Graph-aware retrieval for your codebase.
          </h1>
          <p className="text-[15.5px] leading-[1.65] text-[#F9F9F9]/62">
            Turn a repository into a graph-aware vector database any model can query for
            pennies — implementation, callers and dependencies, in a handful of tokens.
          </p>
          <div className="mt-[26px] flex flex-wrap gap-2">
            {FEATURE_CHIPS.map((chip) => (
              <span
                key={chip}
                className="rounded-full border border-[#F9F9F9]/16 px-[11px] py-[5px] font-mono text-xs font-medium text-[#F9F9F9]/70"
              >
                {chip}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="relative flex items-center justify-center p-6 sm:p-12">
        <button
          type="button"
          onClick={toggleTheme}
          title="Toggle theme"
          className="absolute top-[22px] right-[22px] grid h-[38px] w-[38px] place-items-center rounded-[10px] border border-border bg-surface text-muted-foreground"
        >
          {/* CSS-driven (not React-state-driven) so server/client markup never mismatches. */}
          <Sun size={18} strokeWidth={1.7} className="hidden dark:block" />
          <Moon size={18} strokeWidth={1.7} className="block dark:hidden" />
        </button>

        <div className="w-full max-w-[378px] animate-[dk-fadeup_0.5s_ease_both]">
          <h2 className="mb-1.5 font-display text-[26px] font-bold tracking-[-0.02em] text-foreground">
            Sign in
          </h2>
          <p className="mb-[26px] text-sm text-muted-foreground">Sign in with your dokkai account.</p>

          {error && (
            <div className="mb-4 flex animate-[dk-fade_0.2s_ease] items-center gap-2 rounded-[10px] border border-[color:var(--accent-ring)] bg-[color:var(--accent-weak)] px-3 py-2.5 text-[13px] text-[color:var(--accent-press)]">
              <CircleAlert size={15} strokeWidth={2} className="shrink-0" />
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} noValidate>
            <Label htmlFor="username" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
              Username
            </Label>
            <Input
              id="username"
              name="username"
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="mb-4 h-11 rounded-[11px] border-border-strong bg-surface px-3.5 text-[15px] dark:bg-surface"
            />

            <Label htmlFor="password" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
              Password
            </Label>
            <div className="relative mb-[22px]">
              <Input
                id="password"
                name="password"
                type={showPassword ? "text" : "password"}
                autoComplete="current-password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="h-11 rounded-[11px] border-border-strong bg-surface px-3.5 pr-11 text-[15px] dark:bg-surface"
              />
              <button
                type="button"
                tabIndex={-1}
                onClick={() => setShowPassword((v) => !v)}
                aria-label={showPassword ? "Hide password" : "Show password"}
                className="absolute top-1.5 right-1.5 grid h-8 w-8 place-items-center rounded-lg border-none bg-transparent text-[color:var(--text-faint)]"
              >
                {showPassword ? <EyeOff size={17} strokeWidth={1.7} /> : <Eye size={17} strokeWidth={1.7} />}
              </button>
            </div>

            <Button
              type="submit"
              disabled={submitting}
              className={cn(
                "h-[46px] w-full rounded-[11px] border border-accent bg-accent text-[15px] font-semibold text-white",
                "shadow-[0_1px_2px_rgba(255,74,74,.4)] hover:bg-accent hover:brightness-105",
              )}
            >
              {submitting ? (
                <>
                  <LoaderCircle size={17} strokeWidth={2.4} className="animate-[dk-spin_0.7s_linear_infinite]" />
                  Signing in…
                </>
              ) : (
                "Sign in"
              )}
            </Button>
          </form>

          <div className="mt-5 flex items-center justify-between gap-3 font-mono text-xs text-[color:var(--text-faint)]">
            <span>self-hosted</span>
            {authStatus?.default_admin_active && (
              <span>Default admin credentials are active (admin / admin)</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
