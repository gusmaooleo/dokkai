"use client";

/**
 * Auth context: the current session (bearer token + hydrated user) plus
 * login/logout/refresh actions. Token lives in localStorage under
 * `dokkai.token` (decision 12e); `lib/api.ts` reads it directly for every
 * request and already redirects to `/login` on a 401 from a protected
 * call, so this provider mostly tracks UI-facing state.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { useRouter } from "next/navigation";

import { authApi, clearToken, getToken, setToken as storeToken } from "./api";
import type { Me } from "./types";

interface AuthContextValue {
  /** The hydrated `/auth/me` payload, or `null` before hydration / when signed out. */
  user: Me | null;
  token: string | null;
  /** True until the initial token→user hydration (or its absence) has resolved. */
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  /** Re-fetches `/auth/me` — e.g. to clear `default_admin_active` after a password change. */
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const router = useRouter();
  // Lazy initializers: `getToken()` is a no-op returning `null` during SSR
  // (no `window`). With a stored token the client's first render diverges
  // from SSR (token set, loading true) — accepted: the provider renders
  // children unconditionally, and consumers must gate on `loading` after
  // mount rather than branch during the first render. This also avoids a
  // synchronous setState in the effect below just to reflect "no stored
  // token" (see the react-hooks set-state-in-effect rule).
  const [token, setTokenState] = useState<string | null>(() => getToken());
  const [user, setUser] = useState<Me | null>(null);
  const [loading, setLoading] = useState<boolean>(() => getToken() !== null);

  // Hydrate from a stored token on mount (client-only — localStorage isn't
  // available during SSR, so this intentionally runs post-hydration).
  useEffect(() => {
    if (!getToken()) return;

    let cancelled = false;
    authApi
      .me()
      .then((me) => {
        if (!cancelled) setUser(me);
      })
      .catch(() => {
        // A 401 here already cleared the token and redirected via
        // lib/api.ts's request(); any other failure (e.g. the API being
        // temporarily unreachable) just leaves the user unhydrated.
        if (!cancelled) setUser(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    const res = await authApi.login(username, password);
    storeToken(res.token);
    setTokenState(res.token);
    const me = await authApi.me();
    setUser(me);
  }, []);

  const logout = useCallback(async () => {
    try {
      await authApi.logout();
    } catch {
      // Best-effort — the session is discarded client-side regardless.
    }
    clearToken();
    setTokenState(null);
    setUser(null);
    router.push("/login");
  }, [router]);

  const refresh = useCallback(async () => {
    const me = await authApi.me();
    setUser(me);
  }, []);

  return (
    <AuthContext.Provider value={{ user, token, loading, login, logout, refresh }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
