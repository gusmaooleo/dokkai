"use client";

/**
 * Tiny theme toggle: `data-theme` on `<html>` is the single source of truth
 * (set synchronously pre-hydration by the inline script in `app/layout.tsx`
 * so there's no flash of wrong theme). This context just reads it on mount
 * and flips it — light/dark are the only two values dokkai supports.
 *
 * Context-backed (not per-instance state) so every consumer — sidebar,
 * login, graph canvas — re-renders together when the sidebar's toggle
 * button flips the theme.
 */

import { createContext, useCallback, useContext, useState, type ReactNode } from "react";

const THEME_KEY = "dokkai.theme";

export type Theme = "light" | "dark";

interface ThemeContextValue {
  theme: Theme;
  isDark: boolean;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

function readTheme(): Theme {
  if (typeof document === "undefined") return "light";
  return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  // Lazy initializer: reads the real `data-theme` (set pre-hydration by the
  // inline script in `app/layout.tsx`) directly, rather than starting at
  // "light" and syncing via an effect — one render, no theme-flip flicker.
  // SSR has no `document`, so the server markup assumes "light"; consumers
  // that render theme-dependent DOM should mark it `suppressHydrationWarning`
  // (same escape hatch `layout.tsx` uses on `<html>`).
  const [theme, setTheme] = useState<Theme>(() => readTheme());

  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      const next: Theme = prev === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      window.localStorage.setItem(THEME_KEY, next);
      return next;
    });
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, isDark: theme === "dark", toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within a ThemeProvider");
  return ctx;
}
