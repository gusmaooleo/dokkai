"use client";

/**
 * Tiny theme toggle: `data-theme` on `<html>` is the single source of truth
 * (set synchronously pre-hydration by the inline script in `app/layout.tsx`
 * so there's no flash of wrong theme). This hook just reads it on mount and
 * flips it — light/dark are the only two values dokkai supports.
 */

import { useCallback, useState } from "react";

const THEME_KEY = "dokkai.theme";

export type Theme = "light" | "dark";

function readTheme(): Theme {
  if (typeof document === "undefined") return "light";
  return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

export function useTheme() {
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

  return { theme, isDark: theme === "dark", toggleTheme };
}
