"use client";

/**
 * Minimal toast system, styled after the prototype's single bottom-center
 * toast (`dokkai-ui-prototype.html` ~L478). Supports several toasts at once
 * (stacked) since the shell can trigger more than one in quick succession;
 * each auto-dismisses independently after ~4s.
 */

import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import { CircleAlert } from "lucide-react";

import { cn } from "./utils";

export type ToastVariant = "default" | "error";

interface ToastItem {
  id: number;
  message: string;
  variant: ToastVariant;
}

interface ToastContextValue {
  toast: (message: string, variant?: ToastVariant) => void;
}

const ToastContext = createContext<ToastContextValue | undefined>(undefined);

let nextId = 0;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const toast = useCallback((message: string, variant: ToastVariant = "default") => {
    const id = ++nextId;
    setToasts((prev) => [...prev, { id, message, variant }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      <div className="pointer-events-none fixed bottom-[22px] left-1/2 z-[60] flex -translate-x-1/2 flex-col items-center gap-2 px-4">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={cn(
              "pointer-events-auto flex animate-[dk-fadeup_0.2s_ease] items-center gap-2 rounded-[11px] px-[18px] py-[11px] text-[13.5px] shadow-[var(--shadow-lg)]",
              t.variant === "error"
                ? "bg-[color:var(--accent-press)] text-white"
                : "bg-[#211F1B] text-[#F9F9F9]",
            )}
          >
            {t.variant === "error" && <CircleAlert size={15} strokeWidth={2} className="shrink-0" />}
            {t.message}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used within a ToastProvider");
  return ctx;
}
