"use client";

/**
 * Provider health card, faithful to the prototype's health row
 * (`dokkai-ui-prototype.html` ~L428): a status dot + label and a
 * "Run health check" button with the prototype's spinner state
 * (`healthChecking`). Runs once on mount, then again on demand.
 */

import { useCallback, useEffect, useState } from "react";
import { LoaderCircle } from "lucide-react";

import { ApiError, configApi } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { ProviderHealth } from "@/lib/types";
import { Button } from "@/components/ui/button";

// `refreshKey` re-runs the check when the LLM config is saved, so the
// displayed provider/model never goes stale after a provider switch.
export function HealthCard({ refreshKey = 0 }: { refreshKey?: number }) {
  const [health, setHealth] = useState<ProviderHealth | null>(null);
  const [checking, setChecking] = useState(false);
  const [notConfigured, setNotConfigured] = useState(false);

  const check = useCallback(async () => {
    setChecking(true);
    try {
      const res = await configApi.health();
      setHealth(res);
      setNotConfigured(false);
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        setHealth(null);
        setNotConfigured(true);
      }
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    check();
  }, [check, refreshKey]);

  const status = notConfigured ? "none" : health ? health.status : "unknown";

  const dotClass = cn(
    "size-[9px] shrink-0 rounded-full",
    status === "healthy" && "bg-[#3E9C93]",
    status === "unhealthy" && "bg-accent",
    (status === "none" || status === "unknown") && "bg-[color:var(--text-faint)]",
  );

  const labelClass = cn(
    "font-mono text-xs font-semibold",
    status === "healthy" && "text-[#2f8f82]",
    status === "unhealthy" && "text-accent",
    (status === "none" || status === "unknown") && "font-normal text-[color:var(--text-faint)]",
  );

  const label = notConfigured
    ? "No LLM configured"
    : health
      ? health.status === "healthy"
        ? `Healthy · ${health.provider_name}/${health.model}`
        : `Unhealthy · ${health.message}`
      : "Not checked yet";

  return (
    <div className="flex items-center justify-between gap-3 rounded-[14px] border border-border bg-surface p-5 shadow-[var(--shadow-sm)]">
      <div>
        <div className="text-[13px] font-semibold text-foreground">Provider health</div>
        <div className="mt-[5px] inline-flex items-center gap-[7px]">
          <span className={dotClass} />
          <span className={labelClass}>{label}</span>
        </div>
      </div>
      <Button type="button" variant="outline" onClick={check} disabled={checking}>
        {checking ? (
          <>
            <LoaderCircle size={15} strokeWidth={2.4} className="animate-[dk-spin_0.7s_linear_infinite]" />
            Checking…
          </>
        ) : (
          "Check now"
        )}
      </Button>
    </div>
  );
}
