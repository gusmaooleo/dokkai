"use client";

/**
 * Settings screen, faithful to the prototype's Settings screen
 * (`dokkai-ui-prototype.html` ~L413-433): an "LLM provider" section (tabs +
 * form + save) and a provider-health row. The Account card is new
 * (decision 12n) — not in the prototype — and visible to every role.
 */

import { useState } from "react";

import { AccountCard } from "./account-card";
import { HealthCard } from "./health-card";
import { LlmCard } from "./llm-card";

export function SettingsScreen() {
  const [healthRefreshKey, setHealthRefreshKey] = useState(0);

  return (
    <div className="mx-auto max-w-[720px] px-3.5 py-[18px] sm:px-6.5 sm:py-[30px]">
      <div className="mb-3 font-mono text-[11px] font-semibold tracking-[0.07em] text-[color:var(--text-faint)] uppercase">
        LLM provider
      </div>
      <div className="mb-8">
        <LlmCard onSaved={() => setHealthRefreshKey((k) => k + 1)} />
      </div>

      <div className="mb-3 font-mono text-[11px] font-semibold tracking-[0.07em] text-[color:var(--text-faint)] uppercase">
        Provider health
      </div>
      <div className="mb-8">
        <HealthCard refreshKey={healthRefreshKey} />
      </div>

      <div className="mb-3 font-mono text-[11px] font-semibold tracking-[0.07em] text-[color:var(--text-faint)] uppercase">
        Account
      </div>
      <AccountCard />
    </div>
  );
}
