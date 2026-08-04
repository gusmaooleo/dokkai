"use client";

/**
 * LLM provider card, faithful to the prototype's Settings screen
 * (`dokkai-ui-prototype.html` ~L413-430): provider tabs, a base-URL input
 * for Ollama or an API-key input for cloud providers, a chat-model field
 * and a Save button.
 *
 * Deviations from the prototype, per decision 12i: this is the ONLY
 * chat-LLM config (the prototype's separate Describe/Embedding model
 * selects are dropped — there's a single active provider/model).
 *
 * Feature 22 (C5) — the provider tabs are no longer a closed
 * ollama/openai/anthropic set: they're Ollama (unchanged, still its own
 * hardcoded local path) + `GET /config/llm/providers` (the built-ins the
 * registry knows about, incl. Gemini, plus anything registered in
 * `config/providers.json`) + a final "Other" tab accepting a free-form
 * provider id, model, key and base URL for any OpenAI-compatible endpoint
 * `validate_and_save_config` can still resolve without a code change.
 *
 * Two display decisions from that endpoint (see its docstring,
 * `src/controllers/config.py`):
 *   - A `registered_via_file` provider already carries its own base URL
 *     (and usually its own key) from `config/providers.json` — the API
 *     key input for one is never marked required unless `key_required`
 *     says the file's own `${ENV_VAR}` key genuinely isn't resolved yet,
 *     and the hint text says the key is pre-configured rather than
 *     "not set".
 *   - `default_base_url` (a built-in's default only, e.g. Gemini's
 *     non-guessable OpenAI-compat endpoint — always null for a
 *     file-registered provider, whose own baseUrl may legitimately carry
 *     credentials) is rendered read-only for whichever registered
 *     provider is selected, straight from the endpoint — no value is
 *     hardcoded here.
 *
 * Base-URL syntax (scheme, host, control characters, ...) is validated
 * ONLY server-side (`services/llm_provider.py::validate_base_url`) — an
 * earlier version of this card hand-rolled a client-side mirror of that
 * parser and it disagreed with the real one on a dozen edge cases (both
 * directions: accepting things the server rejects and rejecting things
 * it accepts). This repo has no frontend test runner to keep such a
 * mirror honest, so the "Other" tab only checks that a base URL was
 * entered at all (a plain non-empty check, not a syntax mirror) and lets
 * a malformed one come back as the server's own actionable 400 in a
 * toast — a settings form is saved rarely, so that round trip is cheap
 * compared to carrying a permanent drift surface.
 *
 * The backend (`services/llm_config.py::validate_and_save_config`) requires
 * a non-empty API key on EVERY save for a provider that requires one —
 * there's no "leave blank to keep the existing key" support for THOSE, so
 * the key input is always required for them, even if one is already
 * configured (reflected honestly below via a "configured" hint + a
 * required, always-empty-on-load password field). A `registered_via_file`
 * provider with its own key already resolved is the one case where a
 * blank key IS meaningful (it means "use the file's own key").
 *
 * The model field is a `<select>` when a source of model names exists for
 * the selected tab: `GET /config/llm/models`'s live catalog for the
 * currently ACTIVE provider (that endpoint only ever lists models for the
 * active one — no per-provider variant exists), or otherwise the selected
 * provider's declared `models[]` from `GET /config/llm/providers`
 * (`config/providers.json`, feature 22, C3/C5) — which is what lets a
 * not-yet-active file-registered provider get a dropdown too. Free text
 * only when neither source has anything for the selected tab.
 *
 * Save is admin-only (decision 12p) — non-admins get a read-only summary
 * of the current config instead of the interactive form.
 */

import { useCallback, useEffect, useState } from "react";

import { ApiError, configApi } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useToast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import type { AvailableModel, LLMConfig, LLMProviderInfo } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const DEFAULT_OLLAMA_URL = "http://localhost:11434";

/**
 * Sentinel tab id for the free-form "Other" provider entry. Never sent to
 * the API (the actual typed-in provider id is sent instead — see
 * `handleSave`).
 *
 * Deliberately `"@other"`, not e.g. `"__other__"`: the loader's provider-
 * id charset (`services/llm_provider.py`'s `_PROVIDER_ID_RE`) is
 * `[a-z0-9._-]+` — '@' is outside it, so `config/providers.json` can
 * NEVER register an id equal to this sentinel (the loader rejects it at
 * boot, same as a space or any other out-of-charset character). That
 * makes the two tabs structurally unable to collide, rather than relying
 * on two independently maintained regexes happening to agree — the
 * charset stays underscore-inclusive on the backend (`lm_studio` is a
 * legitimate id) precisely because THIS value is the one kept out of the
 * id space instead. See `scripts/test_providers_file.py`'s dedicated
 * check, which reads this exact constant back out of this file and
 * proves the loader still rejects it.
 */
const OTHER_TAB = "@other";

/**
 * Deliberate, hand-maintained copy of the built-in id/label list — used
 * ONLY as the degraded-mode fallback below when `GET /config/llm/providers`
 * itself is unreachable, never on the normal path. This is NOT a source
 * of truth (that's the registry in `services/llm_provider.py`, reached
 * through the endpoint) and it CAN go stale if a built-in is ever added
 * there without a matching edit here — an acceptable, bounded risk for a
 * fallback that only activates when the real listing can't be fetched at
 * all, versus the alternative of having no degraded mode. `key_required:
 * true` for all three (always true for the built-ins in practice, and
 * the safe direction to be wrong in — asks for a key the save might not
 * strictly need, rather than silently skipping one it does),
 * `default_base_url: null` (no note shown, rather than a hardcoded value
 * that could go stale on top of the id list already being one). This
 * exists so a real, already-configured `openai`/`gemini`/`anthropic`
 * config still resolves to its own tab instead of silently falling
 * through to "Other" with a blank, marked as *required*, base URL field
 * the admin never intended to fill in.
 */
const FALLBACK_BUILTIN_PROVIDERS: LLMProviderInfo[] = [
  { id: "openai", label: "OpenAI", key_required: true, registered_via_file: false, default_base_url: null, models: [] },
  { id: "gemini", label: "Gemini", key_required: true, registered_via_file: false, default_base_url: null, models: [] },
  { id: "anthropic", label: "Anthropic", key_required: true, registered_via_file: false, default_base_url: null, models: [] },
];

export function LlmCard({ onSaved }: { onSaved?: () => void }) {
  const { user } = useAuth();
  const { toast } = useToast();
  const canEdit = user?.role === "admin";

  const [config, setConfig] = useState<LLMConfig | null>(null);
  const [registeredProviders, setRegisteredProviders] = useState<LLMProviderInfo[]>([]);
  const [models, setModels] = useState<AvailableModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedTab, setSelectedTab] = useState<string>("ollama");
  const [customProviderId, setCustomProviderId] = useState("");
  const [baseUrl, setBaseUrl] = useState(DEFAULT_OLLAMA_URL);
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const cfg = await configApi.getLlm().catch((err) => {
        if (err instanceof ApiError && err.status === 404) return null;
        throw err;
      });
      const providersRes = await configApi.listProviders().catch(() => {
        toast(
          "Could not load the registered provider list — showing built-in providers only.",
          "error",
        );
        return { providers: FALLBACK_BUILTIN_PROVIDERS };
      });

      setConfig(cfg);
      setRegisteredProviders(providersRes.providers);
      setApiKey("");

      if (cfg === null) {
        setSelectedTab("ollama");
        setCustomProviderId("");
        setBaseUrl(DEFAULT_OLLAMA_URL);
        setModel("");
        setModels([]);
      } else {
        setModel(cfg.model);
        if (cfg.provider_name === "ollama") {
          setSelectedTab("ollama");
          setCustomProviderId("");
          setBaseUrl(cfg.base_url ?? DEFAULT_OLLAMA_URL);
        } else if (providersRes.providers.some((p) => p.id === cfg.provider_name)) {
          setSelectedTab(cfg.provider_name);
          setCustomProviderId("");
          setBaseUrl("");
        } else {
          setSelectedTab(OTHER_TAB);
          setCustomProviderId(cfg.provider_name);
          setBaseUrl(cfg.base_url ?? "");
        }
        try {
          const modelsRes = await configApi.listModels();
          setModels(modelsRes.models);
        } catch {
          setModels([]);
        }
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Failed to load LLM configuration");
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    load();
  }, [load]);

  const tabs: { id: string; label: string; tag: string }[] = [
    { id: "ollama", label: "Ollama", tag: "Local" },
    ...[...registeredProviders]
      .sort((a, b) => a.id.localeCompare(b.id))
      .map((p) => ({ id: p.id, label: p.label, tag: "Cloud" })),
    { id: OTHER_TAB, label: "Other", tag: "Custom" },
  ];

  const currentProviderInfo = registeredProviders.find((p) => p.id === selectedTab);
  const currentProviderName = selectedTab === OTHER_TAB ? customProviderId.trim() : selectedTab;
  const isLocalSelected = selectedTab === "ollama";
  const isActiveProvider =
    config !== null && config.provider_name === currentProviderName && config.is_local === isLocalSelected;

  /**
   * Model dropdown source, in priority order: (1) the ACTIVE provider's
   * live catalog (`GET /config/llm/models`, most accurate — it reflects
   * what the provider itself reports right now), (2) the selected
   * provider's declared `models[]` from `GET /config/llm/providers`
   * (`config/providers.json`) — this is what makes a not-yet-active
   * file-registered provider get a dropdown too, not just the active
   * one. `null` (free-text input) when neither source has anything.
   */
  const modelDropdownOptions: string[] | null =
    isActiveProvider && models.length > 0
      ? models.map((m) => m.name)
      : currentProviderInfo && currentProviderInfo.models.length > 0
        ? currentProviderInfo.models
        : null;

  function selectProvider(id: string) {
    setSelectedTab(id);
    setApiKey("");
    if (id !== OTHER_TAB) setCustomProviderId("");

    const providerName = id === OTHER_TAB ? customProviderId.trim() : id;
    if (config && config.provider_name === providerName && config.is_local === (id === "ollama")) {
      setModel(config.model);
      setBaseUrl(config.base_url ?? (id === "ollama" ? DEFAULT_OLLAMA_URL : ""));
    } else {
      setModel("");
      setBaseUrl(id === "ollama" ? DEFAULT_OLLAMA_URL : "");
    }
  }

  async function handleSave() {
    if (!model.trim()) {
      toast("Enter a model.", "error");
      return;
    }

    const providerName = currentProviderName;

    if (selectedTab === OTHER_TAB) {
      if (!providerName) {
        toast("Enter a provider id.", "error");
        return;
      }
      const lowerProviderName = providerName.toLowerCase();
      if (lowerProviderName === "ollama") {
        toast("Provider id 'ollama' is only reachable through the Ollama tab.", "error");
        return;
      }
      const alreadyRegistered = registeredProviders.find((p) => p.id === lowerProviderName);
      if (alreadyRegistered) {
        toast(`'${alreadyRegistered.label}' is already registered — use its own tab instead of Other.`, "error");
        return;
      }
      // Base URL syntax is validated server-side only (see the module
      // docstring) — this is just "was something entered at all", the
      // one part of validate_and_save_config's custom-provider rule
      // that's a plain presence check, not a parser to keep in sync.
      if (!baseUrl.trim()) {
        toast("A base URL is required for a custom provider.", "error");
        return;
      }
    } else if (selectedTab !== "ollama") {
      const keyRequired = currentProviderInfo ? currentProviderInfo.key_required : true;
      if (keyRequired && !apiKey.trim()) {
        toast(`An API key is required to save ${providerName}.`, "error");
        return;
      }
    }

    setSaving(true);
    try {
      const saved = await configApi.setLlm({
        is_local: selectedTab === "ollama",
        provider_data: {
          provider_name: providerName,
          model: model.trim(),
          key: selectedTab === "ollama" ? undefined : apiKey.trim() || undefined,
          base_url:
            selectedTab === "ollama"
              ? baseUrl.trim() || undefined
              : selectedTab === OTHER_TAB
                ? baseUrl.trim()
                : undefined,
        },
      });
      toast(`LLM config saved: ${saved.provider_name}/${saved.model}`);
      await load();
      onSaved?.();
    } catch (err) {
      toast(err instanceof ApiError ? err.detail : "Failed to save LLM configuration", "error");
    } finally {
      setSaving(false);
    }
  }

  if (!canEdit) {
    return (
      <div className="rounded-[14px] border border-border bg-surface p-5 shadow-[var(--shadow-sm)]">
        {loading && <p className="text-[13.5px] text-muted-foreground">Loading…</p>}
        {!loading && error && <p className="text-[13.5px] text-muted-foreground">{error}</p>}
        {!loading && !error && !config && (
          <p className="text-[13.5px] text-[color:var(--text-faint)]">No LLM provider configured.</p>
        )}
        {!loading && !error && config && (
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap items-baseline gap-2">
              <span className="text-[14.5px] font-semibold text-foreground capitalize">
                {config.provider_name}
              </span>
              <span className="font-mono text-[10.5px] text-[color:var(--text-faint)]">
                {config.is_local ? "Local" : "Cloud"}
              </span>
            </div>
            <div className="grid grid-cols-[repeat(auto-fit,minmax(180px,1fr))] gap-3 font-mono text-[13px] text-muted-foreground">
              <span>
                model: <span className="text-foreground">{config.model}</span>
              </span>
              {config.is_local ? (
                <span>
                  base url: <span className="text-foreground">{config.base_url}</span>
                </span>
              ) : (
                <span>key: {config.has_key ? "configured" : "not set"}</span>
              )}
            </div>
            <p className="text-[11.5px] text-[color:var(--text-faint)]">
              Only admins can change the LLM configuration.
            </p>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3.5">
      <div className="flex flex-wrap gap-2.5">
        {tabs.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => selectProvider(t.id)}
            className={cn(
              "flex min-w-[118px] flex-1 flex-col gap-0.5 rounded-xl border px-[15px] py-[13px] text-left",
              selectedTab === t.id
                ? "border-[color:var(--accent-ring)] bg-[color:var(--accent-weak)]"
                : "border-border bg-surface",
            )}
          >
            <span className="text-[14.5px] font-semibold text-foreground">{t.label}</span>
            <span className="font-mono text-[10.5px] text-[color:var(--text-faint)]">{t.tag}</span>
          </button>
        ))}
      </div>

      <div className="rounded-[14px] border border-border bg-surface p-5 shadow-[var(--shadow-sm)]">
        {loading ? (
          <p className="text-[13.5px] text-muted-foreground">Loading…</p>
        ) : error ? (
          <p className="text-[13.5px] text-muted-foreground">{error}</p>
        ) : (
          <>
            {selectedTab === "ollama" ? (
              <div className="mb-[18px]">
                <Label htmlFor="llm-base-url" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
                  Base URL
                </Label>
                <Input
                  id="llm-base-url"
                  value={baseUrl}
                  onChange={(e) => setBaseUrl(e.target.value)}
                  placeholder={DEFAULT_OLLAMA_URL}
                  className="h-[42px] rounded-[10px] border-border-strong bg-background px-3 font-mono text-[14px]"
                />
              </div>
            ) : selectedTab === OTHER_TAB ? (
              <>
                <div className="mb-[18px]">
                  <Label htmlFor="llm-provider-id" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
                    Provider id
                  </Label>
                  <Input
                    id="llm-provider-id"
                    value={customProviderId}
                    onChange={(e) => setCustomProviderId(e.target.value)}
                    placeholder="e.g. groq, together, my-gateway"
                    className="h-[42px] rounded-[10px] border-border-strong bg-background px-3 font-mono text-[14px]"
                  />
                </div>
                <div className="mb-[18px]">
                  <Label htmlFor="llm-base-url" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
                    Base URL
                  </Label>
                  <Input
                    id="llm-base-url"
                    value={baseUrl}
                    onChange={(e) => setBaseUrl(e.target.value)}
                    placeholder="http://localhost:1234/v1"
                    className="h-[42px] rounded-[10px] border-border-strong bg-background px-3 font-mono text-[14px]"
                  />
                  <p className="mt-1.5 text-[11.5px] text-muted-foreground">
                    Required — a full http:// or https:// URL for an OpenAI-compatible endpoint.
                  </p>
                </div>
                <div className="mb-[18px]">
                  <Label htmlFor="llm-api-key" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
                    API key
                  </Label>
                  <Input
                    id="llm-api-key"
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="sk-… (optional)"
                    className="h-[42px] rounded-[10px] border-border-strong bg-background px-3 font-mono text-[14px]"
                  />
                  <p className="mt-1.5 text-[11.5px] text-muted-foreground">
                    Optional — leave blank for an endpoint that doesn&apos;t require one.
                  </p>
                </div>
              </>
            ) : (
              <div className="mb-[18px]">
                <Label htmlFor="llm-api-key" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
                  API key
                </Label>
                <Input
                  id="llm-api-key"
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-…"
                  className="h-[42px] rounded-[10px] border-border-strong bg-background px-3 font-mono text-[14px]"
                />
                <p className="mt-1.5 text-[11.5px] text-muted-foreground">
                  {currentProviderInfo?.registered_via_file
                    ? currentProviderInfo.key_required
                      ? "Registered in config/providers.json, but its key isn't resolved yet (its ${ENV_VAR} isn't set) — export it, or enter a key here to override it for this save."
                      : "Already has its own key configured in config/providers.json — leave blank to use it, or enter one here to override it for this save."
                    : isActiveProvider && config?.has_key
                      ? "A key is already configured for this provider — the key is required on every save, so re-enter it to save any change."
                      : "Required to save this provider."}
                </p>
                {currentProviderInfo?.default_base_url && (
                  <p className="mt-1.5 text-[11.5px] text-[color:var(--text-faint)]">
                    Uses {currentProviderInfo.default_base_url} by default.
                  </p>
                )}
              </div>
            )}

            <div className="mb-1">
              <Label htmlFor="llm-model" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
                Chat model
              </Label>
              {modelDropdownOptions ? (
                <Select value={model} onValueChange={setModel}>
                  <SelectTrigger id="llm-model" className="h-[42px] w-full rounded-[10px] border-border-strong bg-background px-3 font-mono text-[13.5px]">
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent>
                    {modelDropdownOptions.map((name) => (
                      <SelectItem key={name} value={name}>
                        {name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              ) : (
                <>
                  <Input
                    id="llm-model"
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    placeholder="e.g. gpt-4o-mini, claude-sonnet-4, qwen2.5-coder:3b"
                    className="h-[42px] rounded-[10px] border-border-strong bg-background px-3 font-mono text-[13.5px]"
                  />
                  <p className="mt-1.5 text-[11.5px] text-muted-foreground">
                    {isActiveProvider
                      ? "No known models for this provider — type the model name."
                      : "No declared model list for this provider yet — type the model name, or save it once to pick from its live models next time."}
                  </p>
                </>
              )}
            </div>

            <div className="flex justify-end gap-2.5 border-t border-border pt-[14px] mt-[16px]">
              <Button type="button" onClick={handleSave} disabled={saving}>
                {saving ? "Saving…" : "Save changes"}
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
