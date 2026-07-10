"use client";

/**
 * LLM provider card, faithful to the prototype's Settings screen
 * (`dokkai-ui-prototype.html` ~L413-430): provider tabs (Ollama/OpenAI/
 * Anthropic), a base-URL input for Ollama or an API-key input for cloud
 * providers, a chat-model field and a Save button.
 *
 * Deviations from the prototype, per decision 12i: this is the ONLY
 * chat-LLM config (the prototype's separate Describe/Embedding model
 * selects are dropped — there's a single active provider/model).
 *
 * The backend (`services/llm_config.py::validate_and_save_config`) requires
 * a non-empty API key on EVERY save for a cloud provider — there's no
 * "leave blank to keep the existing key" support, so the key input is
 * always required when saving openai/anthropic, even if one is already
 * configured (reflected honestly below via a "configured" hint + a
 * required, always-empty-on-load password field).
 *
 * The model field is a `<select>` sourced from `GET /config/llm/models`
 * only when the selected tab matches the currently ACTIVE provider — that
 * endpoint only ever lists models for the active provider (no per-provider
 * variant exists). Any other tab gets a free-text model input instead.
 *
 * Save is admin-only (decision 12p) — non-admins get a read-only summary
 * of the current config instead of the interactive form.
 */

import { useCallback, useEffect, useState } from "react";

import { ApiError, configApi } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useToast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import type { AvailableModel, LLMConfig } from "@/lib/types";
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

type ProviderId = "ollama" | "openai" | "anthropic";

const DEFAULT_OLLAMA_URL = "http://localhost:11434";

const PROVIDERS: { id: ProviderId; label: string; tag: string }[] = [
  { id: "ollama", label: "Ollama", tag: "Local" },
  { id: "openai", label: "OpenAI", tag: "Cloud" },
  { id: "anthropic", label: "Anthropic", tag: "Cloud" },
];

export function LlmCard({ onSaved }: { onSaved?: () => void }) {
  const { user } = useAuth();
  const { toast } = useToast();
  const canEdit = user?.role === "admin";

  const [config, setConfig] = useState<LLMConfig | null>(null);
  const [models, setModels] = useState<AvailableModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [provider, setProvider] = useState<ProviderId>("ollama");
  const [baseUrl, setBaseUrl] = useState(DEFAULT_OLLAMA_URL);
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const cfg = await configApi.getLlm();
      setConfig(cfg);
      setProvider(cfg.provider_name as ProviderId);
      setModel(cfg.model);
      setBaseUrl(cfg.base_url ?? DEFAULT_OLLAMA_URL);
      setApiKey("");
      try {
        const modelsRes = await configApi.listModels();
        setModels(modelsRes.models);
      } catch {
        setModels([]);
      }
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        setConfig(null);
        setModels([]);
      } else {
        setError(err instanceof ApiError ? err.detail : "Failed to load LLM configuration");
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  function selectProvider(id: ProviderId) {
    setProvider(id);
    setApiKey("");
    if (config && config.provider_name === id) {
      setModel(config.model);
      setBaseUrl(config.base_url ?? DEFAULT_OLLAMA_URL);
    } else {
      setModel("");
      if (id === "ollama") setBaseUrl(DEFAULT_OLLAMA_URL);
    }
  }

  const isActiveProvider = config?.provider_name === provider;

  async function handleSave() {
    if (!model.trim()) {
      toast("Enter a model.", "error");
      return;
    }
    if (provider !== "ollama" && !apiKey.trim()) {
      toast(`An API key is required to save ${provider}.`, "error");
      return;
    }

    setSaving(true);
    try {
      const saved = await configApi.setLlm({
        is_local: provider === "ollama",
        provider_data: {
          provider_name: provider,
          model: model.trim(),
          key: provider === "ollama" ? undefined : apiKey.trim(),
          base_url: provider === "ollama" ? baseUrl.trim() || undefined : undefined,
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
        {PROVIDERS.map((pt) => (
          <button
            key={pt.id}
            type="button"
            onClick={() => selectProvider(pt.id)}
            className={cn(
              "flex min-w-[118px] flex-1 flex-col gap-0.5 rounded-xl border px-[15px] py-[13px] text-left",
              provider === pt.id
                ? "border-[color:var(--accent-ring)] bg-[color:var(--accent-weak)]"
                : "border-border bg-surface",
            )}
          >
            <span className="text-[14.5px] font-semibold text-foreground">{pt.label}</span>
            <span className="font-mono text-[10.5px] text-[color:var(--text-faint)]">{pt.tag}</span>
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
            {provider === "ollama" ? (
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
                  {isActiveProvider && config?.has_key
                    ? "A key is already configured for this provider — the key is required on every save, so re-enter it to save any change."
                    : "Required to save this provider."}
                </p>
              </div>
            )}

            <div className="mb-1">
              <Label htmlFor="llm-model" className="mb-[7px] block text-[12.5px] font-semibold text-muted-foreground">
                Chat model
              </Label>
              {isActiveProvider && models.length > 0 ? (
                <Select value={model} onValueChange={setModel}>
                  <SelectTrigger id="llm-model" className="h-[42px] w-full rounded-[10px] border-border-strong bg-background px-3 font-mono text-[13.5px]">
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent>
                    {models.map((m) => (
                      <SelectItem key={m.name} value={m.name}>
                        {m.name}
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
                    The model list is only available for the currently active provider — save this
                    provider once to pick from its models next time.
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
