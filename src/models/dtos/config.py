"""
DTOs for the LLM configuration endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProviderData(BaseModel):
    provider_name: str = Field(
        ...,
        description=(
            "Ignored when is_local=true (Ollama is always used, forced "
            "regardless of this value — Ollama is not reachable through "
            "is_local=false). When is_local=false: one of the built-in "
            "remote ids ('openai', 'gemini', 'anthropic') or any custom id "
            "paired with base_url (an OpenAI-compatible endpoint, e.g. "
            "Groq, LM Studio, vLLM)"
        ),
    )
    model: str = Field(..., description="Model name, e.g. 'gpt-4o-mini', 'claude-sonnet-4', 'llama3'")
    key: str = Field(
        default="",
        description=(
            "API key. Empty for local (Ollama) providers and optional for "
            "custom endpoints that don't require one; required for the "
            "built-in remote providers (openai, gemini, anthropic)"
        ),
    )
    base_url: str | None = Field(
        default=None,
        description=(
            "Base URL override — a full http:// or https:// URL with a "
            "host. Required when is_local=false and provider_name isn't "
            "one of the built-in remote ids (openai, gemini, anthropic)"
        ),
    )


class LLMConfigRequest(BaseModel):
    is_local: bool = Field(..., description="True for Ollama (local), False for remote providers")
    provider_data: ProviderData


class LLMConfigResponse(BaseModel):
    is_local: bool
    provider_name: str
    model: str
    has_key: bool
    base_url: str | None = None


class ProviderHealthResponse(BaseModel):
    provider_name: str
    model: str
    is_local: bool
    status: str  # "healthy" | "unhealthy"
    message: str


class AvailableModel(BaseModel):
    name: str
    provider: str


class ModelsListResponse(BaseModel):
    provider_name: str
    models: list[AvailableModel]


class LLMProviderInfo(BaseModel):
    """
    One entry of ``GET /config/llm/providers`` (feature 22, C5) — lets the
    Settings UI discover which ids it can offer without hardcoding a
    closed list. Never carries a key or anything derived from one — only
    ``key_required`` (a boolean), matching the ``has_key``-only discipline
    the rest of this module already follows.
    """

    id: str
    label: str = Field(
        description=(
            "Human-readable display name for the UI — the built-in's fixed "
            "name ('OpenAI', 'Gemini', 'Anthropic'), or a file-registered "
            "provider's id exactly as written in config/providers.json "
            "(original case preserved, not a mangled str.title())."
        )
    )
    key_required: bool = Field(
        description=(
            "Whether a key must be supplied when saving this provider. "
            "True for the built-in remotes (openai, gemini, anthropic) "
            "and for a file-registered provider whose declared "
            "'${ENV_VAR}' key isn't currently set. False for a "
            "file-registered provider that already resolves a key "
            "(a literal apiKey or an already-exported env var) — the UI "
            "must not demand one for it."
        )
    )
    registered_via_file: bool = Field(
        description=(
            "True for a provider registered from config/providers.json "
            "rather than one of the three built-ins. Such a provider "
            "already carries its own base URL (and usually its own key) "
            "from the file — the UI should say so rather than asking for "
            "either, and never present it as 'unconfigured'."
        )
    )
    default_base_url: str | None = Field(
        default=None,
        description=(
            "The DEFAULT base URL for a BUILT-IN id only (e.g. Gemini's "
            "non-guessable OpenAI-compat endpoint) — always null for a "
            "file-registered provider, deliberately: its own baseUrl is "
            "admin-authored and may legitimately carry embedded Basic-auth "
            "userinfo, and this endpoint is readable by any authenticated "
            "user, not just an admin. Informational/read-only — display "
            "only, never used to build a request."
        ),
    )
    models: list[str] = Field(
        default_factory=list,
        description=(
            "This provider's declared models[] from config/providers.json "
            "(feature 22, C3) — empty for a built-in (they never populate "
            "it) or a file-registered provider that declared none, both of "
            "which mean 'free-text model field' to the UI. Unlike "
            "default_base_url, NOT restricted to built-ins: a model name "
            "carries no secret and isn't admin-authored connection info, "
            "so there's no equivalent leak surface to guard against."
        ),
    )


class LLMProvidersListResponse(BaseModel):
    providers: list[LLMProviderInfo]
