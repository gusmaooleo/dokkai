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
