"""
DTOs for the LLM configuration endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProviderData(BaseModel):
    provider_name: str = Field(..., description="'openai', 'anthropic', or 'ollama'")
    model: str = Field(..., description="Model name, e.g. 'gpt-4o-mini', 'claude-sonnet-4', 'llama3'")
    key: str = Field(default="", description="API key (empty for local providers)")
    base_url: str | None = Field(default=None, description="Custom base URL override")


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
