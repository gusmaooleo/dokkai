"""
LLM configuration API endpoints.

POST /config/llm          — set the LLM provider configuration
GET  /config/llm          — get the current configuration
GET  /config/llm/models   — list available models for the current provider
GET  /config/llm/health   — check provider connectivity
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from models.dtos.config import (
    LLMConfigRequest,
    LLMConfigResponse,
    ProviderHealthResponse,
    ModelsListResponse,
    AvailableModel,
)
from services.auth import require_auth, require_role
from services.llm_config import (
    get_config_store,
    validate_and_save_config,
    get_active_provider,
)

router = APIRouter(prefix="/config", dependencies=[Depends(require_auth)])

# Global LLM config writes are admin-only (6k).
require_admin = require_role("admin")


@router.post("/llm", response_model=LLMConfigResponse, dependencies=[Depends(require_admin)])
async def set_llm_config(data: LLMConfigRequest):
    """
    Configure the LLM provider.

    When ``is_local=true``, the system will look for an Ollama instance
    at the default port and verify the model is installed.

    When ``is_local=false``, it validates connectivity with the remote
    provider; ``provider_data.key`` is required for the built-in remote
    providers (openai, gemini, anthropic) and optional for a custom
    OpenAI-compatible endpoint.
    """
    success, message = await validate_and_save_config(
        is_local=data.is_local,
        provider_name=data.provider_data.provider_name,
        model=data.provider_data.model,
        key=data.provider_data.key,
        base_url=data.provider_data.base_url,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    config = get_config_store().get_llm_config()
    if config is None:
        raise HTTPException(status_code=404, detail="No LLM provider configured. POST to /config/llm first.")
    return LLMConfigResponse(
        is_local=config.is_local,
        provider_name=config.provider_name,
        model=config.model,
        has_key=bool(config.key),
        base_url=config.base_url,
    )


@router.get("/llm", response_model=LLMConfigResponse)
async def get_llm_config():
    """Return the current LLM configuration."""
    config = get_config_store().get_llm_config()

    if config is None:
        raise HTTPException(
            status_code=404,
            detail="No LLM provider configured. POST to /config/llm first.",
        )

    return LLMConfigResponse(
        is_local=config.is_local,
        provider_name=config.provider_name,
        model=config.model,
        has_key=bool(config.key),
        base_url=config.base_url,
    )


@router.get("/llm/models", response_model=ModelsListResponse)
async def list_models():
    """List available models for the currently configured provider."""
    try:
        provider, _ = get_active_provider()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        model_names = await provider.list_models()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to list models: {e}",
        )

    config = get_config_store().get_llm_config()
    if config is None:
        raise HTTPException(status_code=404, detail="No LLM provider configured. POST to /config/llm first.")

    return ModelsListResponse(
        provider_name=config.provider_name,
        models=[
            AvailableModel(name=name, provider=config.provider_name)
            for name in model_names
        ],
    )


@router.get("/llm/health", response_model=ProviderHealthResponse)
async def check_health():
    """Check connectivity with the currently configured LLM provider."""
    config = get_config_store().get_llm_config()

    if config is None:
        raise HTTPException(
            status_code=404,
            detail="No LLM provider configured. POST to /config/llm first.",
        )

    try:
        provider, _ = get_active_provider()
        is_healthy, message = await provider.health_check(config.model)
    except Exception as e:
        is_healthy = False
        message = str(e)

    return ProviderHealthResponse(
        provider_name=config.provider_name,
        model=config.model,
        is_local=config.is_local,
        status="healthy" if is_healthy else "unhealthy",
        message=message,
    )
