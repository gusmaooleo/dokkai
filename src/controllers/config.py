"""
LLM configuration API endpoints.

POST /config/llm            — set the LLM provider configuration
GET  /config/llm            — get the current configuration
GET  /config/llm/providers  — list registry-servable provider ids for the UI
GET  /config/llm/models     — list available models for the current provider
GET  /config/llm/health     — check provider connectivity
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from models.dtos.config import (
    LLMConfigRequest,
    LLMConfigResponse,
    LLMProviderInfo,
    LLMProvidersListResponse,
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
from services.llm_provider import (
    default_base_url_for,
    display_name_for,
    get_builtin_provider_ids,
    get_registered_provider_ids,
    key_env_for,
    models_for,
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


@router.get("/llm/providers", response_model=LLMProvidersListResponse)
async def list_providers():
    """
    List the provider ids the registry can serve directly (feature 22,
    C5): the built-ins (openai, gemini, anthropic) plus anything
    registered from ``config/providers.json``, sourced from
    ``get_registered_provider_ids()`` — the same registry
    ``validate_and_save_config`` resolves ``provider_name`` against, so a
    ``key_required=false`` id here never demands a key that the server
    would then also not require. ``registered_via_file`` is derived from
    ``get_builtin_provider_ids()`` (code, frozen before any file loads) —
    never a second, driftable copy of the three built-in ids.

    Read-only, same auth posture as the other GET /config/llm* endpoints
    (any authenticated user — writes are the only admin-gated part of this
    router). Never returns a key, a key value, or anything derived from
    one. Excludes ``ollama`` — it isn't part of this registry (see
    ``get_registered_provider_ids``'s docstring) and the Settings UI keeps
    its own hardcoded local/Ollama path unchanged.

    Does NOT include a custom "Other" entry (free provider id + base_url)
    — that stays a client-side-only option the UI offers regardless of
    what this endpoint returns, per feature 22's design (an unregistered
    id remains servable as long as a base_url is supplied).

    ``models`` is this id's declared ``models[]`` from
    ``config/providers.json`` (``models_for()``) — lets the Settings
    card's model dropdown work for a not-yet-active file-registered
    provider, not only the currently active one (which is all
    ``GET /config/llm/models`` alone can do). Populated for any
    registered id, not restricted to built-ins the way
    ``default_base_url`` is: a model NAME carries no secret.
    """
    providers = [
        LLMProviderInfo(
            id=pid,
            label=display_name_for(pid) or pid,
            key_required=key_env_for(pid) is not None,
            registered_via_file=pid not in get_builtin_provider_ids(),
            default_base_url=default_base_url_for(pid),
            models=list(models_for(pid)),
        )
        for pid in sorted(get_registered_provider_ids())
    ]
    return LLMProvidersListResponse(providers=providers)


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
