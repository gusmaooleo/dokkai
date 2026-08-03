"""
LLM configuration service with storage abstraction.

Manages the active LLM provider configuration. Uses an in-memory store
by default, but the ``ConfigStore`` interface makes it trivial to swap
in a database-backed implementation later.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from services.db import DatabaseUnavailableError, get_pool
from services.llm_provider import (
    effective_base_url,
    get_provider,
    get_registered_provider_ids,
    key_env_for,
    validate_base_url,
    LLMProvider,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Current LLM provider configuration."""

    is_local: bool
    provider_name: str
    model: str
    key: str = ""
    base_url: str | None = None


# -----------------------------------------------------------------------
# Abstract store interface — implement this for DB support later
# -----------------------------------------------------------------------

class ConfigStore(ABC):
    """
    Interface for persisting configuration.

    Implementations:
        - InMemoryConfigStore (current)
        - DatabaseConfigStore (future — same interface, zero code changes)
    """

    @abstractmethod
    def get_llm_config(self) -> LLMConfig | None:
        """Return the current LLM config, or None if not set."""
        ...

    @abstractmethod
    def set_llm_config(self, config: LLMConfig) -> None:
        """Persist the LLM config."""
        ...

    @abstractmethod
    def clear_llm_config(self) -> None:
        """Remove the current LLM config."""
        ...


# -----------------------------------------------------------------------
# In-memory implementation
# -----------------------------------------------------------------------

class InMemoryConfigStore(ConfigStore):
    """Stores config in application memory. Resets on restart."""

    def __init__(self) -> None:
        self._llm_config: LLMConfig | None = None

    def get_llm_config(self) -> LLMConfig | None:
        return self._llm_config

    def set_llm_config(self, config: LLMConfig) -> None:
        self._llm_config = config

    def clear_llm_config(self) -> None:
        self._llm_config = None


# -----------------------------------------------------------------------
# Singleton config store instance
# -----------------------------------------------------------------------

_config_store: ConfigStore = InMemoryConfigStore()


def get_config_store() -> ConfigStore:
    """Return the global config store instance."""
    return _config_store


def set_config_store(store: ConfigStore) -> None:
    """Replace the global config store (e.g., switch to DB-backed)."""
    global _config_store
    _config_store = store


# -----------------------------------------------------------------------
# Postgres persistence (write-through) — the DB row is the durable copy of
# the active config; InMemoryConfigStore remains the runtime source of truth
# and is what every read (GET /config/llm, get_active_provider) consults.
# -----------------------------------------------------------------------

async def load_persisted_config() -> LLMConfig | None:
    """
    Load the persisted LLM config from Postgres, if a row exists.

    Never raises: returns None if there is no saved row yet, or if Postgres
    is unreachable at all — a pool that never got created (get_pool raises
    DatabaseUnavailableError) *or* a pool that was created at boot but whose
    connection has since died (raw asyncpg/OS errors from ``fetchrow``, e.g.
    ``asyncpg.InterfaceError``, which isn't even a ``PostgresError``). Callers
    (boot) should fall back to the env seed in either case. Logs a warning
    when the reason is a down Postgres.
    """
    try:
        pool = await get_pool()
        row = await pool.fetchrow(
            "SELECT is_local, provider_name, model, key, base_url FROM llm_config WHERE id = 1"
        )
    except DatabaseUnavailableError as e:
        logger.warning(
            "LLM config: Postgres unavailable, falling back to env seed: %s", e
        )
        return None
    except Exception as e:
        logger.warning(
            "LLM config: failed to load persisted config, falling back to env seed: %s", e
        )
        return None

    if row is None:
        return None

    return LLMConfig(
        is_local=row["is_local"],
        provider_name=row["provider_name"],
        model=row["model"],
        key=row["key"] or "",
        base_url=row["base_url"],
    )


async def save_persisted_config(config: LLMConfig) -> None:
    """
    Upsert the active LLM config into the single-row ``llm_config`` table.

    Raises if Postgres is unreachable: :class:`DatabaseUnavailableError` when
    no pool has ever been created, or a raw asyncpg/OS error when a
    previously-created pool's connection has since died. Callers decide how
    to degrade (the in-memory config is the runtime source of truth
    regardless).
    """
    pool = await get_pool()
    await pool.execute(
        """
        INSERT INTO llm_config (id, is_local, provider_name, model, key, base_url, updated_at)
        VALUES (1, $1, $2, $3, $4, $5, now())
        ON CONFLICT (id) DO UPDATE SET
            is_local = EXCLUDED.is_local,
            provider_name = EXCLUDED.provider_name,
            model = EXCLUDED.model,
            key = EXCLUDED.key,
            base_url = EXCLUDED.base_url,
            updated_at = now()
        """,
        config.is_local,
        config.provider_name,
        config.model,
        config.key,
        config.base_url,
    )


# -----------------------------------------------------------------------
# High-level helpers
# -----------------------------------------------------------------------

async def validate_and_save_config(
    is_local: bool,
    provider_name: str,
    model: str,
    key: str = "",
    base_url: str | None = None,
) -> tuple[bool, str]:
    """
    Validate the LLM config and save it if valid.

    For local (Ollama) configs, verifies that:
    1. Ollama is reachable
    2. The requested model is actually installed

    For remote configs, verifies that:
    1. The provider id and base_url combination is resolvable (a built-in
       id, or a custom id paired with an http(s) base_url)
    2. An API key is provided, when the provider requires one (built-in
       remotes always do; a custom OpenAI-compatible endpoint may not)
    3. The provider is reachable

    Returns (success, message).
    """
    name = provider_name.lower().strip()

    if is_local:
        # Force Ollama for local configs
        name = "ollama"
        resolved_base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )

        # Check Ollama is reachable. The probe itself is a USE-time HTTP call
        # (services.llm_provider.effective_base_url rewrites localhost ->
        # host.docker.internal inside the API container), but the value we
        # go on to PERSIST below is resolved_base_url, unrewritten — the
        # 14-A invariant is "never rewrite the stored value", not "never
        # probe a rewritten one".
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{effective_base_url(resolved_base_url)}/api/tags")
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            return False, f"Cannot reach Ollama at {resolved_base_url}: {e}"

        # Check the model is installed
        installed_models = [m["name"] for m in data.get("models", [])]
        # Ollama model names can have `:latest` suffix
        model_found = any(
            m == model or m.startswith(f"{model}:") or model.startswith(f"{m.split(':')[0]}")
            for m in installed_models
        )

        if not model_found:
            return False, (
                f"Model '{model}' is not installed in Ollama. "
                f"Installed models: {', '.join(installed_models) or '(none)'}"
            )

        config = LLMConfig(
            is_local=True,
            provider_name="ollama",
            model=model,
            key="",
            base_url=resolved_base_url,
        )

    else:
        # Remote provider — resolved through the registry built in
        # services.llm_provider (feature 22): a built-in id (openai,
        # gemini, anthropic) uses its own default base URL and requires
        # its key; any other id is servable through the openai-completions
        # shape as soon as a base_url is supplied, and its key is optional
        # (LM Studio/vLLM/llama.cpp and similar local servers commonly
        # don't require one). Every check below runs before any network
        # call, so a malformed config never costs a live request.
        if not name:
            return False, "provider_name is required"

        if name == "ollama":
            # Ollama is always local — it isn't one of this branch's
            # registered ids (get_registered_provider_ids() excludes it)
            # and isn't a valid "custom" id either; a base_url here would
            # silently skip the is_local branch's /api/tags model check.
            return False, (
                "provider_name 'ollama' is only reachable with "
                "is_local=true (Ollama is always local) — retry with "
                "is_local=true."
            )

        registered = get_registered_provider_ids()

        if base_url:
            normalized_base_url, url_error = validate_base_url(base_url)
            if url_error:
                return False, url_error
            base_url = normalized_base_url

        if name not in registered and not base_url:
            return False, (
                f"Unknown provider '{provider_name}'. Registered providers: "
                f"{', '.join(sorted(registered))}. A custom provider needs "
                "an OpenAI-compatible base_url."
            )

        if key_env_for(name) and not key:
            return False, f"API key is required for {name}"

        # Quick connectivity check
        try:
            provider = get_provider(name, api_key=key, base_url=base_url)
            is_healthy, msg = await provider.health_check(model)
            if not is_healthy:
                return False, msg
        except Exception as e:
            return False, f"Provider validation failed: {e}"

        config = LLMConfig(
            is_local=False,
            provider_name=name,
            model=model,
            key=key,
            base_url=base_url,
        )

    store = get_config_store()
    store.set_llm_config(config)

    # Write-through to Postgres. Best-effort: the in-memory config (the
    # runtime source of truth) is already updated above, so a down Postgres
    # doesn't block the request — just log it, since the change won't
    # survive a restart until persistence is back. Broad except: a pool
    # created at boot but since disconnected raises raw asyncpg/OS errors
    # from ``execute`` (not DatabaseUnavailableError, which only fires when
    # no pool exists at all) — a failed persist must never fail the POST.
    try:
        await save_persisted_config(config)
    except Exception as e:
        logger.warning(
            "LLM config: saved in memory but failed to persist to Postgres "
            "(will not survive a restart): %s", e,
        )

    return True, f"LLM config saved: {name}/{model}"


def get_active_provider() -> tuple[LLMProvider, str]:
    """
    Build an LLMProvider from the current saved config.

    Returns (provider_instance, model_name).
    Raises ValueError if no config is set.
    """
    store = get_config_store()
    config = store.get_llm_config()

    if config is None:
        raise ValueError(
            "No LLM provider configured. "
            "POST to /config/llm to set one up first."
        )

    provider = get_provider(
        config.provider_name,
        api_key=config.key,
        base_url=config.base_url,
    )
    return provider, config.model
