"""
LLM configuration service with storage abstraction.

Manages the active LLM provider configuration. Uses an in-memory store
by default, but the ``ConfigStore`` interface makes it trivial to swap
in a database-backed implementation later.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from services.llm_provider import get_provider, LLMProvider


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
    1. An API key is provided
    2. The provider is reachable

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

        # Check Ollama is reachable
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{resolved_base_url}/api/tags")
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
        # Remote provider
        if name not in ("openai", "anthropic"):
            return False, f"Unknown remote provider: '{provider_name}'. Use 'openai' or 'anthropic'."

        if not key:
            return False, f"API key is required for {name}"

        # Quick connectivity check
        try:
            provider = get_provider(name, api_key=key, base_url=base_url)
            is_healthy, msg = await provider.health_check()
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
