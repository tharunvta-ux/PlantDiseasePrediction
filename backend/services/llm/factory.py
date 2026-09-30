"""
backend/services/llm/factory.py

Builds the configured LLM provider from settings.
"""

from __future__ import annotations

import threading

from backend.config import Settings, get_settings
from backend.services.llm.base import LLMNotConfiguredError, LLMProvider

_provider: LLMProvider | None = None
_provider_key: tuple | None = None
_lock = threading.Lock()


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """
    Return a provider for the current settings (reused while unchanged).

    Raises:
        LLMNotConfiguredError: Missing key or unsupported provider.
    """

    global _provider, _provider_key

    settings = settings or get_settings()

    if not settings.llm_api_key:
        raise LLMNotConfiguredError("LLM_API_KEY is not set.")

    key = (
        settings.llm_provider,
        settings.llm_api_key,
        settings.llm_model,
        settings.llm_fallback_model,
        settings.llm_timeout_seconds,
        settings.llm_temperature,
    )

    with _lock:
        if _provider is not None and _provider_key == key:
            return _provider

        if settings.llm_provider == "gemini":
            from backend.services.llm.gemini_provider import GeminiProvider

            _provider = GeminiProvider(
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                fallback_model=settings.llm_fallback_model,
                timeout_seconds=settings.llm_timeout_seconds,
                temperature=settings.llm_temperature,
            )
        else:
            raise LLMNotConfiguredError(
                f"Unsupported LLM_PROVIDER: {settings.llm_provider!r}"
            )

        _provider_key = key

        return _provider
