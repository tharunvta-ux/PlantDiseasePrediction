"""
LLM provider layer.

Use `get_llm_provider()` to obtain the configured provider; route and
service code should depend only on `LLMProvider` and `LLMError`.
"""

from backend.services.llm.base import (
    LLMError,
    LLMInvalidResponseError,
    LLMNotConfiguredError,
    LLMProvider,
    LLMTimeoutError,
    LLMUnavailableError,
    LLMUpstreamError,
)
from backend.services.llm.factory import get_llm_provider

__all__ = [
    "LLMError",
    "LLMInvalidResponseError",
    "LLMNotConfiguredError",
    "LLMProvider",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "LLMUpstreamError",
    "get_llm_provider",
]
