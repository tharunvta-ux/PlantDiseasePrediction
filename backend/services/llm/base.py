"""
backend/services/llm/base.py

Provider-neutral LLM interface and error types.

Each error carries a stable `code` and an HTTP status so routes can
return consistent JSON errors without knowing the provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass(frozen=True)
class LLMResponse:
    """Raw provider output plus the model that actually produced it."""

    text: str
    model: str


class LLMError(Exception):
    """Base class for all LLM failures."""

    code = "LLM_ERROR"
    http_status = 502
    public_message = "The treatment guidance service failed. Please try again."

    def __init__(self, detail: str = "") -> None:
        super().__init__(detail or self.public_message)
        self.detail = detail


class LLMNotConfiguredError(LLMError):
    """No API key / unsupported provider configured."""

    code = "LLM_NOT_CONFIGURED"
    http_status = 503
    public_message = (
        "Treatment guidance is not available: the LLM service is not "
        "configured on the server."
    )


class LLMTimeoutError(LLMError):
    """Provider did not answer in time."""

    code = "LLM_TIMEOUT"
    http_status = 504
    public_message = "The treatment guidance service timed out. Please try again."


class LLMUnavailableError(LLMError):
    """Network failure reaching the provider."""

    code = "LLM_UNAVAILABLE"
    http_status = 503
    public_message = (
        "Could not reach the treatment guidance service. Check the "
        "server's internet connection and try again."
    )


class LLMUpstreamError(LLMError):
    """Provider returned an error (auth, quota, server error...)."""

    code = "LLM_UPSTREAM_ERROR"
    http_status = 502

    def __init__(
        self,
        detail: str = "",
        code: str | None = None,
        public_message: str | None = None,
        status: int | None = None,
    ) -> None:
        super().__init__(detail)
        self.status = status  # provider HTTP status, if known
        if code:
            self.code = code
        if public_message:
            self.public_message = public_message


class LLMInvalidResponseError(LLMError):
    """Provider answered but the output failed validation."""

    code = "LLM_INVALID_RESPONSE"
    http_status = 502
    public_message = (
        "The treatment guidance service returned an invalid response. "
        "Please try again."
    )


class LLMProvider(Protocol):
    """Minimal interface every provider implements."""

    name: str
    model: str

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Dict[str, Any],
    ) -> LLMResponse:
        """
        Generate a JSON document conforming to `response_schema`.

        Returns:
            The raw JSON text (validated by the caller) and the model used.

        Raises:
            LLMError subclasses on any failure.
        """
        ...
