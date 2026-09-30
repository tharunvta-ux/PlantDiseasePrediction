"""
backend/services/llm/gemini_provider.py

Google Gemini implementation of `LLMProvider` (google-genai SDK).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import httpx
from google import genai
from google.genai import errors, types

from backend.services.llm.base import (
    LLMInvalidResponseError,
    LLMNotConfiguredError,
    LLMTimeoutError,
    LLMUnavailableError,
    LLMUpstreamError,
)

logger = logging.getLogger(__name__)

# Original request + one retry (the SDK default is 5 attempts, which
# makes the UI wait far too long on outages).
RETRY_ATTEMPTS = 2

MAX_OUTPUT_TOKENS = 2048


class GeminiProvider:
    """Calls the Gemini API with structured JSON output."""

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float,
        temperature: float,
        base_url: str | None = None,
    ) -> None:
        """
        Args:
            base_url: Optional API endpoint override (proxies, tests).
        """

        if not api_key:
            raise LLMNotConfiguredError("Gemini API key is missing.")

        self.model = model
        self.temperature = temperature

        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                base_url=base_url,
                timeout=int(timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(attempts=RETRY_ATTEMPTS),
            ),
        )

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Dict[str, Any],
    ) -> str:
        """Generate JSON text; map SDK/network errors to LLMError."""

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_json_schema=response_schema,
        )

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config,
            )

        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # Could not establish a connection at all (offline, DNS,
            # firewall, refused) - a network failure, not a slow API.
            logger.warning("Gemini unreachable: %s", exc)
            raise LLMUnavailableError(str(exc)) from exc

        except httpx.TimeoutException as exc:
            logger.warning("Gemini request timed out: %s", exc)
            raise LLMTimeoutError(str(exc)) from exc

        except httpx.TransportError as exc:
            logger.warning("Gemini network error: %s", exc)
            raise LLMUnavailableError(str(exc)) from exc

        except errors.APIError as exc:
            logger.warning("Gemini API error %s: %s", exc.code, exc.message)
            raise _map_api_error(exc) from exc

        text = getattr(response, "text", None)

        if not text:
            finish = None
            if getattr(response, "candidates", None):
                finish = response.candidates[0].finish_reason
            raise LLMInvalidResponseError(
                f"Empty response from Gemini (finish_reason={finish})."
            )

        return text


def _map_api_error(exc: errors.APIError) -> LLMUpstreamError:
    """Translate a Gemini HTTP error into a stable error code."""

    status = exc.code or 0

    if status in (401, 403) or (status == 400 and "API key" in str(exc.message)):
        return LLMUpstreamError(
            str(exc),
            code="LLM_AUTH_FAILED",
            public_message=(
                "The treatment guidance service rejected the server's API "
                "key. The administrator needs to check LLM_API_KEY."
            ),
        )

    if status == 402:
        return LLMUpstreamError(
            str(exc),
            code="LLM_BILLING",
            public_message=(
                "The treatment guidance service is unavailable: the LLM "
                "account has no remaining credit. The administrator needs "
                "to check billing in Google AI Studio."
            ),
        )

    if status == 404:
        return LLMUpstreamError(
            str(exc),
            code="LLM_MODEL_NOT_FOUND",
            public_message=(
                "The configured LLM model is not available. The "
                "administrator needs to check LLM_MODEL."
            ),
        )

    if status == 429:
        return LLMUpstreamError(
            str(exc),
            code="LLM_RATE_LIMITED",
            public_message=(
                "The treatment guidance service is busy (rate limit "
                "reached). Please try again in a minute."
            ),
        )

    return LLMUpstreamError(str(exc))
