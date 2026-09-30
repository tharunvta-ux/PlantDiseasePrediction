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
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
    LLMUpstreamError,
)

logger = logging.getLogger(__name__)

# Original request + one retry (the SDK default is 5 attempts, which
# makes the UI wait far too long on outages).
RETRY_ATTEMPTS = 2

MAX_OUTPUT_TOKENS = 2048

# Overloaded / rate-limited: worth trying the fallback model.
FALLBACK_STATUSES = frozenset({429, 503})


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
        fallback_model: str | None = None,
    ) -> None:
        """
        Args:
            base_url: Optional API endpoint override (proxies, tests).
            fallback_model: Model tried once when the primary model is
                overloaded or rate-limited (HTTP 503 / 429).
        """

        if not api_key:
            raise LLMNotConfiguredError("Gemini API key is missing.")

        self.model = model
        self.fallback_model = (
            fallback_model if fallback_model and fallback_model != model else None
        )
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
    ) -> LLMResponse:
        """
        Generate JSON text, falling back to `fallback_model` once if the
        primary model is overloaded or rate-limited.
        """

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_json_schema=response_schema,
            # No tools are used; disabling AFC also silences SDK warnings.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

        try:
            return self._generate(self.model, user_prompt, config)

        except LLMUpstreamError as exc:
            if not self.fallback_model or exc.status not in FALLBACK_STATUSES:
                raise

            logger.warning(
                "Model %s unavailable (%s); falling back to %s",
                self.model,
                exc.status,
                self.fallback_model,
            )

            return self._generate(self.fallback_model, user_prompt, config)

    def _generate(
        self,
        model: str,
        user_prompt: str,
        config: types.GenerateContentConfig,
    ) -> LLMResponse:
        """One generate_content call with error mapping."""

        try:
            response = self._client.models.generate_content(
                model=model,
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

        return LLMResponse(text=text, model=model)


def _map_api_error(exc: errors.APIError) -> LLMUpstreamError:
    """Translate a Gemini HTTP error into a stable error code."""

    status = exc.code or 0

    def error(code: str | None = None, message: str | None = None) -> LLMUpstreamError:
        return LLMUpstreamError(str(exc), code=code, public_message=message, status=status)

    if status in (401, 403) or (status == 400 and "API key" in str(exc.message)):
        return error(
            "LLM_AUTH_FAILED",
            "The treatment guidance service rejected the server's API "
            "key. The administrator needs to check LLM_API_KEY.",
        )

    if status == 402:
        return error(
            "LLM_BILLING",
            "The treatment guidance service is unavailable: the LLM "
            "account has no remaining credit. The administrator needs "
            "to check billing in Google AI Studio.",
        )

    if status == 404:
        return error(
            "LLM_MODEL_NOT_FOUND",
            "The configured LLM model is not available. The "
            "administrator needs to check LLM_MODEL.",
        )

    if status == 429:
        return error(
            "LLM_RATE_LIMITED",
            "The treatment guidance service is busy (rate limit "
            "reached). Please try again in a minute.",
        )

    if status == 503:
        return error(
            "LLM_BUSY",
            "The treatment guidance service is temporarily overloaded. "
            "Please try again in a minute.",
        )

    return error()
