"""
backend/services/llm/gemini_provider.py

Google Gemini implementation of `LLMProvider` (google-genai SDK).
"""

from __future__ import annotations

import io
import logging
import wave
from dataclasses import dataclass
from typing import Any, Dict

import httpx
from google import genai
from google.genai import errors, types

from backend.services.llm.base import (
    ImageInput,
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

# Prebuilt Gemini TTS voice (multilingual).
TTS_VOICE = "Kore"

TTS_SAMPLE_RATE = 24000


@dataclass(frozen=True)
class SpeechAudio:
    """Synthesised speech."""

    wav: bytes
    model: str


def as_wav(data: bytes, sample_rate: int = TTS_SAMPLE_RATE) -> bytes:
    """Return WAV bytes; raw 16-bit mono PCM gets a WAV header."""

    if data[:4] == b"RIFF":
        return data

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(data)

    return buffer.getvalue()


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
        tts_model: str | None = None,
        tts_fallback_model: str | None = None,
    ) -> None:
        """
        Args:
            base_url: Optional API endpoint override (proxies, tests).
            fallback_model: Model tried once when the primary model is
                overloaded or rate-limited (HTTP 503 / 429).
            tts_model / tts_fallback_model: Text-to-speech models.
        """

        if not api_key:
            raise LLMNotConfiguredError("Gemini API key is missing.")

        self.model = model
        self.fallback_model = (
            fallback_model if fallback_model and fallback_model != model else None
        )
        self.tts_model = tts_model
        self.tts_fallback_model = (
            tts_fallback_model if tts_fallback_model and tts_fallback_model != tts_model else None
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
        image: ImageInput | None = None,
    ) -> LLMResponse:
        """
        Generate JSON text (optionally about an image), falling back to
        `fallback_model` once if the primary model is overloaded or
        rate-limited.
        """

        contents: Any = user_prompt

        if image is not None:
            contents = [
                types.Part.from_bytes(data=image.data, mime_type=image.mime_type),
                user_prompt,
            ]

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
            return self._generate(self.model, contents, config)

        except LLMUpstreamError as exc:
            if not self.fallback_model or exc.status not in FALLBACK_STATUSES:
                raise

            logger.warning(
                "Model %s unavailable (%s); falling back to %s",
                self.model,
                exc.status,
                self.fallback_model,
            )

            return self._generate(self.fallback_model, contents, config)

    def synthesize_speech(self, text: str, language_tag: str) -> SpeechAudio:
        """
        Text-to-speech with the TTS model (falls back once on 503/429).

        Returns:
            WAV audio (24 kHz mono 16-bit).
        """

        if not self.tts_model:
            raise LLMNotConfiguredError("No text-to-speech model configured.")

        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                language_code=language_tag,
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE)
                ),
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        try:
            return self._speech(self.tts_model, text, config)

        except LLMUpstreamError as exc:
            if not self.tts_fallback_model or exc.status not in FALLBACK_STATUSES:
                raise

            logger.warning(
                "TTS model %s unavailable (%s); falling back to %s",
                self.tts_model,
                exc.status,
                self.tts_fallback_model,
            )

            return self._speech(self.tts_fallback_model, text, config)

    def _speech(
        self,
        model: str,
        text: str,
        config: types.GenerateContentConfig,
    ) -> SpeechAudio:
        """One TTS call; returns WAV bytes."""

        response = self._request(model, text, config)

        try:
            part = response.candidates[0].content.parts[0].inline_data
            data = part.data
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMInvalidResponseError("No audio in TTS response.") from exc

        if not data:
            raise LLMInvalidResponseError("Empty audio in TTS response.")

        return SpeechAudio(wav=as_wav(data), model=model)

    def _request(self, model: str, contents: Any, config: types.GenerateContentConfig) -> Any:
        """generate_content with network / API error mapping."""

        try:
            return self._client.models.generate_content(
                model=model,
                contents=contents,
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

    def _generate(
        self,
        model: str,
        contents: Any,
        config: types.GenerateContentConfig,
    ) -> LLMResponse:
        """One generate_content call returning JSON text."""

        response = self._request(model, contents, config)

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
