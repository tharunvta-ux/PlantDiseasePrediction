"""
backend/services/llm/gemini_provider.py

Google Gemini implementation of `LLMProvider` (google-genai SDK).
"""

from __future__ import annotations

import io
import logging
import re
import threading
import time
import wave
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, TypeVar

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

# Overloaded / rate-limited / retired (404): try the next model in the chain.
FALLBACK_STATUSES = frozenset({404, 429, 503})

# The SDK retries these on the same model. 429/503 are left out: moving
# to the next free model is faster than waiting on an exhausted one.
SDK_RETRY_STATUSES = [408, 500, 502, 504]

# How long a model is skipped after 429 / 503 when the error does not
# say when to retry. Free-tier daily quotas say so ("retry in 16h49m").
DEFAULT_COOLDOWN_SECONDS = {404: 24 * 3600.0, 429: 60.0, 503: 20.0}
MAX_COOLDOWN_SECONDS = 24 * 3600.0

_RETRY_IN = re.compile(r"retry in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", re.IGNORECASE)
_RETRY_DELAY = re.compile(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", re.IGNORECASE)

T = TypeVar("T")

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


def model_chain(primary: str | None, fallbacks: str | Sequence[str] | None) -> List[str]:
    """
    Primary model followed by the fallbacks, without duplicates.
    `fallbacks` may be a comma-separated string.
    """

    if isinstance(fallbacks, str):
        fallbacks = fallbacks.split(",")

    chain: List[str] = []

    for name in [primary, *(fallbacks or [])]:
        name = (name or "").strip()
        if name and name not in chain:
            chain.append(name)

    return chain


def retry_after_seconds(detail: str, status: int | None) -> float:
    """Cooldown for a rate-limited / overloaded model, from the error text."""

    match = _RETRY_DELAY.search(detail)
    if match:
        seconds = float(match.group(1))
    else:
        match = _RETRY_IN.search(detail)
        if match and any(match.groups()):
            hours, minutes, secs = match.groups()
            seconds = int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(secs or 0)
        else:
            seconds = DEFAULT_COOLDOWN_SECONDS.get(status or 0, 30.0)

    return min(max(seconds, 1.0), MAX_COOLDOWN_SECONDS)


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
        fallback_model: str | Sequence[str] | None = None,
        tts_model: str | None = None,
        tts_fallback_model: str | Sequence[str] | None = None,
        tts_timeout_seconds: float = 60.0,
    ) -> None:
        """
        Args:
            base_url: Optional API endpoint override (proxies, tests).
            fallback_model: Models tried in order when the previous one is
                overloaded or rate-limited (HTTP 503 / 429). A list or a
                comma-separated string. Free-tier quotas are per model, so
                each free model adds its own daily allowance.
            tts_model / tts_fallback_model: Text-to-speech models (same).
        """

        if not api_key:
            raise LLMNotConfiguredError("Gemini API key is missing.")

        self.model = model
        self.models = model_chain(model, fallback_model)
        self.tts_model = tts_model
        self.tts_models = model_chain(tts_model, tts_fallback_model) if tts_model else []
        self.temperature = temperature

        # model -> time.monotonic() until which it is skipped
        self._cooldown_until: Dict[str, float] = {}
        self._cooldown_lock = threading.Lock()

        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                base_url=base_url,
                timeout=int(timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(
                    attempts=RETRY_ATTEMPTS,
                    http_status_codes=SDK_RETRY_STATUSES,
                ),
            ),
        )

        # Speech responses are large audio payloads: own, longer timeout
        # and a single attempt (a retry would double the user's wait).
        self._tts_client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                base_url=base_url,
                timeout=int(tts_timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
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
        Generate JSON text (optionally about an image), moving down the
        model chain while models are overloaded or rate-limited.
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

        return self._first_available(
            self.models,
            lambda model: self._generate(model, contents, config),
            "Model",
        )

    def _first_available(self, models: List[str], call: Callable[[str], T], label: str) -> T:
        """
        Run `call` on the first model in `models` that is not cooling down;
        on 404 / 429 / 503 cool that model down and try the next one.
        """

        now = time.monotonic()

        with self._cooldown_lock:
            ready = [m for m in models if self._cooldown_until.get(m, 0.0) <= now]

        if not ready:
            raise LLMUpstreamError(
                "Every configured model is rate-limited or overloaded.",
                code="LLM_RATE_LIMITED",
                public_message=(
                    "The AI service has reached its free usage limit or is busy. "
                    "Please try again later."
                ),
                status=429,
            )

        for index, model in enumerate(ready):
            try:
                return call(model)

            except LLMUpstreamError as exc:
                if exc.status not in FALLBACK_STATUSES:
                    raise

                cooldown = retry_after_seconds(exc.detail, exc.status)

                with self._cooldown_lock:
                    self._cooldown_until[model] = time.monotonic() + cooldown

                if index == len(ready) - 1:
                    raise

                logger.warning(
                    "%s %s unavailable (%s), skipped for %.0f s; falling back to %s",
                    label,
                    model,
                    exc.status,
                    cooldown,
                    ready[index + 1],
                )

        raise AssertionError("unreachable")

    def synthesize_speech(self, text: str, language_tag: str) -> SpeechAudio:
        """
        Text-to-speech, moving down the TTS model chain on 503/429.

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

        return self._first_available(
            self.tts_models,
            lambda model: self._speech(model, text, config),
            "TTS model",
        )

    def _speech(
        self,
        model: str,
        text: str,
        config: types.GenerateContentConfig,
    ) -> SpeechAudio:
        """One TTS call; returns WAV bytes."""

        response = self._request(model, text, config, client=self._tts_client)

        try:
            part = response.candidates[0].content.parts[0].inline_data
            data = part.data
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMInvalidResponseError("No audio in TTS response.") from exc

        if not data:
            raise LLMInvalidResponseError("Empty audio in TTS response.")

        return SpeechAudio(wav=as_wav(data), model=model)

    def _request(
        self,
        model: str,
        contents: Any,
        config: types.GenerateContentConfig,
        client: Any = None,
    ) -> Any:
        """generate_content with network / API error mapping."""

        client = client or self._client

        try:
            return client.models.generate_content(
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
