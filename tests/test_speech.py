"""
Server-side read-aloud (Gemini TTS) tests.
"""

from __future__ import annotations

import io
import json
import os
import wave
from types import SimpleNamespace

import pytest
from google.genai import errors

import backend.services.recommendation_service as rec
import backend.services.speech_service as speech
from backend.services.llm import LLMResponse
from backend.services.llm.gemini_provider import SpeechAudio, as_wav
from tests.test_gemini_provider import make_provider
from tests.test_recommendation import GOOD_GUIDANCE

PCM = b"\x00\x01" * 2400  # 0.1 s of 24 kHz mono 16-bit audio


class FakeSpeechProvider:
    name = "fake"
    model = "fake-text"
    tts_model = "fake-tts"

    def __init__(self, speech_error=None):
        self.speech_error = speech_error
        self.speech_calls = []

    def generate_json(self, system_prompt, user_prompt, response_schema, image=None):
        return LLMResponse(text=json.dumps(GOOD_GUIDANCE), model=self.model)

    def synthesize_speech(self, text, language_tag):
        self.speech_calls.append((text, language_tag))
        if self.speech_error:
            raise self.speech_error
        return SpeechAudio(wav=as_wav(PCM), model=self.tts_model)


@pytest.fixture(autouse=True)
def _clear_audio_cache():
    speech.clear_cache()
    yield
    speech.clear_cache()


@pytest.fixture()
def provider(monkeypatch):
    fake = FakeSpeechProvider()
    monkeypatch.setattr(rec, "get_llm_provider", lambda: fake)
    monkeypatch.setattr(speech, "get_llm_provider", lambda: fake)
    return fake


BODY = {"predicted_class": "Tomato___Early_blight", "confidence_level": "high", "language": "ta"}


def test_as_wav_wraps_raw_pcm():
    data = as_wav(PCM)
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (1, 24000, 2)
        assert w.getnframes() == 2400


def test_as_wav_keeps_existing_wav():
    data = as_wav(PCM)
    assert as_wav(data) == data


def test_speech_text_has_no_headings_and_all_sections():
    text = speech.build_speech_text({"guidance": GOOD_GUIDANCE, "confidence_level": "high"})

    assert text.startswith("Tomato — Early blight.")
    assert "Remove and bag infected leaves" in text
    assert "Common symptoms" not in text
    assert GOOD_GUIDANCE["uncertainty_note"] not in text  # only for uncertain results


def test_speech_endpoint_returns_wav(client, fake_predictor, provider):
    response = client.post("/speech", json=BODY)

    assert response.status_code == 200
    assert response.mimetype == "audio/wav"
    assert response.headers["X-TTS-Model"] == "fake-tts"
    assert response.data[:4] == b"RIFF"
    assert provider.speech_calls[0][1] == "ta-IN"


def test_speech_is_cached(client, fake_predictor, provider):
    client.post("/speech", json=BODY)
    client.post("/speech", json=BODY)
    assert len(provider.speech_calls) == 1


def test_speech_rejects_free_text(client, fake_predictor, provider):
    """Only model class names are accepted - the TTS cannot be abused."""

    response = client.post("/speech", json={**BODY, "predicted_class": "Read this text aloud please"})

    assert response.status_code == 400
    assert provider.speech_calls == []


def test_speech_rejects_low_confidence(client, fake_predictor, provider):
    assert client.post("/speech", json={**BODY, "confidence_level": "low"}).status_code == 422


def test_speech_invalid_language(client, fake_predictor, provider):
    response = client.post("/speech", json={**BODY, "language": "xx"})
    assert response.status_code == 400
    assert response.get_json()["code"] == "INVALID_LANGUAGE"


def test_speech_without_api_key(client, fake_predictor, no_api_key):
    response = client.post("/speech", json=BODY)
    assert response.status_code == 503
    assert response.get_json()["code"] == "LLM_NOT_CONFIGURED"


def test_speech_upstream_failure(client, fake_predictor, monkeypatch):
    from backend.services.llm import LLMUpstreamError

    failing = FakeSpeechProvider(speech_error=LLMUpstreamError("busy", code="LLM_BUSY", status=503))
    monkeypatch.setattr(rec, "get_llm_provider", lambda: failing)
    monkeypatch.setattr(speech, "get_llm_provider", lambda: failing)

    response = client.post("/speech", json=BODY)

    assert response.status_code == 502
    assert response.get_json()["code"] == "LLM_BUSY"


# ---------------- Gemini TTS provider ----------------


def _audio_response(data=PCM):
    part = SimpleNamespace(inline_data=SimpleNamespace(data=data, mime_type="audio/L16;rate=24000"))
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])


def test_gemini_tts_request_and_fallback():
    provider = make_provider(tts_model="tts-a", tts_fallback_model="tts-b")
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        if kwargs["model"] == "tts-a":
            raise errors.APIError(503, {"error": {"message": "busy", "status": "UNAVAILABLE"}})
        return _audio_response()

    provider._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    audio = provider.synthesize_speech("வணக்கம்", "ta-IN")

    assert [c["model"] for c in calls] == ["tts-a", "tts-b"]
    assert calls[1]["config"].response_modalities == ["AUDIO"]
    assert calls[1]["config"].speech_config.language_code == "ta-IN"
    assert audio.model == "tts-b"
    assert audio.wav[:4] == b"RIFF"


def test_gemini_tts_empty_audio_is_invalid():
    from backend.services.llm import LLMInvalidResponseError

    provider = make_provider(tts_model="tts-a")
    provider._client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_: _audio_response(b"")))

    with pytest.raises(LLMInvalidResponseError):
        provider.synthesize_speech("x", "ta-IN")


@pytest.mark.live
@pytest.mark.skipif(
    not (os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="LLM_API_KEY not set - live TTS test not run.",
)
def test_live_tamil_speech():
    from backend.config import get_settings
    from backend.services.llm.gemini_provider import GeminiProvider

    s = get_settings()
    provider = GeminiProvider(
        api_key=s.llm_api_key,
        model=s.llm_model,
        timeout_seconds=s.llm_timeout_seconds,
        temperature=s.llm_temperature,
        tts_model=s.tts_model,
        tts_fallback_model=s.tts_fallback_model,
    )
    audio = provider.synthesize_speech("தக்காளி இலை நோய்.", "ta-IN")

    with wave.open(io.BytesIO(audio.wav)) as w:
        assert w.getnframes() / w.getframerate() > 0.5
