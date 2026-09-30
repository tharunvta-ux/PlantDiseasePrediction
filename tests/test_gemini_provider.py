"""
GeminiProvider tests.

Network failures use REAL sockets (no mocks):
- a closed local port -> connection refused
- a local server that accepts but never answers -> timeout
API error mapping uses a stub client because provoking specific HTTP
statuses from Google is not possible offline.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from types import SimpleNamespace

import pytest
from google.genai import errors

from backend.services.llm import (
    LLMInvalidResponseError,
    LLMNotConfiguredError,
    LLMTimeoutError,
    LLMUnavailableError,
    LLMUpstreamError,
)
from backend.services.llm.gemini_provider import GeminiProvider

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


def make_provider(**kwargs) -> GeminiProvider:
    options = dict(api_key="test-key", model="gemini-test", timeout_seconds=2, temperature=0)
    options.update(kwargs)
    return GeminiProvider(**options)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_missing_key_rejected():
    with pytest.raises(LLMNotConfiguredError):
        make_provider(api_key="")


def test_connection_refused_is_unavailable():
    provider = make_provider(base_url=f"http://127.0.0.1:{free_port()}")

    with pytest.raises(LLMUnavailableError):
        provider.generate_json("s", "u", SCHEMA)


def test_silent_server_times_out():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    port = server.getsockname()[1]
    connections = []

    def accept_forever():
        while True:
            try:
                conn, _ = server.accept()
                connections.append(conn)  # hold open, never reply
            except OSError:
                return

    threading.Thread(target=accept_forever, daemon=True).start()

    try:
        provider = make_provider(base_url=f"http://127.0.0.1:{port}", timeout_seconds=1)
        start = time.time()

        with pytest.raises(LLMTimeoutError):
            provider.generate_json("s", "u", SCHEMA)

        assert time.time() - start < 20  # retries are bounded
    finally:
        server.close()
        for conn in connections:
            conn.close()


def _stub(provider, fn):
    provider._client = SimpleNamespace(models=SimpleNamespace(generate_content=fn))


@pytest.mark.parametrize(
    "status, code",
    [
        (401, "LLM_AUTH_FAILED"),
        (402, "LLM_BILLING"),
        (403, "LLM_AUTH_FAILED"),
        (404, "LLM_MODEL_NOT_FOUND"),
        (429, "LLM_RATE_LIMITED"),
        (500, "LLM_UPSTREAM_ERROR"),
        (503, "LLM_BUSY"),
    ],
)
def test_api_errors_are_mapped(status, code):
    provider = make_provider()  # no fallback configured

    def fail(**_kwargs):
        raise errors.APIError(status, {"error": {"message": "boom", "status": "X"}})

    _stub(provider, fail)

    with pytest.raises(LLMUpstreamError) as info:
        provider.generate_json("s", "u", SCHEMA)

    assert info.value.code == code


def test_empty_response_is_invalid():
    provider = make_provider()
    _stub(provider, lambda **_: SimpleNamespace(text=None, candidates=[]))

    with pytest.raises(LLMInvalidResponseError):
        provider.generate_json("s", "u", SCHEMA)


def test_request_uses_json_schema_and_system_prompt():
    provider = make_provider()
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(text='{"a": "b"}')

    _stub(provider, capture)

    response = provider.generate_json("SYSTEM", "USER", SCHEMA)

    assert response.text == '{"a": "b"}'
    assert response.model == "gemini-test"
    assert seen["model"] == "gemini-test"
    assert seen["contents"] == "USER"
    assert seen["config"].system_instruction == "SYSTEM"
    assert seen["config"].response_mime_type == "application/json"
    assert seen["config"].response_json_schema == SCHEMA


# ---------------- fallback model ----------------


def _failing_primary(status, calls):
    def generate(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "gemini-test":
            raise errors.APIError(status, {"error": {"message": "busy", "status": "X"}})
        return SimpleNamespace(text='{"a": "b"}')

    return generate


@pytest.mark.parametrize("status", [503, 429])
def test_falls_back_when_primary_overloaded(status):
    provider = make_provider(fallback_model="gemini-fallback")
    calls = []
    _stub(provider, _failing_primary(status, calls))

    response = provider.generate_json("s", "u", SCHEMA)

    assert calls == ["gemini-test", "gemini-fallback"]
    assert response.model == "gemini-fallback"


@pytest.mark.parametrize("status", [400, 401, 402, 404, 500])
def test_no_fallback_for_other_errors(status):
    provider = make_provider(fallback_model="gemini-fallback")
    calls = []
    _stub(provider, _failing_primary(status, calls))

    with pytest.raises(LLMUpstreamError):
        provider.generate_json("s", "u", SCHEMA)

    assert calls == ["gemini-test"]


def test_fallback_failure_is_reported():
    provider = make_provider(fallback_model="gemini-fallback")

    def always_busy(**_kwargs):
        raise errors.APIError(503, {"error": {"message": "busy", "status": "X"}})

    _stub(provider, always_busy)

    with pytest.raises(LLMUpstreamError) as info:
        provider.generate_json("s", "u", SCHEMA)

    assert info.value.code == "LLM_BUSY"


# ---------------- live (opt-in) ----------------


@pytest.mark.live
@pytest.mark.skipif(
    not (os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="LLM_API_KEY not set - live Gemini test not run.",
)
def test_live_gemini_recommendation():
    from backend.config import get_settings
    from backend.services.recommendation_service import generate_recommendation

    settings = get_settings()
    provider = GeminiProvider(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        fallback_model=settings.llm_fallback_model,
        timeout_seconds=settings.llm_timeout_seconds,
        temperature=settings.llm_temperature,
    )

    result = generate_recommendation("Tomato___Late_blight", "high", [], provider=provider)
    guidance = result["guidance"]

    assert guidance["recommended_treatment"]
    assert guidance["when_to_seek_expert_help"]
