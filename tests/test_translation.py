"""
Translation of plan / case texts and the read-aloud endpoints.
"""

from __future__ import annotations

import json

import pytest

import backend.services.speech_service as speech
import backend.services.translation_service as translation
from backend.services.llm import LLMResponse, LLMTimeoutError
from backend.services.llm.gemini_provider import SpeechAudio, as_wav
from backend.services.localize_service import localize_cases
from backend.services.ui_text import UI_STRINGS
from tests.test_tracking_api import check_in, register, start_case

PCM = b"\x00\x01" * 2400


class FakeTranslator:
    """Prefixes "TA:" (keeps numbers); `break_numbers` drops all digits."""

    name = "fake"
    model = "fake-translate"
    tts_model = "fake-tts"

    def __init__(self, break_numbers=False, error=None):
        self.break_numbers = break_numbers
        self.error = error
        self.batches = []
        self.spoken = []

    def generate_json(self, system_prompt, user_prompt, response_schema, image=None):
        if self.error:
            raise self.error
        texts = json.loads(user_prompt[user_prompt.index("{"):])["texts"]
        self.batches.append(texts)
        out = [
            "TA:" + ("".join(c for c in t if not c.isdigit()) if self.break_numbers else t)
            for t in texts
        ]
        return LLMResponse(text=json.dumps({"translations": out}), model=self.model)

    def synthesize_speech(self, text, language_tag):
        self.spoken.append((text, language_tag))
        return SpeechAudio(wav=as_wav(PCM), model=self.tts_model)


@pytest.fixture()
def translator(monkeypatch):
    fake = FakeTranslator()
    monkeypatch.setattr(translation, "get_llm_provider", lambda: fake)
    monkeypatch.setattr(speech, "get_llm_provider", lambda: fake)
    speech.clear_cache()
    yield fake
    speech.clear_cache()


# ---------- translation service ----------


@pytest.mark.parametrize(
    "source, translated, safe",
    [
        ("Spray 2.5 g per litre", "TA 2.5 g", True),
        ("Spray 2.5 g per litre", "TA 25 g", False),
        ("Check again in {days}", "TA {days}", True),
        ("Check again in {days}", "TA {dias}", False),
        ("Call 1800-180-1551", "TA 1800-180-1551", True),
        ("Anything", "", False),
    ],
)
def test_numbers_and_placeholders_must_survive(source, translated, safe):
    assert translation.is_safe_translation(source, translated) is safe


def test_english_is_not_sent(translator):
    assert translation.translate_texts(["Hello"], "en") == {"Hello": "Hello"}
    assert translator.batches == []


def test_translations_are_cached(translator):
    first = translation.translate_texts(["Remove leaves", "Water at the base"], "ta")
    second = translation.translate_texts(["Remove leaves"], "ta")

    assert first == {"Remove leaves": "TA:Remove leaves", "Water at the base": "TA:Water at the base"}
    assert second == {"Remove leaves": "TA:Remove leaves"}
    assert len(translator.batches) == 1


def test_translation_changing_a_dose_is_rejected(monkeypatch):
    fake = FakeTranslator(break_numbers=True)
    monkeypatch.setattr(translation, "get_llm_provider", lambda: fake)

    result = translation.translate_texts(["Spray 2 g per litre", "Remove leaves"], "ta")

    assert result["Spray 2 g per litre"] == "Spray 2 g per litre"  # English kept
    assert result["Remove leaves"] == "TA:Remove leaves"


def test_llm_failure_falls_back_to_english(monkeypatch):
    fake = FakeTranslator(error=LLMTimeoutError("slow"))
    monkeypatch.setattr(translation, "get_llm_provider", lambda: fake)

    assert translation.translate_texts(["Remove leaves"], "ta") == {"Remove leaves": "Remove leaves"}


def test_translations_persist_in_the_database(client, translator):
    with client.application.app_context():
        translation.translate_texts(["Remove leaves"], "ta")

    translation.clear_cache()  # e.g. a server restart
    translator.batches.clear()

    with client.application.app_context():
        assert translation.translate_texts(["Remove leaves"], "ta") == {"Remove leaves": "TA:Remove leaves"}

    assert translator.batches == []


# ---------- what is translated ----------


def test_cases_keep_product_names_doses_and_notes(client, fake_predictor, translator):
    import backend.routes.tracking_routes as routes

    routes_review = routes.review_note
    routes.review_note = lambda *a, **k: None
    try:
        register(client)
        case_id = start_case(client).get_json()["case"]["id"]
        check_in(client, case_id, "not_worked", notes="my own words")
    finally:
        routes.review_note = routes_review

    case = client.get("/cases?language=ta").get_json()["cases"][0]
    step = case["current_step"]
    product = step["products"][0]

    assert case["display_name"].startswith("TA:")
    assert step["title"].startswith("TA:") and step["actions"][0].startswith("TA:")
    assert product["kind"] == "TA:fungicide"
    assert product["name"] == "Zineb 75% WP" and product["dose"] == "800 g per acre"
    assert case["steps"][0]["notes"] == "my own words"
    assert case["current_step"] is not None and case["current_step"]["title"] == step["title"]
    assert "TA:TA:" not in json.dumps(case)  # shared objects translated once


def test_localize_cases_english_is_unchanged():
    cases = [{"display_name": "X", "steps": []}]
    assert localize_cases(cases, "en") is cases


def test_ui_text_translates_the_allowlist(client, translator):
    body = client.get("/ui-text?language=ta").get_json()

    assert body["strings"]["It worked"] == "TA:It worked"
    assert set(body["strings"]) <= set(UI_STRINGS)
    assert client.get("/ui-text?language=xx").status_code == 400


def test_plan_preview_is_translated(client, fake_predictor, translator):
    body = client.post(
        "/treatment-plan",
        json={"predicted_class": "Potato___Late_blight", "confidence_level": "high", "language": "ta"},
    ).get_json()

    assert body["step"]["title"].startswith("TA:")
    assert body["step"]["products"][0]["name"] == "Mancozeb"


# ---------- read aloud ----------


def test_case_speech_parts_then_audio(client, fake_predictor, translator):
    register(client)
    case_id = start_case(client).get_json()["case"]["id"]

    parts = client.post(f"/cases/{case_id}/speech", json={"language": "ta"}).get_json()
    assert parts["speech_lang"] == "ta-IN"
    assert parts["parts"] and parts["parts"][0].startswith("TA:")
    assert any("Azoxystrobin 23% SC" in p for p in parts["parts"])  # product read as listed

    audio = client.post(f"/cases/{case_id}/speech", json={"language": "ta", "part": 0})
    assert audio.mimetype == "audio/wav"
    assert audio.headers["X-Speech-Parts"] == str(len(parts["parts"]))
    assert translator.spoken[0][1] == "ta-IN"


def test_case_speech_is_private(client, fake_predictor, translator):
    register(client, email="one@example.com")
    case_id = start_case(client).get_json()["case"]["id"]
    client.post("/auth/logout")

    assert client.post(f"/cases/{case_id}/speech", json={"language": "ta"}).status_code == 401
    register(client, email="two@example.com")
    assert client.post(f"/cases/{case_id}/speech", json={"language": "ta"}).status_code == 404


def test_plan_speech(client, fake_predictor, translator):
    body = {"predicted_class": "Tomato___Early_blight", "confidence_level": "high", "extent": "most"}

    parts = client.post("/treatment-plan/speech", json={**body, "language": "en"}).get_json()
    assert parts["speech_lang"] == "en-IN"
    assert "Azoxystrobin 23% SC" in " ".join(parts["parts"])
    assert any("Recovery is unlikely" in p for p in parts["parts"])  # outlook included

    bad = client.post("/treatment-plan/speech", json={**body, "language": "ta", "part": 99})
    assert bad.status_code == 400


def test_plan_speech_rejects_untrackable(client, fake_predictor, translator):
    response = client.post(
        "/treatment-plan/speech",
        json={"predicted_class": "Tomato___healthy", "confidence_level": "high", "language": "ta"},
    )
    assert response.status_code == 400
