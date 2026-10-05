"""
Hybrid analysis (Option B) tests: CNN + vision-LLM plant identification.
"""

from __future__ import annotations

import io
import json
import os

import numpy as np
import pytest
from PIL import Image

import backend.services.plant_identification_service as ident
from backend.prediction.confidence import CalibrationConfig
from backend.services.analysis_service import (
    ROUTE_MISMATCH,
    ROUTE_NOT_PLANT,
    ROUTE_SUPPORTED,
    ROUTE_UNSUPPORTED,
    ROUTE_UNVERIFIED,
    combine_results,
    supported_crops,
)
from backend.services.llm import (
    LLMInvalidResponseError,
    LLMNotConfiguredError,
    LLMResponse,
)
from backend.services.prediction_service import PredictionRun, build_prediction_result
from tests.conftest import CLASS_NAMES, first_val_image, one_hot_like
from tests.test_predict_api import textured_png

CALIBRATION = CalibrationConfig(temperature=1.0, high_threshold=0.9, low_threshold=0.5, provisional=True)

CROPS = supported_crops(CLASS_NAMES)


def make_run(probs: np.ndarray) -> PredictionRun:
    result = build_prediction_result(probs, CLASS_NAMES, CALIBRATION, [], "test")
    return PredictionRun(result, probs, CLASS_NAMES, CALIBRATION)


def identification(**overrides):
    base = {
        "is_plant": True,
        "plant_common_name": "Tomato",
        "plant_scientific_name": "Solanum lycopersicum",
        "supported_crop": "Tomato",
        "identification_confidence": "high",
        "plant_part": "leaf",
        "appears_healthy": "no",
        "visible_symptoms": ["Brown spots"],
        "possible_issues": ["Possibly early blight"],
        "general_advice": ["Remove affected leaves"],
        "image_notes": "",
    }
    base.update(overrides)
    return base


# ---------------- supported crops ----------------


def test_supported_crops_are_the_14_crops():
    assert len(CROPS) == 14
    assert "Tomato" in CROPS and "Pepper, bell" in CROPS and "Corn (maize)" in CROPS


# ---------------- routing ----------------


def test_supported_crop_uses_cnn():
    run = make_run(one_hot_like("Tomato___Early_blight", 0.97))
    out = combine_results(run, identification())

    assert out["route"] == ROUTE_SUPPORTED
    assert out["diagnosis"]["predicted_class"] == "Tomato___Early_blight"
    assert out["diagnosis"]["confidence_level"] == "high"
    assert out["diagnosis"]["source"] == "cnn"
    assert out["show_llm_observation"] is False


def test_crop_mismatch_keeps_cnn_but_uncertain():
    """The CNN wins crop disagreements (the LLM mislabels similar leaves,
    e.g. tomato as potato), but the result can no longer be 'high'."""

    probs = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    probs[CLASS_NAMES.index("Potato___Late_blight")] = 0.96
    probs[CLASS_NAMES.index("Tomato___Late_blight")] = 0.03
    probs[CLASS_NAMES.index("Tomato___healthy")] = 0.01
    out = combine_results(make_run(probs), identification(supported_crop="Tomato"))

    diag = out["diagnosis"]
    assert out["route"] == ROUTE_MISMATCH
    assert diag["predicted_class"] == "Potato___Late_blight"
    assert diag["source"] == "cnn"
    assert diag["confidence_level"] == "uncertain"

    alt = diag["alternative_if_ai_is_right"]
    assert alt["predicted_class"] == "Tomato___Late_blight"
    assert alt["confidence"] == pytest.approx(75.0, abs=0.01)  # 0.03 / 0.04
    assert alt["model_crop_probability"] == pytest.approx(4.0, abs=0.01)
    assert any("Tomato" in note for note in out["notes"])
    assert out["show_llm_observation"] is True


def test_unsupported_without_high_confidence_keeps_cnn():
    run = make_run(one_hot_like("Tomato___Early_blight", 0.97))
    out = combine_results(
        run,
        identification(plant_common_name="Eggplant", supported_crop="none", identification_confidence="medium"),
    )

    assert out["route"] == ROUTE_MISMATCH
    assert out["diagnosis"]["predicted_class"] == "Tomato___Early_blight"
    assert out["diagnosis"]["confidence_level"] == "uncertain"
    assert "alternative_if_ai_is_right" not in out["diagnosis"]


def test_unsupported_plant_has_no_cnn_diagnosis():
    run = make_run(one_hot_like("Corn_(maize)___healthy", 0.99))
    out = combine_results(run, identification(plant_common_name="Rose", supported_crop="none"))

    assert out["route"] == ROUTE_UNSUPPORTED
    assert out["diagnosis"] is None
    assert out["show_llm_observation"] is True
    assert "Rose" in out["message"]


def test_not_plant_has_no_diagnosis():
    run = make_run(one_hot_like("Corn_(maize)___healthy", 0.999))  # the 'green rectangle' case
    out = combine_results(run, identification(is_plant=False, supported_crop="none"))

    assert out["route"] == ROUTE_NOT_PLANT
    assert out["diagnosis"] is None
    assert out["show_llm_observation"] is False


def test_identification_unavailable_falls_back_to_cnn():
    run = make_run(one_hot_like("Tomato___Early_blight", 0.97))
    out = combine_results(run, None, LLMNotConfiguredError("no key"))

    assert out["route"] == ROUTE_UNVERIFIED
    assert out["diagnosis"]["predicted_class"] == "Tomato___Early_blight"
    assert out["identification_error"]["code"] == "LLM_NOT_CONFIGURED"
    assert out["notes"]


def test_non_leaf_photo_caps_confidence():
    run = make_run(one_hot_like("Tomato___Early_blight", 0.97))
    out = combine_results(run, identification(plant_part="fruit"))

    assert out["diagnosis"]["confidence_level"] == "uncertain"
    assert any("leaf photos" in note for note in out["notes"])


def test_low_confidence_supported_shows_observation():
    probs = np.full(len(CLASS_NAMES), 1 / len(CLASS_NAMES), dtype=np.float32)
    probs[CLASS_NAMES.index("Tomato___Early_blight")] += 0.01
    probs /= probs.sum()
    out = combine_results(make_run(probs), identification())

    assert out["diagnosis"]["confidence_level"] == "low"
    assert out["show_llm_observation"] is True


# ---------------- identification service ----------------


class FakeVisionProvider:
    name = "fake"
    model = "fake-vision"

    def __init__(self, reply: dict | str | None = None, error=None):
        self.reply = reply if reply is not None else identification()
        self.error = error
        self.calls = []

    def generate_json(self, system_prompt, user_prompt, response_schema, image=None):
        self.calls.append((system_prompt, user_prompt, response_schema, image))
        if self.error:
            raise self.error
        text = self.reply if isinstance(self.reply, str) else json.dumps(self.reply)
        return LLMResponse(text=text, model=self.model)


@pytest.fixture(autouse=True)
def _clear_identification_cache():
    ident.clear_cache()
    yield
    ident.clear_cache()


@pytest.fixture()
def big_photo(tmp_path):
    path = tmp_path / "photo.png"
    Image.open(io.BytesIO(textured_png((3000, 2000)))).save(path)
    return str(path)


def test_image_is_sent_downscaled_as_jpeg(big_photo):
    provider = FakeVisionProvider()
    ident.identify_plant(big_photo, CROPS, provider=provider)

    _, _, schema, image = provider.calls[0]
    sent = Image.open(io.BytesIO(image.data))

    assert image.mime_type == "image/jpeg"
    assert max(sent.size) == ident.MAX_IMAGE_SIDE
    assert schema["properties"]["supported_crop"]["enum"][-1] == "none"
    assert "Tomato" in schema["properties"]["supported_crop"]["enum"]


def test_crop_name_is_normalised(big_photo):
    provider = FakeVisionProvider(identification(supported_crop="  tomato "))
    assert ident.identify_plant(big_photo, CROPS, provider=provider)["supported_crop"] == "Tomato"


def test_unknown_crop_becomes_none(big_photo):
    provider = FakeVisionProvider(identification(supported_crop="Mango"))
    assert ident.identify_plant(big_photo, CROPS, provider=provider)["supported_crop"] == "none"


def test_not_plant_forces_none(big_photo):
    provider = FakeVisionProvider(identification(is_plant=False, supported_crop="Tomato"))
    assert ident.identify_plant(big_photo, CROPS, provider=provider)["supported_crop"] == "none"


def test_dosages_removed_from_advice(big_photo):
    provider = FakeVisionProvider(
        identification(general_advice=["Spray 5 ml per litre", "Remove affected leaves"])
    )
    result = ident.identify_plant(big_photo, CROPS, provider=provider)

    assert result["general_advice"] == ["Remove affected leaves"]
    assert result["safety"]["redacted_items"] == 1


def test_invalid_identification_json(big_photo):
    provider = FakeVisionProvider('{"is_plant": "maybe"}')

    with pytest.raises(LLMInvalidResponseError):
        ident.identify_plant(big_photo, CROPS, provider=provider)


def test_identification_is_cached(big_photo):
    provider = FakeVisionProvider()
    ident.identify_plant(big_photo, CROPS, provider=provider)
    second = ident.identify_plant(big_photo, CROPS, provider=provider)

    assert len(provider.calls) == 1
    assert second["cached"] is True


# ---------------- /analyze endpoint ----------------


def post_analyze(client, data: bytes, name="leaf.png"):
    return client.post(
        "/analyze",
        data={"file": (io.BytesIO(data), name)},
        content_type="multipart/form-data",
    )


def test_analyze_endpoint_supported(client, fake_predictor, monkeypatch):
    provider = FakeVisionProvider()
    monkeypatch.setattr(ident, "get_llm_provider", lambda: provider)

    body = post_analyze(client, textured_png()).get_json()

    assert body["route"] == ROUTE_SUPPORTED
    assert body["diagnosis"]["predicted_class"] == "Tomato___Early_blight"
    assert {"predicted_class", "confidence", "top3_predictions"} <= set(body["prediction"])


def test_analyze_without_api_key_still_works(client, fake_predictor, no_api_key):
    response = post_analyze(client, textured_png())
    body = response.get_json()

    assert response.status_code == 200
    assert body["route"] == ROUTE_UNVERIFIED
    assert body["diagnosis"] is not None


def test_analyze_rejects_bad_uploads(client, fake_predictor):
    assert post_analyze(client, b"not an image", "x.jpg").status_code == 400
    assert post_analyze(client, b"GIF89a", "x.gif").status_code == 400
    assert client.post("/analyze").status_code == 400


def test_analyze_oversized_upload(client, fake_predictor):
    assert post_analyze(client, b"0" * (10 * 1024 * 1024 + 1), "big.jpg").status_code == 413


def test_gemini_receives_image_part():
    from types import SimpleNamespace

    from backend.services.llm import ImageInput
    from tests.test_gemini_provider import make_provider

    provider = make_provider()
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(text="{}")

    provider._client = SimpleNamespace(models=SimpleNamespace(generate_content=capture))
    provider.generate_json("S", "U", {}, image=ImageInput(b"\xff\xd8jpeg", "image/jpeg"))

    part, text = seen["contents"]
    assert part.inline_data.mime_type == "image/jpeg"
    assert text == "U"


# ---------------- live (opt-in) ----------------


@pytest.mark.live
@pytest.mark.skipif(
    not (os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="LLM_API_KEY not set - live identification test not run.",
)
def test_live_recognises_a_leaf_as_a_plant():
    """Only 'is a plant' is asserted: the exact crop is not reliable
    (measured 15/20 correct on PlantVillage leaves; tomato is sometimes
    called potato), which is why the CNN wins crop disagreements."""

    image = first_val_image("Potato___Late_blight")
    result = ident.identify_plant(str(image), CROPS)

    assert result["is_plant"] is True
