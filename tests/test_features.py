"""
Tests for: explanation heat maps, model performance page/report,
multi-language guidance.
"""

from __future__ import annotations

import base64
import io
import json

import numpy as np
import pytest
from PIL import Image

import backend.services.plant_identification_service as ident
import backend.services.recommendation_service as rec
from backend.prediction.explain import _positions, log_odds, occlusion_map
from backend.services.languages import LANGUAGES, normalise_language
from tests.conftest import CLASS_NAMES
from tests.test_analysis import FakeVisionProvider
from tests.test_predict_api import textured_png
from tests.test_recommendation import FakeProvider


# ---------------- occlusion heat map ----------------


def test_positions_cover_whole_image():
    starts = _positions(256, 64, 48)
    assert starts == [0, 48, 96, 144, 192]
    assert starts[-1] + 64 == 256


def test_log_odds_is_finite_when_saturated():
    probs = np.zeros((1, 3), dtype=np.float32)
    probs[0, 0] = 1.0
    assert np.isfinite(log_odds(probs, 0)).all()


def test_occlusion_finds_the_important_region():
    """A fake model whose confidence depends only on fine detail in the
    top-left 64x64 block: the heat map must peak there."""

    rng = np.random.default_rng(0)
    image = np.full((1, 256, 256, 3), 0.5, dtype=np.float32)
    image[0, :64, :64] = rng.random((64, 64, 3))  # detailed 'lesion'

    def predict_batch(batch):
        detail = batch[:, :64, :64].std(axis=(1, 2, 3))
        p = 1 / (1 + np.exp(-(detail * 40 - 2)))
        return np.stack([p, 1 - p], axis=1)

    result = occlusion_map(image, 0, predict_batch)

    assert result.heat.shape == (5, 5)
    assert np.unravel_index(result.heat.argmax(), result.heat.shape) == (0, 0)
    assert result.heat[4, 4] == 0


def test_explain_endpoint(client, fake_predictor):
    response = client.post(
        "/explain",
        data={"file": (io.BytesIO(textured_png()), "leaf.png"), "target_class": "Tomato___Early_blight"},
        content_type="multipart/form-data",
    )
    body = response.get_json()

    assert response.status_code == 200
    assert body["grid"] == [5, 5]
    assert body["overlay_png"].startswith("data:image/png;base64,")
    Image.open(io.BytesIO(base64.b64decode(body["overlay_png"].split(",")[1]))).verify()


def test_explain_rejects_unknown_class(client, fake_predictor):
    response = client.post(
        "/explain",
        data={"file": (io.BytesIO(textured_png()), "leaf.png"), "target_class": "Banana___Panama"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["code"] == "INVALID_CLASS"


def test_explain_rejects_bad_upload(client, fake_predictor):
    response = client.post(
        "/explain",
        data={"file": (io.BytesIO(b"nope"), "x.jpg"), "target_class": "Tomato___Early_blight"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


# ---------------- model performance ----------------


def test_model_report(client):
    response = client.get("/model/report")
    body = response.get_json()

    assert response.status_code == 200
    assert body["summary"]["test_images"] > 0
    assert 0 < body["summary"]["accuracy"] <= 1
    assert len(body["per_class"]) == 38
    matrix = np.array(body["confusion_matrix"])
    assert matrix.shape == (38, 38)
    assert matrix.sum() == body["summary"]["test_images"]
    assert body["provisional"] is True


def test_model_figures(client):
    response = client.get("/model/figures/reliability_diagram")
    assert response.status_code == 200
    assert response.mimetype == "image/png"


@pytest.mark.parametrize("name", ["../../.env", "secret", "training_accuracy.png"])
def test_model_figures_whitelist(client, name):
    assert client.get(f"/model/figures/{name}").status_code == 404


def test_model_page(client):
    response = client.get("/app/model")
    assert response.status_code == 200
    assert b"Model performance" in response.data


# ---------------- languages ----------------


def test_languages_endpoint(client):
    body = client.get("/languages").get_json()
    codes = {item["code"] for item in body}

    assert {"en", "hi", "ta", "te"} <= codes
    assert all(item["speech_lang"] for item in body)


@pytest.mark.parametrize("value, expected", [(None, "en"), ("", "en"), ("HI", "hi"), ("xx", None), (5, None)])
def test_normalise_language(value, expected):
    assert normalise_language(value) == expected


def test_recommendation_in_hindi(client, fake_predictor, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(rec, "get_llm_provider", lambda: provider)

    body = client.post(
        "/recommendation",
        json={"predicted_class": "Tomato___Early_blight", "confidence_level": "high", "language": "hi"},
    ).get_json()

    _, user_prompt, _ = provider.calls[0]
    assert "Hindi" in user_prompt
    assert body["language"] == "hi"
    assert body["speech_lang"] == LANGUAGES["hi"][2]


def test_languages_are_cached_separately(client, fake_predictor, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(rec, "get_llm_provider", lambda: provider)

    for lang in ("en", "ta", "en"):
        client.post(
            "/recommendation",
            json={"predicted_class": "Tomato___Early_blight", "confidence_level": "high", "language": lang},
        )

    assert len(provider.calls) == 2


def test_invalid_language_rejected(client, fake_predictor, monkeypatch):
    monkeypatch.setattr(rec, "get_llm_provider", lambda: FakeProvider())

    response = client.post(
        "/recommendation",
        json={"predicted_class": "Tomato___Early_blight", "confidence_level": "high", "language": "klingon"},
    )

    assert response.status_code == 400
    assert response.get_json()["code"] == "INVALID_LANGUAGE"


def test_analyze_passes_language_to_identification(client, fake_predictor, monkeypatch):
    provider = FakeVisionProvider()
    monkeypatch.setattr(ident, "get_llm_provider", lambda: provider)

    client.post(
        "/analyze",
        data={"file": (io.BytesIO(textured_png()), "leaf.png"), "language": "ta"},
        content_type="multipart/form-data",
    )

    _, user_prompt, _, _ = provider.calls[0]
    assert "Tamil" in user_prompt


def test_analyze_invalid_language(client, fake_predictor):
    response = client.post(
        "/analyze",
        data={"file": (io.BytesIO(textured_png()), "leaf.png"), "language": "zz"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
