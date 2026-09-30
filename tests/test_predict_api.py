"""
/predict endpoint tests with a fake predictor (no TensorFlow inference).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

from tests.conftest import CLASS_NAMES, one_hot_like

ORIGINAL_KEYS = {"predicted_class", "confidence", "top3_predictions"}


def textured_png(size=(400, 400), seed=0) -> bytes:
    """Sharp, mid-brightness synthetic image (passes quality checks)."""

    rng = np.random.default_rng(seed)
    # Same noise in every channel so grey-level contrast stays high.
    grey = rng.integers(40, 216, size=(size[1], size[0], 1), dtype=np.uint8)
    pixels = np.repeat(grey, 3, axis=2)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def degraded_jpeg() -> bytes:
    """Small, blurred, dark image - a 'low quality' photo."""

    image = Image.open(io.BytesIO(textured_png((400, 400))))
    image = image.filter(ImageFilter.GaussianBlur(8)).point(lambda v: v * 0.25)
    image = image.resize((120, 120))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=30)
    return buffer.getvalue()


def post_image(client, data: bytes, name: str):
    return client.post(
        "/predict",
        data={"file": (io.BytesIO(data), name)},
        content_type="multipart/form-data",
    )


def test_high_confidence_response_is_backward_compatible(client, fake_predictor):
    response = post_image(client, textured_png(), "leaf.png")
    body = response.get_json()

    assert response.status_code == 200
    assert ORIGINAL_KEYS <= set(body)
    assert body["predicted_class"] == "Tomato___Early_blight"
    assert body["confidence_level"] == "high"
    assert body["quality_warnings"] == []
    assert body["thresholds_provisional"] is True


def test_low_confidence_prediction(client, fake_predictor):
    fake_predictor.probabilities = np.full(len(CLASS_NAMES), 1 / len(CLASS_NAMES), dtype=np.float32)

    body = post_image(client, textured_png(), "leaf.png").get_json()

    assert body["confidence_level"] == "low"
    assert body["health_status"] == "uncertain"
    assert "clearer" in body["message"]


def test_low_quality_image_is_flagged_and_capped(client, fake_predictor):
    fake_predictor.probabilities = one_hot_like("Tomato___Early_blight", 0.999)

    body = post_image(client, degraded_jpeg(), "blurry.jpg").get_json()
    codes = {w["code"] for w in body["quality_warnings"]}

    assert {"LOW_RESOLUTION", "BLURRY", "TOO_DARK"} <= codes
    assert body["confidence_level"] == "uncertain"  # would be high otherwise


def test_unsupported_extension(client, fake_predictor):
    response = post_image(client, b"GIF89a", "leaf.gif")
    assert response.status_code == 400
    assert response.get_json()["error"] == "Unsupported file type"


def test_non_image_with_image_extension(client, fake_predictor):
    response = post_image(client, b"this is not an image", "leaf.jpg")
    assert response.status_code == 400
    assert "not a valid image" in response.get_json()["error"]


def test_missing_file(client, fake_predictor):
    assert client.post("/predict").status_code == 400


def test_oversized_upload_rejected(client, fake_predictor):
    big = b"0" * (10 * 1024 * 1024 + 1)
    response = post_image(client, big, "big.jpg")

    assert response.status_code == 413
    assert "10 MB" in response.get_json()["error"]


def test_server_error_does_not_leak_details(client, fake_predictor):
    def boom(_path):
        raise RuntimeError("secret internal path C:/x")

    fake_predictor.predict_probabilities = boom
    response = post_image(client, textured_png(), "leaf.png")

    assert response.status_code == 500
    assert "secret" not in response.get_json()["error"]


def test_upload_is_deleted(client, fake_predictor, tmp_path):
    post_image(client, textured_png(), "leaf.png")
    assert list(tmp_path.iterdir()) == []
