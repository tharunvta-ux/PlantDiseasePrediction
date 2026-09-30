"""
Integration tests with the real trained model (marker: model).

Real-world / downloaded images are read from tests/fixtures/real_world
and tests/fixtures/web (see tests/fixtures/README.md). When those
folders are empty the tests are SKIPPED - never reported as passing.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from backend.config import PROJECT_ROOT
from tests.conftest import CLASS_NAMES, FIXTURES_DIR, first_val_image

pytestmark = pytest.mark.model

PREDICTIONS_FILE = PROJECT_ROOT / "results" / "calibration" / "val_predictions.npz"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@pytest.fixture(scope="module")
def predictor(real_model_available):
    from backend.prediction.predictor import get_predictor

    return get_predictor()


def upload(client, path: Path):
    return client.post(
        "/predict",
        data={"file": (io.BytesIO(path.read_bytes()), path.name)},
        content_type="multipart/form-data",
    )


def fixture_images(folder: str):
    """(class_name, path) pairs from tests/fixtures/<folder>/<class>/."""

    root = FIXTURES_DIR / folder
    items = []

    if root.exists():
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for image in sorted(class_dir.iterdir()):
                if image.suffix.lower() in IMAGE_EXTENSIONS:
                    items.append((class_dir.name, image))

    return items


# ---------------- PlantVillage ----------------


def test_preprocessing_unchanged_for_plantvillage_images(predictor):
    """New PIL bilinear/EXIF path equals the old load_img path at 256x256."""

    import tensorflow as tf

    path = first_val_image("Apple___Apple_scab")
    old = tf.keras.utils.img_to_array(
        tf.keras.utils.load_img(path, target_size=(256, 256))
    ) / 255.0

    assert np.allclose(predictor.preprocess_image(str(path))[0], old)


def test_matches_cached_validation_predictions(predictor):
    if not PREDICTIONS_FILE.exists():
        pytest.skip("Run backend.evaluation.collect_predictions first.")

    data = np.load(PREDICTIONS_FILE)
    paths = [str(p) for p in data["paths"]]

    for index in (0, len(paths) // 2, len(paths) - 1):
        probs = predictor.predict_probabilities(str(PROJECT_ROOT / paths[index]))
        assert np.allclose(probs, data["probabilities"][index], atol=1e-4)


def test_plantvillage_image_end_to_end(client, real_model_available):
    path = first_val_image("Tomato___Late_blight")
    response = upload(client, path)
    body = response.get_json()

    assert response.status_code == 200
    assert {"predicted_class", "confidence", "top3_predictions"} <= set(body)
    assert body["predicted_class"] in CLASS_NAMES
    assert body["confidence_level"] in {"high", "uncertain", "low"}


def test_healthy_plantvillage_image(client, real_model_available):
    path = first_val_image("Tomato___healthy")
    body = upload(client, path).get_json()

    assert body["predicted_class"] == "Tomato___healthy"
    assert body["is_healthy"] is True
    assert body["health_status"] == "healthy"


def test_degraded_image_is_not_high_confidence(client, real_model_available, tmp_path):
    source = first_val_image("Potato___Early_blight")

    with Image.open(source) as image:
        degraded = (
            image.convert("RGB")
            .filter(ImageFilter.GaussianBlur(6))
            .point(lambda v: v * 0.3)
            .resize((96, 96))
        )
    target = tmp_path / "degraded.jpg"
    degraded.save(target, quality=25)

    body = upload(client, target).get_json()

    assert body["quality_warnings"]
    assert body["confidence_level"] != "high"


# ---------------- Real-world / downloaded images ----------------


@pytest.mark.parametrize("folder", ["real_world", "web"])
def test_no_confidently_wrong_predictions(client, real_model_available, folder):
    """
    Safety property on real-world photos: the model may be wrong, but it
    must not be wrong at the 'high' level. Labels come from folder names
    (use "_unsupported" for images of plants outside the 38 classes).
    """

    images = fixture_images(folder)

    if not images:
        pytest.skip(f"No images in tests/fixtures/{folder}/ - real-world behaviour NOT tested.")

    confidently_wrong = []

    for label, path in images:
        body = upload(client, path).get_json()

        if body["confidence_level"] == "high" and body["predicted_class"] != label:
            confidently_wrong.append((path.name, label, body["predicted_class"]))

    assert confidently_wrong == []
