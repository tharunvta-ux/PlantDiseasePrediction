"""
Shared pytest fixtures.

Most API tests use a fake predictor (fixed probability vectors) so they
run fast and deterministically; tests marked `model` load the real
TensorFlow model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from backend.config import MODEL_ARTIFACTS_ROOT, PLANTVILLAGE_DIR, get_settings  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
ARTIFACTS_DIR = MODEL_ARTIFACTS_ROOT / "plant_disease_cnn"
CLASS_NAMES: List[str] = json.loads(
    (ARTIFACTS_DIR / "class_names.json").read_text(encoding="utf-8")
)
VAL_DIR = PLANTVILLAGE_DIR / "val"


def one_hot_like(class_name: str, top: float) -> np.ndarray:
    """Probability vector with `top` mass on class_name, rest spread evenly."""

    probs = np.full(len(CLASS_NAMES), (1 - top) / (len(CLASS_NAMES) - 1))
    probs[CLASS_NAMES.index(class_name)] = top

    return probs.astype(np.float32)


class FakePredictor:
    """Stands in for PlantDiseasePredictor without TensorFlow."""

    def __init__(self) -> None:
        self.class_names = CLASS_NAMES
        self.artifacts_dir = ARTIFACTS_DIR
        self.model_version = "plant_disease_cnn"
        self.probabilities = one_hot_like("Tomato___Early_blight", 0.99)

    def predict_probabilities(self, image_path: str) -> np.ndarray:
        return self.probabilities


@pytest.fixture()
def fake_predictor(monkeypatch) -> FakePredictor:
    """Patch every get_predictor() consumer with a FakePredictor."""

    fake = FakePredictor()

    import backend.routes.recommendation_routes as rec_routes
    import backend.services.prediction_service as pred_service

    monkeypatch.setattr(pred_service, "get_predictor", lambda: fake)
    monkeypatch.setattr(rec_routes, "get_predictor", lambda: fake)

    return fake


@pytest.fixture()
def client(tmp_path):
    """Flask test client with an isolated upload folder."""

    from backend.app import create_app

    app = create_app()
    app.config.update(TESTING=True, UPLOAD_FOLDER=str(tmp_path))

    return app.test_client()


@pytest.fixture(autouse=True)
def _clear_recommendation_cache():
    """Each test starts with an empty LLM cache."""

    from backend.services.recommendation_service import clear_cache

    clear_cache()
    yield
    clear_cache()


@pytest.fixture()
def no_api_key(monkeypatch):
    """Simulate a server without LLM credentials (ignores local .env)."""

    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")


def first_val_image(class_name: str) -> Path:
    """First validation image for a class (skips if dataset absent)."""

    folder = VAL_DIR / class_name

    if not folder.exists():
        pytest.skip(f"PlantVillage val data not available: {folder}")

    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[0]


@pytest.fixture(scope="session")
def real_model_available() -> bool:
    """Skip model tests when the trained model is missing."""

    if not get_settings().model_path.exists():
        pytest.skip("Trained model file not present.")

    return True
