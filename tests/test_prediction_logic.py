"""
Unit tests for class parsing, confidence policy, quality checks and the
calibration helpers (no TensorFlow model needed).
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.evaluation.calibrate import low_threshold, youden_threshold
from backend.prediction.class_info import parse_class_name
from backend.prediction.confidence import (
    CalibrationConfig,
    apply_temperature,
    confidence_level,
    load_calibration,
)
from backend.prediction.image_quality import QualityMetrics, assess_quality
from backend.services.prediction_service import build_prediction_result
from tests.conftest import ARTIFACTS_DIR, CLASS_NAMES, one_hot_like

CALIBRATION = CalibrationConfig(
    temperature=1.0,
    high_threshold=0.9,
    low_threshold=0.5,
    provisional=True,
)


# ---------------- class names ----------------


@pytest.mark.parametrize(
    "raw, crop, condition, healthy",
    [
        ("Tomato___Early_blight", "Tomato", "Early blight", False),
        ("Corn_(maize)___healthy", "Corn (maize)", "Healthy", True),
        ("Pepper,_bell___Bacterial_spot", "Pepper, bell", "Bacterial spot", False),
        ("Corn_(maize)___Common_rust_", "Corn (maize)", "Common rust", False),
    ],
)
def test_parse_class_name(raw, crop, condition, healthy):
    info = parse_class_name(raw)
    assert (info.crop, info.condition, info.is_healthy) == (crop, condition, healthy)


def test_all_model_classes_parse():
    assert len(CLASS_NAMES) == 38
    for name in CLASS_NAMES:
        parse_class_name(name)


def test_parse_rejects_unknown_format():
    with pytest.raises(ValueError):
        parse_class_name("not a class")


# ---------------- calibration / levels ----------------


def test_calibration_file_is_valid_and_provisional():
    cal = load_calibration(ARTIFACTS_DIR)
    assert cal is not None
    assert 0 < cal.low_threshold <= cal.high_threshold < 1
    assert cal.provisional is True


def test_apply_temperature_one_is_identity():
    probs = np.array([0.7, 0.2, 0.1])
    assert np.allclose(apply_temperature(probs, 1.0), probs)


def test_higher_temperature_softens():
    probs = np.array([0.9, 0.05, 0.05])
    assert apply_temperature(probs, 2.0).max() < 0.9


@pytest.mark.parametrize(
    "conf, warnings, expected",
    [
        (0.95, False, "high"),
        (0.95, True, "uncertain"),  # quality warnings cap the level
        (0.70, False, "uncertain"),
        (0.30, False, "low"),
    ],
)
def test_confidence_levels(conf, warnings, expected):
    assert confidence_level(conf, CALIBRATION, warnings) == expected


def test_no_calibration_never_high():
    assert confidence_level(0.999, None, False) == "uncertain"


def test_youden_threshold_separates():
    conf = np.array([0.99, 0.95, 0.9, 0.6, 0.5, 0.4])
    correct = np.array([True, True, True, False, False, False])
    assert youden_threshold(conf, correct) == pytest.approx(0.9)


def test_low_threshold_needs_enough_samples():
    conf = np.linspace(0.9, 1.0, 50)
    correct = np.ones(50, dtype=bool)
    assert low_threshold(conf, correct)["value"] is None


# ---------------- quality ----------------

REFERENCE = {
    "blur_score_p01": 35.0,
    "brightness_p01": 78.0,
    "brightness_p99": 172.0,
    "contrast_p01": 17.0,
}


def _metrics(**overrides):
    base = dict(width=1000, height=800, blur_score=500.0, brightness=120.0, contrast=50.0)
    base.update(overrides)
    return QualityMetrics(**base)


def test_good_image_has_no_warnings():
    assert assess_quality(_metrics(), REFERENCE) == []


@pytest.mark.parametrize(
    "overrides, code",
    [
        ({"blur_score": 5.0}, "BLURRY"),
        ({"brightness": 30.0}, "TOO_DARK"),
        ({"brightness": 230.0}, "TOO_BRIGHT"),
        ({"contrast": 3.0}, "LOW_CONTRAST"),
        ({"width": 120, "height": 90}, "LOW_RESOLUTION"),
    ],
)
def test_quality_warning_codes(overrides, code):
    codes = [w["code"] for w in assess_quality(_metrics(**overrides), REFERENCE)]
    assert code in codes


# ---------------- response building ----------------


def test_result_keeps_original_fields():
    probs = one_hot_like("Apple___Apple_scab", 0.97)
    result = build_prediction_result(probs, CLASS_NAMES, CALIBRATION, [], "v")

    assert result["predicted_class"] == "Apple___Apple_scab"
    assert result["confidence"] == pytest.approx(97.0, abs=0.01)
    assert len(result["top3_predictions"]) == 3
    assert set(result["top3_predictions"][0]) == {"class", "confidence"}


def test_result_high_disease():
    probs = one_hot_like("Apple___Apple_scab", 0.97)
    result = build_prediction_result(probs, CLASS_NAMES, CALIBRATION, [], "v")

    assert result["confidence_level"] == "high"
    assert result["crop"] == "Apple"
    assert result["health_status"] == "diseased"
    assert result["disease"] == "Apple scab"


def test_result_healthy():
    probs = one_hot_like("Tomato___healthy", 0.97)
    result = build_prediction_result(probs, CLASS_NAMES, CALIBRATION, [], "v")

    assert result["is_healthy"] is True
    assert result["health_status"] == "healthy"
    assert result["disease"] is None


def test_result_low_confidence_withholds_diagnosis():
    probs = np.full(len(CLASS_NAMES), 1 / len(CLASS_NAMES), dtype=np.float32)
    result = build_prediction_result(probs, CLASS_NAMES, CALIBRATION, [], "v")

    assert result["confidence_level"] == "low"
    assert result["health_status"] == "uncertain"
    assert "uncertain" in result["message"].lower()


def test_crop_confidence_sums_classes():
    probs = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    probs[CLASS_NAMES.index("Tomato___Early_blight")] = 0.4
    probs[CLASS_NAMES.index("Tomato___Late_blight")] = 0.35
    probs[CLASS_NAMES.index("Potato___Early_blight")] = 0.25
    result = build_prediction_result(probs, CLASS_NAMES, CALIBRATION, [], "v")

    assert result["crop"] == "Tomato"
    assert result["crop_confidence"] == pytest.approx(75.0, abs=0.01)
    assert result["confidence_level"] == "low"  # 0.40 < low threshold
