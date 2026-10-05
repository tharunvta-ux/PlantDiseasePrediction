"""
backend/services/prediction_service.py

Turns raw model output into the API prediction result:
quality checks -> calibration -> confidence level -> crop / health.

The first three response fields (predicted_class, confidence,
top3_predictions) keep their original meaning: raw softmax percentages.
Everything else is additive.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from backend.prediction.class_info import parse_class_name
from backend.prediction.confidence import (
    LEVEL_HIGH,
    LEVEL_LOW,
    CalibrationConfig,
    apply_temperature,
    confidence_level,
    load_calibration,
)
from backend.prediction.image_quality import (
    assess_quality,
    compute_quality_metrics,
)
from backend.prediction.predictor import get_predictor

logger = logging.getLogger(__name__)

TOP_K = 3


@lru_cache(maxsize=1)
def _calibration_for(artifacts_dir: str) -> CalibrationConfig | None:
    """Cache calibration.json per model."""

    return load_calibration(Path(artifacts_dir))


def _pct(value: float) -> float:
    """0-1 probability -> percentage rounded for the API."""

    return round(float(value) * 100, 2)


def _crop_probabilities(
    probabilities: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """Sum class probabilities per crop."""

    totals: Dict[str, float] = defaultdict(float)

    for name, prob in zip(class_names, probabilities):
        totals[parse_class_name(name).crop] += float(prob)

    return dict(totals)


def _message(level: str, display_name: str, crop: str, provisional: bool) -> str:
    """User-facing summary for each confidence level."""

    if level == LEVEL_HIGH:
        text = f"Likely diagnosis: {display_name}."
        if provisional:
            text += (
                " Confidence thresholds were measured on lab images, "
                "so confirm with a local expert before treating."
            )
        return text

    if level == LEVEL_LOW:
        return (
            "The result is uncertain and no diagnosis is shown. "
            "Please upload a clearer, well-lit photo of a single leaf "
            "that fills most of the frame."
        )

    return (
        f"Possible {display_name}, but the model is not confident. "
        f"The plant appears to be {crop}. Compare the top predictions "
        "below and consider retaking the photo."
    )


def build_prediction_result(
    probabilities: np.ndarray,
    class_names: List[str],
    calibration: CalibrationConfig | None,
    quality_warnings: List[Dict[str, Any]],
    model_version: str,
) -> Dict[str, Any]:
    """
    Assemble the API response from a softmax vector (pure function).
    """

    top_indices = np.argsort(probabilities)[-TOP_K:][::-1]
    top_index = int(top_indices[0])

    calibrated = (
        apply_temperature(probabilities, calibration.temperature)
        if calibration
        else probabilities
    )

    calibrated_conf = float(calibrated[top_index])

    level = confidence_level(
        calibrated_conf,
        calibration,
        has_quality_warnings=bool(quality_warnings),
    )

    info = parse_class_name(class_names[top_index])

    crops = _crop_probabilities(calibrated, class_names)
    crop_name = max(crops, key=crops.get)

    if level == LEVEL_LOW:
        health_status = "uncertain"
    else:
        health_status = "healthy" if info.is_healthy else "diseased"

    return {
        # --- original fields (unchanged semantics) ---
        "predicted_class": class_names[top_index],
        "confidence": _pct(probabilities[top_index]),
        "top3_predictions": [
            {
                "class": class_names[int(i)],
                "confidence": _pct(probabilities[int(i)]),
            }
            for i in top_indices
        ],
        # --- additive fields ---
        "display_name": info.display_name,
        "crop": crop_name,
        "crop_confidence": _pct(crops[crop_name]),
        "disease": None if info.is_healthy else info.condition,
        "is_healthy": info.is_healthy,
        "health_status": health_status,
        "calibrated_confidence": _pct(calibrated_conf),
        "confidence_level": level,
        "thresholds_provisional": calibration.provisional if calibration else True,
        "quality_warnings": quality_warnings,
        "message": _message(
            level,
            info.display_name,
            crop_name,
            calibration.provisional if calibration else True,
        ),
        "model_version": model_version,
    }


@dataclass(frozen=True)
class PredictionRun:
    """API result plus the raw material other services may need."""

    result: Dict[str, Any]
    calibrated_probabilities: np.ndarray
    class_names: List[str]
    calibration: CalibrationConfig | None


def analyze_image(image_path: str) -> Dict[str, Any]:
    """
    Full prediction pipeline for one uploaded image.
    """

    return run_prediction(image_path).result


def run_prediction(image_path: str) -> PredictionRun:
    """
    Run quality checks + model + calibration for one image.
    """

    predictor = get_predictor()
    calibration = _calibration_for(str(predictor.artifacts_dir))

    metrics = compute_quality_metrics(image_path)
    warnings = assess_quality(
        metrics,
        calibration.quality_reference if calibration else None,
    )

    probabilities = predictor.predict_probabilities(image_path)

    result = build_prediction_result(
        probabilities,
        predictor.class_names,
        calibration,
        warnings,
        predictor.model_version,
    )

    logger.info(
        "Prediction %s (raw %.2f%%, calibrated %.2f%%, level=%s, warnings=%s, quality=%s)",
        result["predicted_class"],
        result["confidence"],
        result["calibrated_confidence"],
        result["confidence_level"],
        [w["code"] for w in warnings],
        metrics.to_dict(),
    )

    calibrated = (
        apply_temperature(probabilities, calibration.temperature)
        if calibration
        else probabilities
    )

    return PredictionRun(
        result=result,
        calibrated_probabilities=calibrated,
        class_names=list(predictor.class_names),
        calibration=calibration,
    )
