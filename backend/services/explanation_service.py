"""
backend/services/explanation_service.py

Builds the "where did the model look?" explanation for a CNN class.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from backend.prediction.class_info import parse_class_name
from backend.prediction.explain import occlusion_map, render_overlay
from backend.prediction.predictor import get_predictor

logger = logging.getLogger(__name__)


class ExplanationRequestError(ValueError):
    """Unknown target class (HTTP 400)."""


def explain_image(image_path: str, target_class: str) -> Dict[str, Any]:
    """
    Occlusion-sensitivity heat map for `target_class` on one image.

    Raises:
        ExplanationRequestError: target_class is not a model class.
    """

    predictor = get_predictor()

    if target_class not in predictor.class_names:
        raise ExplanationRequestError("target_class must be one of the model's class names.")

    start = time.time()

    result = occlusion_map(
        predictor.preprocess_image(image_path),
        predictor.class_names.index(target_class),
        predictor.predict_batch,
    )

    peak_drop = float(result.heat.max())
    elapsed = time.time() - start

    logger.info(
        "Explanation for %s: base %.3f, peak log-odds drop %.2f, %d runs in %.2fs",
        target_class,
        result.base_probability,
        peak_drop,
        result.heat.size,
        elapsed,
    )

    return {
        "target_class": target_class,
        "display_name": parse_class_name(target_class).display_name,
        "method": "occlusion_sensitivity_blur",
        "grid": list(result.heat.shape),
        "base_probability": round(result.base_probability * 100, 2),
        "peak_log_odds_drop": round(peak_drop, 3),
        "has_signal": peak_drop > 0,
        "overlay_png": render_overlay(image_path, result),
        "explanation": (
            "Red areas are where blurring part of the photo reduced the model's "
            "evidence for this result the most - the regions it relies on. "
            "If red appears on the background or leaf edges rather than on the "
            "symptoms, the model is partly using the background, a known "
            "weakness of models trained on lab photos."
        ),
        "seconds": round(elapsed, 2),
    }
