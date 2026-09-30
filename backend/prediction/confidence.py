"""
backend/prediction/confidence.py

Confidence calibration and the three-level confidence policy.

The temperature and thresholds are produced by
`backend/evaluation/calibrate.py` and stored in
backend/model_artifacts/<model>/calibration.json - never hardcoded here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_FILE = "calibration.json"

LEVEL_HIGH = "high"
LEVEL_UNCERTAIN = "uncertain"
LEVEL_LOW = "low"


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration parameters for one model."""

    temperature: float
    high_threshold: float
    low_threshold: float
    provisional: bool
    quality_reference: Dict[str, float] = field(default_factory=dict)


def apply_temperature(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """
    Temperature-scale softmax outputs.

    softmax(log(p) / T) equals softmax(logits / T), so this works on the
    stored softmax outputs without access to the logits.
    """

    logits = np.log(np.clip(probabilities, 1e-38, None)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)

    return exp / exp.sum(axis=-1, keepdims=True)


def load_calibration(artifacts_dir: Path) -> Optional[CalibrationConfig]:
    """
    Load calibration.json for a model, or None if it does not exist.
    """

    path = artifacts_dir / CALIBRATION_FILE

    if not path.exists():
        logger.warning("No calibration file at %s", path)
        return None

    data = json.loads(path.read_text(encoding="utf-8"))

    return CalibrationConfig(
        temperature=float(data["temperature"]),
        high_threshold=float(data["thresholds"]["high"]),
        low_threshold=float(data["thresholds"]["low"]),
        provisional=bool(data.get("provisional", True)),
        quality_reference=data.get("quality_reference", {}),
    )


def confidence_level(
    calibrated_confidence: float,
    calibration: Optional[CalibrationConfig],
    has_quality_warnings: bool,
) -> str:
    """
    Map a calibrated top-1 confidence (0-1) to high / uncertain / low.

    - Without calibration data the result is never "high".
    - Quality warnings cap the level at "uncertain": the image lies
      outside what the thresholds were measured on.
    """

    if calibration is None:
        return LEVEL_UNCERTAIN

    if calibrated_confidence < calibration.low_threshold:
        return LEVEL_LOW

    if calibrated_confidence >= calibration.high_threshold and not has_quality_warnings:
        return LEVEL_HIGH

    return LEVEL_UNCERTAIN
