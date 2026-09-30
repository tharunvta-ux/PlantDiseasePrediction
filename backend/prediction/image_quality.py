"""
backend/prediction/image_quality.py

Lightweight image-quality checks run before trusting a prediction.

Metrics are computed on the same 256x256 view the model sees, so they
are comparable with the reference statistics measured on the
PlantVillage calibration split (stored in calibration.json). An image is
flagged when it falls outside the range the model was trained on - no
hand-picked cut-offs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import cv2
import numpy as np
from PIL import Image, ImageOps

MODEL_INPUT_SIZE = (256, 256)


@dataclass(frozen=True)
class QualityMetrics:
    """Quality measurements for one image."""

    width: int
    height: int
    blur_score: float  # variance of the Laplacian (higher = sharper)
    brightness: float  # mean grey level, 0-255
    contrast: float  # std of grey level

    def to_dict(self) -> Dict[str, float]:
        """Serialise for logging / JSON."""

        return asdict(self)


def compute_quality_metrics(image_path: str | Path) -> QualityMetrics:
    """
    Measure size, sharpness, brightness and contrast of an image.
    """

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        resized = image.resize(MODEL_INPUT_SIZE, Image.Resampling.BILINEAR)
        grey = np.asarray(resized.convert("L"), dtype=np.float64)

    return QualityMetrics(
        width=width,
        height=height,
        blur_score=float(cv2.Laplacian(grey, cv2.CV_64F).var()),
        brightness=float(grey.mean()),
        contrast=float(grey.std()),
    )


def assess_quality(
    metrics: QualityMetrics,
    reference: Mapping[str, float] | None,
) -> List[Dict[str, Any]]:
    """
    Compare metrics with the training-distribution reference.

    Args:
        metrics: Measurements for the uploaded image.
        reference: Percentiles from calibration.json
            (keys like "blur_score_p01"). If None, only the
            resolution check runs.

    Returns:
        A list of {"code", "message"} warnings (empty when fine).
    """

    warnings: List[Dict[str, Any]] = []

    if min(metrics.width, metrics.height) < min(MODEL_INPUT_SIZE):
        warnings.append(
            {
                "code": "LOW_RESOLUTION",
                "message": (
                    "The image is smaller than the model's "
                    f"{MODEL_INPUT_SIZE[0]}x{MODEL_INPUT_SIZE[1]} input. "
                    "Use a larger, closer photo of the leaf."
                ),
            }
        )

    if not reference:
        return warnings

    if metrics.blur_score < reference["blur_score_p01"]:
        warnings.append(
            {
                "code": "BLURRY",
                "message": (
                    "The image looks blurrier than the images the model "
                    "was trained on. Hold the camera steady and focus "
                    "on the leaf."
                ),
            }
        )

    if metrics.brightness < reference["brightness_p01"]:
        warnings.append(
            {
                "code": "TOO_DARK",
                "message": (
                    "The image is darker than the training images. "
                    "Take the photo in even daylight."
                ),
            }
        )
    elif metrics.brightness > reference["brightness_p99"]:
        warnings.append(
            {
                "code": "TOO_BRIGHT",
                "message": (
                    "The image is brighter than the training images. "
                    "Avoid direct glare or overexposure."
                ),
            }
        )

    if metrics.contrast < reference["contrast_p01"]:
        warnings.append(
            {
                "code": "LOW_CONTRAST",
                "message": (
                    "The image has very low contrast. Make sure the leaf "
                    "fills most of the frame and is clearly visible."
                ),
            }
        )

    return warnings
