"""
backend/prediction/severity.py

Rough disease severity from a photo: the share of visible leaf area that
is not healthy green (spots, yellowing, browning, white powder).

    leaf region  green or yellow plant tissue, closed and filled so spots
                 and powder inside a leaf count as part of it
    affected     leaf-region pixels that are not healthy green

This is the classic "% leaf area diseased" measure, done with colour
thresholds - deterministic, so photos taken a week apart can be
compared. It is an approximation: shadows, reflections or a different
camera angle shift it by several percentage points, so results are
reported as grades (mild / moderate / severe), not exact numbers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import cv2
import numpy as np

ANALYSIS_SIDE = 512

# OpenCV hue is 0-179 (degrees / 2).
GREEN_HUE = (35, 90)
PLANT_HUE = (15, 90)  # yellow to green: yellowing leaves are still leaves
MIN_SATURATION = 45
# Dull or shiny green is still healthy leaf.
MIN_GREEN_SATURATION = 20
MIN_VALUE = 35
# Bright, colourless pixels inside a leaf: white powder (mildew).
# Pale powdery mildew and virus mottling barely change colour, so this
# measure underestimates them; the grower's own estimate covers that.
POWDER_MIN_VALUE = 160

# Leaf region must cover at least this share of the photo to be trusted.
MIN_LEAF_COVERAGE = 0.08
MIN_COMPONENT_SHARE = 0.01

GRADE_MILD = "mild"
GRADE_MODERATE = "moderate"
GRADE_SEVERE = "severe"

# Upper bounds (affected % of leaf area) for each grade.
MILD_MAX_PERCENT = 10.0
MODERATE_MAX_PERCENT = 30.0


@dataclass(frozen=True)
class SeverityMeasurement:
    affected_percent: float  # of the visible leaf area
    leaf_coverage_percent: float  # of the photo
    grade: str
    reliable: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def grade_for_percent(percent: float) -> str:
    if percent < MILD_MAX_PERCENT:
        return GRADE_MILD
    if percent < MODERATE_MAX_PERCENT:
        return GRADE_MODERATE
    return GRADE_SEVERE


def _load(image_path: str) -> np.ndarray:
    data = np.fromfile(image_path, dtype=np.uint8)  # works with non-ASCII paths
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    scale = ANALYSIS_SIDE / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return image


def _leaf_region(plant: np.ndarray) -> np.ndarray:
    """Close gaps and fill each sizeable plant blob (spots become interior)."""

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(plant, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_COMPONENT_SHARE * plant.size

    region = np.zeros_like(plant)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(region, [contour], -1, 255, thickness=cv2.FILLED)

    return region


def measure_severity(image_path: str) -> SeverityMeasurement:
    """Share of the visible leaf area that is diseased."""

    hsv = cv2.cvtColor(_load(image_path), cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    lit = val >= MIN_VALUE
    is_green_hue = (hue >= GREEN_HUE[0]) & (hue <= GREEN_HUE[1])
    coloured = (sat >= MIN_SATURATION) & lit

    green = is_green_hue & (sat >= MIN_GREEN_SATURATION) & lit
    plant = coloured & (hue >= PLANT_HUE[0]) & (hue <= PLANT_HUE[1])
    # Yellow, brown, orange or red spots / patches.
    lesion = coloured & ~is_green_hue
    powder = (sat < MIN_SATURATION) & (val >= POWDER_MIN_VALUE)

    region = _leaf_region(plant.astype(np.uint8) * 255) > 0
    coverage = float(region.sum()) / region.size

    # Grey and very dark pixels (background showing through, shadows)
    # count neither way.
    healthy = region & green
    affected = region & ~green & (lesion | powder)
    leaf_pixels = int(healthy.sum() + affected.sum())

    if leaf_pixels == 0:
        return SeverityMeasurement(0.0, 0.0, GRADE_MILD, reliable=False)

    percent = 100.0 * float(affected.sum()) / leaf_pixels

    return SeverityMeasurement(
        affected_percent=round(percent, 1),
        leaf_coverage_percent=round(100.0 * coverage, 1),
        grade=grade_for_percent(percent),
        reliable=coverage >= MIN_LEAF_COVERAGE,
    )
