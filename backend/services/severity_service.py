"""
backend/services/severity_service.py

Severity grade, trend between check-ins, and recovery outlook.

Severity combines two estimates and keeps the worse one:
- the photo measurement (share of leaf area spotted / yellow / brown /
  powdery, see backend/prediction/severity.py), and
- the grower's answer to "How much of the plant has this?" - one photo
  shows a few leaves; the grower sees the whole plant.

The outlook is a plain-language category (good / fair / poor) with its
reason, deliberately NOT a percentage: there is no data to back a number
like "90% chance", and growers would rely on it.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from backend.prediction.severity import (
    GRADE_MILD,
    GRADE_MODERATE,
    GRADE_SEVERE,
)
from backend.services.treatment_plan_service import is_viral, plan_entry

EXTENT_FEW = "few_leaves"
EXTENT_HALF = "about_half"
EXTENT_MOST = "most"

EXTENT_GRADE = {EXTENT_FEW: GRADE_MILD, EXTENT_HALF: GRADE_MODERATE, EXTENT_MOST: GRADE_SEVERE}
GRADE_ORDER = [GRADE_MILD, GRADE_MODERATE, GRADE_SEVERE]

OUTLOOK_GOOD = "good"
OUTLOOK_FAIR = "fair"
OUTLOOK_POOR = "poor"
OUTLOOK_ORDER = [OUTLOOK_POOR, OUTLOOK_FAIR, OUTLOOK_GOOD]

TREND_IMPROVING = "improving"
TREND_SAME = "same"
TREND_WORSENING = "worsening"

# A photo change must be this large (relative and absolute) to count.
TREND_RELATIVE = 0.3
TREND_MIN_POINTS = 3.0


def clean_extent(value: Any) -> str | None:
    return value if value in EXTENT_GRADE else None


def clean_photo_severity(value: Any) -> Dict[str, Any] | None:
    """A photo measurement sent back by the client (preview only)."""

    if not isinstance(value, Mapping):
        return None

    percent = value.get("affected_percent")
    grade = value.get("grade")

    if not isinstance(percent, (int, float)) or not 0 <= percent <= 100 or grade not in GRADE_ORDER:
        return None

    return {"affected_percent": float(percent), "grade": grade, "reliable": bool(value.get("reliable"))}


def combine(photo: Mapping[str, Any] | None, extent: str | None) -> Dict[str, Any] | None:
    """Severity from the photo measurement and the grower's estimate."""

    photo_ok = bool(photo and photo.get("reliable"))
    photo_grade = photo["grade"] if photo_ok else None
    extent_grade = EXTENT_GRADE.get(extent) if extent else None

    grades = [g for g in (photo_grade, extent_grade) if g]
    if not grades:
        return None

    return {
        "grade": max(grades, key=GRADE_ORDER.index),
        "photo_grade": photo_grade,
        "affected_percent": photo["affected_percent"] if photo_ok else None,
        "extent": extent,
    }


def trend(previous: Mapping[str, Any] | None, current: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    """Did the disease shrink or spread since the last measurement?"""

    if not previous or not current:
        return None

    before, after = previous.get("affected_percent"), current.get("affected_percent")

    if before is not None and after is not None:
        change = after - before
        if change <= -TREND_MIN_POINTS and after <= before * (1 - TREND_RELATIVE):
            direction = TREND_IMPROVING
        elif change >= TREND_MIN_POINTS and after >= before * (1 + TREND_RELATIVE):
            direction = TREND_WORSENING
        else:
            direction = TREND_SAME
        text = f"about {before:.0f}% of the leaf affected before, about {after:.0f}% now"
    else:
        step = GRADE_ORDER.index(current["grade"]) - GRADE_ORDER.index(previous["grade"])
        direction = TREND_IMPROVING if step < 0 else TREND_WORSENING if step > 0 else TREND_SAME
        text = f"{previous['grade']} before, {current['grade']} now"

    return {"direction": direction, "text": text}


def outlook(
    predicted_class: str | None,
    severity: Mapping[str, Any] | None,
    change: Mapping[str, Any] | None = None,
) -> Dict[str, Any] | None:
    """Recovery outlook: good / fair / poor, with the reason."""

    if predicted_class and (is_viral(predicted_class) or plan_entry(predicted_class).get("incurable")):
        return {
            "level": OUTLOOK_POOR,
            "text": (
                "This disease cannot be cured on the plant. Remove infected plants so "
                "it does not spread to healthy ones."
            ),
        }

    if not severity:
        return None

    level = {GRADE_MILD: OUTLOOK_GOOD, GRADE_MODERATE: OUTLOOK_FAIR, GRADE_SEVERE: OUTLOOK_POOR}[severity["grade"]]
    direction = change["direction"] if change else TREND_SAME

    index = OUTLOOK_ORDER.index(level)
    if direction == TREND_IMPROVING:
        level = OUTLOOK_ORDER[min(index + 1, 2)]
    elif direction == TREND_WORSENING:
        level = OUTLOOK_ORDER[max(index - 1, 0)]

    reason = {
        TREND_IMPROVING: "the disease is shrinking",
        TREND_WORSENING: "the disease is spreading",
    }.get(direction, {
        GRADE_MILD: "the problem is still small",
        GRADE_MODERATE: "a fair part of the plant is affected",
        GRADE_SEVERE: "much of the plant is affected",
    }[severity["grade"]])

    text = {
        OUTLOOK_GOOD: f"Good chance of recovery - {reason}. Follow the steps and check on time.",
        OUTLOOK_FAIR: f"Fair chance of recovery - {reason}. Act now; following the steps closely gives the best result.",
        OUTLOOK_POOR: (
            f"Recovery is unlikely - {reason}. Protect nearby plants first, and ask an "
            "expert whether this plant is worth treating or should be removed."
        ),
    }[level]

    return {"level": level, "text": text}


def assess(
    predicted_class: str | None,
    photo: Mapping[str, Any] | None,
    extent: str | None,
    previous: Mapping[str, Any] | None = None,
) -> Dict[str, Any] | None:
    """Severity + trend + outlook, as stored on cases and check-ins."""

    severity = combine(photo, extent)
    change = trend(previous, severity)
    view = outlook(predicted_class, severity, change)

    if severity is None and view is None:
        return None

    return {**(severity or {}), "trend": change, "outlook": view}
