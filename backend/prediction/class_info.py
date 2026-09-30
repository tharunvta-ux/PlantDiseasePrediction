"""
backend/prediction/class_info.py

Helpers for interpreting PlantVillage class names such as
"Tomato___Early_blight" or "Corn_(maize)___healthy".
"""

from __future__ import annotations

from dataclasses import dataclass

CLASS_SEPARATOR = "___"
HEALTHY_LABEL = "healthy"


@dataclass(frozen=True)
class ClassInfo:
    """Parsed information about one model class."""

    raw_name: str
    crop: str
    condition: str
    is_healthy: bool

    @property
    def display_name(self) -> str:
        """Human-readable name, e.g. 'Tomato — Early blight'."""

        return f"{self.crop} — {self.condition}"


def _humanize(part: str) -> str:
    """Turn 'Corn_(maize)' into 'Corn (maize)'."""

    text = part.replace("_", " ").strip()
    text = " ".join(text.split())

    return text[:1].upper() + text[1:]


def parse_class_name(raw_name: str) -> ClassInfo:
    """
    Split a PlantVillage class name into crop and condition.

    Raises:
        ValueError: If the name does not follow "<crop>___<condition>".
    """

    if CLASS_SEPARATOR not in raw_name:
        raise ValueError(f"Unrecognised class name: {raw_name!r}")

    crop_part, condition_part = raw_name.split(CLASS_SEPARATOR, 1)

    is_healthy = condition_part.strip("_").lower() == HEALTHY_LABEL

    return ClassInfo(
        raw_name=raw_name,
        crop=_humanize(crop_part),
        condition="Healthy" if is_healthy else _humanize(condition_part),
        is_healthy=is_healthy,
    )
