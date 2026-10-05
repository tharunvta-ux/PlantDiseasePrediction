"""
backend/services/plant_identification_service.py

Stage 1 of the hybrid pipeline: a vision LLM (Gemini) looks at the photo
and reports
- whether it shows a plant at all,
- which plant it is and whether it is one of the crops our CNN supports,
- which plant part is visible,
- visible symptoms / possible issues (used ONLY for plants the CNN does
  not support, and always labelled as an unverified AI observation).

Disease diagnosis for supported crops stays with the trained CNN.
"""

from __future__ import annotations

import hashlib
import io
import logging
from typing import Any, Dict, List, Literal, Sequence

from PIL import Image, ImageOps
from pydantic import BaseModel, Field, ValidationError

from backend.services.languages import DEFAULT_LANGUAGE, language_instruction
from backend.services.llm import (
    ImageInput,
    LLMInvalidResponseError,
    LLMProvider,
    get_llm_provider,
)
from backend.services.recommendation_service import DOSAGE_PATTERN, LRUCache

logger = logging.getLogger(__name__)

NOT_SUPPORTED = "none"

# Images are downscaled before upload: faster, cheaper, enough detail.
MAX_IMAGE_SIDE = 1024
JPEG_QUALITY = 85

CACHE_SIZE = 256


class PlantIdentification(BaseModel):
    """Structured output of the vision model."""

    is_plant: bool = Field(description="True if the photo mainly shows a real plant or plant part.")
    plant_common_name: str = Field(description="Common name, or empty string if unknown / not a plant.")
    plant_scientific_name: str = Field(description="Scientific name, or empty string if unsure.")
    supported_crop: str = Field(
        description="Exactly one value from the allowed list if the plant is that crop, otherwise 'none'."
    )
    identification_confidence: Literal["high", "medium", "low"] = Field(
        description="How sure you are about the plant identity."
    )
    plant_part: Literal["leaf", "fruit", "flower", "stem", "whole_plant", "other", "none"] = Field(
        description="Main plant part visible in the photo."
    )
    appears_healthy: Literal["yes", "no", "unclear"] = Field(
        description="Whether the visible plant looks healthy."
    )
    visible_symptoms: List[str] = Field(description="Symptoms actually visible in the photo (empty if none).")
    possible_issues: List[str] = Field(
        description="Possible causes of the symptoms, each phrased as a possibility, most likely first."
    )
    general_advice: List[str] = Field(
        description="Low-risk, general care steps. No product names, no doses."
    )
    image_notes: str = Field(description="Short note on photo quality or framing problems, or empty string.")


SYSTEM_PROMPT = """\
You identify plants in photos for a plant-health app.

Tasks:
1. Decide if the photo mainly shows a real plant or plant part. Drawings, \
plain coloured areas, objects, people or animals are not plants.
2. Identify the plant (common and scientific name) as precisely as the \
photo allows. Say "low" confidence when unsure; never invent certainty.
3. Set supported_crop to one value from the allowed list ONLY if the \
plant is that crop; otherwise "none".
4. Describe only symptoms you can actually see. List possible issues as \
possibilities ("possibly ...", "could be ..."), not diagnoses.
5. General advice must be low-risk (inspection, hygiene, watering, \
removing affected parts, getting expert confirmation). NEVER give product \
or brand names, doses, rates or mixing ratios.

If it is not a plant: is_plant=false, supported_crop="none", empty lists.
Respond only with JSON matching the schema.
"""


def build_schema(supported_crops: Sequence[str]) -> Dict[str, Any]:
    """JSON schema with supported_crop restricted to the allowed values."""

    schema = PlantIdentification.model_json_schema()
    schema["properties"]["supported_crop"]["enum"] = [*supported_crops, NOT_SUPPORTED]

    return schema


def prepare_image(image_path: str) -> ImageInput:
    """Upright, RGB, at most MAX_IMAGE_SIDE px, re-encoded as JPEG."""

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY)

    return ImageInput(data=buffer.getvalue(), mime_type="image/jpeg")


def _normalise_crop(value: str, supported_crops: Sequence[str]) -> str:
    """Map the model's answer onto an allowed crop name or 'none'."""

    lookup = {crop.lower(): crop for crop in supported_crops}

    return lookup.get(value.strip().lower(), NOT_SUPPORTED)


def _redact(items: List[str]) -> tuple[List[str], int]:
    kept = [item for item in items if not DOSAGE_PATTERN.search(item)]
    return kept, len(items) - len(kept)


_cache = LRUCache(CACHE_SIZE)


def clear_cache() -> None:
    """Empty the identification cache (used by tests)."""

    _cache.clear()


def identify_plant(
    image_path: str,
    supported_crops: Sequence[str],
    provider: LLMProvider | None = None,
    language: str = DEFAULT_LANGUAGE,
) -> Dict[str, Any]:
    """
    Identify the plant in an image.

    Raises:
        LLMError subclasses when the provider fails or answers invalidly.
    """

    provider = provider or get_llm_provider()
    image = prepare_image(image_path)

    key = (
        provider.name,
        provider.model,
        hashlib.sha256(image.data).hexdigest(),
        tuple(supported_crops),
        language,
    )

    cached = _cache.get(key)

    if cached is not None:
        return {**cached, "cached": True}

    crops_text = ", ".join(f'"{c}"' for c in supported_crops)

    response = provider.generate_json(
        SYSTEM_PROMPT,
        f"Allowed supported_crop values: {crops_text}, \"{NOT_SUPPORTED}\".\n"
        "Identify the plant in this photo. Keep plant_common_name in English. "
        "For visible_symptoms, possible_issues, general_advice and image_notes: "
        f"{language_instruction(language)}",
        build_schema(supported_crops),
        image=image,
    )

    try:
        parsed = PlantIdentification.model_validate_json(response.text)
    except ValidationError as exc:
        logger.warning("Plant identification failed validation: %s", exc)
        raise LLMInvalidResponseError(str(exc)) from exc

    result = parsed.model_dump()
    result["supported_crop"] = (
        _normalise_crop(parsed.supported_crop, supported_crops) if parsed.is_plant else NOT_SUPPORTED
    )

    removed = 0
    for field in ("possible_issues", "general_advice"):
        result[field], count = _redact(result[field])
        removed += count

    result["safety"] = {"redacted_items": removed}
    result["generated_by"] = {"provider": provider.name, "model": response.model}

    logger.info(
        "Plant identification: is_plant=%s plant=%r supported_crop=%s confidence=%s part=%s",
        result["is_plant"],
        result["plant_common_name"],
        result["supported_crop"],
        result["identification_confidence"],
        result["plant_part"],
    )

    _cache.set(key, result)

    return {**result, "cached": False}
