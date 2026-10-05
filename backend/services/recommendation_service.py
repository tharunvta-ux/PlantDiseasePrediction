"""
backend/services/recommendation_service.py

Generates structured treatment / prevention guidance for a disease that
the computer-vision model has ALREADY predicted.

The LLM never sees the image and never diagnoses. It receives only:
- the validated class name(s) from the model,
- the confidence level from the calibrated policy,
- curated facts from backend/knowledge/disease_info.json.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence

from pydantic import BaseModel, Field, ValidationError

from backend.prediction.class_info import parse_class_name
from backend.prediction.confidence import LEVEL_HIGH, LEVEL_LOW, LEVEL_UNCERTAIN
from backend.services.languages import (
    DEFAULT_LANGUAGE,
    LANGUAGES,
    language_instruction,
)
from backend.services.llm import (
    LLMInvalidResponseError,
    LLMProvider,
    get_llm_provider,
)

logger = logging.getLogger(__name__)

KNOWLEDGE_FILE = (
    Path(__file__).resolve().parents[1] / "knowledge" / "disease_info.json"
)

CACHE_SIZE = 128

MAX_ALTERNATIVES = 2

DISCLAIMER = (
    "AI-generated guidance based on an automated image prediction. It is "
    "not a confirmed diagnosis. Treatment depends on the crop, severity "
    "and local regulations - always follow product labels and local "
    "agricultural extension advice."
)


# ==========================================================
# Errors
# ==========================================================


class RecommendationRequestError(ValueError):
    """Invalid input to the recommendation service (HTTP 400/422)."""

    def __init__(self, message: str, code: str, http_status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status


# ==========================================================
# Output schema
# ==========================================================


class TreatmentGuidance(BaseModel):
    """Structured guidance returned by the LLM."""

    disease: str = Field(description="Name of the predicted condition and crop.")
    what_it_means: str = Field(
        description="Plain-language explanation of the condition and its cause."
    )
    common_symptoms: List[str] = Field(description="Typical visible symptoms.")
    recommended_treatment: List[str] = Field(
        description=(
            "Treatment options. Generic active-ingredient categories only; "
            "no product names, no dosages."
        )
    )
    preventive_measures: List[str] = Field(description="How to prevent recurrence.")
    immediate_steps: List[str] = Field(
        description="Practical steps the grower can take today."
    )
    when_to_seek_expert_help: List[str] = Field(
        description="Situations where a local agronomist / extension service is needed."
    )
    uncertainty_note: str = Field(
        description="How confident the diagnosis is and what that means for the advice."
    )


RESPONSE_SCHEMA = TreatmentGuidance.model_json_schema()

LIST_SECTIONS = (
    "common_symptoms",
    "recommended_treatment",
    "preventive_measures",
    "immediate_steps",
    "when_to_seek_expert_help",
)


# ==========================================================
# Prompts
# ==========================================================

SYSTEM_PROMPT = """\
You are an agricultural extension assistant. A computer-vision model has \
already classified a photo of a plant leaf. You explain that result and \
give practical, safe guidance. You do NOT diagnose - you never see the \
image and must not contradict or replace the model's prediction.

Rules:
- Use the provided facts (pathogen type, causal agent, key facts). Do not \
contradict them. Match advice to the pathogen type: no fungicides for \
viral, bacterial or pest problems; no "cure" for viral diseases or \
citrus greening.
- Chemical control: name generic active-ingredient categories only \
(e.g. "copper-based bactericide", "fungicide labelled for late blight \
on tomato"). NEVER give product or brand names. NEVER give doses, rates, \
concentrations, mixing ratios or spray intervals.
- Whenever chemical control is mentioned, say to follow the product label \
and local agricultural guidance, and that suitability depends on the crop, \
severity and local regulations.
- Prefer cultural and non-chemical measures first.
- If the confidence level is "uncertain", do not state the diagnosis as \
fact: explain it is a possible diagnosis, mention the alternatives given, \
recommend confirming (clearer photo or local expert) before any chemical \
treatment, and keep immediate steps low-risk.
- If the plant is healthy, say so; give brief care and monitoring advice \
and state that no treatment is needed.
- Be concise: 2-5 short items per list, plain language for farmers and \
home gardeners.
- Respond only with JSON matching the provided schema.
"""


@lru_cache(maxsize=1)
def load_knowledge() -> Dict[str, Any]:
    """Load the curated disease facts."""

    return json.loads(KNOWLEDGE_FILE.read_text(encoding="utf-8"))


def _facts_for(class_name: str) -> Dict[str, Any]:
    """Prompt-ready facts for one class."""

    info = parse_class_name(class_name)
    entry = load_knowledge().get(class_name, {})

    return {
        "crop": info.crop,
        "condition": info.condition,
        "is_healthy": info.is_healthy,
        "pathogen_type": entry.get("pathogen_type", "unknown"),
        "causal_agent": entry.get("causal_agent"),
        "key_facts": entry.get("key_facts", []),
    }


def build_user_prompt(
    predicted_class: str,
    level: str,
    alternatives: Sequence[str],
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """Compose the user prompt from validated model output only."""

    payload = {
        "model_prediction": _facts_for(predicted_class),
        "confidence_level": level,
        "alternative_predictions": [_facts_for(name) for name in alternatives],
    }

    return (
        "Explain this plant-disease prediction and give structured "
        f"guidance. {language_instruction(language)}\n\n"
        + json.dumps(payload, indent=2)
    )


# ==========================================================
# Safety post-processing
# ==========================================================

_UNIT = (
    r"(?:ml|millilit(?:re|er)s?|l|lit(?:re|er)s?|g|grams?|kg|oz|ounces?|lbs?|"
    r"pounds?|%|ppm|tsp|teaspoons?|tbsp|tablespoons?|cups?|gal|gallons?|pints?|"
    r"quarts?|cc)"
)

DOSAGE_PATTERN = re.compile(
    rf"\d+(?:[.,]\d+)?\s*(?:-|to|–)?\s*(?:\d+(?:[.,]\d+)?\s*)?{_UNIT}(?![a-z])"
    rf"|\b\d+(?:[.,]\d+)?\s*{_UNIT}?\s*(?:per|/)\s*"
    r"(?:acre|hectare|ha|gallon|litre|liter|l|plant|tree|m2|sq)\b"
    r"|[®™]",
    re.IGNORECASE,
)


def redact_unsafe_items(guidance: TreatmentGuidance) -> int:
    """
    Remove list items that contain dosages/rates or trademark symbols.

    Returns:
        Number of removed items.
    """

    removed = 0

    for section in LIST_SECTIONS:
        items = getattr(guidance, section)
        kept = [item for item in items if not DOSAGE_PATTERN.search(item)]
        removed += len(items) - len(kept)
        setattr(guidance, section, kept)

    return removed


# ==========================================================
# Cache
# ==========================================================


class LRUCache:
    """Tiny thread-safe LRU cache."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._data: OrderedDict[tuple, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple) -> Dict[str, Any] | None:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: tuple, value: Dict[str, Any]) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_cache = LRUCache(CACHE_SIZE)


def clear_cache() -> None:
    """Empty the guidance cache (used by tests)."""

    _cache.clear()


# ==========================================================
# Public API
# ==========================================================


def validate_request(
    predicted_class: Any,
    level: Any,
    top_predictions: Any,
    known_classes: Sequence[str],
) -> tuple[str, str, List[str]]:
    """
    Validate and normalise a recommendation request.

    Only exact model class names are accepted, so no free text from the
    client ever reaches the LLM prompt.
    """

    known = set(known_classes)

    if not isinstance(predicted_class, str) or predicted_class not in known:
        raise RecommendationRequestError(
            "predicted_class must be one of the model's class names.",
            code="INVALID_CLASS",
        )

    if level not in (LEVEL_HIGH, LEVEL_UNCERTAIN, LEVEL_LOW):
        raise RecommendationRequestError(
            "confidence_level must be 'high', 'uncertain' or 'low'.",
            code="INVALID_CONFIDENCE_LEVEL",
        )

    if level == LEVEL_LOW:
        raise RecommendationRequestError(
            "Treatment guidance is not generated for low-confidence "
            "predictions. Please upload a clearer image first.",
            code="LOW_CONFIDENCE",
            http_status=422,
        )

    alternatives: List[str] = []

    if top_predictions is not None:
        if not isinstance(top_predictions, list):
            raise RecommendationRequestError(
                "top3_predictions must be a list.",
                code="INVALID_TOP_PREDICTIONS",
            )

        for item in top_predictions[:3]:
            name = item.get("class") if isinstance(item, dict) else None

            if name not in known:
                raise RecommendationRequestError(
                    "top3_predictions contains an unknown class.",
                    code="INVALID_TOP_PREDICTIONS",
                )

            if name != predicted_class and name not in alternatives:
                alternatives.append(name)

    # Alternatives only matter when the prediction is uncertain.
    if level != LEVEL_UNCERTAIN:
        alternatives = []

    return predicted_class, level, alternatives[:MAX_ALTERNATIVES]


def generate_recommendation(
    predicted_class: str,
    level: str,
    alternatives: Sequence[str],
    provider: LLMProvider | None = None,
    language: str = DEFAULT_LANGUAGE,
) -> Dict[str, Any]:
    """
    Produce structured guidance for a validated prediction.

    Raises:
        LLMError subclasses when the provider fails.
    """

    provider = provider or get_llm_provider()

    key = (provider.name, provider.model, predicted_class, level, tuple(alternatives), language)

    cached = _cache.get(key)

    if cached is not None:
        logger.info("Recommendation cache hit for %s (%s, %s)", predicted_class, level, language)
        return {**cached, "cached": True}

    response = provider.generate_json(
        SYSTEM_PROMPT,
        build_user_prompt(predicted_class, level, alternatives, language),
        RESPONSE_SCHEMA,
    )

    try:
        guidance = TreatmentGuidance.model_validate_json(response.text)
    except ValidationError as exc:
        logger.warning("LLM output failed validation: %s", exc)
        raise LLMInvalidResponseError(str(exc)) from exc

    removed = redact_unsafe_items(guidance)

    if removed:
        logger.warning(
            "Removed %d guidance item(s) containing dosages/brands for %s",
            removed,
            predicted_class,
        )

    facts = _facts_for(predicted_class)
    knowledge_meta = load_knowledge().get("_meta", {})

    result = {
        "predicted_class": predicted_class,
        "display_name": parse_class_name(predicted_class).display_name,
        "confidence_level": level,
        "alternatives": [
            {"class": name, "display_name": parse_class_name(name).display_name}
            for name in alternatives
        ],
        "guidance": guidance.model_dump(),
        "grounding": {
            "pathogen_type": facts["pathogen_type"],
            "causal_agent": facts["causal_agent"],
            "knowledge_expert_reviewed": bool(
                knowledge_meta.get("expert_reviewed", False)
            ),
        },
        "safety": {"redacted_items": removed},
        "generated_by": {"provider": provider.name, "model": response.model},
        "disclaimer": DISCLAIMER,
        "language": language,
        "speech_lang": LANGUAGES[language][2],
    }

    _cache.set(key, result)

    return {**result, "cached": False}
