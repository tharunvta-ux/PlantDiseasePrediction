"""
backend/services/analysis_service.py

Hybrid analysis (Option B): combine the trained CNN with vision-LLM
plant identification.

Routes
------
supported_crop     LLM says it is crop X and the CNN's top crop is X
                   -> CNN diagnosis as-is.
crop_mismatch      LLM names a different crop than the CNN (or says the
                   plant is unsupported without high confidence)
                   -> CNN diagnosis kept but capped at "uncertain"; the
                   LLM's view and the CNN's best match within the LLM's
                   crop are shown as notes.
unsupported_plant  LLM is highly confident it is a plant the CNN was not
                   trained on -> no CNN diagnosis; LLM observation.
not_plant          Not a plant -> no diagnosis.
unverified         Identification unavailable (LLM down / no key)
                   -> CNN result as before, flagged as unverified.

Why the CNN wins crop disagreements: on PlantVillage leaves the vision
LLM named the correct crop for only 15 of 20 test images (some wrong
answers marked "high"), e.g. tomato leaves called potato. The CNN was
trained on exactly these crops. The LLM's strength is rejecting
non-plants and recognising plants outside the 14 crops.

The CNN is the only source of disease diagnoses for supported crops.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from backend.prediction.class_info import parse_class_name
from backend.prediction.confidence import LEVEL_HIGH, LEVEL_LOW, LEVEL_UNCERTAIN
from backend.services.languages import DEFAULT_LANGUAGE
from backend.services.llm import LLMError
from backend.services.plant_identification_service import NOT_SUPPORTED, identify_plant
from backend.services.prediction_service import PredictionRun, run_prediction

logger = logging.getLogger(__name__)

ROUTE_SUPPORTED = "supported_crop"
ROUTE_MISMATCH = "crop_mismatch"
ROUTE_UNSUPPORTED = "unsupported_plant"
ROUTE_NOT_PLANT = "not_plant"
ROUTE_UNVERIFIED = "unverified"

# The CNN was trained only on single-leaf photos.
LEAF_PARTS = {"leaf"}


def supported_crops(class_names: List[str]) -> List[str]:
    """Distinct crop names the CNN knows, in class order."""

    crops: List[str] = []

    for name in class_names:
        crop = parse_class_name(name).crop
        if crop not in crops:
            crops.append(crop)

    return crops


def _pct(value: float) -> float:
    return round(float(value) * 100, 2)


def _diagnosis_from_prediction(prediction: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Diagnosis block taken directly from the CNN result."""

    return {
        "source": source,
        "predicted_class": prediction["predicted_class"],
        "display_name": prediction["display_name"],
        "disease": prediction["disease"],
        "is_healthy": prediction["is_healthy"],
        "health_status": prediction["health_status"],
        "confidence": prediction["calibrated_confidence"],
        "confidence_level": prediction["confidence_level"],
        "top3_predictions": prediction["top3_predictions"],
    }


def _diagnosis_within_crop(
    run: PredictionRun,
    crop: str,
) -> Dict[str, Any]:
    """Best CNN class restricted to the crop the LLM identified."""

    probs = run.calibrated_probabilities
    indices = [i for i, name in enumerate(run.class_names) if parse_class_name(name).crop == crop]

    crop_mass = float(probs[indices].sum())
    within = probs[indices] / crop_mass if crop_mass > 0 else np.full(len(indices), 1 / len(indices))

    order = np.argsort(within)[::-1]
    best = indices[int(order[0])]
    info = parse_class_name(run.class_names[best])

    return {
        "source": "cnn_within_identified_crop",
        "predicted_class": run.class_names[best],
        "display_name": info.display_name,
        "disease": None if info.is_healthy else info.condition,
        "is_healthy": info.is_healthy,
        "health_status": "healthy" if info.is_healthy else "diseased",
        # Re-normalised within one crop: NOT a calibrated probability.
        "confidence": _pct(within[order[0]]),
        "confidence_level": LEVEL_UNCERTAIN,
        "model_crop_probability": _pct(crop_mass),
        "top3_predictions": [
            {"class": run.class_names[indices[int(i)]], "confidence": _pct(within[i])}
            for i in order[:3]
        ],
    }


def combine_results(
    run: PredictionRun,
    identification: Dict[str, Any] | None,
    identification_error: LLMError | None = None,
) -> Dict[str, Any]:
    """
    Merge CNN output and plant identification into one decision.
    Pure function (no I/O) so every route is unit-testable.
    """

    prediction = run.result
    notes: List[str] = []
    diagnosis: Dict[str, Any] | None
    llm_observation = False

    if identification is None:
        route = ROUTE_UNVERIFIED
        diagnosis = _diagnosis_from_prediction(prediction, "cnn")
        notes.append(
            "The plant type could not be verified, so the model assumes this "
            "is a leaf of one of its 14 supported crops."
        )
        message = prediction["message"]

    elif not identification["is_plant"]:
        route = ROUTE_NOT_PLANT
        diagnosis = None
        message = (
            "No plant was detected in this photo. Please upload a clear photo "
            "of a plant leaf."
        )

    elif (
        identification["supported_crop"] == NOT_SUPPORTED
        and identification["identification_confidence"] == "high"
    ):
        route = ROUTE_UNSUPPORTED
        diagnosis = None
        llm_observation = True
        name = identification["plant_common_name"] or "This plant"
        message = (
            f"{name} is not one of the 14 crops our disease model was trained "
            "on, so no model diagnosis is shown. Below is an AI observation "
            "of the photo - it is unverified; consult a local expert."
        )

    else:
        crop = identification["supported_crop"]
        diagnosis = _diagnosis_from_prediction(prediction, "cnn")

        if crop == prediction["crop"]:
            route = ROUTE_SUPPORTED
            message = prediction["message"]
        else:
            route = ROUTE_MISMATCH
            llm_observation = True
            ai_plant = identification["plant_common_name"] or crop

            if diagnosis["confidence_level"] == LEVEL_HIGH:
                diagnosis["confidence_level"] = LEVEL_UNCERTAIN

            notes.append(
                f"The AI identified the plant as {ai_plant}, but the disease "
                f"model recognised it as {prediction['crop']}. The model's "
                "result is shown, marked uncertain - the two disagree."
            )

            if crop != NOT_SUPPORTED:
                alternative = _diagnosis_within_crop(run, crop)
                diagnosis["alternative_if_ai_is_right"] = alternative
                notes.append(
                    f"If it is {crop}, the model's best {crop} match is "
                    f"{alternative['display_name']}."
                )

            message = (
                f"Possible {diagnosis['display_name']}. The result is uncertain "
                "because the plant identification and the disease model disagree."
            )

        if identification["plant_part"] not in LEAF_PARTS:
            notes.append(
                "The disease model was trained on leaf photos; this photo mainly "
                f"shows the {identification['plant_part'].replace('_', ' ')}."
            )
            if diagnosis["confidence_level"] == LEVEL_HIGH:
                diagnosis["confidence_level"] = LEVEL_UNCERTAIN

    if diagnosis and diagnosis["confidence_level"] == LEVEL_LOW:
        llm_observation = llm_observation or identification is not None

    return {
        "route": route,
        "message": message,
        "notes": notes,
        "diagnosis": diagnosis,
        "show_llm_observation": bool(llm_observation and identification and identification["is_plant"]),
        "prediction": prediction,
        "plant_identification": identification,
        "identification_error": (
            {"code": identification_error.code, "message": identification_error.public_message}
            if identification_error
            else None
        ),
    }


def analyze_upload(image_path: str, language: str = DEFAULT_LANGUAGE) -> Dict[str, Any]:
    """Run the CNN and plant identification on one image and combine."""

    run = run_prediction(image_path)

    identification = None
    error = None

    try:
        identification = identify_plant(
            image_path,
            supported_crops(run.class_names),
            language=language,
        )
    except LLMError as exc:
        logger.warning("Plant identification unavailable [%s]: %s", exc.code, exc.detail)
        error = exc

    result = combine_results(run, identification, error)

    logger.info("Analysis route=%s diagnosis=%s", result["route"],
                result["diagnosis"]["predicted_class"] if result["diagnosis"] else None)

    return result
