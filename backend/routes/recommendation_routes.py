"""
Treatment recommendation API routes.

POST /recommendation
    Body (JSON), normally the /predict response:
        {
            "predicted_class": "Tomato___Early_blight",
            "confidence_level": "high" | "uncertain",
            "top3_predictions": [{"class": "...", "confidence": 12.3}, ...],
            "language": "en" | "hi" | "ta" | ...
        }

POST /speech
    Same body; returns the guidance read aloud as audio/wav.
"""

import io
import logging

from flask import Blueprint
from flask import jsonify
from flask import request
from flask import send_file

from backend.prediction.predictor import get_predictor
from backend.services.languages import LANGUAGES, normalise_language
from backend.services.llm import LLMError
from backend.services.recommendation_service import (
    RecommendationRequestError,
    generate_recommendation,
    validate_request,
)
from backend.services.speech_service import speak_guidance

recommendation_bp = Blueprint(
    "recommendation",
    __name__
)

logger = logging.getLogger(__name__)


def _error(message: str, code: str, status: int):
    """Uniform JSON error body."""

    return jsonify({"error": message, "code": code}), status


def _validated(payload):
    """
    Validate a recommendation/speech request body.

    Returns:
        (predicted_class, level, alternatives, language)

    Raises:
        RecommendationRequestError
    """

    if not isinstance(payload, dict):
        raise RecommendationRequestError("Request body must be a JSON object.", "INVALID_JSON")

    predicted_class, level, alternatives = validate_request(
        payload.get("predicted_class"),
        payload.get("confidence_level"),
        payload.get("top3_predictions"),
        get_predictor().class_names,
    )

    language = normalise_language(payload.get("language"))

    if language is None:
        raise RecommendationRequestError("Unsupported language.", "INVALID_LANGUAGE")

    return predicted_class, level, alternatives, language


@recommendation_bp.route("/speech", methods=["POST"])
def speech():
    """
    WAV audio of the treatment guidance (same body as /recommendation).
    """

    try:
        predicted_class, level, alternatives, language = _validated(request.get_json(silent=True))

        audio = speak_guidance(predicted_class, level, alternatives, language)

        response = send_file(io.BytesIO(audio["wav"]), mimetype="audio/wav")
        response.headers["X-TTS-Model"] = audio["model"]

        return response

    except RecommendationRequestError as exc:
        return _error(str(exc), exc.code, exc.http_status)

    except LLMError as exc:
        logger.warning("Speech failed [%s]: %s", exc.code, exc.detail)
        return _error(exc.public_message, exc.code, exc.http_status)

    except Exception:
        logger.exception("Speech failed")
        return _error("Could not create the audio.", "SERVER_ERROR", 500)


@recommendation_bp.route("/languages", methods=["GET"])
def languages():
    """Languages available for AI-generated guidance."""

    return jsonify(
        [
            {"code": code, "name": name, "native_name": native, "speech_lang": speech}
            for code, (name, native, speech) in LANGUAGES.items()
        ]
    ), 200


@recommendation_bp.route("/recommendation", methods=["POST"])
def recommendation():
    """
    Generate LLM treatment guidance for a model prediction.
    """

    try:
        predicted_class, level, alternatives, language = _validated(request.get_json(silent=True))

        result = generate_recommendation(
            predicted_class,
            level,
            alternatives,
            language=language,
        )

        return jsonify(result), 200

    except RecommendationRequestError as exc:
        return _error(str(exc), exc.code, exc.http_status)

    except LLMError as exc:
        logger.warning("Recommendation failed [%s]: %s", exc.code, exc.detail)
        return _error(exc.public_message, exc.code, exc.http_status)

    except Exception:
        logger.exception("Recommendation failed")
        return _error(
            "Treatment guidance failed due to a server error.",
            "SERVER_ERROR",
            500,
        )
