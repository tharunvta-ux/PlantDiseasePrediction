"""
Treatment recommendation API routes.

POST /recommendation
    Body (JSON), normally the /predict response:
        {
            "predicted_class": "Tomato___Early_blight",
            "confidence_level": "high" | "uncertain",
            "top3_predictions": [{"class": "...", "confidence": 12.3}, ...]
        }
"""

import logging

from flask import Blueprint
from flask import jsonify
from flask import request

from backend.prediction.predictor import get_predictor
from backend.services.llm import LLMError
from backend.services.recommendation_service import (
    RecommendationRequestError,
    generate_recommendation,
    validate_request,
)

recommendation_bp = Blueprint(
    "recommendation",
    __name__
)

logger = logging.getLogger(__name__)


def _error(message: str, code: str, status: int):
    """Uniform JSON error body."""

    return jsonify({"error": message, "code": code}), status


@recommendation_bp.route("/recommendation", methods=["POST"])
def recommendation():
    """
    Generate LLM treatment guidance for a model prediction.
    """

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object.", "INVALID_JSON", 400)

    try:
        predicted_class, level, alternatives = validate_request(
            payload.get("predicted_class"),
            payload.get("confidence_level"),
            payload.get("top3_predictions"),
            get_predictor().class_names,
        )

        result = generate_recommendation(predicted_class, level, alternatives)

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
