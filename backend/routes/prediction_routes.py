"""
Prediction API routes.
"""

import logging

from flask import Blueprint
from flask import jsonify
from werkzeug.exceptions import HTTPException

from backend.routes.upload_utils import UploadError, allowed_file, saved_image_upload  # noqa: F401
from backend.services.prediction_service import analyze_image

prediction_bp = Blueprint(
    "prediction",
    __name__
)

logger = logging.getLogger(__name__)


@prediction_bp.route("/", methods=["GET"])
def home():
    """
    Health endpoint.
    """

    return jsonify(
        {
            "message": "Plant Disease Prediction API Running"
        }
    ), 200


@prediction_bp.route("/predict", methods=["POST"])
def predict():
    """
    Predict disease from uploaded image.

    Response keeps the original fields (predicted_class, confidence,
    top3_predictions) and adds crop, health, calibrated confidence,
    confidence_level, quality_warnings and message.
    """

    try:
        with saved_image_upload() as filepath:
            return jsonify(analyze_image(filepath)), 200

    except UploadError as exc:
        return exc.response

    except HTTPException:
        raise  # e.g. 413 upload too large -> app error handler

    except Exception:

        logger.exception("Prediction failed")

        return jsonify(
            {
                "error": "Prediction failed due to a server error. "
                         "Please try again."
            }
        ), 500
