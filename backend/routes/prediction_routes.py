"""
Prediction API routes.
"""

import logging
import os
import uuid

from flask import Blueprint
from flask import current_app
from flask import jsonify
from flask import request
from PIL import Image, UnidentifiedImageError

from backend.config import ALLOWED_EXTENSIONS
from backend.services.prediction_service import analyze_image

prediction_bp = Blueprint(
    "prediction",
    __name__
)

logger = logging.getLogger(__name__)


def allowed_file(filename: str) -> bool:
    """
    Check if uploaded file has supported extension.
    """

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


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

    if "file" not in request.files:
        return jsonify(
            {"error": "No file uploaded"}
        ), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify(
            {"error": "No selected file"}
        ), 400

    if not allowed_file(file.filename):
        return jsonify(
            {"error": "Unsupported file type"}
        ), 400

    extension = file.filename.rsplit(".", 1)[1].lower()

    filename = f"{uuid.uuid4()}.{extension}"

    filepath = os.path.join(
        current_app.config["UPLOAD_FOLDER"],
        filename
    )

    try:

        file.save(filepath)

        try:
            with Image.open(filepath) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, SyntaxError):
            return jsonify(
                {"error": "The uploaded file is not a valid image"}
            ), 400

        prediction = analyze_image(filepath)

        return jsonify(prediction), 200

    except Exception:

        logger.exception("Prediction failed")

        return jsonify(
            {
                "error": "Prediction failed due to a server error. "
                         "Please try again."
            }
        ), 500

    finally:

        if os.path.exists(filepath):
            os.remove(filepath)
