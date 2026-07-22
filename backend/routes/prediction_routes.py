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
from PIL import Image

from backend.prediction.predictor import predict_image

prediction_bp = Blueprint(
    "prediction",
    __name__
)

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}


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

        Image.open(filepath).verify()

        prediction = predict_image(filepath)

        return jsonify(
            {
                "predicted_class": prediction["predicted_class"],
                "confidence": prediction["confidence"],
                "top3_predictions": prediction["top3_predictions"]
            }
        ), 200

    except Exception as e:

        logging.exception("Prediction failed")

        return jsonify(
            {
                "error": str(e)
            }
        ), 500

    finally:

        if os.path.exists(filepath):
            os.remove(filepath)