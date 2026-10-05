"""
Hybrid analysis route (CNN + vision-LLM plant identification).

POST /analyze   multipart form with "file" (same as /predict)
"""

import logging

from flask import Blueprint
from flask import jsonify
from flask import request
from werkzeug.exceptions import HTTPException

from backend.routes.upload_utils import UploadError, saved_image_upload
from backend.services.analysis_service import analyze_upload
from backend.services.explanation_service import ExplanationRequestError, explain_image
from backend.services.languages import normalise_language

analysis_bp = Blueprint(
    "analysis",
    __name__
)

logger = logging.getLogger(__name__)


@analysis_bp.route("/analyze", methods=["POST"])
def analyze():
    """
    Identify the plant and diagnose it with the trained model.
    Works without the LLM (route "unverified").
    """

    language = normalise_language(request.form.get("language"))

    if language is None:
        return jsonify({"error": "Unsupported language.", "code": "INVALID_LANGUAGE"}), 400

    try:
        with saved_image_upload() as filepath:
            return jsonify(analyze_upload(filepath, language)), 200

    except UploadError as exc:
        return exc.response

    except HTTPException:
        raise  # e.g. 413 upload too large -> app error handler

    except Exception:
        logger.exception("Analysis failed")

        return jsonify(
            {"error": "Analysis failed due to a server error. Please try again."}
        ), 500


@analysis_bp.route("/explain", methods=["POST"])
def explain():
    """
    Heat map of the image regions the CNN relied on.

    Multipart form: "file" (image) and "target_class" (model class name).
    """

    try:
        with saved_image_upload() as filepath:
            target_class = request.form.get("target_class", "")
            return jsonify(explain_image(filepath, target_class)), 200

    except UploadError as exc:
        return exc.response

    except ExplanationRequestError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_CLASS"}), 400

    except HTTPException:
        raise

    except Exception:
        logger.exception("Explanation failed")

        return jsonify(
            {"error": "Could not create the explanation. Please try again."}
        ), 500
