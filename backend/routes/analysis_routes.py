"""
Hybrid analysis route (CNN + vision-LLM plant identification).

POST /analyze   multipart form with "file" (same as /predict)
"""

import logging

from flask import Blueprint
from flask import jsonify
from werkzeug.exceptions import HTTPException

from backend.routes.upload_utils import UploadError, saved_image_upload
from backend.services.analysis_service import analyze_upload

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

    try:
        with saved_image_upload() as filepath:
            return jsonify(analyze_upload(filepath)), 200

    except UploadError as exc:
        return exc.response

    except HTTPException:
        raise  # e.g. 413 upload too large -> app error handler

    except Exception:
        logger.exception("Analysis failed")

        return jsonify(
            {"error": "Analysis failed due to a server error. Please try again."}
        ), 500
