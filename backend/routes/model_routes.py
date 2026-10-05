"""
Model performance API routes.

GET /model/report           evaluation report of the deployed model
GET /model/figures/<name>   whitelisted evaluation figures (PNG)
"""

import logging

from flask import Blueprint
from flask import abort
from flask import jsonify
from flask import send_file

from backend.services.model_report_service import (
    ReportNotFoundError,
    get_figure_path,
    get_model_report,
)

model_bp = Blueprint(
    "model",
    __name__,
    url_prefix="/model",
)

logger = logging.getLogger(__name__)


@model_bp.route("/report", methods=["GET"])
def report():
    """Accuracy, calibration and confusion data for the deployed model."""

    try:
        return jsonify(get_model_report()), 200
    except ReportNotFoundError:
        return jsonify(
            {"error": "No evaluation report is available for this model."}
        ), 404


@model_bp.route("/figures/<name>", methods=["GET"])
def figure(name: str):
    """Serve one evaluation figure by its whitelisted name."""

    path = get_figure_path(name)

    if path is None:
        abort(404)

    return send_file(path, mimetype="image/png", max_age=3600)
