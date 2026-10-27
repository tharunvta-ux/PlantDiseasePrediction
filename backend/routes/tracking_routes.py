"""
Treatment plan and tracking routes.

POST   /treatment-plan              JSON; first step for a diagnosis (no login)
GET    /cases                       the signed-in user's cases, newest first
POST   /cases                       multipart: file + diagnosis fields; starts tracking
GET    /cases/<id>                  one case with all steps
DELETE /cases/<id>
POST   /cases/<id>/checkin          multipart: outcome, notes, optional new photo
GET    /cases/<id>/photo            case photo (JPEG)
GET    /cases/<id>/steps/<sid>/photo   check-in photo (JPEG)

Diagnosis fields (JSON body or form fields):
    predicted_class    model class name, or empty for a plant the model
                       does not cover
    confidence_level   "high" | "uncertain"
    plant_name         AI-identified plant name (optional for model classes)
    general_advice     AI observation care steps (JSON list; unsupported plants)
"""

from __future__ import annotations

import json
import logging
from contextlib import nullcontext

from flask import Blueprint, Response, jsonify, request
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from werkzeug.exceptions import HTTPException

from backend.database.db import db_session
from backend.database.models import TreatmentCase, TreatmentStep
from backend.prediction.predictor import get_predictor
from backend.routes.auth_routes import current_user, login_required
from backend.routes.upload_utils import UploadError, saved_image_upload
from backend.services.analysis_service import analyze_upload
from backend.services.checkin_note_service import review_note
from backend.services.languages import normalise_language
from backend.services.tracking_service import (
    TrackingRequestError,
    case_to_dict,
    check_in,
    ensure_can_check_in,
    make_thumbnail,
    new_case,
    plan_preview,
    summarise_followup,
    validate_case_request,
    validate_outcome,
)
from backend.services.treatment_plan_service import plan_meta

tracking_bp = Blueprint("tracking", __name__)

logger = logging.getLogger(__name__)


def _error(message: str, code: str, status: int):
    return jsonify({"error": message, "code": code}), status


def _diagnosis_fields(source) -> dict:
    """Read diagnosis fields from a JSON dict or a multipart form."""

    advice = source.get("general_advice")

    if isinstance(advice, str):
        try:
            advice = json.loads(advice)
        except ValueError:
            advice = []

    return validate_case_request(
        source.get("predicted_class") or None,
        source.get("confidence_level") or None,
        source.get("plant_name"),
        advice,
        get_predictor().class_names,
    )


def _has_file() -> bool:
    return "file" in request.files and request.files["file"].filename != ""


def _own_case(case_id: int) -> TreatmentCase | None:
    """Case by id if it belongs to the signed-in user."""

    case = db_session().scalar(
        select(TreatmentCase)
        .where(TreatmentCase.id == case_id, TreatmentCase.user_id == current_user().id)
        .options(selectinload(TreatmentCase.steps))
    )

    return case


def _jpeg(data: bytes | None):
    if not data:
        return _error("No photo.", "NOT_FOUND", 404)

    response = Response(data, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "private, max-age=86400"

    return response


@tracking_bp.route("/treatment-plan", methods=["POST"])
def treatment_plan():
    """First step of the plan for a diagnosis (preview, nothing saved)."""

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object.", "INVALID_JSON", 400)

    try:
        fields = _diagnosis_fields(payload)
    except TrackingRequestError as exc:
        return _error(str(exc), exc.code, exc.http_status)

    return jsonify({**plan_preview(fields), "meta": plan_meta()}), 200


@tracking_bp.route("/cases", methods=["GET"])
@login_required
def list_cases():
    cases = db_session().scalars(
        select(TreatmentCase)
        .where(TreatmentCase.user_id == current_user().id)
        .options(selectinload(TreatmentCase.steps))
        .order_by(TreatmentCase.created_at.desc(), TreatmentCase.id.desc())
    ).all()

    return jsonify({"cases": [case_to_dict(c) for c in cases]}), 200


@tracking_bp.route("/cases", methods=["POST"])
@login_required
def create_case():
    try:
        fields = _diagnosis_fields(request.form)

        with saved_image_upload() as filepath:
            photo = make_thumbnail(filepath)

    except TrackingRequestError as exc:
        return _error(str(exc), exc.code, exc.http_status)

    except UploadError as exc:
        return exc.response

    db = db_session()
    case = new_case(current_user().id, fields, photo)
    db.add(case)
    db.commit()

    return jsonify({"case": case_to_dict(case)}), 201


@tracking_bp.route("/cases/<int:case_id>", methods=["GET"])
@login_required
def get_case(case_id: int):
    case = _own_case(case_id)

    if case is None:
        return _error("Case not found.", "NOT_FOUND", 404)

    return jsonify({"case": case_to_dict(case)}), 200


@tracking_bp.route("/cases/<int:case_id>", methods=["DELETE"])
@login_required
def delete_case(case_id: int):
    case = _own_case(case_id)

    if case is None:
        return _error("Case not found.", "NOT_FOUND", 404)

    db = db_session()
    db.delete(case)
    db.commit()

    return jsonify({"deleted": case_id}), 200


@tracking_bp.route("/cases/<int:case_id>/checkin", methods=["POST"])
@login_required
def case_check_in(case_id: int):
    """
    "Worked" / "did not work" for the current step. A new photo is
    re-analysed and can change the diagnosis.
    """

    case = _own_case(case_id)

    if case is None:
        return _error("Case not found.", "NOT_FOUND", 404)

    language = normalise_language(request.form.get("language"))

    if language is None:
        return _error("Unsupported language.", "INVALID_LANGUAGE", 400)

    try:
        outcome, notes = validate_outcome(request.form.get("outcome"), request.form.get("notes"))
        ensure_can_check_in(case)  # before the (slow) photo analysis

        followup = None
        photo = None

        with saved_image_upload() if _has_file() else nullcontext() as filepath:
            if filepath:
                photo = make_thumbnail(filepath)
                try:
                    followup = summarise_followup(analyze_upload(filepath, language))
                except Exception:
                    logger.exception("Check-in photo analysis failed")
                    followup = {"error": "analysis_failed"}
                followup["has_photo"] = True

        note_review = review_note(case, outcome, notes, language) if notes else None

        feedback = check_in(case, outcome, notes, followup, photo, note_review)

    except TrackingRequestError as exc:
        return _error(str(exc), exc.code, exc.http_status)

    except UploadError as exc:
        return exc.response

    except HTTPException:
        raise

    db_session().commit()

    return jsonify({"feedback": feedback, "case": case_to_dict(case)}), 200


@tracking_bp.route("/cases/<int:case_id>/photo", methods=["GET"])
@login_required
def case_photo(case_id: int):
    photo = db_session().scalar(
        select(TreatmentCase.photo).where(
            TreatmentCase.id == case_id, TreatmentCase.user_id == current_user().id
        )
    )

    return _jpeg(photo)


@tracking_bp.route("/cases/<int:case_id>/steps/<int:step_id>/photo", methods=["GET"])
@login_required
def step_photo(case_id: int, step_id: int):
    photo = db_session().scalar(
        select(TreatmentStep.followup_photo)
        .join(TreatmentCase)
        .where(
            TreatmentStep.id == step_id,
            TreatmentCase.id == case_id,
            TreatmentCase.user_id == current_user().id,
        )
    )

    return _jpeg(photo)
