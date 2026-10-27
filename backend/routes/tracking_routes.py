"""
Treatment plan and tracking routes.

POST   /treatment-plan              JSON; first step for a diagnosis (no login)
POST   /treatment-plan/speech       same body + "part": read the plan aloud
GET    /ui-text?language=ta         translated labels of these screens
GET    /cases                       the signed-in user's cases, newest first
POST   /cases/<id>/speech           JSON {"language", "part"}: read the case aloud
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

Every response that shows plan or case text takes "language" (query,
JSON or form field) and is translated server-side; product names and
doses are never translated. Read-aloud endpoints without "part" return
the text parts (for the browser's own voice); with "part" they return
that part as WAV audio.
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
from backend.prediction.severity import measure_severity
from backend.services.analysis_service import analyze_upload
from backend.services.checkin_note_service import review_note
from backend.services.languages import LANGUAGES, normalise_language
from backend.services.llm import LLMError
from backend.services.localize_service import (
    case_speech_sentences,
    localize_cases,
    localize_preview,
    plan_speech_sentences,
    spoken_text,
)
from backend.services.severity_service import assess, clean_extent, clean_photo_severity
from backend.services.speech_service import SpeechPartError, speak_text, split_for_speech
from backend.services.translation_service import translate_texts
from backend.services.ui_text import UI_STRINGS
from backend.services.tracking_service import (
    TrackingRequestError,
    case_to_dict,
    check_in,
    ensure_can_check_in,
    latest_severity,
    make_thumbnail,
    new_case,
    plan_preview,
    summarise_followup,
    validate_case_request,
    validate_outcome,
)
from backend.services.treatment_plan_service import ai_condition_ids, plan_meta

tracking_bp = Blueprint("tracking", __name__)

logger = logging.getLogger(__name__)


def _error(message: str, code: str, status: int):
    return jsonify({"error": message, "code": code}), status


def _language(value) -> str:
    language = normalise_language(value)

    if language is None:
        raise TrackingRequestError("Unsupported language.", "INVALID_LANGUAGE")

    return language


def _case_json(case: TreatmentCase, language: str) -> dict:
    return localize_cases([case_to_dict(case)], language)[0]


def _speech_response(sentences, language: str, part):
    """Text parts (part is None) or one part as WAV audio."""

    text = spoken_text(sentences, language)

    if part is None:
        return jsonify({
            "parts": split_for_speech(text),
            "language": language,
            "speech_lang": LANGUAGES[language][2],
        }), 200

    if not isinstance(part, int) or isinstance(part, bool):
        return _error("part must be an integer.", "INVALID_PART", 400)

    try:
        audio = speak_text(text, language, part)
    except SpeechPartError as exc:
        return _error(str(exc), "INVALID_PART", 400)
    except LLMError as exc:
        logger.warning("Speech failed [%s]: %s", exc.code, exc.detail)
        return _error(exc.public_message, exc.code, exc.http_status)

    response = Response(audio["wav"], mimetype="audio/wav")
    response.headers["X-Speech-Part"] = str(audio["part"])
    response.headers["X-Speech-Parts"] = str(audio["parts"])

    return response


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
        [*get_predictor().class_names, *ai_condition_ids()],
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


def _photo_severity(filepath: str) -> dict | None:
    try:
        return measure_severity(filepath).to_dict()
    except Exception:
        logger.exception("Severity measurement failed")
        return None


def _jpeg(data: bytes | None):
    if not data:
        return _error("No photo.", "NOT_FOUND", 404)

    response = Response(data, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "private, max-age=86400"

    return response


@tracking_bp.route("/treatment-plan", methods=["POST"])
def treatment_plan():
    """First step of the plan for a diagnosis (preview, nothing saved)."""

    try:
        preview, language, _ = _preview_request()
    except TrackingRequestError as exc:
        return _error(str(exc), exc.code, exc.http_status)

    return jsonify({**localize_preview(preview, language), "meta": plan_meta()}), 200


@tracking_bp.route("/treatment-plan/speech", methods=["POST"])
def treatment_plan_speech():
    """Read "What to do now" aloud (text parts, or one part as WAV)."""

    try:
        preview, language, payload = _preview_request()
    except TrackingRequestError as exc:
        return _error(str(exc), exc.code, exc.http_status)

    if not preview["trackable"]:
        return _error("Nothing to read for this diagnosis.", "NOT_TRACKABLE", 400)

    return _speech_response(plan_speech_sentences(preview), language, payload.get("part"))


def _preview_request():
    """Validated (English) plan preview for a JSON request body."""

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        raise TrackingRequestError("Request body must be a JSON object.", "INVALID_JSON")

    preview = plan_preview(
        _diagnosis_fields(payload),
        clean_photo_severity(payload.get("photo_severity")),
        clean_extent(payload.get("extent")),
    )

    return preview, _language(payload.get("language")), payload


@tracking_bp.route("/ui-text", methods=["GET"])
def ui_text():
    """Translated labels of the plan and history screens."""

    try:
        language = _language(request.args.get("language"))
    except TrackingRequestError as exc:
        return _error(str(exc), exc.code, exc.http_status)

    mapping = translate_texts(UI_STRINGS, language)
    response = jsonify({"language": language, "strings": {k: v for k, v in mapping.items() if v != k}})
    response.headers["Cache-Control"] = "public, max-age=3600"

    return response


@tracking_bp.route("/cases", methods=["GET"])
@login_required
def list_cases():
    try:
        language = _language(request.args.get("language"))
    except TrackingRequestError as exc:
        return _error(str(exc), exc.code, exc.http_status)

    cases = db_session().scalars(
        select(TreatmentCase)
        .where(TreatmentCase.user_id == current_user().id)
        .options(selectinload(TreatmentCase.steps))
        .order_by(TreatmentCase.created_at.desc(), TreatmentCase.id.desc())
    ).all()

    return jsonify({"cases": localize_cases([case_to_dict(c) for c in cases], language)}), 200


@tracking_bp.route("/cases", methods=["POST"])
@login_required
def create_case():
    try:
        fields = _diagnosis_fields(request.form)

        with saved_image_upload() as filepath:
            photo = make_thumbnail(filepath)
            severity = assess(
                fields["predicted_class"],
                _photo_severity(filepath),
                clean_extent(request.form.get("extent")),
            )

    except TrackingRequestError as exc:
        return _error(str(exc), exc.code, exc.http_status)

    except UploadError as exc:
        return exc.response

    language = normalise_language(request.form.get("language")) or "en"

    db = db_session()
    case = new_case(current_user().id, fields, photo, severity)
    db.add(case)
    db.commit()

    return jsonify({"case": _case_json(case, language)}), 201


@tracking_bp.route("/cases/<int:case_id>", methods=["GET"])
@login_required
def get_case(case_id: int):
    case = _own_case(case_id)

    if case is None:
        return _error("Case not found.", "NOT_FOUND", 404)

    try:
        language = _language(request.args.get("language"))
    except TrackingRequestError as exc:
        return _error(str(exc), exc.code, exc.http_status)

    return jsonify({"case": _case_json(case, language)}), 200


@tracking_bp.route("/cases/<int:case_id>/speech", methods=["POST"])
@login_required
def case_speech(case_id: int):
    """Read a case aloud: where it stands and what to do now."""

    case = _own_case(case_id)

    if case is None:
        return _error("Case not found.", "NOT_FOUND", 404)

    payload = request.get_json(silent=True) or {}

    try:
        language = _language(payload.get("language"))
    except TrackingRequestError as exc:
        return _error(str(exc), exc.code, exc.http_status)

    return _speech_response(case_speech_sentences(case_to_dict(case)), language, payload.get("part"))


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
        measured = None
        extent = clean_extent(request.form.get("extent"))

        with saved_image_upload() if _has_file() else nullcontext() as filepath:
            if filepath:
                photo = make_thumbnail(filepath)
                measured = _photo_severity(filepath)
                try:
                    followup = summarise_followup(analyze_upload(filepath, language))
                except Exception:
                    logger.exception("Check-in photo analysis failed")
                    followup = {"error": "analysis_failed"}
                followup["has_photo"] = True

        # Stored in English like all case text, translated when shown.
        note_review = review_note(case, outcome, notes, "en") if notes else None
        severity = (
            assess(case.predicted_class, measured, extent, latest_severity(case))
            if measured or extent
            else None
        )

        feedback = check_in(case, outcome, notes, followup, photo, note_review, severity)

    except TrackingRequestError as exc:
        return _error(str(exc), exc.code, exc.http_status)

    except UploadError as exc:
        return exc.response

    except HTTPException:
        raise

    db_session().commit()

    return jsonify({
        "feedback": translate_texts([feedback], language).get(feedback, feedback),
        "case": _case_json(case, language),
    }), 200


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
