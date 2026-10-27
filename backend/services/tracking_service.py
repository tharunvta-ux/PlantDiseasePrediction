"""
backend/services/tracking_service.py

Treatment tracking: a diagnosed problem becomes a case with a first
step and a "check again on" date. At each check-in the grower answers
"worked" or "did not work" and may add a new photo:

    worked       -> case recovered, positive message
    not worked   -> the new photo (if any) is re-analysed:
                    * a confident, different disease -> plan for that one
                    * the same disease, now confident -> may unlock a
                      TNAU treatment step
                    otherwise the next rung of the plan; after the last
                    rung -> consult an expert nearby
"""

from __future__ import annotations

import io
import logging
from datetime import timedelta
from typing import Any, Dict, List, Sequence

from PIL import Image, ImageOps

from backend.database.models import (
    CASE_ACTIVE,
    CASE_EXPERT,
    CASE_RECOVERED,
    OUTCOME_NOT_WORKED,
    OUTCOME_PENDING,
    OUTCOME_WORKED,
    TreatmentCase,
    TreatmentStep,
    iso,
    utcnow,
)
from backend.prediction.class_info import parse_class_name
from backend.prediction.confidence import LEVEL_HIGH, LEVEL_UNCERTAIN
from backend.prediction.severity import GRADE_SEVERE
from backend.services.recommendation_service import DOSAGE_PATTERN
from backend.services.severity_service import TREND_IMPROVING, TREND_WORSENING, assess
from backend.services.treatment_plan_service import (
    KIND_EXPERT,
    StepPlan,
    expert_step,
    first_step,
    is_ai_condition,
    is_trackable,
    next_step,
)

logger = logging.getLogger(__name__)

# Hard stop, also when re-analysis keeps finding different diseases.
MAX_CASE_STEPS = 6

# Title of a repeated step when the photo shows the disease shrinking;
# a step is only repeated once.
CONTINUE_TITLE = "Keep going - the disease is shrinking"

THUMBNAIL_SIDE = 640
THUMBNAIL_QUALITY = 80

MAX_PLANT_NAME = 120
MAX_ADVICE_ITEMS = 6
MAX_ADVICE_LENGTH = 300
MAX_NOTES = 1000


class TrackingRequestError(ValueError):
    """Invalid tracking request (HTTP 400 / 409)."""

    def __init__(self, message: str, code: str, http_status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status


# ==========================================================
# Validation
# ==========================================================


def validate_case_request(
    predicted_class: Any,
    level: Any,
    plant_name: Any,
    general_advice: Any,
    known_classes: Sequence[str],
) -> Dict[str, Any]:
    """
    Normalise a "start tracking" request.

    predicted_class empty/None = a plant the model does not cover, which
    is tracked with the AI observation's general advice.
    """

    if predicted_class in (None, ""):
        if not isinstance(plant_name, str) or not plant_name.strip():
            raise TrackingRequestError("plant_name is required.", "INVALID_PLANT_NAME")

        name = plant_name.strip()[:MAX_PLANT_NAME]

        return {
            "predicted_class": None,
            "confidence_level": None,
            "plant_name": name,
            "display_name": f"{name} — AI observation",
            "ai_advice": _clean_advice(general_advice),
        }

    if not isinstance(predicted_class, str) or predicted_class not in set(known_classes):
        raise TrackingRequestError(
            "predicted_class must be one of the model's class names.", "INVALID_CLASS"
        )

    if level not in (LEVEL_HIGH, LEVEL_UNCERTAIN):
        raise TrackingRequestError(
            "Only high or uncertain predictions can be tracked. Please upload a "
            "clearer photo first.",
            "INVALID_CONFIDENCE_LEVEL",
        )

    info = parse_class_name(predicted_class)

    if info.is_healthy:
        raise TrackingRequestError(
            "The plant looks healthy - there is nothing to treat.", "HEALTHY"
        )

    name = plant_name.strip()[:MAX_PLANT_NAME] if isinstance(plant_name, str) else ""

    return {
        "predicted_class": predicted_class,
        "confidence_level": level,
        "plant_name": name or info.crop,
        "display_name": info.display_name,
        "ai_advice": [],
    }


def _clean_advice(items: Any) -> List[str]:
    """Client-sent AI advice: short strings only, no doses."""

    if not isinstance(items, list):
        return []

    cleaned = [
        item.strip()[:MAX_ADVICE_LENGTH]
        for item in items
        if isinstance(item, str) and item.strip()
    ]

    return [item for item in cleaned if not DOSAGE_PATTERN.search(item)][:MAX_ADVICE_ITEMS]


def validate_outcome(outcome: Any, notes: Any) -> tuple[str, str | None]:
    if outcome not in (OUTCOME_WORKED, OUTCOME_NOT_WORKED):
        raise TrackingRequestError(
            "outcome must be 'worked' or 'not_worked'.", "INVALID_OUTCOME"
        )

    text = notes.strip()[:MAX_NOTES] if isinstance(notes, str) and notes.strip() else None

    return outcome, text


# ==========================================================
# Photos
# ==========================================================


def make_thumbnail(image_path: str) -> bytes:
    """Upright JPEG, at most THUMBNAIL_SIDE px, for the history view."""

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((THUMBNAIL_SIDE, THUMBNAIL_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=THUMBNAIL_QUALITY)

    return buffer.getvalue()


# ==========================================================
# Cases and steps
# ==========================================================


def add_step(case: TreatmentCase, plan: StepPlan) -> TreatmentStep:
    now = utcnow()

    step = TreatmentStep(
        number=len(case.steps) + 1,
        created_at=now,
        predicted_class=case.predicted_class,
        kind=plan.kind,
        title=plan.title,
        actions=list(plan.actions),
        products=list(plan.products),
        wait_days=plan.wait_days,
        check_question=plan.check_question,
        reason=plan.reason,
        source_url=plan.source_url,
        check_due_at=now + timedelta(days=plan.wait_days) if plan.wait_days else None,
        outcome=OUTCOME_PENDING,
    )

    case.steps.append(step)

    if plan.kind == KIND_EXPERT:
        case.status = CASE_EXPERT

    return step


def new_case(
    user_id: int,
    request: Dict[str, Any],
    photo: bytes,
    severity: Dict[str, Any] | None = None,
) -> TreatmentCase:
    """Case with its first step (not yet added to a session)."""

    case = TreatmentCase(
        user_id=user_id,
        status=CASE_ACTIVE,
        plant_name=request["plant_name"],
        predicted_class=request["predicted_class"],
        display_name=request["display_name"],
        confidence_level=request["confidence_level"],
        photo=photo,
        severity=severity,
    )

    add_step(case, first_step(request["predicted_class"], request["confidence_level"], request["ai_advice"]))

    return case


def summarise_followup(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """What the re-analysis of a check-in photo found."""

    diagnosis = analysis.get("diagnosis")
    identification = analysis.get("plant_identification") or {}
    ai_condition = analysis.get("ai_condition")

    summary = {
        "route": analysis.get("route"),
        "message": analysis.get("message"),
        "plant_name": identification.get("plant_common_name") or None,
        "predicted_class": diagnosis["predicted_class"] if diagnosis else None,
        "display_name": diagnosis["display_name"] if diagnosis else None,
        "confidence_level": diagnosis["confidence_level"] if diagnosis else None,
        "is_healthy": diagnosis["is_healthy"] if diagnosis else None,
    }

    if not diagnosis and ai_condition:
        # AI matches are never "high": switching treatment needs the
        # grower's confirmation, which a check-in photo does not give.
        summary.update(
            predicted_class=ai_condition["id"],
            display_name=ai_condition["display_name"],
            confidence_level=LEVEL_UNCERTAIN,
            is_healthy=False,
        )

    return summary


def _step_history(case: TreatmentCase) -> List[Dict[str, Any]]:
    """Steps already tried for the case's current diagnosis."""

    return [
        {"kind": s.kind, "products": s.products or []}
        for s in case.steps
        if s.predicted_class == case.predicted_class
    ]


def _apply_followup(case: TreatmentCase, followup: Dict[str, Any] | None) -> tuple[List[str], StepPlan | None]:
    """
    Use a re-analysed photo after "did not work".

    Returns messages for the grower and, if a different disease was
    found, the first step of its plan.
    """

    if not followup:
        return [], None

    if followup.get("error"):
        return ["The new photo could not be analysed, so the plan continues as before."], None

    found = followup.get("predicted_class")
    level = followup.get("confidence_level")

    if not found or level not in (LEVEL_HIGH, LEVEL_UNCERTAIN):
        return ["The new photo did not give a clear result, so the plan continues as before."], None

    if followup.get("is_healthy"):
        if level == LEVEL_HIGH:
            return [
                "The new photo looks healthy to the model. If other leaves still show "
                "symptoms, photograph those leaves at the next check."
            ], None
        return [], None

    if found != case.predicted_class:
        if level == LEVEL_HIGH:
            old = case.display_name
            info = parse_class_name(found)
            case.predicted_class = found
            case.display_name = info.display_name
            case.confidence_level = LEVEL_HIGH
            return [
                f"The new photo looks like {info.display_name} rather than {old}. "
                f"Here is a plan for {info.display_name}."
            ], first_step(found, LEVEL_HIGH)

        return [
            f"The new photo might show {followup.get('display_name')} instead, but this "
            "is not certain, so the current plan continues. Mention it to an expert "
            "if the problem stays."
        ], None

    if level == LEVEL_HIGH and case.confidence_level != LEVEL_HIGH:
        case.confidence_level = LEVEL_HIGH
        return [f"The new photo confirms {case.display_name}."], None

    return [], None


def latest_severity(case: TreatmentCase) -> Dict[str, Any] | None:
    """Most recent severity with a grade (check-ins first, then the start)."""

    for step in reversed(case.steps):
        if step.severity and step.severity.get("grade"):
            return step.severity

    return case.severity if case.severity and case.severity.get("grade") else None


def continue_step(step: TreatmentStep, change: Dict[str, Any]) -> StepPlan:
    """Same step again, a little longer - the disease is already shrinking."""

    return StepPlan(
        kind=step.kind,
        title=CONTINUE_TITLE,
        actions=list(step.actions or []),
        products=list(step.products or []),
        wait_days=max(3, (step.wait_days or 6) // 2),
        check_question=step.check_question,
        reason=(
            f"The new photo shows less disease than before ({change['text']}). The "
            "treatment may be working slowly, so keep going a few more days before "
            "changing anything."
        ),
        source_url=step.source_url,
    )


def worked_message(case: TreatmentCase, followup: Dict[str, Any] | None) -> str:
    step = case.current_step
    if step is not None and step.kind == KIND_EXPERT:
        message = f"Great - glad the expert's advice helped your {case.plant_name}!"
    else:
        message = (
            f"Great news - the treatment is working for your {case.plant_name}! "
            "Keep checking the plant every few days for the next two weeks and keep "
            "up the care steps so the problem does not come back."
        )

    if (
        followup
        and followup.get("predicted_class") == case.predicted_class
        and followup.get("confidence_level") == LEVEL_HIGH
    ):
        message += (
            f" Note: the new photo still shows signs of {case.display_name}. Watch it "
            "closely and start a new check if it spreads."
        )

    return message


def ensure_can_check_in(case: TreatmentCase) -> None:
    step = case.current_step

    if step is None or step.outcome != OUTCOME_PENDING or case.status == CASE_RECOVERED:
        raise TrackingRequestError(
            "This case has no step waiting for a check-in.", "NOTHING_TO_CHECK", 409
        )


def check_in(
    case: TreatmentCase,
    outcome: str,
    notes: str | None,
    followup: Dict[str, Any] | None = None,
    followup_photo: bytes | None = None,
    note_review: Dict[str, Any] | None = None,
    severity: Dict[str, Any] | None = None,
) -> str:
    """
    Record the answer for the current step and decide what comes next.

    `note_review` is the AI reply to the grower's note (see
    checkin_note_service); an urgent note sends the case to an expert.
    `severity` (severity_service.assess, with the trend since the last
    measurement) can keep a slowly working step going, or escalate a
    fast-spreading severe problem.

    Returns:
        The feedback message (also stored on the step).
    """

    ensure_can_check_in(case)
    step = case.current_step
    reply = (note_review or {}).get("reply")
    urgent = bool((note_review or {}).get("urgent"))
    change = (severity or {}).get("trend") or {}
    direction = change.get("direction")

    step.outcome = outcome
    step.outcome_at = utcnow()
    step.notes = notes
    step.followup = followup
    step.followup_photo = followup_photo
    step.severity = severity

    if outcome == OUTCOME_WORKED:
        trend_note = None
        if direction == TREND_IMPROVING:
            trend_note = f"The photo agrees: {change['text']}."
        elif direction == TREND_WORSENING:
            trend_note = (
                f"Note: the photo shows more disease than before ({change['text']}). "
                "Watch the plant closely and start a new check if it spreads."
            )
        feedback = " ".join(filter(None, [worked_message(case, followup), trend_note, reply]))
        case.status = CASE_RECOVERED
        step.feedback = feedback
        return feedback

    messages, plan = _apply_followup(case, followup)
    if reply:
        messages.insert(0, reply)

    if urgent and step.kind != KIND_EXPERT:
        plan = expert_step(
            "What you describe sounds serious, so please do not wait for another "
            "step - contact an agricultural expert now."
        )
    elif (
        direction == TREND_WORSENING
        and (severity or {}).get("grade") == GRADE_SEVERE
        and step.kind != KIND_EXPERT
    ):
        plan = expert_step(
            f"The disease is spreading fast ({change['text']}). Please contact an "
            "agricultural expert now instead of trying another step."
        )
    elif (
        plan is None
        and direction == TREND_IMPROVING
        and step.kind != KIND_EXPERT
        and step.title != CONTINUE_TITLE
        and len(case.steps) < MAX_CASE_STEPS
    ):
        plan = continue_step(step, change)
    elif len(case.steps) >= MAX_CASE_STEPS:
        plan = expert_step(
            "Several steps have been tried without success. An expert can examine "
            "the plant in person and find the cause."
        )
    elif plan is None:
        plan = next_step(case.predicted_class, case.confidence_level, _step_history(case))

    add_step(case, plan)

    if plan.kind != KIND_EXPERT:
        case.status = CASE_ACTIVE

    feedback = " ".join([*messages, plan.reason]).strip()
    step.feedback = feedback

    logger.info("Case %s check-in: not worked -> step %s (%s)", case.id, len(case.steps), plan.kind)

    return feedback


# ==========================================================
# Serialisation
# ==========================================================


def step_to_dict(step: TreatmentStep, case_id: int) -> Dict[str, Any]:
    followup = dict(step.followup) if step.followup else None

    return {
        "id": step.id,
        "number": step.number,
        "kind": step.kind,
        "title": step.title,
        "actions": step.actions or [],
        "products": step.products or [],
        "wait_days": step.wait_days,
        "check_question": step.check_question,
        "reason": step.reason,
        "source_url": step.source_url,
        "created_at": iso(step.created_at),
        "check_due_at": iso(step.check_due_at),
        "outcome": step.outcome,
        "outcome_at": iso(step.outcome_at),
        "notes": step.notes,
        "feedback": step.feedback,
        "followup": followup,
        "severity": step.severity,
        "followup_photo_url": (
            f"/cases/{case_id}/steps/{step.id}/photo"
            if followup and followup.get("has_photo")
            else None
        ),
    }


def case_to_dict(case: TreatmentCase) -> Dict[str, Any]:
    steps = [step_to_dict(s, case.id) for s in case.steps]
    current = steps[-1] if steps else None

    return {
        "id": case.id,
        "status": case.status,
        "plant_name": case.plant_name,
        "predicted_class": case.predicted_class,
        "display_name": case.display_name,
        "confidence_level": case.confidence_level,
        # model = trained CNN; ai_match = AI-matched TNAU condition (high =
        # confirmed by the grower); ai_observation = general AI advice only
        "diagnosis_source": (
            "ai_observation"
            if case.predicted_class is None
            else "ai_match" if is_ai_condition(case.predicted_class) else "model"
        ),
        "created_at": iso(case.created_at),
        "updated_at": iso(case.updated_at),
        "photo_url": f"/cases/{case.id}/photo",
        "severity": case.severity,
        "latest_severity": latest_severity(case),
        "current_step": current,
        "awaiting_check_in": bool(
            current and current["outcome"] == OUTCOME_PENDING and case.status != CASE_RECOVERED
        ),
        "steps": steps,
    }


def plan_preview(
    request: Dict[str, Any],
    photo_severity: Dict[str, Any] | None = None,
    extent: str | None = None,
) -> Dict[str, Any]:
    """First step, severity and outlook for a diagnosis, without saving."""

    trackable = is_trackable(request["predicted_class"], request["confidence_level"])

    return {
        "trackable": trackable,
        "step": (
            first_step(request["predicted_class"], request["confidence_level"], request["ai_advice"]).to_dict()
            if trackable
            else None
        ),
        "severity": assess(request["predicted_class"], photo_severity, extent) if trackable else None,
    }
