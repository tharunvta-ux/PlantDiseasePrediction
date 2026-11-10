"""
Treatment-kit shop routes (PayPal sandbox checkout).

GET  /shop/config                       is checkout on? PayPal client id (public), currency
POST /shop/cases/<id>/kit               JSON {area, area_unit, sprays, product}: priced kit
                                        for the case's current step (no field given = the
                                        field from the grower's last order, if any)
POST /shop/cases/<id>/assistant         JSON {message, language, area?, area_unit?, sprays?}:
                                        the AI assistant reads the grower's message and
                                        builds the kit it describes
POST /shop/cases/<id>/orders            JSON {area, area_unit, sprays, product, assistant?}:
                                        creates the PayPal order -> {"paypal_order_id"}
POST /shop/orders/<paypal_id>/capture   after the grower approves in the PayPal window
GET  /shop/orders                       the signed-in user's orders, newest first
POST /shop/orders/<id>/refund           full refund of a paid order (cancel)

Only products the case's current step lists can be ordered, and the
amount is always recomputed on the server from the curated plan and the
catalog: the browser never sends prices.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from flask import Blueprint, jsonify, request
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.config import get_settings
from backend.database.db import db_session
from backend.database.models import (
    CASE_ACTIVE,
    ORDER_CREATED,
    ORDER_FAILED,
    ORDER_PAID,
    ORDER_REFUNDED,
    OUTCOME_PENDING,
    ShopOrder,
    TreatmentCase,
    TreatmentStep,
    utcnow,
)
from backend.routes.auth_routes import current_user, login_required
from backend.services.kit_service import (
    AREA_UNITS,
    Field,
    KitError,
    build_quote,
    catalog_meta,
    currency,
    kit_options,
    validate_field,
)
from backend.services.languages import normalise_language
from backend.services.llm import LLMError
from backend.services.paypal_service import (
    PayPalError,
    capture_details,
    get_paypal_client,
    new_invoice_id,
)
from backend.services.shop_assistant_service import plan_kit

shop_bp = Blueprint("shop", __name__, url_prefix="/shop")

logger = logging.getLogger(__name__)


def _error(message: str, code: str, status: int):
    return jsonify({"error": message, "code": code}), status


def _json_body() -> dict:
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        raise KitError("Request body must be a JSON object.", "INVALID_JSON")

    return payload


def _own_case(case_id: int) -> TreatmentCase | None:
    return db_session().scalar(
        select(TreatmentCase)
        .where(TreatmentCase.id == case_id, TreatmentCase.user_id == current_user().id)
        .options(selectinload(TreatmentCase.steps))
    )


def _orderable_step(case: TreatmentCase) -> TreatmentStep:
    """The case's current step, if it lists products that can be ordered."""

    step = case.current_step

    if case.status != CASE_ACTIVE or step is None or step.outcome != OUTCOME_PENDING:
        raise KitError("This case has no open treatment step.", "STEP_CLOSED", 409)

    if not kit_options(step.products or []):
        raise KitError("This step has no products that can be ordered.", "NOTHING_TO_ORDER", 409)

    return step


def _last_field(case: TreatmentCase) -> Field | None:
    """Field size of the grower's latest order (this case first, then any)."""

    latest = (
        select(ShopOrder)
        .where(ShopOrder.user_id == current_user().id, ShopOrder.status != ORDER_FAILED)
        .order_by(ShopOrder.created_at.desc(), ShopOrder.id.desc())
        .limit(1)
    )

    for query in (latest.where(ShopOrder.case_id == case.id), latest):
        order = db_session().scalar(query)
        if order and order.field:
            try:
                return validate_field(order.field.get("area"), order.field.get("area_unit"), order.field.get("sprays"))
            except KitError:
                return None

    return None


def _field_from(payload: dict, required: bool = True) -> Field | None:
    if not required and payload.get("area") in (None, ""):
        return None

    return validate_field(payload.get("area"), payload.get("area_unit"), payload.get("sprays"))


def _case_orders(case_id: int) -> list:
    orders = db_session().scalars(
        select(ShopOrder)
        .where(ShopOrder.user_id == current_user().id, ShopOrder.case_id == case_id)
        .where(ShopOrder.status != ORDER_CREATED)
        .order_by(ShopOrder.created_at.desc(), ShopOrder.id.desc())
    ).all()

    return [o.to_dict() for o in orders]


@shop_bp.route("/config", methods=["GET"])
def shop_config():
    settings = get_settings()

    return jsonify({
        "enabled": settings.paypal_configured,
        # The client id is public by design (it is in the PayPal JS SDK URL).
        "client_id": settings.paypal_client_id if settings.paypal_configured else None,
        "currency": currency(),
        "sandbox": "sandbox" in settings.paypal_api_base,
        "assistant": settings.llm_configured,
        "area_units": list(AREA_UNITS),
        **catalog_meta(),
    }), 200


@shop_bp.route("/cases/<int:case_id>/kit", methods=["POST"])
@login_required
def case_kit(case_id: int):
    """Priced kit for the current step."""

    case = _own_case(case_id)

    if case is None:
        return _error("Case not found.", "NOT_FOUND", 404)

    try:
        payload = _json_body()
        step = _orderable_step(case)
        field = _field_from(payload, required=False)
        remembered = False

        if field is None:
            field = _last_field(case)
            remembered = field is not None

        quote = build_quote(step.products, field, payload.get("product") or None) if field else None

    except KitError as exc:
        return _error(str(exc), exc.code, exc.http_status)

    return jsonify({
        "step_number": step.number,
        "options": [p["name"] for p in kit_options(step.products)],
        "field": field.to_dict() if field else None,
        "remembered": remembered,
        "quote": quote,
        "orders": _case_orders(case.id),
    }), 200


@shop_bp.route("/cases/<int:case_id>/assistant", methods=["POST"])
@login_required
def case_assistant(case_id: int):
    """The AI assistant turns the grower's own words into a kit."""

    case = _own_case(case_id)

    if case is None:
        return _error("Case not found.", "NOT_FOUND", 404)

    try:
        payload = _json_body()
        language = normalise_language(payload.get("language"))

        if language is None:
            raise KitError("Unsupported language.", "INVALID_LANGUAGE")

        step = _orderable_step(case)
        current = _field_from(payload, required=False) or _last_field(case)
        result = plan_kit(
            payload.get("message"),
            step.products,
            case.plant_name,
            case.display_name,
            current,
            language,
        )

    except KitError as exc:
        return _error(str(exc), exc.code, exc.http_status)

    except LLMError as exc:
        logger.warning("Kit assistant unavailable [%s]: %s", exc.code, exc.detail)
        return _error(
            "The assistant is busy right now. Enter your field size below instead.",
            "ASSISTANT_UNAVAILABLE",
            503,
        )

    return jsonify(result), 200


@shop_bp.route("/cases/<int:case_id>/orders", methods=["POST"])
@login_required
def create_order(case_id: int):
    """Create the PayPal order for the kit (the grower approves it next)."""

    case = _own_case(case_id)

    if case is None:
        return _error("Case not found.", "NOT_FOUND", 404)

    try:
        payload = _json_body()
        step = _orderable_step(case)
        field = _field_from(payload)
        quote = build_quote(step.products, field, payload.get("product") or None)
        client = get_paypal_client()

        invoice_id = new_invoice_id()
        reference = f"case-{case.id}-step-{step.number}"
        paypal_order = client.create_order(
            quote["items"],
            quote["currency"],
            quote["total"],
            reference_id=reference,
            invoice_id=invoice_id,
            description=f"Treatment kit: {case.display_name} ({quote['field_text']})",
        )

    except KitError as exc:
        return _error(str(exc), exc.code, exc.http_status)

    except PayPalError as exc:
        return _error(str(exc), exc.code, exc.http_status)

    assistant = payload.get("assistant")
    order = ShopOrder(
        user_id=current_user().id,
        case_id=case.id,
        step_number=step.number,
        paypal_order_id=paypal_order["id"],
        invoice_id=invoice_id,
        status=ORDER_CREATED,
        currency=quote["currency"],
        total=quote["total"],
        product=quote["product"],
        display_name=case.display_name,
        items=quote["items"],
        field=field.to_dict(),
        quote={k: quote[k] for k in ("field_text", "needed_text", "packed_text", "dose", "group", "kind")},
        assistant={"reply": str(assistant)[:500]} if assistant else None,
    )
    db = db_session()
    db.add(order)
    db.commit()

    return jsonify({"paypal_order_id": order.paypal_order_id, "order": order.to_dict()}), 201


def _own_order(**where) -> ShopOrder | None:
    query = select(ShopOrder).where(ShopOrder.user_id == current_user().id)

    for key, value in where.items():
        query = query.where(getattr(ShopOrder, key) == value)

    return db_session().scalar(query)


@shop_bp.route("/orders/<string:paypal_order_id>/capture", methods=["POST"])
@login_required
def capture_order(paypal_order_id: str):
    """Take the payment once the grower has approved it in PayPal."""

    order = _own_order(paypal_order_id=paypal_order_id)

    if order is None:
        return _error("Order not found.", "NOT_FOUND", 404)

    if order.status == ORDER_PAID:
        return jsonify({"order": order.to_dict()}), 200

    if order.status != ORDER_CREATED:
        return _error("This order can no longer be paid.", "ORDER_CLOSED", 409)

    try:
        details = capture_details(get_paypal_client().capture_order(paypal_order_id))
    except PayPalError as exc:
        if exc.issue == "INSTRUMENT_DECLINED":
            # The PayPal window lets the buyer choose another funding source.
            return _error("The payment method was declined. Choose another one.", "INSTRUMENT_DECLINED", 422)
        return _error(str(exc), exc.code, exc.http_status)

    db = db_session()
    paid = details["capture_status"] in ("COMPLETED", "PENDING")
    amount_ok = (
        details["amount"] is not None
        and Decimal(details["amount"]) == Decimal(order.total)
        and details["currency"] == order.currency
    )

    if not paid or not amount_ok:
        logger.error(
            "Capture of %s not accepted: status=%s amount=%s %s (expected %s %s)",
            paypal_order_id, details["capture_status"], details["amount"], details["currency"],
            order.total, order.currency,
        )
        order.status = ORDER_FAILED
        order.capture_id = details["capture_id"]
        db.commit()
        return _error("The payment could not be completed.", "CAPTURE_FAILED", 502)

    order.status = ORDER_PAID
    order.capture_id = details["capture_id"]
    order.payer = details["payer"]
    order.shipping = details["shipping"]
    order.paid_at = utcnow()
    db.commit()

    return jsonify({"order": order.to_dict(), "capture_status": details["capture_status"]}), 200


@shop_bp.route("/orders", methods=["GET"])
@login_required
def list_orders():
    orders = db_session().scalars(
        select(ShopOrder)
        .where(ShopOrder.user_id == current_user().id, ShopOrder.status != ORDER_CREATED)
        .order_by(ShopOrder.created_at.desc(), ShopOrder.id.desc())
    ).all()

    return jsonify({"orders": [o.to_dict() for o in orders]}), 200


@shop_bp.route("/orders/<int:order_id>/refund", methods=["POST"])
@login_required
def refund_order(order_id: int):
    """Cancel a paid kit and refund it in full."""

    order = _own_order(id=order_id)

    if order is None:
        return _error("Order not found.", "NOT_FOUND", 404)

    if order.status == ORDER_REFUNDED:
        return jsonify({"order": order.to_dict()}), 200

    if order.status != ORDER_PAID or not order.capture_id:
        return _error("Only paid orders can be refunded.", "NOT_REFUNDABLE", 409)

    try:
        refund = get_paypal_client().refund_capture(
            order.capture_id, f"Refund for treatment kit {order.invoice_id}"
        )
    except PayPalError as exc:
        return _error(str(exc), exc.code, exc.http_status)

    if refund.get("status") not in ("COMPLETED", "PENDING"):
        return _error("PayPal did not accept the refund.", "REFUND_FAILED", 502)

    order.status = ORDER_REFUNDED
    order.refund_id = refund.get("id")
    order.refunded_at = utcnow()
    db_session().commit()

    return jsonify({"order": order.to_dict()}), 200
