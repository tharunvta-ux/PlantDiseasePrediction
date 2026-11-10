"""
backend/services/paypal_service.py

Minimal PayPal REST client for the treatment-kit checkout (sandbox).

    POST /v1/oauth2/token                        client-credentials token
    POST /v2/checkout/orders                     create an order (intent CAPTURE)
    POST /v2/checkout/orders/{id}/capture        take the payment after approval
    GET  /v2/checkout/orders/{id}                order details
    POST /v2/payments/captures/{id}/refund       full refund

The browser only ever sees the order id: the order body, amounts and
line items are built here from the server-side kit quote, and every
write uses a PayPal-Request-Id so a retried request is not applied twice.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from decimal import Decimal
from typing import Any, Dict, List

import httpx

from backend.config import Settings, get_settings

logger = logging.getLogger(__name__)

BRAND_NAME = "Verdant"


class PayPalError(Exception):
    """A PayPal call failed; `code` and `http_status` are safe to return."""

    def __init__(
        self,
        message: str,
        code: str = "PAYPAL_ERROR",
        http_status: int = 502,
        issue: str | None = None,
        detail: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.issue = issue  # PayPal's issue code, e.g. INSTRUMENT_DECLINED
        self.detail = detail


class PayPalNotConfiguredError(PayPalError):
    def __init__(self) -> None:
        super().__init__(
            "Checkout is not available: PayPal is not configured on the server.",
            "PAYPAL_NOT_CONFIGURED",
            503,
        )


def _issue(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None

    details = body.get("details") or []
    if details and isinstance(details[0], dict):
        return details[0].get("issue")
    return body.get("name") or body.get("error")


class PayPalClient:
    """Thread-safe client with a cached access token."""

    def __init__(self, settings: Settings, http: httpx.Client | None = None) -> None:
        if not settings.paypal_configured:
            raise PayPalNotConfiguredError()

        self.base = settings.paypal_api_base
        self._auth = (settings.paypal_client_id, settings.paypal_client_secret)
        self._http = http or httpx.Client(timeout=settings.paypal_timeout_seconds)
        self._token: str | None = None
        self._token_expires = 0.0
        self._lock = threading.Lock()

    # ---------- transport ----------

    def _access_token(self) -> str:
        with self._lock:
            if self._token and time.monotonic() < self._token_expires:
                return self._token

            try:
                response = self._http.post(
                    f"{self.base}/v1/oauth2/token",
                    auth=self._auth,
                    data={"grant_type": "client_credentials"},
                    headers={"Accept": "application/json"},
                )
            except httpx.HTTPError as exc:
                raise PayPalError(
                    "Could not reach PayPal. Please try again.", "PAYPAL_UNAVAILABLE", 503, detail=str(exc)
                ) from exc

            if response.status_code != 200:
                logger.error("PayPal token request failed: %s %s", response.status_code, response.text[:300])
                raise PayPalError(
                    "Checkout is not available: the server's PayPal credentials were rejected.",
                    "PAYPAL_AUTH_FAILED",
                    503,
                )

            body = response.json()
            self._token = body["access_token"]
            # Renew a minute early.
            self._token_expires = time.monotonic() + max(int(body.get("expires_in", 300)) - 60, 30)

            return self._token

    def _request(
        self,
        method: str,
        path: str,
        json_body: Dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> tuple[int, Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self._access_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Prefer": "return=representation",
        }
        if request_id:
            headers["PayPal-Request-Id"] = request_id

        try:
            response = self._http.request(method, f"{self.base}{path}", json=json_body, headers=headers)
        except httpx.HTTPError as exc:
            raise PayPalError(
                "Could not reach PayPal. Please try again.", "PAYPAL_UNAVAILABLE", 503, detail=str(exc)
            ) from exc

        if response.status_code == 401:
            with self._lock:
                self._token = None  # expired early; next call fetches a new one

        try:
            body = response.json() if response.content else {}
        except ValueError:
            body = {}

        if response.status_code >= 400:
            issue = _issue(response)
            logger.warning(
                "PayPal %s %s -> %s %s (debug_id=%s)",
                method, path, response.status_code, issue, body.get("debug_id"),
            )
            raise PayPalError(
                "PayPal could not complete the request. Please try again.",
                "PAYPAL_REQUEST_FAILED",
                502 if response.status_code >= 500 else 422,
                issue=issue,
                detail=response.text[:500],
            )

        return response.status_code, body

    # ---------- orders ----------

    def create_order(
        self,
        items: List[Dict[str, Any]],
        currency: str,
        total: str,
        reference_id: str,
        invoice_id: str,
        description: str,
    ) -> Dict[str, Any]:
        """Create a CAPTURE order for the kit; returns PayPal's order."""

        item_total = sum((Decimal(i["unit_price"]) * i["quantity"] for i in items), Decimal(0))

        if f"{item_total:.2f}" != total:
            raise ValueError(f"Item total {item_total} does not match order total {total}")

        body = {
            "intent": "CAPTURE",
            "purchase_units": [
                {
                    "reference_id": reference_id,
                    "invoice_id": invoice_id,
                    "custom_id": reference_id,
                    "description": description[:127],
                    "amount": {
                        "currency_code": currency,
                        "value": total,
                        "breakdown": {"item_total": {"currency_code": currency, "value": total}},
                    },
                    "items": [
                        {
                            "name": item["name"][:127],
                            "sku": item["sku"][:127],
                            "quantity": str(item["quantity"]),
                            "unit_amount": {"currency_code": currency, "value": item["unit_price"]},
                            "category": "PHYSICAL_GOODS",
                        }
                        for item in items
                    ],
                }
            ],
            "payment_source": {
                "paypal": {
                    "experience_context": {
                        "brand_name": BRAND_NAME,
                        # Deliver to the address on the buyer's PayPal account.
                        "shipping_preference": "GET_FROM_FILE",
                        "user_action": "PAY_NOW",
                    }
                }
            },
        }

        _, order = self._request("POST", "/v2/checkout/orders", body, request_id=f"create-{invoice_id}")
        return order

    def capture_order(self, order_id: str) -> Dict[str, Any]:
        """Capture an approved order (idempotent for the same order id)."""

        try:
            _, order = self._request(
                "POST", f"/v2/checkout/orders/{order_id}/capture", {}, request_id=f"capture-{order_id}"
            )
        except PayPalError as exc:
            if exc.issue == "ORDER_ALREADY_CAPTURED":
                return self.get_order(order_id)
            raise

        return order

    def get_order(self, order_id: str) -> Dict[str, Any]:
        _, order = self._request("GET", f"/v2/checkout/orders/{order_id}")
        return order

    def refund_capture(self, capture_id: str, note: str) -> Dict[str, Any]:
        """Full refund of a captured payment."""

        _, refund = self._request(
            "POST",
            f"/v2/payments/captures/{capture_id}/refund",
            {"note_to_payer": note[:255]},
            request_id=f"refund-{capture_id}",
        )
        return refund


def new_invoice_id() -> str:
    """Unique per order across the sandbox account (PayPal rejects repeats)."""

    return f"VV-{uuid.uuid4().hex[:16].upper()}"


def capture_details(order: Dict[str, Any]) -> Dict[str, Any]:
    """Capture id / status / amount, payer and shipping from a captured order."""

    unit = (order.get("purchase_units") or [{}])[0]
    captures = (unit.get("payments") or {}).get("captures") or []
    capture = captures[0] if captures else {}
    payer = order.get("payer") or {}
    name = payer.get("name") or {}
    shipping = unit.get("shipping") or {}

    return {
        "order_status": order.get("status"),
        "capture_id": capture.get("id"),
        "capture_status": capture.get("status"),
        "amount": (capture.get("amount") or {}).get("value"),
        "currency": (capture.get("amount") or {}).get("currency_code"),
        "payer": {
            "email": payer.get("email_address"),
            "name": " ".join(filter(None, [name.get("given_name"), name.get("surname")])) or None,
            "payer_id": payer.get("payer_id"),
        },
        "shipping": {
            "name": (shipping.get("name") or {}).get("full_name"),
            "address": shipping.get("address"),
        }
        if shipping
        else None,
    }


_client: PayPalClient | None = None
_client_key: tuple | None = None
_client_lock = threading.Lock()


def get_paypal_client(settings: Settings | None = None) -> PayPalClient:
    """Client for the current settings (reused while they are unchanged)."""

    global _client, _client_key

    settings = settings or get_settings()
    key = (settings.paypal_client_id, settings.paypal_client_secret, settings.paypal_api_base)

    with _client_lock:
        if _client is None or _client_key != key:
            _client = PayPalClient(settings)
            _client_key = key

        return _client
