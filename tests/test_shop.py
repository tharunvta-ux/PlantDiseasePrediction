"""
Treatment-kit shop: dose parsing, kit sizing, the AI assistant and the
PayPal checkout routes (PayPal and the LLM are faked; no network).
"""

from __future__ import annotations

import json
from decimal import Decimal

import httpx
import pytest

from backend.config import get_settings
from backend.services.kit_service import (
    KitError,
    build_quote,
    choose_packs,
    catalog_entry,
    parse_dose,
    validate_field,
)
from backend.services.llm import LLMResponse, LLMUnavailableError
from backend.services.paypal_service import PayPalClient
from backend.services.shop_assistant_service import plan_kit
from tests.test_tracking_api import check_in, register, start_case


@pytest.fixture(autouse=True)
def no_note_llm(monkeypatch):
    import backend.routes.tracking_routes as routes

    monkeypatch.setattr(routes, "review_note", lambda *args, **kwargs: None)


# ---------- dose parsing and kit sizing ----------


@pytest.mark.parametrize(
    ("dose", "amount", "basis"),
    [
        ("0.25% (2.5 g per litre of water)", 2.5, "litre"),
        ("2-4 g per litre of water", 4.0, "litre"),
        ("1 ml per litre of water", 1.0, "litre"),
        ("50 g per 100 litres of spray fluid at petal fall", 0.5, "litre"),
        ("200-300 g per 100 litres of spray fluid (TNAU schedule)", 3.0, "litre"),
        ("1000 g per hectare", 1000.0, "hectare"),
        ("200-250 ml per hectare", 250.0, "hectare"),
        ("1%", 10.0, "litre"),
        ("0.6%", 6.0, "litre"),
    ],
)
def test_parse_dose(dose, amount, basis):
    rate = parse_dose(dose)
    assert rate.basis == basis
    assert rate.amount == pytest.approx(amount)


def test_parse_dose_per_acre_and_kg():
    assert parse_dose("200 ml per acre").amount == pytest.approx(200 / 0.404686)
    assert parse_dose("4 kg per acre (soil application at tillering)").amount == pytest.approx(4000 / 0.404686)
    assert parse_dose("spray when needed") is None


def test_every_plan_product_can_be_sized():
    from backend.services.treatment_plan_service import load_ai_conditions, load_plans

    entries = [*load_plans()["classes"].values(), *load_ai_conditions()["conditions"].values()]
    for entry in entries:
        for product in entry.get("products", []):
            assert catalog_entry(product["name"]) is not None, product["name"]
            assert parse_dose(product["dose"]) is not None, product["dose"]


def test_choose_packs_is_cheapest_cover():
    from itertools import product as combos

    from backend.services.kit_service import pack_price

    entry = catalog_entry("Mancozeb")

    for needed in (3, 800, 1250, 2600):
        counts = choose_packs(entry, needed)
        cost = sum(pack_price(entry, s) * n for s, n in counts.items())
        assert sum(size * n for size, n in counts.items()) >= needed

        # Brute force over small pack counts finds nothing cheaper.
        sizes = entry["packs"]
        for numbers in combos(range(4), repeat=len(sizes)):
            if sum(s * n for s, n in zip(sizes, numbers)) >= needed:
                assert cost <= sum(pack_price(entry, s) * n for s, n in zip(sizes, numbers))

    assert choose_packs(entry, 3) == {100: 1}


def test_quote_quantity_for_per_litre_dose():
    products = [{"name": "Mancozeb", "dose": "0.25% (2.5 g per litre of water)", "kind": "fungicide", "group": "FRAC M03"}]
    quote = build_quote(products, validate_field(1, "acre", 2))

    # 1 acre ~ 202 litres of spray per round, 2.5 g/l, 2 rounds ~ 1012 g.
    assert quote["needed"] == pytest.approx(0.404686 * 500 * 2.5 * 2, rel=1e-3)
    assert quote["spray_litres"] == pytest.approx(202.3, abs=0.1)
    packed = sum(i["pack_size"] * i["quantity"] for i in quote["items"])
    assert packed >= quote["needed"]
    assert Decimal(quote["total"]) == sum(Decimal(i["line_total"]) for i in quote["items"])


def test_quote_picks_cheapest_unless_product_given():
    products = [
        {"name": "Azoxystrobin 23% SC", "dose": "200 ml per acre", "kind": "fungicide", "group": "FRAC 11"},
        {"name": "Pyraclostrobin 20% WG", "dose": "200 g per acre", "kind": "fungicide", "group": "FRAC 11"},
    ]
    field = validate_field(2, "acre", 1)
    assert build_quote(products, field)["product"] == "Azoxystrobin 23% SC"
    chosen = build_quote(products, field, "Pyraclostrobin 20% WG")
    assert chosen["product"] == "Pyraclostrobin 20% WG"
    assert [a["product"] for a in chosen["alternatives"]] == ["Azoxystrobin 23% SC"]

    with pytest.raises(KitError):
        build_quote(products, field, "Something else")


@pytest.mark.parametrize(
    ("area", "unit", "sprays", "code"),
    [
        (0, "acre", 2, "INVALID_AREA"),
        ("abc", "acre", 2, "INVALID_AREA"),
        (float("nan"), "acre", 2, "INVALID_AREA"),
        (1, "bigha", 2, "INVALID_AREA_UNIT"),
        (500, "acre", 2, "AREA_TOO_LARGE"),
        (1, "acre", 9, "INVALID_SPRAYS"),
        (1, "acre", True, "INVALID_SPRAYS"),
    ],
)
def test_validate_field_rejects(area, unit, sprays, code):
    with pytest.raises(KitError) as exc:
        validate_field(area, unit, sprays)
    assert exc.value.code == code


# ---------- AI assistant ----------


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def generate_json(self, system_prompt, user_prompt, response_schema, image=None):
        self.calls.append((system_prompt, user_prompt))
        if self.error:
            raise self.error
        return LLMResponse(text=json.dumps(self.reply), model=self.model)


TWO_OPTIONS = [
    {"name": "Mancozeb", "dose": "2-4 g per litre of water", "kind": "fungicide", "group": "FRAC M03"},
    {"name": "Zineb", "dose": "2-4 g per litre of water", "kind": "fungicide", "group": "FRAC M03"},
]


def assistant_reply(**overrides):
    return {
        "area": 0.5,
        "area_unit": "acre",
        "sprays": 2,
        "option": 2,
        "already_have_option": 0,
        "reply": "I picked the option you asked for.",
        **overrides,
    }


def test_assistant_builds_kit_from_message():
    provider = FakeProvider(assistant_reply())
    result = plan_kit("half acre maize, use zineb, 2 sprays", TWO_OPTIONS, "Maize", "Common rust", provider=provider)

    assert result["field"] == {"area": 0.5, "area_unit": "acre", "sprays": 2}
    assert result["product"] == "Zineb"
    assert result["quote"]["product"] == "Zineb"
    assert result["reply"] == "I picked the option you asked for."

    prompt = provider.calls[0][1]
    assert "half acre maize" in prompt
    assert '"option": 1' in prompt and '"option": 2' in prompt


def test_assistant_skips_product_already_at_home():
    provider = FakeProvider(assistant_reply(option=1, already_have_option=1))
    result = plan_kit("I already have mancozeb", TWO_OPTIONS, "Maize", "Common rust", provider=provider)

    assert result["skipped_product"] == "Mancozeb"
    assert result["quote"]["product"] == "Zineb"


def test_assistant_ignores_unknown_option_and_doses_in_reply():
    provider = FakeProvider(assistant_reply(option=7, reply="Use 2.5 g per litre of Mancozeb 75% WP."))
    result = plan_kit("half an acre", TWO_OPTIONS, "Maize", "Common rust", provider=provider)

    assert result["quote"]["product"] in {"Mancozeb", "Zineb"}  # cheapest
    assert "g per litre" not in result["reply"]


def test_assistant_asks_for_missing_area():
    provider = FakeProvider(assistant_reply(area=0, reply="How big is your field?"))
    result = plan_kit("please send the medicine", TWO_OPTIONS, "Maize", "Common rust", provider=provider)

    assert result["field"] is None
    assert result["quote"] is None
    assert result["reply"] == "How big is your field?"


class FlakyProvider(FakeProvider):
    """First reply is broken JSON, the second is fenced JSON."""

    def generate_json(self, system_prompt, user_prompt, response_schema, image=None):
        self.calls.append((system_prompt, user_prompt))
        if len(self.calls) == 1:
            return LLMResponse(text='{"area": 0.5, "area_unit": "ac', model="lite")
        return LLMResponse(text="```json\n" + json.dumps(self.reply) + "\n```", model="lite")


def test_assistant_retries_broken_json_and_strips_fences():
    provider = FlakyProvider(assistant_reply())
    result = plan_kit("half acre", TWO_OPTIONS, "Maize", "Common rust", provider=provider)

    assert len(provider.calls) == 2
    assert result["quote"]["product"] == "Zineb"


def test_assistant_rejects_empty_message():
    with pytest.raises(KitError):
        plan_kit("  ", TWO_OPTIONS, "Maize", "Common rust", provider=FakeProvider(assistant_reply()))


# ---------- PayPal checkout routes ----------


class FakePayPal:
    """Answers the PayPal REST calls the shop makes."""

    def __init__(self, capture_amount=None, capture_status="COMPLETED"):
        self.requests = []
        self.orders = {}
        self.capture_amount = capture_amount
        self.capture_status = capture_status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path

        if path == "/v1/oauth2/token":
            return httpx.Response(200, json={"access_token": "token", "expires_in": 3600})

        if path == "/v2/checkout/orders" and request.method == "POST":
            body = json.loads(request.content)
            order_id = f"ORDER{len(self.orders) + 1}"
            self.orders[order_id] = body
            return httpx.Response(201, json={"id": order_id, "status": "CREATED"})

        if path.endswith("/capture"):
            order_id = path.split("/")[-2]
            unit = self.orders[order_id]["purchase_units"][0]
            amount = self.capture_amount or unit["amount"]["value"]
            return httpx.Response(201, json={
                "id": order_id,
                "status": "COMPLETED",
                "payer": {"email_address": "buyer@example.com", "name": {"given_name": "Test", "surname": "Buyer"}, "payer_id": "PAYER1"},
                "purchase_units": [{
                    "shipping": {"name": {"full_name": "Test Buyer"}, "address": {"address_line_1": "1 Farm Road", "admin_area_2": "Coimbatore", "postal_code": "641003", "country_code": "IN"}},
                    "payments": {"captures": [{"id": f"CAP-{order_id}", "status": self.capture_status, "amount": {"currency_code": "USD", "value": amount}}]},
                }],
            })

        if path.endswith("/refund"):
            return httpx.Response(201, json={"id": "REFUND1", "status": "COMPLETED"})

        return httpx.Response(404, json={"name": "RESOURCE_NOT_FOUND"})


@pytest.fixture()
def paypal(monkeypatch):
    monkeypatch.setenv("PAYPAL_CLIENT_ID", "sandbox-client-id")
    monkeypatch.setenv("PAYPAL_CLIENT_SECRET", "sandbox-secret")

    fake = FakePayPal()
    client = PayPalClient(get_settings(), http=httpx.Client(transport=httpx.MockTransport(fake)))

    import backend.routes.shop_routes as routes

    monkeypatch.setattr(routes, "get_paypal_client", lambda: client)
    return fake


FIELD = {"area": 0.5, "area_unit": "acre", "sprays": 2}


def signed_in_case(client, **extra):
    register(client)
    response = start_case(client, **extra)
    assert response.status_code == 201
    return response.get_json()["case"]


def test_config_reports_disabled_without_keys(client, monkeypatch):
    monkeypatch.setenv("PAYPAL_CLIENT_ID", "")
    monkeypatch.setenv("PAYPAL_CLIENT_SECRET", "")
    body = client.get("/shop/config").get_json()
    assert body["enabled"] is False
    assert body["client_id"] is None


def test_config_exposes_only_client_id(client, paypal):
    body = client.get("/shop/config").get_json()
    assert body["enabled"] is True
    assert body["client_id"] == "sandbox-client-id"
    assert body["sandbox"] is True
    assert "sandbox-secret" not in json.dumps(body)


def test_kit_requires_sign_in_and_own_case(client, fake_predictor):
    case = signed_in_case(client)
    client.post("/auth/logout")
    assert client.post(f"/shop/cases/{case['id']}/kit", json=FIELD).status_code == 401

    register(client, email="other@example.com")
    assert client.post(f"/shop/cases/{case['id']}/kit", json=FIELD).status_code == 404


def test_kit_for_current_step(client, fake_predictor):
    case = signed_in_case(client)
    step_products = {p["name"] for p in case["current_step"]["products"]}

    body = client.post(f"/shop/cases/{case['id']}/kit", json=FIELD).get_json()

    assert body["quote"]["product"] in step_products
    assert body["field"] == FIELD
    assert body["remembered"] is False
    # Nothing ordered yet, no field given: no kit, form shown empty.
    assert client.post(f"/shop/cases/{case['id']}/kit", json={}).get_json()["quote"] is None


def test_kit_refuses_products_not_in_step(client, fake_predictor):
    case = signed_in_case(client)
    response = client.post(f"/shop/cases/{case['id']}/kit", json={**FIELD, "product": "Acephate"})
    assert response.status_code == 400
    assert response.get_json()["code"] == "INVALID_PRODUCT"


def test_care_step_has_nothing_to_order(client, fake_predictor):
    case = signed_in_case(client, level="uncertain")
    response = client.post(f"/shop/cases/{case['id']}/kit", json=FIELD)
    assert response.status_code == 409
    assert response.get_json()["code"] == "NOTHING_TO_ORDER"


def test_checkout_capture_and_refund(client, fake_predictor, paypal):
    case = signed_in_case(client)
    quote = client.post(f"/shop/cases/{case['id']}/kit", json=FIELD).get_json()["quote"]

    # A price sent by the browser is ignored: the server recomputes it.
    created = client.post(f"/shop/cases/{case['id']}/orders", json={**FIELD, "total": "0.01"})
    assert created.status_code == 201
    order_id = created.get_json()["paypal_order_id"]

    sent = paypal.orders[order_id]
    unit = sent["purchase_units"][0]
    assert sent["intent"] == "CAPTURE"
    assert unit["amount"]["value"] == quote["total"]
    assert unit["amount"]["breakdown"]["item_total"]["value"] == quote["total"]
    assert sum(Decimal(i["unit_amount"]["value"]) * int(i["quantity"]) for i in unit["items"]) == Decimal(quote["total"])
    assert unit["reference_id"] == f"case-{case['id']}-step-1"
    create_request = next(r for r in paypal.requests if r.url.path == "/v2/checkout/orders")
    assert create_request.headers["PayPal-Request-Id"].startswith("create-VV-")

    # Not listed until paid.
    assert client.get("/shop/orders").get_json()["orders"] == []

    captured = client.post(f"/shop/orders/{order_id}/capture")
    assert captured.status_code == 200
    order = captured.get_json()["order"]
    assert order["status"] == "paid"
    assert order["payer"]["email"] == "buyer@example.com"
    assert order["shipping"]["address"]["admin_area_2"] == "Coimbatore"

    # Capturing again is harmless.
    assert client.post(f"/shop/orders/{order_id}/capture").get_json()["order"]["status"] == "paid"

    listed = client.get("/shop/orders").get_json()["orders"]
    assert [o["paypal_order_id"] for o in listed] == [order_id]

    refunded = client.post(f"/shop/orders/{order['id']}/refund")
    assert refunded.status_code == 200
    assert refunded.get_json()["order"]["status"] == "refunded"
    assert client.post(f"/shop/orders/{order['id']}/refund").status_code == 200


def test_capture_with_wrong_amount_is_not_marked_paid(client, fake_predictor, paypal):
    paypal.capture_amount = "0.01"
    case = signed_in_case(client)
    order_id = client.post(f"/shop/cases/{case['id']}/orders", json=FIELD).get_json()["paypal_order_id"]

    response = client.post(f"/shop/orders/{order_id}/capture")
    assert response.status_code == 502
    assert client.get("/shop/orders").get_json()["orders"][0]["status"] == "failed"


def test_other_users_cannot_capture_or_refund(client, fake_predictor, paypal):
    case = signed_in_case(client)
    order_id = client.post(f"/shop/cases/{case['id']}/orders", json=FIELD).get_json()["paypal_order_id"]
    order = client.post(f"/shop/orders/{order_id}/capture").get_json()["order"]

    client.post("/auth/logout")
    register(client, email="other@example.com")
    assert client.post(f"/shop/orders/{order_id}/capture").status_code == 404
    assert client.post(f"/shop/orders/{order['id']}/refund").status_code == 404


def test_next_step_reuses_field_with_a_different_group(client, fake_predictor, paypal):
    case = signed_in_case(client)
    first_group = {p["group"] for p in case["current_step"]["products"]}
    order_id = client.post(f"/shop/cases/{case['id']}/orders", json=FIELD).get_json()["paypal_order_id"]
    client.post(f"/shop/orders/{order_id}/capture")

    after = check_in(client, case["id"], "not_worked").get_json()["case"]
    assert after["current_step"]["kind"] == "treatment"

    body = client.post(f"/shop/cases/{case['id']}/kit", json={}).get_json()
    assert body["remembered"] is True
    assert body["field"] == FIELD
    assert body["step_number"] == 2
    assert body["quote"]["group"] not in first_group


def test_assistant_route(client, fake_predictor, monkeypatch):
    import backend.services.shop_assistant_service as assistant

    case = signed_in_case(client)
    provider = FakeProvider(assistant_reply(option=1, reply="Ready for your half acre."))
    monkeypatch.setattr(assistant, "get_llm_provider", lambda: provider)

    response = client.post(f"/shop/cases/{case['id']}/assistant", json={"message": "arai acre", "language": "ta"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["field"] == FIELD
    assert body["reply"] == "Ready for your half acre."
    assert "Tamil" in provider.calls[0][1]


def test_assistant_route_when_llm_is_down(client, fake_predictor, monkeypatch):
    import backend.services.shop_assistant_service as assistant

    case = signed_in_case(client)
    monkeypatch.setattr(assistant, "get_llm_provider", lambda: FakeProvider(error=LLMUnavailableError("down")))

    response = client.post(f"/shop/cases/{case['id']}/assistant", json={"message": "half acre", "language": "en"})
    assert response.status_code == 503
    assert response.get_json()["code"] == "ASSISTANT_UNAVAILABLE"


def test_shop_blocks_cross_site_posts(client, fake_predictor):
    case = signed_in_case(client)
    response = client.post(
        f"/shop/cases/{case['id']}/orders", json=FIELD, headers={"Origin": "https://evil.example"}
    )
    assert response.status_code == 403
