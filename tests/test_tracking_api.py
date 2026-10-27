"""
Accounts, treatment tracking and check-in endpoints (fake predictor, no LLM).
"""

from __future__ import annotations

import io

import pytest

from tests.test_predict_api import textured_png


def register(client, email="grower@example.com", password="leafy-green-1"):
    return client.post("/auth/register", json={"email": email, "password": password})


def start_case(client, predicted_class="Tomato___Early_blight", level="high", **extra):
    data = {
        "file": (io.BytesIO(textured_png()), "leaf.png"),
        "predicted_class": predicted_class,
        "confidence_level": level,
        "plant_name": "Tomato",
        **extra,
    }
    return client.post("/cases", data=data, content_type="multipart/form-data")


def check_in(client, case_id, outcome, photo=False, notes=""):
    data = {"outcome": outcome, "notes": notes, "language": "en"}
    if photo:
        data["file"] = (io.BytesIO(textured_png(seed=1)), "again.png")
    return client.post(f"/cases/{case_id}/checkin", data=data, content_type="multipart/form-data")


def fake_followup(monkeypatch, predicted_class, level="high", is_healthy=False):
    """Make the check-in photo analysis return a fixed diagnosis."""

    import backend.routes.tracking_routes as routes
    from backend.prediction.class_info import parse_class_name

    def analyze(_path, _language):
        return {
            "route": "supported_crop",
            "message": "",
            "plant_identification": {"plant_common_name": "Tomato"},
            "diagnosis": {
                "predicted_class": predicted_class,
                "display_name": parse_class_name(predicted_class).display_name,
                "confidence_level": level,
                "is_healthy": is_healthy,
            },
        }

    monkeypatch.setattr(routes, "analyze_upload", analyze)


# ---------- accounts ----------


def test_register_login_logout(client):
    response = register(client)
    assert response.status_code == 201
    assert response.get_json()["user"]["email"] == "grower@example.com"
    assert client.get("/auth/me").get_json()["user"]["email"] == "grower@example.com"

    client.post("/auth/logout")
    assert client.get("/auth/me").get_json()["user"] is None

    bad = client.post("/auth/login", json={"email": "grower@example.com", "password": "wrong-password"})
    assert bad.status_code == 401

    good = client.post("/auth/login", json={"email": "GROWER@example.com ", "password": "leafy-green-1"})
    assert good.status_code == 200


def test_register_validation(client):
    assert register(client, email="not-an-email").status_code == 400
    assert register(client, password="short").status_code == 400
    assert register(client).status_code == 201
    client.post("/auth/logout")
    assert register(client).get_json()["code"] == "EMAIL_TAKEN"


def test_passwords_are_hashed(client):
    register(client)

    from flask import current_app  # noqa: F401
    from sqlalchemy import select

    from backend.database.db import db_session
    from backend.database.models import User

    with client.application.app_context():
        user = db_session().scalar(select(User))
        assert "leafy-green-1" not in user.password_hash


def test_cases_require_sign_in(client, fake_predictor):
    assert client.get("/cases").status_code == 401
    assert start_case(client).status_code == 401


def test_cross_site_post_is_blocked(client):
    response = client.post(
        "/auth/register",
        json={"email": "a@example.com", "password": "leafy-green-1"},
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 403


# ---------- starting a case ----------


def test_start_case_creates_first_step_with_due_date(client, fake_predictor):
    register(client)
    response = start_case(client)

    assert response.status_code == 201
    case = response.get_json()["case"]
    step = case["current_step"]

    assert case["status"] == "active"
    assert case["awaiting_check_in"] is True
    assert step["kind"] == "treatment"
    assert step["products"][0]["name"] == "Azoxystrobin 23% SC"
    assert step["check_due_at"] > step["created_at"]

    photo = client.get(case["photo_url"])
    assert photo.status_code == 200 and photo.mimetype == "image/jpeg"

    listed = client.get("/cases").get_json()["cases"]
    assert [c["id"] for c in listed] == [case["id"]]


@pytest.mark.parametrize(
    "predicted_class, level, code",
    [
        ("Tomato___healthy", "high", "HEALTHY"),
        ("Tomato___Early_blight", "low", "INVALID_CONFIDENCE_LEVEL"),
        ("Not_a_class", "high", "INVALID_CLASS"),
    ],
)
def test_start_case_rejects_untrackable(client, fake_predictor, predicted_class, level, code):
    register(client)
    response = start_case(client, predicted_class, level)

    assert response.status_code == 400
    assert response.get_json()["code"] == code


def test_start_case_for_unsupported_plant(client, fake_predictor):
    register(client)
    response = start_case(
        client,
        predicted_class="",
        level="",
        plant_name="Hibiscus",
        general_advice='["Remove yellow leaves.", "Mix 5 ml per litre of oil"]',
    )

    case = response.get_json()["case"]
    assert response.status_code == 201
    assert case["predicted_class"] is None
    assert case["current_step"]["actions"] == ["Remove yellow leaves."]  # dose removed


def test_users_cannot_see_each_others_cases(client, fake_predictor):
    register(client, email="one@example.com")
    case_id = start_case(client).get_json()["case"]["id"]
    client.post("/auth/logout")

    register(client, email="two@example.com")
    assert client.get(f"/cases/{case_id}").status_code == 404
    assert client.get(f"/cases/{case_id}/photo").status_code == 404
    assert check_in(client, case_id, "worked").status_code == 404
    assert client.get("/cases").get_json()["cases"] == []


# ---------- check-ins ----------


def test_worked_closes_case_with_positive_message(client, fake_predictor):
    register(client)
    case_id = start_case(client).get_json()["case"]["id"]

    response = check_in(client, case_id, "worked", notes="New leaves are clean")
    body = response.get_json()

    assert response.status_code == 200
    assert body["feedback"].startswith("Great news")
    assert body["case"]["status"] == "recovered"
    assert body["case"]["steps"][0]["notes"] == "New leaves are clean"
    assert check_in(client, case_id, "worked").status_code == 409


def test_not_worked_moves_to_next_step_and_finally_expert(client, fake_predictor):
    register(client)
    case_id = start_case(client).get_json()["case"]["id"]

    second = check_in(client, case_id, "not_worked").get_json()["case"]
    assert second["current_step"]["products"][0]["group"] == "FRAC M03"
    assert second["steps"][0]["outcome"] == "not_worked"

    check_in(client, case_id, "not_worked")
    final = check_in(client, case_id, "not_worked").get_json()

    assert final["case"]["status"] == "expert"
    assert final["case"]["current_step"]["kind"] == "expert"
    assert "expert" in final["feedback"]


def test_invalid_outcome(client, fake_predictor):
    register(client)
    case_id = start_case(client).get_json()["case"]["id"]

    assert check_in(client, case_id, "maybe").status_code == 400


def test_new_photo_finding_a_different_disease_switches_plan(client, fake_predictor, monkeypatch):
    register(client)
    case_id = start_case(client).get_json()["case"]["id"]
    fake_followup(monkeypatch, "Tomato___Late_blight")

    body = check_in(client, case_id, "not_worked", photo=True).get_json()
    case = body["case"]

    assert case["predicted_class"] == "Tomato___Late_blight"
    assert "looks like Tomato — Late blight" in body["feedback"]
    assert case["current_step"]["products"][0]["name"] == "Cyazofamid 34.5% SC"
    assert case["steps"][0]["followup"]["predicted_class"] == "Tomato___Late_blight"

    photo_url = case["steps"][0]["followup_photo_url"]
    assert client.get(photo_url).mimetype == "image/jpeg"


def test_new_photo_confirming_uncertain_diagnosis_unlocks_treatment(client, fake_predictor, monkeypatch):
    register(client)
    case = start_case(client, level="uncertain").get_json()["case"]
    assert case["current_step"]["kind"] == "care"

    fake_followup(monkeypatch, "Tomato___Early_blight", level="high")
    body = check_in(client, case["id"], "not_worked", photo=True).get_json()

    assert body["case"]["confidence_level"] == "high"
    assert body["case"]["current_step"]["kind"] == "treatment"
    assert "confirms" in body["feedback"]


def test_failed_photo_analysis_does_not_block_check_in(client, fake_predictor, monkeypatch):
    import backend.routes.tracking_routes as routes

    def broken(_path, _language):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(routes, "analyze_upload", broken)
    register(client)
    case_id = start_case(client).get_json()["case"]["id"]

    body = check_in(client, case_id, "not_worked", photo=True).get_json()

    assert "could not be analysed" in body["feedback"]
    assert body["case"]["current_step"]["number"] == 2


def test_delete_case(client, fake_predictor):
    register(client)
    case_id = start_case(client).get_json()["case"]["id"]

    assert client.delete(f"/cases/{case_id}").status_code == 200
    assert client.get(f"/cases/{case_id}").status_code == 404


# ---------- preview ----------


def test_treatment_plan_preview_needs_no_login(client, fake_predictor):
    response = client.post(
        "/treatment-plan",
        json={"predicted_class": "Potato___Early_blight", "confidence_level": "high"},
    )
    body = response.get_json()

    assert response.status_code == 200
    assert body["trackable"] is True
    assert body["step"]["products"][0]["name"] == "Zineb"
    assert body["meta"]["checked_on"] == "2026-10-27"
