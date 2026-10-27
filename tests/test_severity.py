"""
Severity measurement, grades, trend, outlook and their use in check-ins.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from sqlalchemy import create_engine, inspect, text

from backend.prediction.severity import GRADE_MILD, GRADE_MODERATE, GRADE_SEVERE, measure_severity
from backend.services import severity_service as sev
from backend.services.treatment_plan_service import load_ai_conditions
from tests.test_predict_api import textured_png
from tests.test_tracking_api import check_in, register, start_case

GREEN = (60, 160, 60)  # BGR
BROWN = (40, 80, 140)
GREY = (150, 150, 150)
WHITE = (235, 235, 235)


def leaf_photo(tmp_path, spots=0, spot_colour=BROWN, background=GREY, name="leaf.png"):
    """Grey background, 300x300 green leaf, `spots` 30x30 spots inside it."""

    image = np.full((400, 400, 3), background, dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (349, 349), GREEN, thickness=-1)
    for i in range(spots):
        x, y = 60 + (i % 9) * 32, 60 + (i // 9) * 32
        cv2.rectangle(image, (x, y), (x + 29, y + 29), spot_colour, thickness=-1)
    path = tmp_path / name
    cv2.imwrite(str(path), image)
    return str(path)


# ---------- measurement ----------


def test_healthy_leaf_is_mild(tmp_path):
    result = measure_severity(leaf_photo(tmp_path))

    assert result.reliable
    assert result.affected_percent < 1
    assert result.grade == GRADE_MILD


@pytest.mark.parametrize("spots, expected", [(9, 9.0), (27, 27.0), (54, 54.0)])
def test_brown_spot_share_is_measured(tmp_path, spots, expected):
    # each spot is 30x30 = 900 px of a 300x300 = 90000 px leaf -> 1% each
    result = measure_severity(leaf_photo(tmp_path, spots))

    assert result.affected_percent == pytest.approx(expected, abs=2.5)


def test_white_powder_counts_as_affected(tmp_path):
    result = measure_severity(leaf_photo(tmp_path, 27, spot_colour=WHITE))

    assert result.affected_percent == pytest.approx(27.0, abs=2.5)


def test_no_leaf_is_unreliable(tmp_path):
    image = np.full((300, 300, 3), GREY, dtype=np.uint8)
    path = tmp_path / "grey.png"
    cv2.imwrite(str(path), image)

    assert measure_severity(str(path)).reliable is False


def test_grades():
    from backend.prediction.severity import grade_for_percent

    assert grade_for_percent(5) == GRADE_MILD
    assert grade_for_percent(15) == GRADE_MODERATE
    assert grade_for_percent(45) == GRADE_SEVERE


# ---------- combining, trend, outlook ----------


def photo(percent, reliable=True):
    from backend.prediction.severity import grade_for_percent

    return {"affected_percent": percent, "grade": grade_for_percent(percent), "reliable": reliable}


def test_worse_of_photo_and_grower_estimate_wins():
    assert sev.combine(photo(5), sev.EXTENT_MOST)["grade"] == GRADE_SEVERE
    assert sev.combine(photo(45), sev.EXTENT_FEW)["grade"] == GRADE_SEVERE
    assert sev.combine(photo(45, reliable=False), sev.EXTENT_FEW)["grade"] == GRADE_MILD
    assert sev.combine(None, None) is None


@pytest.mark.parametrize(
    "before, after, direction",
    [(40, 15, "improving"), (15, 40, "worsening"), (20, 18, "same"), (4, 2, "same")],
)
def test_trend_from_photos(before, after, direction):
    change = sev.trend(sev.combine(photo(before), None), sev.combine(photo(after), None))

    assert change["direction"] == direction


def test_trend_from_grades_without_photos():
    change = sev.trend(sev.combine(None, sev.EXTENT_MOST), sev.combine(None, sev.EXTENT_FEW))

    assert change["direction"] == "improving"


def test_outlook_follows_grade_and_trend():
    def level(grade_percent, change=None):
        return sev.outlook("Tomato___Early_blight", sev.combine(photo(grade_percent), None), change)["level"]

    assert level(5) == "good"
    assert level(15) == "fair"
    assert level(45) == "poor"
    assert level(45, {"direction": "improving"}) == "fair"
    assert level(5, {"direction": "worsening"}) == "fair"


def test_virus_outlook_is_poor_even_when_mild():
    view = sev.outlook("Tomato___Tomato_Yellow_Leaf_Curl_Virus", sev.combine(photo(3), None))

    assert view["level"] == "poor"
    assert "cannot be cured" in view["text"]


def test_outlook_text_never_claims_a_percentage():
    for percent in (5, 15, 45):
        assert "%" not in sev.outlook("Rice___Blast", sev.combine(photo(percent), None))["text"]


def test_every_ai_condition_has_two_or_three_checks():
    for key, entry in load_ai_conditions()["conditions"].items():
        assert 2 <= len(entry["checks"]) <= 3, key
        assert all(q.endswith("?") for q in entry["checks"]), key


# ---------- schema upgrade ----------


def test_missing_nullable_columns_are_added(tmp_path):
    from backend.database.db import add_missing_columns
    from backend.database.models import Base

    engine = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    Base.metadata.create_all(engine)
    with engine.begin() as connection:  # simulate a database from before severity existed
        connection.execute(text('ALTER TABLE treatment_cases DROP COLUMN severity'))
        connection.execute(text('ALTER TABLE treatment_steps DROP COLUMN severity'))

    add_missing_columns(engine)

    assert "severity" in {c["name"] for c in inspect(engine).get_columns("treatment_cases")}
    assert "severity" in {c["name"] for c in inspect(engine).get_columns("treatment_steps")}
    add_missing_columns(engine)  # idempotent


# ---------- tracking ----------


@pytest.fixture()
def measured(monkeypatch):
    """Queue of photo measurements returned for uploads, in order."""

    import backend.routes.tracking_routes as routes

    queue = []
    monkeypatch.setattr(routes, "_photo_severity", lambda _path: photo(queue.pop(0)) if queue else None)
    monkeypatch.setattr(routes, "analyze_upload", lambda _path, _language: {"diagnosis": None})
    return queue


def test_case_stores_severity_and_outlook(client, fake_predictor, measured):
    measured.append(18)
    register(client)
    case = start_case(client, extent="about_half").get_json()["case"]

    assert case["severity"]["grade"] == GRADE_MODERATE
    assert case["severity"]["extent"] == "about_half"
    assert case["severity"]["outlook"]["level"] == "fair"


def test_shrinking_disease_keeps_the_step_going_once(client, fake_predictor, measured):
    measured.extend([40, 15, 6])
    register(client)
    case = start_case(client).get_json()["case"]
    products = case["current_step"]["products"]

    first = check_in(client, case["id"], "not_worked", photo=True).get_json()
    step = first["case"]["current_step"]
    assert step["title"].startswith("Keep going")
    assert step["products"] == products
    assert "less disease than before" in first["feedback"]
    assert first["case"]["latest_severity"]["trend"]["direction"] == "improving"

    # Shrinking again, but the step was already repeated once: move on.
    second = check_in(client, case["id"], "not_worked", photo=True).get_json()
    assert not second["case"]["current_step"]["title"].startswith("Keep going")


def test_fast_spreading_severe_disease_goes_to_expert(client, fake_predictor, measured):
    measured.extend([12, 55])
    register(client)
    case_id = start_case(client).get_json()["case"]["id"]

    body = check_in(client, case_id, "not_worked", photo=True).get_json()

    assert body["case"]["status"] == "expert"
    assert "spreading fast" in body["feedback"]


def test_worked_with_shrinking_photo_says_so(client, fake_predictor, measured):
    measured.extend([30, 5])
    register(client)
    case_id = start_case(client).get_json()["case"]["id"]

    body = check_in(client, case_id, "worked", photo=True).get_json()

    assert "The photo agrees" in body["feedback"]
    assert body["case"]["steps"][0]["severity"]["outlook"]["level"] == "good"


def test_extent_only_check_in(client, fake_predictor, measured):
    register(client)
    case_id = start_case(client, extent="most").get_json()["case"]["id"]

    response = client.post(
        f"/cases/{case_id}/checkin",
        data={"outcome": "not_worked", "extent": "few_leaves", "language": "en"},
        content_type="multipart/form-data",
    )
    step = response.get_json()["case"]["steps"][0]

    assert step["severity"]["grade"] == GRADE_MILD
    assert step["severity"]["trend"]["direction"] == "improving"


def test_preview_returns_outlook(client, fake_predictor):
    body = client.post(
        "/treatment-plan",
        json={
            "predicted_class": "Tomato___Early_blight",
            "confidence_level": "high",
            "photo_severity": {"affected_percent": 4, "grade": "mild", "reliable": True},
            "extent": "most",
        },
    ).get_json()

    assert body["severity"]["grade"] == GRADE_SEVERE
    assert body["severity"]["outlook"]["level"] == "poor"


def test_preview_ignores_bad_severity_input(client, fake_predictor):
    body = client.post(
        "/treatment-plan",
        json={
            "predicted_class": "Tomato___Early_blight",
            "confidence_level": "high",
            "photo_severity": {"affected_percent": 500, "grade": "awful"},
            "extent": "everything",
        },
    ).get_json()

    assert body["severity"] is None
