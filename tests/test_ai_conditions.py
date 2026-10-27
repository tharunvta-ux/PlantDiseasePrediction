"""
TNAU plans for diseases the vision LLM matches on crops the CNN does not
cover (papaya, banana, mango, brinjal, chilli, rice, coconut, groundnut).
"""

from __future__ import annotations

import io
import re

import pytest
from PIL import Image

import backend.services.plant_identification_service as ident
from backend.prediction.class_info import parse_class_name
from backend.services.analysis_service import ROUTE_UNSUPPORTED, combine_results
from backend.services.tracking_service import summarise_followup
from backend.services.treatment_plan_service import (
    KIND_CARE,
    KIND_EXPERT,
    KIND_TREATMENT,
    ai_condition_ids,
    first_step,
    load_ai_conditions,
    next_step,
)
from tests.conftest import CLASS_NAMES, one_hot_like
from tests.test_analysis import CROPS, FakeVisionProvider, identification, make_run
from tests.test_predict_api import textured_png
from tests.test_tracking_api import register

CONDITIONS = load_ai_conditions()["conditions"]
NEW_CROPS = {"Papaya", "Banana", "Mango", "Brinjal", "Chilli", "Rice", "Coconut", "Groundnut"}


def papaya(**overrides):
    fields = {
        "plant_common_name": "Papaya",
        "plant_scientific_name": "Carica papaya",
        "supported_crop": "none",
        "identification_confidence": "medium",
        "appears_healthy": "no",
        "matched_condition": "Papaya___Powdery_mildew",
        **overrides,
    }
    return identification(**fields)


# ---------- data ----------


def test_conditions_cover_the_new_crops_only():
    crops = {parse_class_name(key).crop for key in CONDITIONS}

    assert crops == NEW_CROPS
    assert not crops & set(CROPS)  # never overlaps the CNN's crops


@pytest.mark.parametrize("key", list(CONDITIONS))
def test_condition_entries_are_well_formed(key):
    entry = CONDITIONS[key]

    assert entry["source_url"].startswith("https://agritech.tnau.ac.in/")
    assert entry["symptoms"] and entry["care"]
    for product in entry["products"]:
        assert set(product) == {"name", "dose", "kind", "group"}
        assert all(product.values())


def test_banned_or_unverified_chemicals_are_never_suggested():
    excluded = re.compile(
        r"dinocap|karathane|tridemorph|benomyl|carbofuran|methyl dem|edifenphos|"
        r"iprobenphos|chinomethionate|streptomycin|tetracycline|streptocycline|fernoxone|2,4-d",
        re.IGNORECASE,
    )

    for key, entry in CONDITIONS.items():
        for product in entry["products"]:
            assert not excluded.search(product["name"]), key


# ---------- plan ladder ----------


def test_confirmed_match_gets_tnau_treatment():
    step = first_step("Banana___Yellow_sigatoka", "high")

    assert step.kind == KIND_TREATMENT
    assert step.products[0]["name"] == "Propiconazole"
    assert "banana_8" in step.source_url


def test_unconfirmed_match_gets_care_only():
    step = first_step("Banana___Yellow_sigatoka", "uncertain")

    assert step.kind == KIND_CARE
    assert step.products == []
    assert "not confirmed" in step.reason


def test_confirmed_match_switches_group_then_expert():
    cls = "Rice___Blast"
    one = first_step(cls, "high")
    two = next_step(cls, "high", [{"kind": one.kind, "products": one.products}])

    assert one.products[0]["name"] == "Tricyclazole 75% WP"
    assert two.products[0]["group"] != one.products[0]["group"]


def test_viral_match_removes_plants_then_expert():
    one = first_step("Papaya___Ring_spot_virus", "high")

    assert one.kind == KIND_CARE and one.products == []
    assert "spreading" in one.check_question
    assert next_step("Papaya___Ring_spot_virus", "high", [{"kind": one.kind, "products": []}]).kind == KIND_EXPERT


# ---------- identification + routing ----------


@pytest.fixture()
def photo(tmp_path):
    path = tmp_path / "photo.png"
    Image.open(io.BytesIO(textured_png())).save(path)
    return str(path)


@pytest.fixture(autouse=True)
def _clear_identification_cache():
    ident.clear_cache()
    yield
    ident.clear_cache()


def test_prompt_and_schema_list_known_conditions(photo):
    provider = FakeVisionProvider(papaya())
    result = ident.identify_plant(photo, CROPS, provider=provider)

    _, user_prompt, schema, _ = provider.calls[0]
    assert result["matched_condition"] == "Papaya___Powdery_mildew"
    assert "Papaya___Powdery_mildew" in user_prompt
    assert "White powdery growth" in user_prompt
    assert schema["properties"]["matched_condition"]["enum"] == [*ai_condition_ids(), "none"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"appears_healthy": "yes"},                      # healthy plant: no disease
        {"matched_condition": "Papaya___Not_a_thing"},   # unknown id
    ],
)
def test_match_is_dropped_when_not_trustworthy(photo, overrides):
    provider = FakeVisionProvider(papaya(**overrides))

    assert ident.identify_plant(photo, CROPS, provider=provider)["matched_condition"] == "none"


def test_match_is_dropped_for_model_crops(photo):
    reply = identification(matched_condition="Papaya___Powdery_mildew")  # a tomato
    provider = FakeVisionProvider(reply)

    assert ident.identify_plant(photo, CROPS, provider=provider)["matched_condition"] == "none"


def test_matched_condition_routes_to_ai_check_even_at_medium_confidence():
    run = make_run(one_hot_like("Tomato___Early_blight", 0.97))
    out = combine_results(run, papaya())

    assert out["route"] == ROUTE_UNSUPPORTED
    assert out["diagnosis"] is None
    assert out["ai_condition"]["id"] == "Papaya___Powdery_mildew"
    assert out["ai_condition"]["has_products"] is True
    assert "powdery mildew" in out["message"]


def test_no_match_keeps_previous_routing():
    run = make_run(one_hot_like("Tomato___Early_blight", 0.97))
    out = combine_results(run, papaya(matched_condition="none"))

    assert out["ai_condition"] is None
    assert out["route"] != ROUTE_UNSUPPORTED  # medium confidence, as before


def test_followup_photo_match_is_never_high():
    analysis = {
        "route": ROUTE_UNSUPPORTED,
        "diagnosis": None,
        "plant_identification": {"plant_common_name": "Papaya"},
        "ai_condition": {"id": "Papaya___Powdery_mildew", "display_name": "Papaya — Powdery mildew"},
    }

    summary = summarise_followup(analysis)

    assert summary["predicted_class"] == "Papaya___Powdery_mildew"
    assert summary["confidence_level"] == "uncertain"


# ---------- tracking ----------


def test_track_confirmed_ai_match(client, fake_predictor, monkeypatch):
    import backend.routes.tracking_routes as routes

    monkeypatch.setattr(routes, "review_note", lambda *a, **k: None)
    register(client)
    response = client.post(
        "/cases",
        data={
            "file": (io.BytesIO(textured_png()), "papaya.png"),
            "predicted_class": "Papaya___Powdery_mildew",
            "confidence_level": "high",
            "plant_name": "Papaya",
        },
        content_type="multipart/form-data",
    )
    case = response.get_json()["case"]

    assert response.status_code == 201
    assert case["diagnosis_source"] == "ai_match"
    assert case["display_name"] == "Papaya — Powdery mildew"
    assert case["current_step"]["products"][0]["name"] == "Wettable sulphur"


def test_preview_for_unconfirmed_ai_match(client, fake_predictor):
    body = client.post(
        "/treatment-plan",
        json={"predicted_class": "Chilli___Cercospora_leaf_spot", "confidence_level": "uncertain"},
    ).get_json()

    assert body["trackable"] is True
    assert body["step"]["kind"] == "care"
    assert body["step"]["products"] == []
