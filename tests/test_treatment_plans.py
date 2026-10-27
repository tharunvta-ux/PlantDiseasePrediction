"""
Treatment plan ladder and the curated TNAU plan data.
"""

from __future__ import annotations

import re

import pytest

from backend.prediction.class_info import parse_class_name
from backend.services.treatment_plan_service import (
    KIND_CARE,
    KIND_EXPERT,
    KIND_TREATMENT,
    MAX_STEPS,
    first_step,
    is_trackable,
    load_plans,
    next_step,
)
from tests.conftest import CLASS_NAMES

DISEASE_CLASSES = [c for c in CLASS_NAMES if not parse_class_name(c).is_healthy]


def as_previous(*steps):
    return [{"kind": s.kind, "products": s.products} for s in steps]


# ---------- curated data ----------


def test_every_disease_class_has_a_plan():
    plans = load_plans()["classes"]

    assert sorted(plans) == sorted(DISEASE_CLASSES)


@pytest.mark.parametrize("name", DISEASE_CLASSES)
def test_plan_entries_are_well_formed(name):
    entry = load_plans()["classes"][name]

    assert entry["care"], "every plan needs low-risk care steps"
    assert isinstance(entry["wait_days"], int)

    for product in entry["products"]:
        assert set(product) == {"name", "dose", "kind", "group"}
        assert all(product.values())

    if entry["products"]:
        assert entry["source_url"].startswith("https://agritech.tnau.ac.in/"), (
            "chemical options must come from a TNAU page"
        )


def test_banned_or_excluded_chemicals_are_never_suggested():
    excluded = re.compile(
        r"captafol|tridemorph|streptocycline|streptomycin|tetracycline|dicofol|"
        r"methyl demeton|monocrotophos|methomyl|dinocap",
        re.IGNORECASE,
    )

    for name, entry in load_plans()["classes"].items():
        for product in entry["products"]:
            assert not excluded.search(product["name"]), name


# ---------- ladder ----------


def test_confident_diagnosis_with_tnau_products_starts_with_treatment():
    step = first_step("Tomato___Early_blight", "high")

    assert step.kind == KIND_TREATMENT
    assert {p["group"] for p in step.products} == {"FRAC 11"}
    assert [p["name"] for p in step.products] == ["Azoxystrobin 23% SC", "Pyraclostrobin 20% WG"]
    assert step.wait_days == 7
    assert step.source_url.endswith("tomato_diseases_2.html")
    assert any("label" in a for a in step.actions)


def test_uncertain_diagnosis_gets_care_only():
    step = first_step("Tomato___Early_blight", "uncertain")

    assert step.kind == KIND_CARE
    assert step.products == []
    assert "not certain" in step.reason


def test_failed_treatment_switches_chemical_group_then_ends_at_expert():
    cls = "Tomato___Early_blight"
    one = first_step(cls, "high")
    two = next_step(cls, "high", as_previous(one))
    three = next_step(cls, "high", as_previous(one, two))
    four = next_step(cls, "high", as_previous(one, two, three))

    assert two.kind == KIND_TREATMENT and [p["group"] for p in two.products] == ["FRAC M03"]
    assert "different chemical group" in two.title
    assert three.kind == KIND_TREATMENT and [p["group"] for p in three.products] == ["FRAC 2"]
    assert MAX_STEPS == 3
    assert four.kind == KIND_EXPERT


def test_single_chemical_group_goes_to_expert_after_one_failure():
    cls = "Potato___Late_blight"  # mancozeb and zineb are both FRAC M03
    one = first_step(cls, "high")

    assert one.kind == KIND_TREATMENT and len(one.products) == 2
    assert next_step(cls, "high", as_previous(one)).kind == KIND_EXPERT


def test_care_then_confirmed_diagnosis_unlocks_treatment():
    cls = "Tomato___Late_blight"
    care = first_step(cls, "uncertain")

    assert next_step(cls, "uncertain", as_previous(care)).kind == KIND_EXPERT

    treatment = next_step(cls, "high", as_previous(care))
    assert treatment.kind == KIND_TREATMENT
    assert treatment.products[0]["name"] == "Cyazofamid 34.5% SC"


def test_no_tnau_product_means_care_then_expert():
    cls = "Tomato___Leaf_Mold"
    care = first_step(cls, "high")

    assert care.kind == KIND_CARE and care.products == []
    assert next_step(cls, "high", as_previous(care)).kind == KIND_EXPERT


def test_viral_disease_targets_the_vector_and_then_expert():
    cls = "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
    one = first_step(cls, "high")

    assert one.kind == KIND_TREATMENT
    assert all("whitefly" in p["kind"] for p in one.products)
    assert "spreading" in one.check_question
    assert next_step(cls, "high", as_previous(one)).kind == KIND_EXPERT


def test_incurable_disease_goes_straight_to_expert():
    step = first_step("Orange___Haunglongbing_(Citrus_greening)", "high")

    assert step.kind == KIND_EXPERT
    assert any("no cure" in a.lower() for a in step.actions)


def test_unsupported_plant_uses_ai_advice_then_expert():
    one = first_step(None, None, ["Remove yellow leaves."])

    assert one.kind == KIND_CARE and one.actions == ["Remove yellow leaves."]
    assert one.products == []
    assert next_step(None, None, as_previous(one)).kind == KIND_EXPERT


def test_expert_step_lists_kisan_call_centre():
    step = next_step(None, None, as_previous(first_step(None, None)))

    assert any("1800-180-1551" in a for a in step.actions)


def test_trackable():
    assert is_trackable("Tomato___Early_blight", "high")
    assert is_trackable("Tomato___Early_blight", "uncertain")
    assert not is_trackable("Tomato___Early_blight", "low")
    assert not is_trackable("Tomato___healthy", "high")
    assert is_trackable(None, None)
