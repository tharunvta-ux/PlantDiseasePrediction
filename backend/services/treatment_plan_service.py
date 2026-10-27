"""
backend/services/treatment_plan_service.py

Step-by-step treatment plans for treatment tracking.

A plan is a short ladder of steps; the grower reports after each step
whether it worked:

    care        low-risk cultural steps (always first when the diagnosis
                is uncertain or no TNAU product is listed)
    treatment   care steps plus ONE of the TNAU-listed products of one
                chemical group (only for high-confidence diagnoses)
    expert      consult an agricultural expert nearby

"Not worked" moves to the next rung: care -> treatment (if confident)
-> treatment with a different chemical group -> expert. Viral diseases
cannot be cured, so after one step they go to an expert.

Pure functions over backend/knowledge/treatment_plans.json - products
and doses never come from the LLM.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence

from backend.prediction.class_info import parse_class_name
from backend.prediction.confidence import LEVEL_HIGH
from backend.services.recommendation_service import load_knowledge

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"
PLANS_FILE = KNOWLEDGE_DIR / "treatment_plans.json"
# Diseases of crops the CNN does not cover, matched by the vision LLM and
# confirmed by the grower. Ids use the same "Crop___Condition" form.
AI_CONDITIONS_FILE = KNOWLEDGE_DIR / "ai_condition_plans.json"

KIND_CARE = "care"
KIND_TREATMENT = "treatment"
KIND_EXPERT = "expert"

# Non-expert steps before the plan always ends at an expert.
MAX_STEPS = 3

CARE_WAIT_DAYS = 3

KISAN_CALL_CENTRE = "1800-180-1551"

PRODUCT_SAFETY = (
    "Use only ONE of the listed products. Read and follow the product label "
    "(dose, safety gear, waiting period before harvest). Doses are as listed "
    "by TNAU - check that the product is registered for your crop locally."
)

EXPERT_ACTIONS = [
    "Take the plant (or clear photos of affected leaves) to your nearest "
    "Krishi Vigyan Kendra (KVK) or agriculture / horticulture department office.",
    f"Call the Kisan Call Centre (free, in local languages): {KISAN_CALL_CENTRE}.",
    "Show them this treatment history so they know what has already been tried.",
]

GENERIC_CARE = [
    "Remove and destroy badly affected leaves (do not compost them).",
    "Water at the base of the plant, not over the leaves.",
    "Check the plant every day and note any change.",
]


@dataclass
class StepPlan:
    """One step of a treatment plan (stored on TreatmentStep)."""

    kind: str
    title: str
    actions: List[str]
    products: List[Dict[str, str]] = field(default_factory=list)
    wait_days: int = 0
    check_question: str = ""
    reason: str = ""
    source_url: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def load_plans() -> Dict[str, Any]:
    """Load the curated treatment plans."""

    return json.loads(PLANS_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_ai_conditions() -> Dict[str, Any]:
    """Curated plans for AI-matched diseases of other crops."""

    return json.loads(AI_CONDITIONS_FILE.read_text(encoding="utf-8"))


def ai_condition_ids() -> List[str]:
    return list(load_ai_conditions()["conditions"])


def is_ai_condition(key: str | None) -> bool:
    return bool(key) and key in load_ai_conditions()["conditions"]


def ai_condition_summary(key: str) -> Dict[str, Any]:
    """What the grower sees before confirming an AI-matched disease."""

    entry = load_ai_conditions()["conditions"][key]
    info = parse_class_name(key)

    return {
        "id": key,
        "crop": info.crop,
        "disease": info.condition,
        "display_name": info.display_name,
        "symptoms": entry["symptoms"],
        "source_url": entry["source_url"],
        "has_products": bool(entry["products"]),
    }


def plan_entry(predicted_class: str) -> Dict[str, Any]:
    """Curated entry for a model class or AI condition (empty dict if none)."""

    return (
        load_plans()["classes"].get(predicted_class)
        or load_ai_conditions()["conditions"].get(predicted_class)
        or {}
    )


def is_viral(predicted_class: str) -> bool:
    entry = load_knowledge().get(predicted_class) or plan_entry(predicted_class)
    return entry.get("pathogen_type") == "viral"


def is_trackable(predicted_class: str | None, level: str | None) -> bool:
    """Healthy plants and low-confidence results get no treatment plan."""

    if predicted_class is None:
        return True  # unsupported plant with an AI observation

    return level in ("high", "uncertain") and not parse_class_name(predicted_class).is_healthy


def _groups(products: Sequence[Dict[str, str]]) -> List[str]:
    """Distinct chemical groups in listing order."""

    seen: List[str] = []
    for product in products:
        if product["group"] not in seen:
            seen.append(product["group"])
    return seen


def _treatment_step(
    predicted_class: str,
    entry: Dict[str, Any],
    group: str,
    switched: bool,
) -> StepPlan:
    products = [p for p in entry["products"] if p["group"] == group]
    kind_word = products[0]["kind"].split(" ")[0]
    viral = is_viral(predicted_class)

    if switched:
        title = f"Switch to a {kind_word} from a different chemical group"
        reason = (
            f"The previous {kind_word} did not help enough. Switching to a different "
            "chemical group works better than repeating the same one, because the "
            "pest or disease may tolerate it."
        )
    elif viral:
        title = "Remove infected plants and control the insects that spread the virus"
        reason = (
            "Virus-infected plants cannot be cured; the goal is to stop the virus "
            "spreading to healthy plants."
        )
    elif is_ai_condition(predicted_class):
        title = f"Care steps plus a TNAU-recommended {kind_word}"
        reason = (
            "You confirmed the symptoms match, so TNAU's treatment is suggested. "
            "If unsure, show the plant to an expert before spraying."
        )
    else:
        title = f"Care steps plus a TNAU-recommended {kind_word}"
        reason = "The diagnosis is confident, so a targeted treatment is suggested."

    return StepPlan(
        kind=KIND_TREATMENT,
        title=title,
        actions=[*entry.get("care", []), PRODUCT_SAFETY],
        products=[{k: p[k] for k in ("name", "dose", "kind", "group")} for p in products],
        wait_days=int(entry.get("wait_days") or 7),
        check_question=_check_question(viral),
        reason=reason,
        source_url=entry.get("source_url"),
    )


def _check_question(viral: bool) -> str:
    if viral:
        return "Did the problem stop spreading (no new plants with symptoms)?"
    return "Is the plant recovering (no new spots, new leaves look healthy)?"


def expert_step(reason: str) -> StepPlan:
    return StepPlan(
        kind=KIND_EXPERT,
        title="Consult an agricultural expert nearby",
        actions=list(EXPERT_ACTIONS),
        reason=reason,
        check_question="Did the expert's advice solve the problem?",
    )


def first_step(
    predicted_class: str | None,
    level: str | None,
    ai_advice: Sequence[str] | None = None,
) -> StepPlan:
    """
    First step for a new case.

    predicted_class None means a plant the disease model does not cover;
    `ai_advice` then holds the AI observation's low-risk care steps.
    """

    if predicted_class is None:
        return StepPlan(
            kind=KIND_CARE,
            title="Try these general care steps",
            actions=list(ai_advice or []) or list(GENERIC_CARE),
            wait_days=CARE_WAIT_DAYS,
            check_question=_check_question(False),
            reason=(
                "This plant is not covered by the disease model, so only general, "
                "low-risk care is suggested. No chemicals are suggested without a "
                "confirmed diagnosis."
            ),
        )

    entry = plan_entry(predicted_class)
    viral = is_viral(predicted_class)

    if entry.get("incurable"):
        step = expert_step(
            "This disease cannot be cured on the plant and needs expert "
            "confirmation before any tree is removed."
        )
        step.actions = [*entry.get("care", []), *step.actions]
        step.source_url = entry.get("source_url")
        return step

    groups = _groups(entry.get("products", []))

    if level == LEVEL_HIGH and groups:
        return _treatment_step(predicted_class, entry, groups[0], switched=False)

    reason = "Low-risk care steps first."
    if groups and is_ai_condition(predicted_class):
        reason += (
            " The symptoms are not confirmed, so the TNAU treatment is only shown "
            "once they are (or after an expert confirms the disease)."
        )
    elif groups:
        reason += (
            " The diagnosis is not certain, so chemical treatment is only suggested "
            "after a clearer photo confirms it."
        )
    else:
        reason += " TNAU lists no chemical option for this condition that this app can suggest."

    return StepPlan(
        kind=KIND_CARE,
        title=(
            "Remove infected plants and stop the spread"
            if viral
            else "Start with these care steps"
        ),
        actions=list(entry.get("care", [])) or list(GENERIC_CARE),
        wait_days=min(int(entry.get("wait_days") or CARE_WAIT_DAYS), 5) or CARE_WAIT_DAYS,
        check_question=_check_question(viral),
        reason=reason,
        source_url=entry.get("source_url"),
    )


def next_step(
    predicted_class: str | None,
    level: str | None,
    previous: Sequence[Dict[str, Any]],
) -> StepPlan:
    """
    Step after the latest one did not work.

    `previous` are the earlier steps (dicts with "kind" and "products"),
    oldest first.
    """

    tried = [s for s in previous if s["kind"] != KIND_EXPERT]

    if previous and previous[-1]["kind"] == KIND_EXPERT:
        return expert_step(
            "The problem continues after expert advice. Contact the expert again, "
            "or ask the Kisan Call Centre for a field visit."
        )

    if len(tried) >= MAX_STEPS or predicted_class is None:
        return expert_step(
            "The suggested steps have not solved the problem. An expert can examine "
            "the plant in person and find the cause."
        )

    if is_viral(predicted_class):
        return expert_step(
            "Virus diseases cannot be cured, and the problem is still spreading. "
            "An expert can advise how to protect the rest of the crop."
        )

    entry = plan_entry(predicted_class)
    groups = _groups(entry.get("products", []))
    used = {p["group"] for s in tried for p in s.get("products", [])}
    remaining = [g for g in groups if g not in used]

    if level == LEVEL_HIGH and remaining:
        return _treatment_step(predicted_class, entry, remaining[0], switched=bool(used))

    if groups and level != LEVEL_HIGH:
        return expert_step(
            "The care steps did not help and the diagnosis is still uncertain. Get "
            "it confirmed by an expert before using any chemical."
        )

    return expert_step(
        "The suggested steps have not solved the problem. An expert can examine "
        "the plant in person and find the cause."
    )


def plan_meta() -> Dict[str, Any]:
    """Source information shown next to product suggestions."""

    meta = load_plans()["_meta"]
    ai_meta = load_ai_conditions()["_meta"]

    return {
        "source": meta["source"],
        "checked_on": meta["checked_on"],
        "expert_reviewed": meta["expert_reviewed"] and ai_meta["expert_reviewed"],
        "excluded_from_tnau": meta["excluded_from_tnau"] + ai_meta["excluded_from_tnau"],
    }
