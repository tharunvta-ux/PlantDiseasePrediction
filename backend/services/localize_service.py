"""
backend/services/localize_service.py

Puts tracking data (cases, steps, plan previews, AI-matched diseases)
into the grower's language, and builds the text that is read aloud.

Translated: titles, reasons, actions, questions, feedback, outlook and
trend texts, product *kinds*. Never translated: product names and doses
(kept exactly as TNAU lists them) and the grower's own notes.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Iterable, List

from backend.services.languages import DEFAULT_LANGUAGE
from backend.services.translation_service import translate_texts

STEP_TEXT_FIELDS = ("title", "reason", "check_question", "feedback")


def _step_texts(step: Dict[str, Any]) -> Iterable[str]:
    for field in STEP_TEXT_FIELDS:
        yield step.get(field)
    yield from step.get("actions") or []
    for product in step.get("products") or []:
        yield product.get("kind")
    yield from _severity_texts(step.get("severity"))


def _severity_texts(severity: Dict[str, Any] | None) -> Iterable[str]:
    if not severity:
        return
    yield (severity.get("outlook") or {}).get("text")
    yield (severity.get("trend") or {}).get("text")


def _apply_step(step: Dict[str, Any], tr: Callable[[str], str], severity: bool = True) -> None:
    for field in STEP_TEXT_FIELDS:
        if step.get(field):
            step[field] = tr(step[field])
    step["actions"] = [tr(a) for a in step.get("actions") or []]
    for product in step.get("products") or []:
        if product.get("kind"):
            product["kind"] = tr(product["kind"])  # name and dose untouched
    if severity:
        _apply_severity(step.get("severity"), tr)


def _apply_severity(severity: Dict[str, Any] | None, tr: Callable[[str], str]) -> None:
    if not severity:
        return
    for key in ("outlook", "trend"):
        part = severity.get(key)
        if part and part.get("text"):
            part["text"] = tr(part["text"])


def _translator(texts: Iterable[str], language: str) -> Callable[[str], str]:
    mapping = translate_texts([t for t in texts if t], language)
    return lambda text: mapping.get(text, text)


def localize_cases(cases: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
    """Translated copies of serialised cases (one LLM batch for all)."""

    if language == DEFAULT_LANGUAGE or not cases:
        return cases

    cases = copy.deepcopy(cases)
    texts: List[str] = []

    for case in cases:
        texts += [case.get("display_name"), case.get("plant_name")]
        texts += list(_severity_texts(case.get("severity")))
        texts += list(_severity_texts(case.get("latest_severity")))
        for step in case["steps"]:
            texts += list(_step_texts(step))

    tr = _translator(texts, language)

    for case in cases:
        case["display_name"] = tr(case["display_name"])
        case["plant_name"] = tr(case["plant_name"])
        # Several dicts can be the same object (current_step is the last
        # step, latest_severity a step's severity): translate each once.
        seen: set[int] = set()
        for severity in (case.get("severity"), case.get("latest_severity")):
            if severity is not None and id(severity) not in seen:
                seen.add(id(severity))
                _apply_severity(severity, tr)
        for step in case["steps"]:
            shared = step.get("severity") is not None and id(step["severity"]) in seen
            _apply_step(step, tr, severity=not shared)
        case["current_step"] = case["steps"][-1] if case["steps"] else None

    return cases


def localize_preview(preview: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Translated copy of a /treatment-plan preview."""

    if language == DEFAULT_LANGUAGE:
        return preview

    preview = copy.deepcopy(preview)
    step = preview.get("step") or {}
    tr = _translator(list(_step_texts(step)) + list(_severity_texts(preview.get("severity"))), language)

    if step:
        _apply_step(step, tr)
    _apply_severity(preview.get("severity"), tr)

    return preview


def localize_ai_condition(condition: Dict[str, Any] | None, language: str) -> Dict[str, Any] | None:
    """Translated symptom description and yes/no checks of an AI match."""

    if not condition or language == DEFAULT_LANGUAGE:
        return condition

    condition = dict(condition)
    tr = _translator(
        [condition["display_name"], condition["crop"], condition["disease"], condition["symptoms"], *condition["checks"]],
        language,
    )

    for key in ("display_name", "crop", "disease", "symptoms"):
        condition[key] = tr(condition[key])
    condition["checks"] = [tr(q) for q in condition["checks"]]

    return condition


# ---------- read aloud ----------


def _product_sentence(products: List[Dict[str, str]]) -> str | None:
    if not products:
        return None
    options = "; or ".join(f"{p['name']}, {p['dose']}" for p in products)
    return f"Use one of these: {options}. Follow the product label."


def step_speech_sentences(step: Dict[str, Any]) -> List[str]:
    """English sentences describing one step (translated before speaking)."""

    sentences = [step.get("title"), step.get("reason"), *(step.get("actions") or [])]
    sentences.append(_product_sentence(step.get("products") or []))

    if step.get("wait_days"):
        days = step["wait_days"]
        sentences.append(f"Then wait {days} {'day' if days == 1 else 'days'} and check: {step.get('check_question')}")

    return [s for s in sentences if s]


def case_speech_sentences(case: Dict[str, Any]) -> List[str]:
    """What the read-aloud button says for a tracked case."""

    sentences = [case.get("display_name")]
    severity = case.get("latest_severity") or case.get("severity") or {}

    if case.get("status") == "recovered":
        sentences.append(case["steps"][-1].get("feedback"))
        return [s for s in sentences if s]

    sentences.append((severity.get("outlook") or {}).get("text"))

    if len(case["steps"]) > 1:
        sentences.append(case["steps"][-2].get("feedback"))

    sentences += step_speech_sentences(case["current_step"])

    return [s for s in sentences if s]


def plan_speech_sentences(preview: Dict[str, Any]) -> List[str]:
    """What the read-aloud button says on "What to do now"."""

    sentences = step_speech_sentences(preview["step"]) if preview.get("step") else []
    outlook = ((preview.get("severity") or {}).get("outlook") or {}).get("text")

    return [s for s in [*sentences, outlook] if s]


def spoken_text(sentences: List[str], language: str) -> str:
    """One text in the grower's language, ready for speech synthesis."""

    tr = _translator(sentences, language)
    parts = [tr(s).strip().rstrip(".।") for s in sentences]

    return ". ".join(p for p in parts if p) + "."
