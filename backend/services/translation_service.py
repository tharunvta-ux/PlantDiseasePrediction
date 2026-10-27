"""
backend/services/translation_service.py

Translates app-generated text (treatment steps, outlook, feedback, UI
labels) into the grower's language with the LLM.

Safety:
- Only text the server produced is translated - never arbitrary client
  text - so the API key cannot be used as a free translation service.
- Numbers, units, phone numbers and {placeholders} must survive: if a
  translation loses or changes any of them, the English original is
  kept. A translation can therefore never change a dose.
- Product names and doses are not sent at all (callers skip them).

Translations are cached in the database (and in memory), so each text
costs one LLM call per language, ever.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Dict, Iterable, List, Sequence

from flask import has_app_context
from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.services.languages import DEFAULT_LANGUAGE, LANGUAGES
from backend.services.llm import LLMError, LLMProvider, get_llm_provider
from backend.services.recommendation_service import LRUCache

logger = logging.getLogger(__name__)

BATCH_SIZE = 40
MAX_TEXT_CHARS = 2000
MEMORY_CACHE_SIZE = 4096

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")


class TranslationBatch(BaseModel):
    translations: List[str]


RESPONSE_SCHEMA = TranslationBatch.model_json_schema()

SYSTEM_PROMPT = """\
You translate short texts from a plant-health app for farmers.

Rules:
- Use simple, everyday words a farmer would use.
- Keep every number, percentage, unit and phone number exactly as \
written, using the same digits (0-9).
- Keep placeholders in curly braces, e.g. {days}, exactly as written.
- Keep chemical and product names (e.g. Mancozeb, Bordeaux mixture) and \
abbreviations (TNAU, KVK, FRAC) as they are, in Latin letters.
- Keep emoji and symbols.
- Return exactly one translation per input text, in the same order.
Respond only with JSON matching the schema.
"""

_memory = LRUCache(MEMORY_CACHE_SIZE)


def clear_cache() -> None:
    """Empty the in-memory cache (used by tests)."""

    _memory.clear()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_safe_translation(source: str, translated: str) -> bool:
    """All numbers and placeholders of the source survive translation."""

    if not translated or not translated.strip():
        return False

    numbers = sorted(_NUMBER.findall(source))
    kept = _NUMBER.findall(translated)

    if any(kept.count(n) < numbers.count(n) for n in set(numbers)):
        return False

    return sorted(_PLACEHOLDER.findall(source)) == sorted(_PLACEHOLDER.findall(translated))


def _from_db(language: str, hashes: Sequence[str]) -> Dict[str, str]:
    if not hashes or not has_app_context():
        return {}

    from backend.database.db import db_session
    from backend.database.models import Translation

    rows = db_session().execute(
        select(Translation.source_hash, Translation.text).where(
            Translation.language == language, Translation.source_hash.in_(list(hashes))
        )
    )

    return {source_hash: text for source_hash, text in rows}


def _to_db(language: str, items: Dict[str, str]) -> None:
    if not items or not has_app_context():
        return

    from backend.database.db import db_session
    from backend.database.models import Translation

    session = db_session()

    for source, text in items.items():
        session.add(Translation(language=language, source_hash=_hash(source), source_text=source, text=text))

    try:
        session.commit()
    except IntegrityError:  # another request stored the same text first
        session.rollback()


def _translate_batch(texts: List[str], language: str, provider: LLMProvider) -> Dict[str, str]:
    name = LANGUAGES[language][0]
    response = provider.generate_json(
        SYSTEM_PROMPT,
        f"Translate each text into {name}.\n" + json.dumps({"texts": texts}, ensure_ascii=False),
        RESPONSE_SCHEMA,
    )

    try:
        batch = TranslationBatch.model_validate_json(response.text)
    except ValidationError as exc:
        logger.warning("Translation failed validation: %s", exc)
        return {}

    if len(batch.translations) != len(texts):
        logger.warning("Translation returned %d items for %d texts", len(batch.translations), len(texts))
        return {}

    result = {}
    for source, translated in zip(texts, batch.translations):
        if is_safe_translation(source, translated):
            result[source] = translated.strip()
        else:
            logger.warning("Dropped a translation that changed numbers or placeholders")

    return result


def translate_texts(
    texts: Iterable[str],
    language: str,
    provider: LLMProvider | None = None,
) -> Dict[str, str]:
    """
    Map each text to its translation (the original when not available).

    Never raises: if the LLM fails, the English text is used.
    """

    unique = [t for t in dict.fromkeys(texts) if isinstance(t, str) and t.strip()]

    if language == DEFAULT_LANGUAGE or language not in LANGUAGES or not unique:
        return {t: t for t in unique}

    result: Dict[str, str] = {}
    missing: List[str] = []

    for text in unique:
        cached = _memory.get((language, text))
        if cached is not None:
            result[text] = cached["text"]
        else:
            missing.append(text)

    if missing:
        stored = _from_db(language, [_hash(t) for t in missing])
        still_missing = []
        for text in missing:
            if _hash(text) in stored:
                result[text] = stored[_hash(text)]
                _memory.set((language, text), {"text": result[text]})
            else:
                still_missing.append(text)
        missing = still_missing

    translatable = [t for t in missing if len(t) <= MAX_TEXT_CHARS]

    if translatable:
        fresh: Dict[str, str] = {}
        try:
            provider = provider or get_llm_provider()
            for start in range(0, len(translatable), BATCH_SIZE):
                fresh.update(_translate_batch(translatable[start:start + BATCH_SIZE], language, provider))
        except LLMError as exc:
            logger.warning("Translation unavailable [%s]: %s", exc.code, exc.detail)

        for text, translated in fresh.items():
            result[text] = translated
            _memory.set((language, text), {"text": translated})

        _to_db(language, fresh)

    return {t: result.get(t, t) for t in unique}
