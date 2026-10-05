"""
backend/services/languages.py

Languages the LLM guidance can be written in. Codes are what the
frontend sends; the BCP-47 tag is used for browser text-to-speech.
"""

from __future__ import annotations

from typing import Dict, Tuple

DEFAULT_LANGUAGE = "en"

# code -> (English name, native name, BCP-47 speech tag)
LANGUAGES: Dict[str, Tuple[str, str, str]] = {
    "en": ("English", "English", "en-IN"),
    "hi": ("Hindi", "हिन्दी", "hi-IN"),
    "ta": ("Tamil", "தமிழ்", "ta-IN"),
    "te": ("Telugu", "తెలుగు", "te-IN"),
    "kn": ("Kannada", "ಕನ್ನಡ", "kn-IN"),
    "ml": ("Malayalam", "മലയാളം", "ml-IN"),
    "mr": ("Marathi", "मराठी", "mr-IN"),
    "bn": ("Bengali", "বাংলা", "bn-IN"),
    "gu": ("Gujarati", "ગુજરાતી", "gu-IN"),
    "pa": ("Punjabi", "ਪੰਜਾਬੀ", "pa-IN"),
    "es": ("Spanish", "Español", "es-ES"),
    "fr": ("French", "Français", "fr-FR"),
}


def normalise_language(code: object) -> str | None:
    """Return a supported language code, or None if unsupported."""

    if code is None or code == "":
        return DEFAULT_LANGUAGE

    if isinstance(code, str) and code.strip().lower() in LANGUAGES:
        return code.strip().lower()

    return None


def language_instruction(code: str) -> str:
    """Prompt line telling the LLM which language to write in."""

    name = LANGUAGES[code][0]

    if code == DEFAULT_LANGUAGE:
        return "Write all text values in English."

    return (
        f"Write all text values in {name}, using simple words a farmer would "
        "use. Keep the JSON keys in English. Scientific names may stay in "
        "Latin. The no-dose / no-brand rules apply in every language."
    )
