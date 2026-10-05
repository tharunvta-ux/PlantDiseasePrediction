"""
backend/config.py

Central configuration for the Plant Disease Detection backend.

All secrets and deployment-specific values are read from environment
variables (optionally from a local, git-ignored `.env` file). Nothing in
this module is ever sent to the frontend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load `.env` once at import time. Existing environment variables win.
load_dotenv(PROJECT_ROOT / ".env", override=False)

# -----------------------------------------------------
# Model
# -----------------------------------------------------

DEFAULT_MODEL_PATH = PROJECT_ROOT / "saved_models" / "plant_disease_cnn.keras"

# Per-model metadata (class names, calibration) lives in version control,
# next to the code, because saved_models/ is git-ignored.
MODEL_ARTIFACTS_ROOT = PROJECT_ROOT / "backend" / "model_artifacts"

PLANTVILLAGE_DIR = PROJECT_ROOT / "dataset" / "raw" / "PlantVillage"

# -----------------------------------------------------
# Uploads
# -----------------------------------------------------

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS = frozenset({"jpg", "jpeg", "png"})

# -----------------------------------------------------
# LLM defaults
# -----------------------------------------------------

# Verified against https://ai.google.dev/gemini-api/docs/models and
# /pricing on 2026-09-30: stable model with a free API tier.
DEFAULT_LLM_MODEL = "gemini-3.8-flash"

# Used once when the primary model is overloaded (503) or rate-limited
# (429). Also free tier; set LLM_FALLBACK_MODEL= (empty) to disable.
DEFAULT_LLM_FALLBACK_MODEL = "gemini-3.5-flash-lite"

# Text-to-speech (read-aloud in languages devices often lack voices for).
# Both have a free tier and support all app languages (verified 2026-10-06).
DEFAULT_TTS_MODEL = "gemini-3.8-flash-lite-tts"
DEFAULT_TTS_FALLBACK_MODEL = "gemini-3.8-flash-tts"


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back on bad input."""

    raw = os.getenv(name)

    if raw is None or raw.strip() == "":
        return default

    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from the environment."""

    model_path: Path
    llm_provider: str
    llm_api_key: str | None
    llm_model: str
    llm_fallback_model: str | None
    tts_model: str | None
    tts_fallback_model: str | None
    llm_timeout_seconds: float
    llm_temperature: float

    @property
    def model_artifacts_dir(self) -> Path:
        """Directory holding class_names.json / calibration.json."""

        return MODEL_ARTIFACTS_ROOT / self.model_path.stem

    @property
    def model_version(self) -> str:
        """Identifier of the deployed model (its file stem)."""

        return self.model_path.stem

    @property
    def llm_configured(self) -> bool:
        """True when an LLM API key is available."""

        return bool(self.llm_api_key)


def get_settings() -> Settings:
    """
    Resolve settings from the current environment.

    Read on every call (cheap) so tests can override variables.
    """

    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    api_key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")

    return Settings(
        model_path=model_path,
        llm_provider=os.getenv("LLM_PROVIDER", "gemini").strip().lower(),
        llm_api_key=api_key.strip() if api_key else None,
        llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL).strip(),
        llm_fallback_model=(
            os.getenv("LLM_FALLBACK_MODEL", DEFAULT_LLM_FALLBACK_MODEL).strip() or None
        ),
        tts_model=os.getenv("LLM_TTS_MODEL", DEFAULT_TTS_MODEL).strip() or None,
        tts_fallback_model=(
            os.getenv("LLM_TTS_FALLBACK_MODEL", DEFAULT_TTS_FALLBACK_MODEL).strip() or None
        ),
        llm_timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 30.0),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.2),
    )
