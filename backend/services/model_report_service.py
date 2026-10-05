"""
backend/services/model_report_service.py

Serves the evaluation results of the deployed model (produced by
backend/evaluation/calibrate.py) for the model performance page.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from backend.config import PROJECT_ROOT, get_settings
from backend.prediction.class_info import parse_class_name

logger = logging.getLogger(__name__)

CALIBRATION_DIR = PROJECT_ROOT / "results" / "calibration"
REPORT_FILE = "calibration_report.json"

ORIGINAL_MODEL = "plant_disease_cnn"


def figure_files(model_version: str) -> Dict[str, Path]:
    """Whitelisted figure files for a model."""

    return {
        "reliability_diagram": CALIBRATION_DIR / model_version / "reliability_diagram.png",
        "accuracy_vs_coverage": CALIBRATION_DIR / model_version / "accuracy_vs_coverage.png",
        "training_accuracy": PROJECT_ROOT / "results" / "training_accuracy.png",
        "training_loss": PROJECT_ROOT / "results" / "training_loss.png",
    }


class ReportNotFoundError(FileNotFoundError):
    """No evaluation report exists for the deployed model."""


def _load(model_version: str) -> Dict[str, Any] | None:
    path = CALIBRATION_DIR / model_version / REPORT_FILE

    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def _summary(report: Dict[str, Any]) -> Dict[str, Any]:
    metrics = report["test_metrics"]
    size = report.get("model_size_bytes")

    return {
        "model_version": report["model_version"],
        "test_images": report["data"]["test_images"],
        "accuracy": metrics["accuracy"],
        "macro_class_accuracy": metrics["macro_class_accuracy"],
        "ece": metrics["ece_calibrated"],
        "levels": metrics["levels"],
        "thresholds": {k: report["thresholds"][k] for k in ("high", "low")},
        "model_size_mb": round(size / 2**20, 1) if size else None,
    }


@lru_cache(maxsize=4)
def _build(model_version: str) -> Dict[str, Any]:

    report = _load(model_version)

    if report is None:
        raise ReportNotFoundError(f"No evaluation report for {model_version}")

    metrics = report["test_metrics"]
    names = metrics["class_names"]
    counts = metrics.get("class_counts", {})

    per_class = [
        {
            "class": name,
            "display_name": parse_class_name(name).display_name,
            "accuracy": metrics["per_class_accuracy"].get(name),
            "test_images": (counts.get("test") or {}).get(name),
            "train_images": (counts.get("train") or {}).get(name),
        }
        for name in names
    ]

    comparison = None

    if model_version != ORIGINAL_MODEL:
        original = _load(ORIGINAL_MODEL)
        if original:
            comparison = _summary(original)

    return {
        "summary": _summary(report),
        "created": report["created"],
        "data": report["data"],
        "provisional": report["provisional"],
        "provisional_reason": report["provisional_reason"],
        "temperature": report["temperature"],
        "ece_raw": metrics["ece_raw"],
        "quality_flags": metrics.get("quality_flags"),
        "per_class": per_class,
        "class_display_names": [parse_class_name(n).display_name for n in names],
        "confusion_matrix": metrics["confusion_matrix"],
        "top_confusions": [
            {
                **item,
                "true_display": parse_class_name(item["true"]).display_name,
                "predicted_display": parse_class_name(item["predicted"]).display_name,
            }
            for item in metrics["top_confusions"]
        ],
        "comparison_original": comparison,
        "figures": sorted(
            name for name, path in figure_files(model_version).items() if path.exists()
        ),
    }


def get_model_report() -> Dict[str, Any]:
    """Evaluation report for the currently configured model."""

    return _build(get_settings().model_version)


def get_figure_path(name: str) -> Path | None:
    """Path of a whitelisted figure, or None."""

    path = figure_files(get_settings().model_version).get(name)

    return path if path and path.exists() else None
