"""
backend/training/export_tflite.py

Export a trained Keras model to a compact TensorFlow Lite file for
low-memory deployment (e.g. Render free plan, 512 MB RAM).

The source .keras model is only read, never modified. The exported
model gets its own artifacts folder (class names now; calibration must
be re-measured with collect_predictions + calibrate before serving).

Variants:
    int8     dynamic-range quantization (weights int8, ~4x smaller)
    float16  weights stored as float16 (~2x smaller)
    float32  no quantization

Usage:
    python -m backend.training.export_tflite --variant int8
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import tensorflow as tf

from backend.config import DEFAULT_MODEL_PATH, MODEL_ARTIFACTS_ROOT
from backend.prediction.predictor import CLASS_NAMES_FILE

logger = logging.getLogger(__name__)

VARIANTS = ("int8", "float16", "float32")


def export(model_path: Path, variant: str) -> Path:
    """Convert model_path to <stem>_<variant>.tflite next to it."""

    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if variant in ("int8", "float16"):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if variant == "float16":
        converter.target_spec.supported_types = [tf.float16]

    target = model_path.with_name(f"{model_path.stem}_{variant}.tflite")
    target.write_bytes(converter.convert())

    source_artifacts = MODEL_ARTIFACTS_ROOT / model_path.stem
    target_artifacts = MODEL_ARTIFACTS_ROOT / target.stem
    target_artifacts.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_artifacts / CLASS_NAMES_FILE, target_artifacts / CLASS_NAMES_FILE)

    logger.info(
        "Exported %s (%.1f MB) | class names -> %s",
        target,
        target.stat().st_size / 2**20,
        target_artifacts,
    )
    logger.info(
        "Next: MODEL_PATH=%s python -m backend.evaluation.collect_predictions "
        "--output results/calibration/%s.npz, then calibrate.",
        target,
        target.stem,
    )

    return target


def main() -> None:

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--variant", choices=VARIANTS, default="int8")
    args = parser.parse_args()

    export(args.model, args.variant)


if __name__ == "__main__":
    main()
