"""
backend/evaluation/collect_predictions.py

Run the deployed model over an image folder (default: the PlantVillage
validation split) using the exact inference preprocessing, and cache the
softmax outputs plus image-quality metrics for calibration/evaluation.

Also exports the model's class names to
backend/model_artifacts/<model>/class_names.json.

Usage:
    python -m backend.evaluation.collect_predictions [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from backend.config import PLANTVILLAGE_DIR, PROJECT_ROOT
from backend.prediction.image_quality import compute_quality_metrics
from backend.prediction.predictor import (
    CLASS_NAMES_FILE,
    PlantDiseasePredictor,
)

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "calibration" / "val_predictions.npz"

BATCH_SIZE = 64


def list_images(
    data_dir: Path,
    class_names: List[str],
) -> Tuple[List[Path], np.ndarray]:
    """
    List images in <data_dir>/<class_name>/ with integer labels.

    Folders whose name is not a model class are rejected so labels can
    never silently shift.
    """

    unknown = [
        folder.name
        for folder in data_dir.iterdir()
        if folder.is_dir() and folder.name not in class_names
    ]

    if unknown:
        raise ValueError(f"Folders are not model classes: {unknown}")

    paths: List[Path] = []
    labels: List[int] = []

    for index, name in enumerate(class_names):
        folder = data_dir / name

        if not folder.exists():
            continue

        for file in sorted(folder.iterdir()):
            if file.suffix.lower() in VALID_EXTENSIONS:
                paths.append(file)
                labels.append(index)

    return paths, np.asarray(labels, dtype=np.int32)


def export_class_names(predictor: PlantDiseasePredictor) -> Path:
    """Write the predictor's class list next to the model metadata."""

    predictor.artifacts_dir.mkdir(parents=True, exist_ok=True)

    target = predictor.artifacts_dir / CLASS_NAMES_FILE

    target.write_text(
        json.dumps(predictor.class_names, indent=2),
        encoding="utf-8",
    )

    return target


def collect(data_dir: Path, output: Path) -> None:
    """Predict every image in data_dir and save the results."""

    predictor = PlantDiseasePredictor()

    class_file = export_class_names(predictor)
    logger.info("Class names written to %s", class_file)

    paths, labels = list_images(data_dir, predictor.class_names)
    logger.info("Found %d images in %s", len(paths), data_dir)

    probabilities = np.zeros(
        (len(paths), len(predictor.class_names)),
        dtype=np.float32,
    )
    quality = np.zeros((len(paths), 5), dtype=np.float64)

    start = time.time()

    for begin in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[begin:begin + BATCH_SIZE]

        batch = np.concatenate(
            [predictor.preprocess_image(str(p)) for p in batch_paths]
        )

        probabilities[begin:begin + len(batch_paths)] = (
            predictor.model.predict(batch, verbose=0)
        )

        for offset, path in enumerate(batch_paths):
            m = compute_quality_metrics(path)
            quality[begin + offset] = (
                m.width,
                m.height,
                m.blur_score,
                m.brightness,
                m.contrast,
            )

        done = begin + len(batch_paths)

        if (begin // BATCH_SIZE) % 20 == 0:
            logger.info(
                "%d/%d images (%.0fs)",
                done,
                len(paths),
                time.time() - start,
            )

    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        probabilities=probabilities,
        labels=labels,
        paths=np.asarray(
            [str(p.relative_to(PROJECT_ROOT)) for p in paths]
        ),
        quality=quality,
        quality_columns=np.asarray(
            ["width", "height", "blur_score", "brightness", "contrast"]
        ),
        class_names=np.asarray(predictor.class_names),
        model_version=np.asarray(predictor.model_version),
    )

    accuracy = float((probabilities.argmax(axis=1) == labels).mean())

    logger.info(
        "Saved %s | top-1 accuracy %.4f | %.0fs",
        output,
        accuracy,
        time.time() - start,
    )


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PLANTVILLAGE_DIR / "val",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    collect(args.data_dir, args.output)


if __name__ == "__main__":
    main()
