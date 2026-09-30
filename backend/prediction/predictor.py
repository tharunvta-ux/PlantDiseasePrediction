"""
backend/prediction/predictor.py

Plant Disease Prediction using trained CNN model.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageOps

from backend.config import (
    MODEL_ARTIFACTS_ROOT,
    PLANTVILLAGE_DIR,
    get_settings,
)
from backend.prediction.model_backends import load_backend

# -----------------------------------------------------
# Logging
# -----------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------
# Paths
# -----------------------------------------------------

MODEL_PATH = get_settings().model_path

TRAIN_DIR = PLANTVILLAGE_DIR / "train"

IMAGE_SIZE = (256, 256)

CLASS_NAMES_FILE = "class_names.json"


def load_class_names(artifacts_dir: Path) -> List[str]:
    """
    Load the model's class names in output-index order.

    Prefers the versioned `class_names.json` so the API does not depend
    on the training dataset being present. Falls back to the sorted
    training folder names (the order Keras used during training).
    """

    class_file = artifacts_dir / CLASS_NAMES_FILE

    if class_file.exists():
        return json.loads(class_file.read_text(encoding="utf-8"))

    if TRAIN_DIR.exists():
        logger.warning(
            "%s not found; reading class names from %s",
            class_file,
            TRAIN_DIR,
        )
        return sorted(
            folder.name
            for folder in TRAIN_DIR.iterdir()
            if folder.is_dir()
        )

    raise FileNotFoundError(
        f"Class names not found:\n{class_file}"
    )


class PlantDiseasePredictor:
    """
    Predict plant disease from a single leaf image.
    """

    def __init__(self, model_path: Path | None = None) -> None:

        settings = get_settings()
        model_path = Path(model_path or settings.model_path)

        logger.info("Loading model...")

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found:\n{model_path}"
            )

        # .keras -> TensorFlow, .tflite -> lightweight LiteRT runtime.
        self.backend = load_backend(model_path, IMAGE_SIZE)
        self.model_version = model_path.stem

        # Input size comes from the model so other architectures
        # (e.g. 224x224 transfer-learning models) need no code change.
        self.image_size = self.backend.input_size

        logger.info("Model loaded successfully.")

        self.artifacts_dir = MODEL_ARTIFACTS_ROOT / model_path.stem

        self.class_names = load_class_names(self.artifacts_dir)

        num_outputs = self.backend.num_classes

        if num_outputs != len(self.class_names):
            raise ValueError(
                f"Model has {num_outputs} outputs but "
                f"{len(self.class_names)} class names were loaded."
            )

        logger.info(
            "Loaded %d classes.",
            len(self.class_names),
        )

    def preprocess_image(
        self,
        image_path: str,
    ) -> np.ndarray:
        """
        Load, orient, resize and normalize an image.

        - EXIF orientation is applied so phone photos are upright.
        - Bilinear resize matches the training pipeline
          (image_dataset_from_directory); PIL antialiases when
          downscaling large photos.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image = image.resize(self.image_size, Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32)

        array /= 255.0

        return np.expand_dims(array, axis=0)

    def predict_probabilities(
        self,
        image_path: str,
    ) -> np.ndarray:
        """
        Return the raw softmax vector for one image.
        """

        image = self.preprocess_image(image_path)

        return self.predict_batch(image)[0]

    def predict_batch(
        self,
        images: np.ndarray,
    ) -> np.ndarray:
        """
        Softmax outputs for a preprocessed [N, H, W, 3] batch.
        """

        return self.backend.predict_batch(images)

    def predict(
        self,
        image_path: str,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predict disease.
        """

        predictions = self.predict_probabilities(image_path)

        predicted_index = int(np.argmax(predictions))

        predicted_class = self.class_names[predicted_index]

        confidence = float(
            predictions[predicted_index] * 100
        )

        top3_indices = np.argsort(predictions)[-3:][::-1]

        top3_predictions = []

        for index in top3_indices:
            top3_predictions.append(
                (
                    self.class_names[index],
                    float(predictions[index] * 100),
                )
            )

        return (
            predicted_class,
            confidence,
            top3_predictions,
        )


# -----------------------------------------------------
# Shared Predictor Instance
# -----------------------------------------------------

_predictor: PlantDiseasePredictor | None = None
_predictor_lock = threading.Lock()


def get_predictor() -> PlantDiseasePredictor:
    """
    Return the process-wide predictor, loading the model on first use.
    """

    global _predictor

    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                _predictor = PlantDiseasePredictor()

    return _predictor


def main() -> None:

    print("\n" + "=" * 60)
    print("Plant Disease Prediction")
    print("=" * 60)

    predictor = get_predictor()

    image_path = input(
        "\nEnter image path:\n> "
    ).strip()

    disease, confidence, top3 = predictor.predict(
        image_path
    )

    print("\n" + "=" * 60)
    print("Prediction Result")
    print("=" * 60)

    print(f"\nPredicted Disease : {disease}")
    print(f"Confidence         : {confidence:.2f}%")

    print("\nTop 3 Predictions")
    print("-" * 60)

    for i, (name, score) in enumerate(
        top3,
        start=1,
    ):
        print(f"{i}. {name:<40} {score:.2f}%")

    print("=" * 60)


# -----------------------------------------------------
# Reusable Prediction Function
# -----------------------------------------------------


def predict_image(image_path: str) -> dict:
    """
    Predict plant disease for an image and return
    the result as a dictionary for the Flask API.
    """

    predicted_class, confidence, top3 = get_predictor().predict(image_path)

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "top3_predictions": [
            {
                "class": name,
                "confidence": round(score, 2),
            }
            for name, score in top3
        ],
    }


if __name__ == "__main__":
    main()
