"""
backend/prediction/predictor.py

Plant Disease Prediction using trained CNN model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "saved_models" / "plant_disease_cnn.keras"

TRAIN_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "PlantVillage"
    / "train"
)

IMAGE_SIZE = (256, 256)


class PlantDiseasePredictor:
    """
    Predict plant disease from a single leaf image.
    """

    def __init__(self) -> None:

        logger.info("Loading model...")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found:\n{MODEL_PATH}"
            )

        self.model = tf.keras.models.load_model(MODEL_PATH)

        logger.info("Model loaded successfully.")

        # Read class names directly from folder names
        self.class_names = sorted(
            [
                folder.name
                for folder in TRAIN_DIR.iterdir()
                if folder.is_dir()
            ]
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
        Resize and normalize image.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        image = tf.keras.utils.load_img(
            image_path,
            target_size=IMAGE_SIZE,
        )

        image = tf.keras.utils.img_to_array(image)

        image = image.astype(np.float32)

        image /= 255.0

        image = np.expand_dims(image, axis=0)

        return image

    def predict(
        self,
        image_path: str,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predict disease.
        """

        image = self.preprocess_image(image_path)

        predictions = self.model.predict(
            image,
            verbose=0,
        )[0]

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


def main() -> None:

    print("\n" + "=" * 60)
    print("Plant Disease Prediction")
    print("=" * 60)

    predictor = PlantDiseasePredictor()

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


if __name__ == "__main__":
    main()