"""
backend/services/data_loader.py

Production-ready TensorFlow Data Loader

Responsibilities:
- Load training and validation datasets
- Resize images
- Normalize pixel values
- Batch images
- Shuffle only training dataset
- Cache datasets
- Prefetch datasets
- Print dataset information
"""

from pathlib import Path
from typing import Tuple

import tensorflow as tf

# ==========================================================
# Configuration
# ==========================================================

IMAGE_SIZE: tuple[int, int] = (256, 256)
BATCH_SIZE: int = 32
SEED: int = 42
NORMALIZATION_DIVISOR: float = 255.0

# ==========================================================
# Dataset Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = PROJECT_ROOT / "dataset" / "raw" / "PlantVillage" / "train"
VAL_DIR = PROJECT_ROOT / "dataset" / "raw" / "PlantVillage" / "val"


class DataLoader:
    """
    Loads and prepares TensorFlow datasets.

    Features
    --------
    - Automatic label inference
    - Image resizing
    - Image normalization
    - Batching
    - Training shuffle
    - Cache
    - Prefetch
    """

    def __init__(self) -> None:
        """Initialize the DataLoader."""
        self.class_names: list[str] = []

        self._validate_dataset_paths()

    # ======================================================
    # Private Methods
    # ======================================================

    def _validate_dataset_paths(self) -> None:
        """
        Ensure dataset directories exist.
        """

        if not TRAIN_DIR.exists():
            raise FileNotFoundError(
                f"Training directory not found:\n{TRAIN_DIR}"
            )

        if not VAL_DIR.exists():
            raise FileNotFoundError(
                f"Validation directory not found:\n{VAL_DIR}"
            )

    @staticmethod
    def _normalize_images(
        images: tf.Tensor,
        labels: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Normalize pixel values from [0,255] to [0,1].
        """

        images = tf.cast(images, tf.float32)
        images = images / NORMALIZATION_DIVISOR

        return images, labels

    def _load_directory(
        self,
        directory: Path,
        shuffle: bool,
    ) -> tf.data.Dataset:
        """
        Load one image directory.

        Parameters
        ----------
        directory : Path
            Path to dataset folder.

        shuffle : bool
            Whether dataset should be shuffled.

        Returns
        -------
        tf.data.Dataset
        """

        dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="int",
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            seed=SEED,
        )

        if not self.class_names:
            self.class_names = dataset.class_names

        dataset = dataset.map(
            self._normalize_images,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        dataset = dataset.cache()

        dataset = dataset.prefetch(
            buffer_size=tf.data.AUTOTUNE
        )

        return dataset

    # ======================================================
    # Public Methods
    # ======================================================

    def load_datasets(
        self,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Load both training and validation datasets.

        Returns
        -------
        tuple
            (train_dataset, validation_dataset)
        """

        train_dataset = self._load_directory(
            TRAIN_DIR,
            shuffle=True,
        )

        validation_dataset = self._load_directory(
            VAL_DIR,
            shuffle=False,
        )

        return train_dataset, validation_dataset

    def print_dataset_info(
        self,
        dataset: tf.data.Dataset,
        dataset_name: str,
    ) -> None:
        """
        Print useful dataset information.
        """

        print("\n" + "=" * 60)
        print(dataset_name)
        print("=" * 60)

        image_batch, label_batch = next(iter(dataset))

        print(f"Batch Shape      : {image_batch.shape}")
        print(f"Label Shape      : {label_batch.shape}")
        print(f"Image Data Type  : {image_batch.dtype}")
        print(f"Label Data Type  : {label_batch.dtype}")
        print(f"Image Shape      : {image_batch.shape[1:]}")
        print(f"Batch Size       : {image_batch.shape[0]}")
        print(f"Total Classes    : {len(self.class_names)}")

        print("\nClass Names")
        print("-" * 60)

        for index, class_name in enumerate(self.class_names):
            print(f"{index:02d} -> {class_name}")

        print()

# ==========================================================
# Main Function
# ==========================================================


def main() -> None:
    """
    Test the DataLoader.
    """

    loader = DataLoader()

    train_dataset, validation_dataset = loader.load_datasets()

    loader.print_dataset_info(
        train_dataset,
        "Training Dataset",
    )

    loader.print_dataset_info(
        validation_dataset,
        "Validation Dataset",
    )


if __name__ == "__main__":
    main()