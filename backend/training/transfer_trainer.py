"""
backend/training/transfer_trainer.py

Stage B: transfer-learning model aimed at better real-world robustness.

- ImageNet-pretrained EfficientNetV2-B0 backbone, 224x224 input.
- Accepts the same [0, 1] input as the existing CNN, so the API
  predictor works unchanged (it reads the input size from the model).
- Strong in-model augmentation (active only during training): framing
  (zoom-out, shift, rotation, flips), lighting/colour jitter, blur.
- Class-balanced loss weights (the training set is ~36x imbalanced).
- Two phases: train the new head with the backbone frozen, then
  fine-tune the top of the backbone at a low learning rate.
- Early stopping uses only the CALIBRATION half of PlantVillage val; the
  held-out TEST half (see backend/evaluation/calibrate.py) is untouched.
- Saves to a NEW file; the existing plant_disease_cnn.keras is never
  modified.

Full training is slow on CPU (native Windows TensorFlow has no GPU
support); use Google Colab / a GPU machine for real runs.

Usage:
    python -m backend.training.transfer_trainer
    python -m backend.training.transfer_trainer --smoke-test   # CPU check
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from backend.config import MODEL_ARTIFACTS_ROOT, PLANTVILLAGE_DIR, PROJECT_ROOT
from backend.evaluation.calibrate import stratified_split
from backend.evaluation.collect_predictions import list_images
from backend.prediction.predictor import CLASS_NAMES_FILE

logger = logging.getLogger(__name__)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = 15
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_LAYERS = 40  # top backbone layers unfrozen in phase 2
EARLY_STOPPING_PATIENCE = 3

MODEL_NAME = "plant_disease_effnetv2b0"
MODEL_SAVE_PATH = PROJECT_ROOT / "saved_models" / f"{MODEL_NAME}.keras"
RESULTS_DIR = PROJECT_ROOT / "results" / MODEL_NAME


# ==========================================================
# Data
# ==========================================================


def load_image(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Decode, resize (antialiased bilinear) and scale to [0, 1]."""

    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear", antialias=True)

    return image / 255.0, label


def make_dataset(
    paths: List[Path],
    labels: np.ndarray,
    shuffle: bool,
) -> tf.data.Dataset:
    """tf.data pipeline from explicit file lists."""

    dataset = tf.data.Dataset.from_tensor_slices(
        ([str(p) for p in paths], labels.astype(np.int32))
    )

    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    return (
        dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def class_weights(labels: np.ndarray, num_classes: int) -> dict:
    """Balanced weights: n_samples / (n_classes * n_class_samples)."""

    counts = np.bincount(labels, minlength=num_classes)

    return {
        i: float(len(labels) / (num_classes * c)) if c else 0.0
        for i, c in enumerate(counts)
    }


# ==========================================================
# Model
# ==========================================================


def build_augmentation() -> keras.Sequential:
    """Real-world-oriented augmentation, applied on [0, 255] images."""

    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.5, fill_mode="reflect"),
            # Negative = zoom in, positive = zoom out (leaf smaller in frame).
            keras.layers.RandomZoom((-0.2, 0.4), fill_mode="reflect"),
            keras.layers.RandomTranslation(0.15, 0.15, fill_mode="reflect"),
            keras.layers.RandomColorJitter(
                value_range=(0, 255),
                brightness_factor=0.3,
                contrast_factor=0.3,
                saturation_factor=(0.3, 0.7),
                hue_factor=0.05,
            ),
            keras.layers.RandomGaussianBlur(
                factor=0.5,
                kernel_size=5,
                sigma=(0.1, 2.0),
                value_range=(0, 255),
            ),
        ],
        name="real_world_augmentation",
    )


def build_model(num_classes: int, weights: str | None) -> Tuple[keras.Model, keras.Model]:
    """
    Returns:
        (model, backbone) - model takes [0, 1] RGB images.
    """

    backbone = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=weights,
        input_shape=(*IMAGE_SIZE, 3),
        include_preprocessing=True,  # expects [0, 255]
    )
    backbone.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="image_0_1")
    x = keras.layers.Rescaling(255.0)(inputs)
    x = build_augmentation()(x)
    x = backbone(x, training=False)  # keep BatchNorm in inference mode
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name=MODEL_NAME), backbone


def compile_model(model: keras.Model, learning_rate: float) -> None:

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


# ==========================================================
# Training
# ==========================================================


def train(smoke_test: bool = False) -> None:

    train_dir = PLANTVILLAGE_DIR / "train"
    val_dir = PLANTVILLAGE_DIR / "val"

    class_names = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    num_classes = len(class_names)

    train_paths, train_labels = list_images(train_dir, class_names)
    val_paths, val_labels = list_images(val_dir, class_names)

    # Early stopping on the calibration half only; test half untouched.
    calib_idx, _ = stratified_split(val_labels)
    val_paths = [val_paths[i] for i in calib_idx]
    val_labels = val_labels[calib_idx]

    head_epochs, fine_epochs = HEAD_EPOCHS, FINE_TUNE_EPOCHS
    weights = "imagenet"
    save_path = MODEL_SAVE_PATH

    if smoke_test:
        # Tiny, offline run that only checks the pipeline end to end.
        rng = np.random.default_rng(SEED)
        pick = rng.choice(len(train_paths), 64, replace=False)
        train_paths = [train_paths[i] for i in pick]
        train_labels = train_labels[pick]
        val_paths, val_labels = val_paths[:32], val_labels[:32]
        head_epochs, fine_epochs = 1, 1
        weights = None  # no download
        save_path = RESULTS_DIR / "smoke_test_model.keras"

    logger.info("Train images: %d | early-stopping images: %d", len(train_paths), len(val_paths))

    train_ds = make_dataset(train_paths, train_labels, shuffle=True)
    val_ds = make_dataset(val_paths, val_labels, shuffle=False)
    weights_by_class = class_weights(train_labels, num_classes)

    model, backbone = build_model(num_classes, weights)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(save_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    start = time.time()

    # Phase 1: head only.
    compile_model(model, HEAD_LEARNING_RATE)
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=head_epochs,
        class_weight=weights_by_class,
        callbacks=callbacks,
    )

    # Phase 2: fine-tune the top of the backbone (BatchNorm stays frozen).
    backbone.trainable = True
    for layer in backbone.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, FINE_TUNE_LEARNING_RATE)
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_epochs,
        class_weight=weights_by_class,
        callbacks=callbacks,
    )

    # save_path already holds the best checkpoint across BOTH phases
    # (ModelCheckpoint keeps its best score between fit() calls), so the
    # final in-memory weights are deliberately not saved over it.

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    history = {
        phase: {k: [float(v) for v in values] for k, values in h.history.items()}
        for phase, h in (("head", history_head), ("fine_tune", history_fine))
    }
    (RESULTS_DIR / ("smoke_history.json" if smoke_test else "history.json")).write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    if not smoke_test:
        artifacts = MODEL_ARTIFACTS_ROOT / MODEL_NAME
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / CLASS_NAMES_FILE).write_text(
            json.dumps(class_names, indent=2), encoding="utf-8"
        )

    logger.info("Saved %s in %.0fs", save_path, time.time() - start)


def main() -> None:

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    train(smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
