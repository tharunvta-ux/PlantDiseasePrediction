"""
trainer.py

Trains the custom CNN model for Plant Disease Detection.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from backend.models.cnn_model import CNNModel
from backend.services.data_loader import DataLoader


# ==========================================================
# Configuration
# ==========================================================

EPOCHS = 20

MODEL_SAVE_PATH = Path("saved_models/plant_disease_cnn.keras")
RESULTS_DIR = Path("results")

MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Data Augmentation
# ==========================================================

data_augmentation = Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
    ],
    name="data_augmentation",
)


def augment_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Apply augmentation only to training dataset."""

    return dataset.map(
        lambda images, labels: (
            data_augmentation(images, training=True),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


# ==========================================================
# Plot Utilities
# ==========================================================

def save_training_plots(history) -> None:

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_accuracy.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_loss.png")
    plt.close()


# ==========================================================
# Training
# ==========================================================

def train_model() -> None:

    print("\nLoading datasets...\n")

    loader = DataLoader()

    train_ds, val_ds = loader.load_datasets()

    class_names = loader.class_names

    train_ds = augment_dataset(train_ds)

    print(f"Detected {len(class_names)} classes.")

    cnn = CNNModel(num_classes=len(class_names))
    model = cnn.build()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = ModelCheckpoint(
    filepath="saved_models/best_model.weights.h5",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

    print("\nStarting Training...\n")

    start_time = time.time()

    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping],
)
    model.save("saved_models/plant_disease_cnn.keras")
    training_time = time.time() - start_time

    save_training_plots(history)

    print("\n========================================")
    print("Training Completed Successfully")
    print("========================================")
    print(f"Training Accuracy     : {history.history['accuracy'][-1]:.4f}")
    print(f"Validation Accuracy   : {history.history['val_accuracy'][-1]:.4f}")
    print(f"Training Loss         : {history.history['loss'][-1]:.4f}")
    print(f"Validation Loss       : {history.history['val_loss'][-1]:.4f}")
    print(f"Best Validation Acc   : {max(history.history['val_accuracy']):.4f}")
    print(f"Training Time         : {training_time:.2f} seconds")
    print(f"Model Saved At        : {MODEL_SAVE_PATH}")
    print(f"Plots Saved In        : {RESULTS_DIR}")
    print("========================================")


if __name__ == "__main__":
    train_model()