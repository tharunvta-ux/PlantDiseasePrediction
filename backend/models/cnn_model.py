"""
cnn_model.py

Defines the CNN architecture for Plant Disease Detection.

Author: Your Name
Project: Plant Disease Detection and Treatment Recommendation System
"""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)


class CNNModel:
    """
    Builds and compiles the CNN model.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (256, 256, 3),
        num_classes: int = 38,
    ) -> None:
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self) -> tf.keras.Model:
        """
        Build and compile the CNN model.

        Returns:
            Compiled TensorFlow model.
        """

        model = Sequential(
            [
                # Input Layer
                Input(shape=self.input_shape),

                # -----------------------------
                # Convolution Block 1
                # -----------------------------
                Conv2D(
                    filters=32,
                    kernel_size=(3, 3),
                    activation="relu",
                    padding="same",
                ),
                MaxPooling2D(pool_size=(2, 2)),

                # -----------------------------
                # Convolution Block 2
                # -----------------------------
                Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    activation="relu",
                    padding="same",
                ),
                MaxPooling2D(pool_size=(2, 2)),

                # -----------------------------
                # Convolution Block 3
                # -----------------------------
                Conv2D(
                    filters=128,
                    kernel_size=(3, 3),
                    activation="relu",
                    padding="same",
                ),
                MaxPooling2D(pool_size=(2, 2)),

                # Convert 3D feature maps to 1D vector
                Flatten(),

                # Dense Layer
                Dense(
                    units=256,
                    activation="relu",
                ),

                # Prevent overfitting
                Dropout(rate=0.5),

                # Output Layer
                Dense(
                    units=self.num_classes,
                    activation="softmax",
                ),
            ],
            name="PlantDiseaseCNN",
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        return model


if __name__ == "__main__":
    cnn = CNNModel()
    model = cnn.build()

    model.summary()