"""
backend/prediction/model_backends.py

Inference backends for the predictor:

- KerasBackend:  .keras files, needs full TensorFlow (~1.2 GB RAM).
- TFLiteBackend: .tflite files, uses the small LiteRT runtime
  (ai-edge-litert) so TensorFlow is never imported - suitable for
  low-memory hosting.

TensorFlow is imported lazily so a .tflite deployment does not load it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ModelBackend(Protocol):
    """What the predictor needs from a model."""

    input_size: Tuple[int, int]  # (width, height)
    num_classes: int

    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        """[N, H, W, 3] float32 in [0, 1] -> [N, num_classes] softmax."""
        ...


class KerasBackend:
    """Full TensorFlow / Keras model."""

    def __init__(self, model_path: Path, default_size: Tuple[int, int]) -> None:

        import tensorflow as tf

        self.model = tf.keras.models.load_model(model_path)

        height, width = self.model.input_shape[1:3]
        self.input_size = (int(width or default_size[0]), int(height or default_size[1]))
        self.num_classes = int(self.model.output_shape[-1])

    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        return self.model.predict(images, verbose=0)


def _tflite_interpreter_class():
    """Prefer the lightweight LiteRT runtime; fall back to TensorFlow's."""

    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter
    except ImportError:
        import tensorflow as tf

        logger.warning("ai-edge-litert not installed; using tf.lite.Interpreter")
        return tf.lite.Interpreter


class TFLiteBackend:
    """TensorFlow Lite model run with LiteRT (one image per invoke)."""

    def __init__(self, model_path: Path, default_size: Tuple[int, int]) -> None:

        interpreter_class = _tflite_interpreter_class()

        self.interpreter = interpreter_class(model_path=str(model_path))
        self.interpreter.allocate_tensors()

        self._input = self.interpreter.get_input_details()[0]
        self._output = self.interpreter.get_output_details()[0]

        _, height, width, _ = self._input["shape"]
        self.input_size = (int(width or default_size[0]), int(height or default_size[1]))
        self.num_classes = int(self._output["shape"][-1])

    def predict_batch(self, images: np.ndarray) -> np.ndarray:

        outputs = []

        for image in images:
            self.interpreter.set_tensor(
                self._input["index"],
                image[np.newaxis].astype(self._input["dtype"]),
            )
            self.interpreter.invoke()
            outputs.append(self.interpreter.get_tensor(self._output["index"])[0].copy())

        return np.stack(outputs)


def load_backend(model_path: Path, default_size: Tuple[int, int]) -> ModelBackend:
    """Pick the backend from the file extension."""

    if model_path.suffix.lower() == ".tflite":
        return TFLiteBackend(model_path, default_size)

    return KerasBackend(model_path, default_size)
