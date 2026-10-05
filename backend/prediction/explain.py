"""
backend/prediction/explain.py

"Where did the model look?" - occlusion sensitivity maps.

A square window is slid over the image and the pixels inside it are
replaced by a heavily blurred copy of the same region; for each position
we measure how much the model's evidence (log-odds) for the target class
drops. Regions whose blurring causes a large drop are the ones the
prediction depends on.

Blurring (instead of a flat grey square) removes fine detail such as
lesions while keeping colours, so the occluded image stays close to what
the model saw in training and the map reflects the model, not the
occluder.

Model-agnostic (no gradients), so it works with the TFLite deployment.
Reference: Zeiler & Fergus, "Visualizing and Understanding
Convolutional Networks" (2014).
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# 64 px patch, 48 px stride on a 256 px input -> 5x5 = 25 model runs,
# covering the full image (4 * 48 + 64 = 256).
DEFAULT_PATCH = 64
DEFAULT_STRIDE = 48

# Gaussian blur radius (px at 256x256) used as the occluder.
BLUR_RADIUS = 12

OVERLAY_MAX_SIDE = 512
OVERLAY_ALPHA = 0.55


@dataclass(frozen=True)
class OcclusionResult:
    """Sensitivity grid for one image and class."""

    heat: np.ndarray  # [rows, cols] log-odds drops, >= 0
    base_probability: float
    positions: List[Tuple[int, int]]
    patch: int


def _positions(size: int, patch: int, stride: int) -> List[int]:
    """Patch start offsets that tile [0, size) completely."""

    starts = list(range(0, max(size - patch, 0) + 1, stride))

    if starts[-1] + patch < size:
        starts.append(size - patch)

    return starts


def occlusion_map(
    image: np.ndarray,
    class_index: int,
    predict_batch: Callable[[np.ndarray], np.ndarray],
    patch: int = DEFAULT_PATCH,
    stride: int = DEFAULT_STRIDE,
) -> OcclusionResult:
    """
    Args:
        image: [1, H, W, 3] preprocessed model input.
        class_index: Class whose probability is tracked.
        predict_batch: The model ([N, H, W, 3] -> [N, classes]).
    """

    _, height, width, _ = image.shape
    rows = _positions(height, patch, stride)
    cols = _positions(width, patch, stride)

    blurred = np.asarray(
        Image.fromarray((image[0] * 255).round().astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(BLUR_RADIUS)
        ),
        dtype=np.float32,
    ) / 255.0

    batch = np.repeat(image, len(rows) * len(cols), axis=0)
    positions = []

    for k, (y, x) in enumerate((y, x) for y in rows for x in cols):
        batch[k, y:y + patch, x:x + patch, :] = blurred[y:y + patch, x:x + patch, :]
        positions.append((y, x))

    base_probs = predict_batch(image)
    occluded_probs = predict_batch(batch)

    # Log-odds instead of raw probability: a confident model sits at
    # p ~ 1.0 where occluding one patch barely moves p, but the log-odds
    # still change, so the map stays informative.
    base_score = log_odds(base_probs, class_index)[0]
    occluded_score = log_odds(occluded_probs, class_index)

    heat = np.clip(base_score - occluded_score, 0, None).reshape(len(rows), len(cols))

    return OcclusionResult(
        heat=heat,
        base_probability=float(base_probs[0][class_index]),
        positions=positions,
        patch=patch,
    )


def log_odds(probabilities: np.ndarray, class_index: int) -> np.ndarray:
    """log(p_c / sum of other classes), numerically safe near p = 1."""

    probs = probabilities.astype(np.float64)
    target = probs[:, class_index]
    others = probs.sum(axis=1) - target  # avoids 1 - p rounding to 0

    return np.log(np.clip(target, 1e-38, None)) - np.log(np.clip(others, 1e-38, None))


def _colormap(values: np.ndarray) -> np.ndarray:
    """0..1 -> RGB: yellow (weak) to red (strong)."""

    low = np.array([255, 221, 0], dtype=np.float32)
    high = np.array([214, 31, 31], dtype=np.float32)

    return low + (high - low) * values[..., None]


def render_overlay(image_path: str, result: OcclusionResult) -> str:
    """
    Blend the normalised heat map over the photo.

    Returns:
        PNG as a data URL.
    """

    with Image.open(image_path) as photo:
        photo = ImageOps.exif_transpose(photo).convert("RGB")
        photo.thumbnail((OVERLAY_MAX_SIDE, OVERLAY_MAX_SIDE))

    peak = float(result.heat.max())
    norm = result.heat / peak if peak > 0 else np.zeros_like(result.heat)

    heat_img = Image.fromarray((norm * 255).astype(np.uint8), mode="L").resize(
        photo.size, Image.Resampling.BICUBIC
    )
    heat = np.asarray(heat_img, dtype=np.float32) / 255.0

    base = np.asarray(photo, dtype=np.float32)
    alpha = (heat * OVERLAY_ALPHA)[..., None]
    blended = base * (1 - alpha) + _colormap(heat) * alpha

    buffer = io.BytesIO()
    Image.fromarray(blended.clip(0, 255).astype(np.uint8)).save(buffer, format="PNG", optimize=True)

    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
