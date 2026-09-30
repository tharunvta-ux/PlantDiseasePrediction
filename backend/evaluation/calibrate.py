"""
backend/evaluation/calibrate.py

Derive the confidence policy for the deployed model from data.

Steps
-----
1. Split the cached validation predictions 50/50 (stratified, fixed
   seed) into a *calibration* half and a held-out *test* half.
2. Fit temperature scaling on the calibration half (minimise NLL).
3. Choose thresholds on the calibration half:
   - high: the calibrated confidence that best separates correct from
     incorrect predictions (maximum Youden's J = TPR - FPR).
   - low:  the largest confidence below which the top-1 prediction is
     wrong at least as often as it is right (accuracy <= 50%).
4. Report accuracy / calibration / selective accuracy on the test half.
5. Record image-quality reference percentiles.

The thresholds are marked PROVISIONAL: they are measured on PlantVillage
validation images only (the same split used for early stopping), which
are cleaner than real-world photos.

Usage:
    python -m backend.evaluation.calibrate
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import date
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

from backend.config import MODEL_ARTIFACTS_ROOT, PROJECT_ROOT  # noqa: E402
from backend.evaluation.collect_predictions import DEFAULT_OUTPUT  # noqa: E402
from backend.prediction.confidence import (  # noqa: E402
    CALIBRATION_FILE,
    apply_temperature,
)

logger = logging.getLogger(__name__)

SEED = 42
ECE_BINS = 15
MIN_SAMPLES_FOR_LOW_THRESHOLD = 20
REPORT_DIR = PROJECT_ROOT / "results" / "calibration"


# ==========================================================
# Metrics
# ==========================================================


def stratified_split(
    labels: np.ndarray,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices 50/50 within every class."""

    rng = np.random.default_rng(seed)
    calib, test = [], []

    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        half = len(idx) // 2
        calib.append(idx[:half])
        test.append(idx[half:])

    return np.sort(np.concatenate(calib)), np.sort(np.concatenate(test))


def nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood."""

    picked = probabilities[np.arange(len(labels)), labels]

    return float(-np.log(np.clip(picked, 1e-38, None)).mean())


def expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    bins: int = ECE_BINS,
) -> float:
    """Standard equal-width-bin ECE on top-1 confidence."""

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)

        if mask.any():
            gap = abs(correct[mask].mean() - confidence[mask].mean())
            ece += mask.mean() * gap

    return float(ece)


def wilson_interval(successes: int, total: int) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion."""

    if total == 0:
        return (math.nan, math.nan)

    z = 1.96
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    margin /= denom

    return (centre - margin, centre + margin)


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Temperature minimising NLL on the calibration half."""

    result = minimize_scalar(
        lambda t: nll(apply_temperature(probabilities, t), labels),
        bounds=(0.05, 20.0),
        method="bounded",
    )

    return float(result.x)


# ==========================================================
# Thresholds
# ==========================================================


def youden_threshold(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Confidence maximising TPR - FPR for 'prediction is correct'."""

    n_pos = correct.sum()
    n_neg = (~correct).sum()

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both correct and incorrect predictions.")

    order = np.argsort(-confidence)
    conf_sorted = confidence[order]
    correct_sorted = correct[order]

    tpr = np.cumsum(correct_sorted) / n_pos
    fpr = np.cumsum(~correct_sorted) / n_neg
    j = tpr - fpr

    # Only evaluate at the last index of tied confidences.
    last_of_tie = np.r_[conf_sorted[1:] != conf_sorted[:-1], True]
    j = np.where(last_of_tie, j, -np.inf)

    return float(conf_sorted[int(np.argmax(j))])


def low_threshold(
    confidence: np.ndarray,
    correct: np.ndarray,
) -> Dict[str, Any]:
    """
    Largest threshold t such that predictions with confidence < t are
    correct at most 50% of the time (with enough samples to say so).
    """

    order = np.argsort(confidence)
    conf_sorted = confidence[order]
    correct_sorted = correct[order]

    counts = np.arange(1, len(conf_sorted) + 1)
    acc_below = np.cumsum(correct_sorted) / counts

    valid = (acc_below <= 0.5) & (counts >= MIN_SAMPLES_FOR_LOW_THRESHOLD)

    if not valid.any():
        return {
            "value": None,
            "samples_below": int((conf_sorted < 0.5).sum()),
            "note": "Too few low-confidence errors to estimate.",
        }

    last = int(np.flatnonzero(valid)[-1])

    # Threshold sits just above the last included sample.
    value = float(conf_sorted[last + 1]) if last + 1 < len(conf_sorted) else 1.0

    return {
        "value": value,
        "samples_below": int(counts[last]),
        "accuracy_below": float(acc_below[last]),
    }


def summarise_levels(
    confidence: np.ndarray,
    correct: np.ndarray,
    high: float,
    low: float,
) -> Dict[str, Any]:
    """Coverage and accuracy of each confidence level."""

    levels = {
        "high": confidence >= high,
        "uncertain": (confidence >= low) & (confidence < high),
        "low": confidence < low,
    }

    summary = {}

    for name, mask in levels.items():
        n = int(mask.sum())
        k = int(correct[mask].sum())
        lo, hi = wilson_interval(k, n)
        summary[name] = {
            "count": n,
            "coverage": round(n / len(confidence), 4),
            "accuracy": round(k / n, 4) if n else None,
            "accuracy_95ci": [round(lo, 4), round(hi, 4)] if n else None,
        }

    return summary


# ==========================================================
# Plots
# ==========================================================


def save_plots(
    raw_conf: np.ndarray,
    cal_conf: np.ndarray,
    correct: np.ndarray,
    high: float,
    low: float,
    out_dir: Path,
) -> None:
    """Reliability diagram and risk-coverage curve for the test half."""

    edges = np.linspace(0, 1, ECE_BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for conf, label in ((raw_conf, "Raw softmax"), (cal_conf, "Temperature scaled")):
        accs = [
            correct[(conf > a) & (conf <= b)].mean()
            if ((conf > a) & (conf <= b)).any() else np.nan
            for a, b in zip(edges[:-1], edges[1:])
        ]
        plt.plot(centres, accs, "o-", label=label)

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability diagram (held-out test half)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "reliability_diagram.png")
    plt.close()

    order = np.argsort(-cal_conf)
    cum_acc = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    coverage = np.arange(1, len(order) + 1) / len(order)

    plt.figure(figsize=(7, 5))
    plt.plot(coverage, cum_acc)

    for value, name, colour in ((high, "high", "tab:green"), (low, "low", "tab:red")):
        cov = (cal_conf >= value).mean()
        plt.axvline(
            cov,
            linestyle="--",
            color=colour,
            label=f"{name} threshold ({value:.3f})",
        )

    plt.xlabel("Coverage (fraction of images accepted)")
    plt.ylabel("Accuracy of accepted images")
    plt.title("Accuracy vs coverage (held-out test half)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_coverage.png")
    plt.close()


# ==========================================================
# Main
# ==========================================================


def calibrate(predictions_file: Path) -> Dict[str, Any]:
    """Run the full calibration and write artifacts."""

    data = np.load(predictions_file)
    probabilities = data["probabilities"].astype(np.float64)
    labels = data["labels"]
    class_names = [str(c) for c in data["class_names"]]
    model_version = str(data["model_version"])
    quality = data["quality"]
    quality_columns = [str(c) for c in data["quality_columns"]]

    calib_idx, test_idx = stratified_split(labels)

    temperature = fit_temperature(probabilities[calib_idx], labels[calib_idx])

    calibrated = apply_temperature(probabilities, temperature)
    cal_conf = calibrated.max(axis=1)
    raw_conf = probabilities.max(axis=1)
    correct = calibrated.argmax(axis=1) == labels

    high = youden_threshold(cal_conf[calib_idx], correct[calib_idx])
    low_info = low_threshold(cal_conf[calib_idx], correct[calib_idx])

    # If there is not enough evidence for a separate "low" band, fall
    # back to the high threshold (i.e. only high vs. not-high).
    low = low_info["value"] if low_info["value"] is not None else high
    low = min(low, high)

    t_conf, t_correct = cal_conf[test_idx], correct[test_idx]
    t_labels = labels[test_idx]
    t_pred = calibrated[test_idx].argmax(axis=1)

    per_class = {
        name: round(float((t_pred[t_labels == i] == i).mean()), 4)
        for i, name in enumerate(class_names)
        if (t_labels == i).any()
    }

    q_calib = quality[calib_idx]
    col = {name: i for i, name in enumerate(quality_columns)}
    quality_reference = {
        "blur_score_p01": float(np.percentile(q_calib[:, col["blur_score"]], 1)),
        "brightness_p01": float(np.percentile(q_calib[:, col["brightness"]], 1)),
        "brightness_p99": float(np.percentile(q_calib[:, col["brightness"]], 99)),
        "contrast_p01": float(np.percentile(q_calib[:, col["contrast"]], 1)),
    }

    # How often quality checks fire on held-out lab images, and whether
    # flagged images are actually less accurate.
    q_test = quality[test_idx]
    flagged = (
        (q_test[:, col["blur_score"]] < quality_reference["blur_score_p01"])
        | (q_test[:, col["brightness"]] < quality_reference["brightness_p01"])
        | (q_test[:, col["brightness"]] > quality_reference["brightness_p99"])
        | (q_test[:, col["contrast"]] < quality_reference["contrast_p01"])
    )
    high_mask = t_conf >= high
    quality_flag_stats = {
        "flagged_fraction": round(float(flagged.mean()), 4),
        "accuracy_flagged": round(float(t_correct[flagged].mean()), 4) if flagged.any() else None,
        "accuracy_unflagged": round(float(t_correct[~flagged].mean()), 4),
        "high_coverage_without_quality_cap": round(float(high_mask.mean()), 4),
        "high_coverage_with_quality_cap": round(float((high_mask & ~flagged).mean()), 4),
        "high_accuracy_with_quality_cap": round(
            float(t_correct[high_mask & ~flagged].mean()), 4
        ),
    }

    report = {
        "model_version": model_version,
        "created": date.today().isoformat(),
        "data": {
            "source": str(predictions_file.relative_to(PROJECT_ROOT)),
            "calibration_images": int(len(calib_idx)),
            "test_images": int(len(test_idx)),
            "split": "stratified 50/50 of PlantVillage val, seed 42",
            "caveat": (
                "The PlantVillage val split was also used for early "
                "stopping, so these numbers are optimistic."
            ),
        },
        "temperature": temperature,
        "thresholds": {
            "high": high,
            "low": low,
            "method_high": "max Youden's J (correct vs incorrect) on calibration half",
            "method_low": "largest t with accuracy(conf < t) <= 50% on calibration half",
            "low_details": low_info,
        },
        "provisional": True,
        "provisional_reason": (
            "Derived from PlantVillage validation images only (lab "
            "photos, plain backgrounds). No labelled real-world images "
            "were available; real-world accuracy at these thresholds is "
            "unmeasured."
        ),
        "quality_reference": quality_reference,
        "test_metrics": {
            "accuracy": round(float(t_correct.mean()), 4),
            "macro_class_accuracy": round(float(np.mean(list(per_class.values()))), 4),
            "nll_raw": round(nll(probabilities[test_idx], t_labels), 4),
            "nll_calibrated": round(nll(calibrated[test_idx], t_labels), 4),
            "ece_raw": round(expected_calibration_error(raw_conf[test_idx], t_correct), 4),
            "ece_calibrated": round(expected_calibration_error(t_conf, t_correct), 4),
            "levels": summarise_levels(t_conf, t_correct, high, low),
            "quality_flags": quality_flag_stats,
            "per_class_accuracy": per_class,
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    (REPORT_DIR / "calibration_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    save_plots(raw_conf[test_idx], t_conf, t_correct, high, low, REPORT_DIR)

    # Runtime config consumed by the API (subset of the report).
    runtime = {
        key: report[key]
        for key in (
            "model_version",
            "created",
            "temperature",
            "provisional",
            "provisional_reason",
            "quality_reference",
        )
    }
    runtime["thresholds"] = {"high": high, "low": low}

    artifacts_dir = MODEL_ARTIFACTS_ROOT / model_version
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    (artifacts_dir / CALIBRATION_FILE).write_text(
        json.dumps(runtime, indent=2), encoding="utf-8"
    )

    return report


def main() -> None:

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = calibrate(args.predictions)

    summary = {k: v for k, v in report["test_metrics"].items() if k != "per_class_accuracy"}

    print(json.dumps(
        {
            "temperature": report["temperature"],
            "thresholds": report["thresholds"],
            "test_metrics": summary,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
