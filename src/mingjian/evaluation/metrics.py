"""Dependency-light metrics used by tests and training reports."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mingjian.labels import LABEL_FAKE, LABEL_REAL


def _as_arrays(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    scores = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if labels.size == 0:
        raise ValueError("metrics require at least one sample")
    if not np.isin(labels, [LABEL_REAL, LABEL_FAKE]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(scores).all():
        raise ValueError("y_score must contain finite values")
    return labels, scores


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC with average ranks for tied scores."""

    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(labels.size, dtype=np.float64)
    start = 0
    while start < labels.size:
        end = start
        while end + 1 < labels.size and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        average_rank = (start + end + 2) / 2.0  # 1-based ranks
        ranks[order[start : end + 1]] = average_rank
        start = end + 1

    rank_sum = float(ranks[labels == LABEL_FAKE].sum())
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int | dict[str, int]]:
    """Return common binary metrics for real(0)/fake(1) predictions.

    ``y_score`` is interpreted as P(fake). Set ``threshold`` accordingly.
    """

    labels, scores = _as_arrays(y_true, y_score)
    predictions = (scores >= threshold).astype(np.int64)

    tp = int(((predictions == LABEL_FAKE) & (labels == LABEL_FAKE)).sum())
    tn = int(((predictions == LABEL_REAL) & (labels == LABEL_REAL)).sum())
    fp = int(((predictions == LABEL_FAKE) & (labels == LABEL_REAL)).sum())
    fn = int(((predictions == LABEL_REAL) & (labels == LABEL_FAKE)).sum())

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    accuracy = _safe_divide(tp + tn, labels.size)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": float(_roc_auc(labels, scores)),
        "threshold": float(threshold),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def expected_calibration_error(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute top-label Expected Calibration Error (ECE)."""

    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    labels, probs = _as_arrays(y_true, y_prob)
    if ((probs < 0) | (probs > 1)).any():
        raise ValueError("y_prob must be in [0, 1] for ECE")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(probs, bin_edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        if not mask.any():
            continue
        bin_accuracy = float(labels[mask].mean())
        bin_confidence = float(probs[mask].mean())
        ece += (mask.sum() / labels.size) * abs(bin_accuracy - bin_confidence)
    return float(ece)