from __future__ import annotations

import numpy as np

from mingjian.evaluation.metrics import (
    binary_classification_metrics,
    expected_calibration_error,
)
from mingjian.labels import LABEL_FAKE, LABEL_REAL


def test_binary_metrics_known_values() -> None:
    labels = np.array([LABEL_REAL, LABEL_REAL, LABEL_FAKE, LABEL_FAKE])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = binary_classification_metrics(labels, scores, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["auc"] == 1.0


def test_roc_auc_handles_ties() -> None:
    labels = np.array([LABEL_REAL, LABEL_FAKE, LABEL_REAL, LABEL_FAKE])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    metrics = binary_classification_metrics(labels, scores)
    assert abs(float(metrics["auc"]) - 0.5) < 1e-12


def test_ece_is_zero_for_perfect_calibration() -> None:
    labels = np.array([LABEL_REAL, LABEL_REAL, LABEL_FAKE, LABEL_FAKE])
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(labels, probabilities, n_bins=10) == 0.0