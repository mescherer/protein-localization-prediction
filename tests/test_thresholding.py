"""
Unit tests for threshold optimization, immutability, and edge cases.
"""

import pytest
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef

from protein_loc.thresholding import (
    optimize_thresholds_mcc,
    apply_thresholds,
    optimize_thresholds_oof,
)


def test_mcc_threshold_optimizer_synthetic_separable():
    """Verify threshold optimizer recovers optimal threshold on separable distributions."""
    np.random.seed(42)
    n = 200
    # Class 0: positives at ~0.8, negatives at ~0.2
    y_c0 = np.array([1] * 100 + [0] * 100)
    p_c0 = np.array([0.8 + 0.05 * np.random.randn() for _ in range(100)] +
                    [0.2 + 0.05 * np.random.randn() for _ in range(100)])

    # Class 1: positives at ~0.6, negatives at ~0.1
    y_c1 = np.array([1] * 50 + [0] * 150)
    p_c1 = np.array([0.6 + 0.05 * np.random.randn() for _ in range(50)] +
                    [0.1 + 0.05 * np.random.randn() for _ in range(150)])

    probs = np.column_stack([p_c0, p_c1])
    y_true = np.column_stack([y_c0, y_c1])

    thresholds = optimize_thresholds_mcc(probs, y_true)
    assert len(thresholds) == 2
    assert 0.3 <= thresholds[0] <= 0.7
    assert 0.2 <= thresholds[1] <= 0.5

    # Check resulting MCC
    preds = apply_thresholds(probs, thresholds)
    mcc_0 = matthews_corrcoef(y_c0, preds[:, 0])
    mcc_1 = matthews_corrcoef(y_c1, preds[:, 1])
    assert mcc_0 == pytest.approx(1.0, abs=0.01)
    assert mcc_1 == pytest.approx(1.0, abs=0.01)


def test_threshold_optimizer_no_inplace_mutation():
    """Verify that threshold calibration and application do NOT mutate input probabilities."""
    probs_orig = np.random.uniform(0.1, 0.9, size=(50, 5))
    probs_copy = probs_orig.copy()
    y_true = np.random.binomial(1, 0.3, size=(50, 5))

    thresholds = optimize_thresholds_mcc(probs_orig, y_true)
    preds = apply_thresholds(probs_orig, thresholds)

    # Invariant: probs_orig must remain identical to probs_copy
    np.testing.assert_allclose(probs_orig, probs_copy, err_msg="optimize_thresholds_mcc modified input probs in-place!")


def test_threshold_optimizer_degenerate_classes():
    """Verify degenerate single-class edge cases default gracefully to 0.5."""
    probs = np.random.uniform(0.1, 0.9, size=(50, 3))
    # All zeros for class 0, all ones for class 1, normal for class 2
    y_true = np.zeros((50, 3), dtype=int)
    y_true[:, 1] = 1
    y_true[:20, 2] = 1

    thresholds = optimize_thresholds_mcc(probs, y_true)
    assert thresholds[0] == 0.5
    assert thresholds[1] == 0.5
    assert 0.01 <= thresholds[2] <= 0.99
