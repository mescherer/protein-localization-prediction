"""
Unit tests for multi-label metric calculations against scikit-learn references.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import (
    hamming_loss,
    jaccard_score,
    f1_score,
    matthews_corrcoef,
)

from protein_loc.metrics import (
    compute_multilabel_metrics,
    format_metrics_table,
    format_per_class_table,
)
from protein_loc.data import SUBCELLULAR_LOCATIONS


def test_metrics_perfect_prediction():
    """Verify metrics on perfect predictions evaluate to 1.0."""
    y_true = np.array([
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
    ])
    y_pred = y_true.copy()
    locations = ["A", "B", "C", "D"]

    res = compute_multilabel_metrics(y_true, y_pred, locations=locations)
    assert res["exact_match"] == 1.0
    assert res["hamming_accuracy"] == 1.0
    assert res["jaccard"] == 1.0
    assert res["micro_f1"] == 1.0
    assert res["macro_f1"] == 1.0
    assert res["mean_mcc"] == 1.0


def test_metrics_against_sklearn():
    """Verify computed metrics match scikit-learn ground-truth implementations."""
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.4, size=(100, 11))
    y_pred = np.random.binomial(1, 0.4, size=(100, 11))

    res = compute_multilabel_metrics(y_true, y_pred, locations=SUBCELLULAR_LOCATIONS)

    # Scikit-learn ground truth
    expected_exact = np.all(y_true == y_pred, axis=1).mean()
    expected_hamming = 1.0 - hamming_loss(y_true, y_pred)
    expected_jaccard = jaccard_score(y_true, y_pred, average="samples", zero_division=0)
    expected_micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    expected_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    assert res["exact_match"] == pytest.approx(expected_exact)
    assert res["hamming_accuracy"] == pytest.approx(expected_hamming)
    assert res["jaccard"] == pytest.approx(expected_jaccard)
    assert res["micro_f1"] == pytest.approx(expected_micro_f1)
    assert res["macro_f1"] == pytest.approx(expected_macro_f1)

    for i, loc in enumerate(SUBCELLULAR_LOCATIONS):
        expected_mcc = matthews_corrcoef(y_true[:, i], y_pred[:, i])
        assert res["per_class"][loc]["mcc"] == pytest.approx(expected_mcc)


def test_format_metrics_tables():
    """Verify formatting utilities return non-empty DataFrames."""
    y_true = np.random.binomial(1, 0.3, size=(20, 11))
    y_pred = np.random.binomial(1, 0.3, size=(20, 11))
    res = compute_multilabel_metrics(y_true, y_pred, locations=SUBCELLULAR_LOCATIONS)

    df_overall = format_metrics_table(res)
    assert isinstance(df_overall, pd.DataFrame)
    assert len(df_overall) == 6

    df_pc = format_per_class_table(res)
    assert isinstance(df_pc, pd.DataFrame)
    assert len(df_pc) == 11
