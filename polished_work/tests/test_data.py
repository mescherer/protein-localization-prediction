"""
Unit tests for data loading, embedding extraction, and homology partition invariants.
"""

import os
import pytest
import numpy as np
import pandas as pd
import torch

from protein_loc.data import (
    SUBCELLULAR_LOCATIONS,
    HPA_TEST_LOCATIONS,
    NUCLEAR_LOCATIONS,
    load_subcellular_data,
    load_nuclear_data,
    get_homology_partitions,
    create_kfold_splits,
    compute_class_weights,
    SubcellularDataset,
    get_embedding_columns,
)


@pytest.fixture
def mock_subcellular_csv(tmp_path):
    """Create a small mock subcellular CSV file."""
    csv_file = tmp_path / "mock_subcellular.csv"
    n_samples = 50
    data = {
        "Uniprot": [f"P{i:05d}" for i in range(n_samples)],
        "Kingdom": ["Eukaryota"] * n_samples,
        "Partition": [1 if i % 2 == 0 else 2 for i in range(n_samples)],
    }
    for loc in SUBCELLULAR_LOCATIONS:
        data[loc] = np.random.binomial(1, 0.3, size=n_samples)
    for d in range(1280):
        data[f"emb_{d}"] = np.random.randn(n_samples).astype(np.float32)

    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def mock_hpa_test_csv(tmp_path):
    """Create a small mock HPA test CSV (8 classes present)."""
    csv_file = tmp_path / "mock_hpa_test.csv"
    n_samples = 20
    data = {
        "sid": [f"SID_{i}" for i in range(n_samples)],
        "Lengths": [300] * n_samples,
    }
    for loc in HPA_TEST_LOCATIONS:
        data[loc] = np.random.binomial(1, 0.25, size=n_samples)
    for d in range(1280):
        data[f"emb_{d}"] = np.random.randn(n_samples).astype(np.float32)

    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return str(csv_file)


def test_load_subcellular_data_shapes(mock_subcellular_csv):
    """Verify loaded feature and label dimensions."""
    df, X, y = load_subcellular_data(mock_subcellular_csv)
    assert X.shape == (50, 1280)
    assert y.shape == (50, 11)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_load_hpa_test_alignment(mock_hpa_test_csv):
    """Verify test set label alignment to full 11-class space with zero-filling."""
    df, X, y = load_subcellular_data(mock_hpa_test_csv, is_test=True)
    assert X.shape == (20, 1280)
    assert y.shape == (20, 11)

    # Absent classes should be all zeros
    absent_classes = ["Membrane", "Extracellular", "Plastid"]
    for cls_name in absent_classes:
        cls_idx = SUBCELLULAR_LOCATIONS.index(cls_name)
        assert np.all(y[:, cls_idx] == 0.0), f"Class {cls_name} should be zero-filled in test set."


def test_homology_partition_disjointness(mock_subcellular_csv):
    """Verify that every fold strictly partitions data with zero train/val overlap."""
    df, _, _ = load_subcellular_data(mock_subcellular_csv)
    splits = get_homology_partitions(df, partition_col="Partition")
    assert len(splits) == 2

    for train_idx, val_idx in splits:
        assert len(set(train_idx).intersection(set(val_idx))) == 0, "Train and validation sets must be disjoint."
        assert len(train_idx) + len(val_idx) == len(df), "Total partition indices must cover entire dataset."


def test_homology_partitions_scaling():
    """Verify dynamic scaling to 5 partitions when full dataset is provided."""
    n_samples = 100
    df = pd.DataFrame({"Partition": [1, 2, 3, 4, 5] * 20})
    splits = get_homology_partitions(df, partition_col="Partition")
    assert len(splits) == 5
    for train_idx, val_idx in splits:
        assert len(val_idx) == 20
        assert len(train_idx) == 80


def test_class_weights_calculation():
    """Verify inverse positive frequency class weights."""
    y = np.array([
        [1, 0, 1],
        [1, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    weights = compute_class_weights(y)
    assert weights.shape == (3,)
    assert weights[0] == pytest.approx(4.0 / 2.0)
    assert weights[1] == pytest.approx(4.0 / 1.0)
    assert weights[2] == pytest.approx(4.0 / 2.0)


def test_dataset_tensor_conversion():
    """Verify SubcellularDataset returns proper PyTorch float tensors."""
    X = np.random.randn(10, 1280).astype(np.float32)
    y = np.random.binomial(1, 0.5, size=(10, 11)).astype(np.float32)
    ds = SubcellularDataset(X, y)
    assert len(ds) == 10
    x_t, y_t = ds[0]
    assert isinstance(x_t, torch.Tensor) and x_t.shape == (1280,) and x_t.dtype == torch.float32
    assert isinstance(y_t, torch.Tensor) and y_t.shape == (11,) and y_t.dtype == torch.float32
