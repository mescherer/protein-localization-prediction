"""
Unit tests for seed determinism and end-to-end training reproducibility.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from protein_loc.models import MLP
from protein_loc.training import set_seed, train_model, predict_probs
from protein_loc.data import get_dataloader
from protein_loc.losses import ridge_penalty


def test_seed_determinism_model_initialization():
    """Verify that setting the seed produces bitwise identical initial weights."""
    set_seed(123)
    m1 = MLP(input_dim=1280, num_classes=11, hidden_dims=(128, 64))

    set_seed(123)
    m2 = MLP(input_dim=1280, num_classes=11, hidden_dims=(128, 64))

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2), "Model parameters initialized with same seed must be identical."


def test_seed_determinism_training_loss():
    """Verify identical training loss curves across repeated seeded runs."""
    x = np.random.randn(32, 1280).astype(np.float32)
    y = np.random.binomial(1, 0.3, size=(32, 11)).astype(np.float32)

    set_seed(42)
    loader1 = get_dataloader(x, y, batch_size=16, shuffle=False)
    m1 = MLP(input_dim=1280, num_classes=11, hidden_dims=(64,))
    m1, hist1 = train_model(m1, loader1, epochs=5, lr=1e-3)

    set_seed(42)
    loader2 = get_dataloader(x, y, batch_size=16, shuffle=False)
    m2 = MLP(input_dim=1280, num_classes=11, hidden_dims=(64,))
    m2, hist2 = train_model(m2, loader2, epochs=5, lr=1e-3)

    np.testing.assert_allclose(
        hist1["train_loss"],
        hist2["train_loss"],
        rtol=1e-5,
        err_msg="Loss trajectories diverged under identical random seed.",
    )


def test_end_to_end_training_and_prediction():
    """Verify end-to-end forward/backward training pass and probability prediction."""
    x_tr = np.random.randn(20, 1280).astype(np.float32)
    y_tr = np.random.binomial(1, 0.4, size=(20, 11)).astype(np.float32)
    x_val = np.random.randn(10, 1280).astype(np.float32)
    y_val = np.random.binomial(1, 0.4, size=(10, 11)).astype(np.float32)

    loader = get_dataloader(x_tr, y_tr, batch_size=8, shuffle=True)
    model = MLP(input_dim=1280, num_classes=11, hidden_dims=(64,))
    reg = ridge_penalty(1e-4)

    trained_model, history = train_model(
        model,
        loader,
        val_x=x_val,
        val_y=y_val,
        epochs=3,
        regularizer=reg,
    )

    assert len(history["train_loss"]) == 3
    assert len(history["val_loss"]) == 3
    assert not np.isnan(history["train_loss"]).any()

    probs = predict_probs(trained_model, x_val)
    assert probs.shape == (10, 11)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
