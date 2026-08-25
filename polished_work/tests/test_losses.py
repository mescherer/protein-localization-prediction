"""
Unit tests for loss functions and ridge regularizer invariants.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_loc.models import MLP
from protein_loc.losses import (
    standard_bce,
    weighted_bce,
    pos_weighted_bce,
    focal_loss,
    ridge_penalty,
)


def test_standard_and_weighted_bce():
    """Verify standard and class-weighted BCE loss computation."""
    logits = torch.tensor([[2.0, -1.0], [-2.0, 1.0]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    crit_std = standard_bce()
    loss_std = crit_std(logits, targets)
    assert loss_std.item() > 0.0

    weights = torch.tensor([2.0, 0.5])
    crit_w = weighted_bce(weights)
    loss_w = crit_w(logits, targets)
    assert loss_w.item() > 0.0
    assert not torch.isclose(loss_std, loss_w)


def test_focal_loss_reduction():
    """Verify focal loss computes valid positive scalar loss."""
    logits = torch.randn(10, 11)
    targets = torch.randint(0, 2, (10, 11)).float()

    fl = focal_loss(logits, targets, gamma=2.0)
    assert fl.ndim == 0
    assert fl.item() >= 0.0
    assert not torch.isnan(fl)


def test_ridge_penalty_manual_equivalence():
    """Verify ridge penalty matches manual squared Frobenius norm summation."""
    model = MLP(input_dim=1280, num_classes=11, hidden_dims=(128, 64))
    lambda_reg = 0.01

    reg_fn = ridge_penalty(lambda_reg)
    computed_penalty = reg_fn(model)

    # Manual calculation
    manual_sum = 0.0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            manual_sum += torch.sum(m.weight ** 2).item()
    expected_penalty = lambda_reg * manual_sum

    assert torch.isclose(computed_penalty, torch.tensor(expected_penalty, dtype=torch.float32), rtol=1e-4)


def test_zero_ridge_penalty():
    """Verify lambda=0 returns exact zero penalty."""
    model = MLP(input_dim=1280, num_classes=11)
    reg_fn = ridge_penalty(0.0)
    assert reg_fn(model).item() == 0.0
