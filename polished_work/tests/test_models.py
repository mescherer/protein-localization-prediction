"""
Unit tests for model architectures and forward pass dimensions.
"""

import pytest
import torch
import torch.nn as nn

from protein_loc.models import LogisticRegression, MLP, get_model


@pytest.mark.parametrize("batch_size", [1, 16, 64])
def test_logistic_regression_forward_shape(batch_size):
    """Verify LogisticRegression output dimensions."""
    model = LogisticRegression(input_dim=1280, num_classes=11)
    x = torch.randn(batch_size, 1280)
    out = model(x)
    assert out.shape == (batch_size, 11)


@pytest.mark.parametrize("hidden_dims", [(64,), (128, 64), (256, 128, 64)])
def test_mlp_forward_shapes(hidden_dims):
    """Verify MLP forward shapes with various hidden layer depths."""
    model = MLP(input_dim=1280, num_classes=11, hidden_dims=hidden_dims)
    x = torch.randn(8, 1280)
    out = model(x)
    assert out.shape == (8, 11)


def test_get_model_factory():
    """Verify model factory instantiates expected architecture types."""
    m1 = get_model("logreg", input_dim=1280, num_classes=11)
    assert isinstance(m1, LogisticRegression)

    m2 = get_model("mlp_1h", input_dim=1280, num_classes=11)
    assert isinstance(m2, MLP)
    assert m2.hidden_dims == (64,)

    m3 = get_model("mlp_2h", input_dim=1280, num_classes=11)
    assert isinstance(m3, MLP)
    assert m3.hidden_dims == (128, 64)


def test_model_gradients_flow():
    """Verify backward gradient pass flows to all parameters."""
    model = MLP(input_dim=1280, num_classes=11, hidden_dims=(128, 64))
    x = torch.randn(4, 1280)
    y = torch.ones(4, 11)
    loss = nn.BCEWithLogitsLoss()(model(x), y)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} should not be None."
        assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN."
