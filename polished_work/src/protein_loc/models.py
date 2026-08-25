"""
Neural network architectures for multi-label protein localization.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """Multi-label Logistic Regression (single linear layer to logits)."""

    def __init__(self, input_dim: int = 1280, num_classes: int = 11):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLP(nn.Module):
    """Multi-Layer Perceptron with arbitrary hidden layers and ReLU activations.

    Standard configurations:
    - 1-Hidden-Layer: hidden_dims=(64,)      (1280 -> 64 -> ReLU -> 11)
    - 2-Hidden-Layer: hidden_dims=(128, 64)  (1280 -> 128 -> ReLU -> 64 -> ReLU -> 11)
    """

    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 11,
        hidden_dims: Tuple[int, ...] = (64,),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = tuple(hidden_dims)
        self.dropout_rate = dropout

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def get_model(
    model_type: str,
    input_dim: int = 1280,
    num_classes: int = 11,
    **kwargs,
) -> nn.Module:
    """Factory helper to build model instances by name."""
    model_type = model_type.lower().strip()
    if model_type in ("logreg", "logistic_regression", "linear"):
        return LogisticRegression(input_dim=input_dim, num_classes=num_classes)
    elif model_type in ("mlp_1h", "mlp64", "mlp_1_hidden"):
        return MLP(input_dim=input_dim, num_classes=num_classes, hidden_dims=(64,), **kwargs)
    elif model_type in ("mlp_2h", "mlp128_64", "mlp_2_hidden"):
        return MLP(input_dim=input_dim, num_classes=num_classes, hidden_dims=(128, 64), **kwargs)
    elif model_type == "mlp":
        hidden_dims = kwargs.pop("hidden_dims", (64,))
        return MLP(input_dim=input_dim, num_classes=num_classes, hidden_dims=hidden_dims, **kwargs)
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Expected 'logreg', 'mlp_1h', 'mlp_2h', or 'mlp'.")
