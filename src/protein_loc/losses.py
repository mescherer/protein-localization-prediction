"""
Loss functions and explicit regularization penalties.
"""

from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def standard_bce() -> nn.BCEWithLogitsLoss:
    """Standard unweighted multi-label Binary Cross-Entropy with Logits loss."""
    return nn.BCEWithLogitsLoss()


def weighted_bce(weights: torch.Tensor) -> nn.BCEWithLogitsLoss:
    """BCEWithLogitsLoss with element-wise class weights (rescales all terms per class)."""
    return nn.BCEWithLogitsLoss(weight=torch.as_tensor(weights, dtype=torch.float32))


def pos_weighted_bce(weights: torch.Tensor) -> nn.BCEWithLogitsLoss:
    """BCEWithLogitsLoss with positive-class upweighting (rescales positive terms only)."""
    return nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(weights, dtype=torch.float32))


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Sigmoid Focal Loss (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * ((1 - p_t) ** gamma) * ce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def ridge_penalty(lambda_reg: float = 1e-4) -> Callable[[nn.Module], torch.Tensor]:
    """L2 (Ridge) Frobenius regularizer over all Linear layer weight matrices.

    Computes: lambda_reg * sum_{m in Linear} ||W_m||_F^2

    Returns:
        Callable penalty(model) returning scalar regularization tensor.
    """
    def penalty(model: nn.Module) -> torch.Tensor:
        if lambda_reg == 0.0:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        total_sq_norm = sum(
            torch.linalg.matrix_norm(layer.weight) ** 2
            for layer in model.modules()
            if isinstance(layer, nn.Linear)
        )
        return lambda_reg * total_sq_norm

    return penalty
