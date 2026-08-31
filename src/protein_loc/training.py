"""
Deterministic model training loop, history tracking, and prediction routines.
"""

from typing import Dict, List, Tuple, Optional, Callable, Union
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for PyTorch, NumPy, and random to ensure determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_probs(
    model: nn.Module,
    x: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Compute multi-label sigmoid probabilities for input features.

    Args:
        model: PyTorch classification model.
        x: (N, D) input feature array or tensor.
        device: PyTorch device.

    Returns:
        (N, C) numpy array of predicted probabilities in [0, 1].
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)

    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
    return probs.cpu().numpy()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_x: Optional[np.ndarray] = None,
    val_y: Optional[np.ndarray] = None,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epochs: int = 100,
    lr: float = 1e-3,
    regularizer: Optional[Callable[[nn.Module], torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train a PyTorch model tracking train and validation loss curves.

    The loss tracked in history is the unregularized criterion loss to ensure
    train and validation curves are directly comparable across regularizer choices.

    Args:
        model: PyTorch model.
        train_loader: DataLoader for training split.
        val_x: Optional validation features array for validation loss tracking.
        val_y: Optional validation targets array.
        criterion: Loss function (defaults to nn.BCEWithLogitsLoss).
        optimizer: PyTorch optimizer (defaults to Adam with lr).
        epochs: Number of training epochs (default: 100).
        lr: Learning rate if default Adam optimizer is constructed.
        regularizer: Optional callable returning a regularization penalty tensor.
        device: PyTorch device.
        verbose: If True, prints epoch progress.

    Returns:
        model: Trained PyTorch model in eval mode.
        history: Dict with "train_loss" and "val_loss" lists.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    has_val = val_x is not None and val_y is not None
    if has_val:
        val_x_t = torch.as_tensor(val_x, dtype=torch.float32, device=device)
        val_y_t = torch.as_tensor(val_y, dtype=torch.float32, device=device)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        n_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            objective = loss + regularizer(model) if regularizer is not None else loss
            objective.backward()
            optimizer.step()

            batch_size = x_batch.size(0)
            running_train_loss += loss.item() * batch_size
            n_samples += batch_size

        epoch_train_loss = running_train_loss / max(n_samples, 1)
        history["train_loss"].append(epoch_train_loss)

        if has_val:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_x_t)
                val_loss = criterion(val_logits, val_y_t).item()
            history["val_loss"].append(val_loss)

        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            val_str = f", Val Loss: {history['val_loss'][-1]:.4f}" if has_val else ""
            print(f"Epoch {epoch+1:3d}/{epochs} - Train Loss: {epoch_train_loss:.4f}{val_str}")

    model.eval()
    return model, history
