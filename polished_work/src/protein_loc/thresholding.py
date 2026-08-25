"""
Decision threshold optimization algorithms for multi-label classification.
"""

from typing import Optional, Callable, Dict, Any
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold


def optimize_thresholds_mcc(
    probs: np.ndarray,
    y_true: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Optimize per-class decision thresholds maximizing Matthews Correlation Coefficient (MCC).

    For each class c:
        tau_c* = argmax_{tau in grid} MCC(y_true[:, c], (probs[:, c] > tau))

    Args:
        probs: (N, C) array of predicted probabilities in [0, 1].
        y_true: (N, C) array of binary ground-truth labels in {0, 1}.
        grid: Optional 1D array of candidate thresholds. Defaults to np.linspace(0.01, 0.99, 99).

    Returns:
        thresholds: (C,) numpy array of optimal thresholds.
    """
    if torch.is_tensor(probs):
        probs = probs.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()

    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int32)

    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)

    num_classes = probs.shape[1]
    thresholds = np.full(num_classes, 0.5, dtype=np.float64)

    for c in range(num_classes):
        # If class has no positive or no negative instances, leave at 0.5
        if len(np.unique(y_true[:, c])) < 2:
            continue

        best_mcc = -2.0  # MCC range is [-1, 1]
        best_tau = 0.5
        y_c = y_true[:, c]
        p_c = probs[:, c]

        for tau in grid:
            pred_c = (p_c > tau).astype(np.int32)
            # Compute MCC
            try:
                mcc = matthews_corrcoef(y_c, pred_c)
                if mcc > best_mcc:
                    best_mcc = mcc
                    best_tau = tau
            except Exception:
                continue

        thresholds[c] = best_tau

    return thresholds


def apply_thresholds(
    probs: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Apply per-class decision thresholds to produce binary multi-label predictions.

    Args:
        probs: (N, C) array of probabilities.
        thresholds: (C,) array or scalar threshold.

    Returns:
        preds: (N, C) binary integer array in {0, 1}.
    """
    if torch.is_tensor(probs):
        probs = probs.detach().cpu().numpy()
    thresholds = np.asarray(thresholds, dtype=np.float64)
    return (probs > thresholds).astype(np.int32)


def optimize_thresholds_oof(
    model_factory: Callable[[], torch.nn.Module],
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_fn: Callable[[torch.nn.Module, np.ndarray, np.ndarray], torch.nn.Module],
    predict_fn: Callable[[torch.nn.Module, np.ndarray], np.ndarray],
    n_inner_folds: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Compute out-of-fold predictions on training data to select leakage-free thresholds.

    Performs an internal K-fold cross-validation on (x_train, y_train), accumulates
    out-of-fold probability estimates, and selects the per-class MCC-maximizing thresholds.

    Args:
        model_factory: Callable returning a fresh PyTorch model instance.
        x_train: (N_train, D) training feature embeddings.
        y_train: (N_train, C) training multi-label targets.
        train_fn: Callable (model, x_fold_tr, y_fold_tr) -> trained_model.
        predict_fn: Callable (model, x_fold_val) -> (N_val, C) probabilities.
        n_inner_folds: Number of inner CV folds (default: 3).
        seed: Random seed for inner fold splitting.

    Returns:
        thresholds: (C,) array of calibrated decision thresholds.
    """
    kf = KFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)
    oof_probs = np.zeros_like(y_train, dtype=np.float64)

    for fold_train_idx, fold_val_idx in kf.split(x_train):
        x_tr_inner, y_tr_inner = x_train[fold_train_idx], y_train[fold_train_idx]
        x_val_inner = x_train[fold_val_idx]

        model = model_factory()
        trained_model = train_fn(model, x_tr_inner, y_tr_inner)
        fold_probs = predict_fn(trained_model, x_val_inner)
        oof_probs[fold_val_idx] = fold_probs

    return optimize_thresholds_mcc(oof_probs, y_train)
