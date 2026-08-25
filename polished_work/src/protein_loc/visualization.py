"""
Publication-quality plotting and visualization routines.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

from protein_loc.data import SUBCELLULAR_LOCATIONS


def plot_cross_validated_mcc_heatmap(
    mcc_data: Union[np.ndarray, pd.DataFrame],
    model_names: Optional[List[str]] = None,
    locations: Optional[List[str]] = None,
    title: str = "Cross-validated per-class MCC by model (whole-cell, 11 compartments)",
    footnote: str = "Models: LR = logistic regression, M1 = 1-hidden MLP (64), M2 = 2-hidden MLP (128,64). Symbols: ‡ class-reweighted BCE, ★ focal loss, † ridge (λ=1e-4). All at MCC-optimized thresholds; ctrl = random-label control.",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (12, 6.5),
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Render publication-grade annotated heatmap matrix of per-class MCC across models.

    Args:
        mcc_data: (M, C) matrix or DataFrame of mean MCC scores.
        model_names: List of M model label strings.
        locations: List of C compartment label strings.
        title: Figure title.
        footnote: Explanatory text at bottom of figure.
        vmin: Minimum colorbar value.
        vmax: Maximum colorbar value.
        cmap: Matplotlib colormap (default: 'viridis').
        figsize: Figure dimensions (width, height).
        save_path: Optional file path to save PDF/PNG figure.

    Returns:
        (fig, ax) Matplotlib figure and axes.
    """
    if isinstance(mcc_data, pd.DataFrame):
        if model_names is None:
            model_names = list(mcc_data.index)
        if locations is None:
            locations = list(mcc_data.columns)
        matrix = mcc_data.to_numpy(dtype=np.float64)
    else:
        matrix = np.asarray(mcc_data, dtype=np.float64)
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(matrix.shape[0])]
        if locations is None:
            locations = SUBCELLULAR_LOCATIONS[: matrix.shape[1]]

    num_models, num_locs = matrix.shape

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Plot heatmap
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Ticks and labels
    ax.set_xticks(np.arange(num_locs))
    ax.set_yticks(np.arange(num_models))
    ax.set_xticklabels(locations, rotation=40, ha="right", fontsize=9.5)
    ax.set_yticklabels(model_names, fontsize=10.5)

    # Annotate numeric values inside each cell
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for i in range(num_models):
        for j in range(num_locs):
            val = matrix[i, j]
            # Use light text for dark cells, dark text for bright cells
            rgba = plt.get_cmap(cmap)(norm(val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.55 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.5,
                fontweight="normal",
            )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("per-class MCC (mean over folds)", fontsize=10)
    cbar.ax.tick_params(labelsize=9.5)

    # Title & Layout
    ax.set_title(title, fontsize=12.5, pad=12, fontweight="medium")

    # Footnote at bottom
    if footnote:
        fig.text(
            0.5,
            0.02,
            footnote,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#333333",
            wrap=True,
        )

    plt.tight_layout(rect=[0, 0.06, 1, 0.98])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training and Validation Loss",
    figsize: Tuple[float, float] = (8, 4),
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot training vs validation loss curves over epochs."""
    train_loss = history["train_loss"]
    val_loss = history.get("val_loss", [])
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", lw=2)
    if val_loss:
        ax.plot(epochs, val_loss, label="Validation Loss", color="#ff7f0e", lw=2)

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="medium")
    ax.legend(fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def plot_error_breakdown_pie(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    title: str = "Multi-label Prediction Error Categories",
    figsize: Tuple[float, float] = (6, 5),
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot multi-label error category pie (Exact match vs Under/Over prediction)."""
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    diff = y_true - y_pred
    has_pos = (diff == 1).any(axis=1)   # missed at least one true label (underprediction)
    has_neg = (diff == -1).any(axis=1)  # predicted false positive label (overprediction)

    sizes = [
        int((~has_pos & ~has_neg).sum()),  # Exact match
        int((has_pos & ~has_neg).sum()),   # Underprediction only
        int((~has_pos & has_neg).sum()),   # Overprediction only
        int((has_pos & has_neg).sum()),    # Wrong both directions
    ]
    labels = ["Exact match", "Underprediction", "Overprediction", "Both error types"]
    colors = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728"]

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"fontsize": 9.5},
    )
    ax.set_title(title, fontsize=11, pad=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def plot_per_class_tp_fn_fp(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    locations: Optional[List[str]] = None,
    title: str = "Per-Compartment TP / FN / FP Breakdown",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, Any]:
    """Plot per-compartment diagnostic pie breakdown of True Positives, False Negatives, False Positives."""
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    if locations is None:
        locations = SUBCELLULAR_LOCATIONS[: y_true.shape[1]]

    num_locs = len(locations)
    cols = 4
    rows = (num_locs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.2 * rows), dpi=300)
    axes_flat = axes.flatten()

    cat_labels = ["TP", "FN", "FP"]
    colors = ["#5590D9", "#FF8C00", "#D94040"]

    for i, loc in enumerate(locations):
        t = y_true[:, i]
        p = y_pred[:, i]

        tp = int(((t == 1) & (p == 1)).sum())
        fn = int(((t == 1) & (p == 0)).sum())
        fp = int(((t == 0) & (p == 1)).sum())
        total = tp + fn + fp

        ax = axes_flat[i]
        if total > 0:
            ax.pie(
                [tp, fn, fp],
                labels=cat_labels,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
                textprops={"fontsize": 8},
            )
        else:
            ax.text(0.5, 0.5, "No positives", ha="center", va="center", fontsize=9)

        ax.set_title(f"{loc}\n(n={total})", fontsize=10.5)

    for j in range(num_locs, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, axes
