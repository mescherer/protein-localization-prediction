"""
Multi-label evaluation metrics suite (Exact-match, Jaccard, F1, and Matthews Correlation Coefficient).
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    hamming_loss,
    jaccard_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)


def compute_multilabel_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    locations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute overall multi-label and per-class localization metrics.

    Args:
        y_true: (N, C) binary ground truth array/tensor.
        y_pred: (N, C) binary predicted array/tensor.
        locations: Optional list of class names (length C).

    Returns:
        Dictionary containing:
            - exact_match: float
            - hamming_accuracy: float
            - jaccard: float
            - micro_f1: float
            - macro_f1: float
            - mean_mcc: float (mean across classes with valid variation)
            - per_class: Dict[str, Dict[str, float]]
    """
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    num_samples, num_classes = y_true.shape
    if locations is None:
        locations = [f"Class_{i}" for i in range(num_classes)]

    exact_match = float(np.all(y_pred == y_true, axis=1).mean())
    hamming_acc = float(1.0 - hamming_loss(y_true, y_pred))
    jaccard = float(jaccard_score(y_true, y_pred, average="samples", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    valid_mccs: List[float] = []

    for i, loc in enumerate(locations):
        y_c = y_true[:, i]
        p_c = y_pred[:, i]
        acc = float((y_c == p_c).mean())
        prec = float(precision_score(y_c, p_c, zero_division=0))
        rec = float(recall_score(y_c, p_c, zero_division=0))
        f1 = float(f1_score(y_c, p_c, zero_division=0))

        if len(np.unique(y_c)) < 2:
            mcc = 0.0
        else:
            try:
                mcc = float(matthews_corrcoef(y_c, p_c))
                valid_mccs.append(mcc)
            except Exception:
                mcc = 0.0

        per_class_metrics[loc] = {
            "accuracy": acc,
            "mcc": mcc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": int(y_c.sum()),
        }

    mean_mcc = float(np.mean(valid_mccs)) if valid_mccs else 0.0

    return {
        "exact_match": exact_match,
        "hamming_accuracy": hamming_acc,
        "jaccard": jaccard,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "mean_mcc": mean_mcc,
        "per_class": per_class_metrics,
    }


def format_metrics_table(results_dict: Dict[str, Any]) -> pd.DataFrame:
    """Format evaluation metrics into a clean, presentation-ready DataFrame."""
    overall_rows = [
        {"Metric": "Exact-match Accuracy", "Value": f"{results_dict['exact_match']:.4f}"},
        {"Metric": "Hamming Accuracy", "Value": f"{results_dict['hamming_accuracy']:.4f}"},
        {"Metric": "Jaccard Index (Samples)", "Value": f"{results_dict['jaccard']:.4f}"},
        {"Metric": "Micro F1", "Value": f"{results_dict['micro_f1']:.4f}"},
        {"Metric": "Macro F1", "Value": f"{results_dict['macro_f1']:.4f}"},
        {"Metric": "Mean per-class MCC", "Value": f"{results_dict['mean_mcc']:.4f}"},
    ]
    return pd.DataFrame(overall_rows)


def format_per_class_table(results_dict: Dict[str, Any]) -> pd.DataFrame:
    """Format per-class metrics into a detailed DataFrame."""
    rows = []
    for loc, m in results_dict["per_class"].items():
        rows.append({
            "Location": loc,
            "Accuracy": f"{m['accuracy']:.4f}",
            "MCC": f"{m['mcc']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1": f"{m['f1']:.4f}",
            "Positives": m["support"],
        })
    return pd.DataFrame(rows)
