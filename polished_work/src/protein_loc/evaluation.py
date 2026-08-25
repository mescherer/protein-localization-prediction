"""
Cross-validation, independent HPA test set evaluation, and sub-nuclear transfer benchmarks.
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from protein_loc.data import (
    SUBCELLULAR_LOCATIONS,
    NUCLEAR_LOCATIONS,
    get_embedding_columns,
    get_homology_partitions,
    create_kfold_splits,
    get_dataloader,
    compute_class_weights,
)
from protein_loc.training import train_model, predict_probs, set_seed
from protein_loc.thresholding import optimize_thresholds_mcc, apply_thresholds, optimize_thresholds_oof
from protein_loc.metrics import compute_multilabel_metrics


def run_cross_validation(
    df: pd.DataFrame,
    model_factory: Callable[[], nn.Module],
    locations: Optional[List[str]] = None,
    partition_col: str = "Partition",
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    criterion_factory: Optional[Callable[[torch.Tensor], nn.Module]] = None,
    regularizer: Optional[Callable[[nn.Module], torch.Tensor]] = None,
    calibrate_thresholds: bool = True,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute leave-one-partition-out cross-validation across all available partitions.

    If the dataframe contains explicit homology partitions (e.g. 1..5 or 1..2),
    it dynamically iterates over all unique partitions. If partition column is absent,
    it falls back to standard 5-fold cross-validation.

    Thresholding Protocol:
        Thresholds are calibrated on the training fold without consulting held-out validation labels,
        protecting against validation-set threshold overfitting.

    Returns:
        Dict with "fold_results", "mean", "std", and "per_class_mcc_mean".
    """
    if locations is None:
        locations = SUBCELLULAR_LOCATIONS

    emb_cols = get_embedding_columns(df)
    X = df[emb_cols].to_numpy(dtype=np.float32)
    y = df[locations].to_numpy(dtype=np.float32)

    # Determine splits dynamically
    if partition_col in df.columns and len(df[partition_col].dropna().unique()) >= 2:
        splits = get_homology_partitions(df, partition_col=partition_col)
    else:
        splits = create_kfold_splits(len(df), n_splits=5, seed=seed)

    fold_metrics: List[Dict[str, Any]] = []
    fold_thresholds: List[np.ndarray] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        if verbose:
            print(f"--- Running Fold {fold_idx + 1}/{len(splits)} (Train: {len(train_idx)}, Val: {len(val_idx)}) ---")

        set_seed(seed + fold_idx)
        x_tr, y_tr = X[train_idx], y[train_idx]
        x_val, y_val = X[val_idx], y[val_idx]

        train_loader = get_dataloader(x_tr, y_tr, batch_size=batch_size, shuffle=True)

        if criterion_factory is not None:
            class_weights = compute_class_weights(y_tr)
            criterion = criterion_factory(class_weights)
        else:
            criterion = nn.BCEWithLogitsLoss()

        model = model_factory()
        trained_model, _ = train_model(
            model=model,
            train_loader=train_loader,
            val_x=x_val,
            val_y=y_val,
            criterion=criterion,
            epochs=epochs,
            lr=lr,
            regularizer=regularizer,
            verbose=False,
        )

        val_probs = predict_probs(trained_model, x_val)

        if calibrate_thresholds:
            # Calibrate thresholds using training predictions
            # We train a quick inner predictor or use training set predictions to select thresholds
            tr_probs = predict_probs(trained_model, x_tr)
            tau = optimize_thresholds_mcc(tr_probs, y_tr)
        else:
            tau = np.full(len(locations), 0.5)

        fold_thresholds.append(tau)
        val_preds = apply_thresholds(val_probs, tau)
        metrics = compute_multilabel_metrics(y_val, val_preds, locations=locations)
        fold_metrics.append(metrics)

    # Aggregate metrics across folds
    scalar_keys = ["exact_match", "hamming_accuracy", "jaccard", "micro_f1", "macro_f1", "mean_mcc"]
    mean_metrics: Dict[str, float] = {}
    std_metrics: Dict[str, float] = {}

    for k in scalar_keys:
        vals = [fm[k] for fm in fold_metrics]
        mean_metrics[k] = float(np.mean(vals))
        std_metrics[k] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    per_class_mcc_mean: Dict[str, float] = {}
    per_class_mcc_std: Dict[str, float] = {}
    for loc in locations:
        mccs = [fm["per_class"][loc]["mcc"] for fm in fold_metrics]
        per_class_mcc_mean[loc] = float(np.mean(mccs))
        per_class_mcc_std[loc] = float(np.std(mccs, ddof=1)) if len(mccs) > 1 else 0.0

    return {
        "n_folds": len(splits),
        "fold_metrics": fold_metrics,
        "mean": mean_metrics,
        "std": std_metrics,
        "per_class_mcc_mean": per_class_mcc_mean,
        "per_class_mcc_std": per_class_mcc_std,
        "thresholds_mean": np.mean(fold_thresholds, axis=0),
        "locations": locations,
    }


def evaluate_hpa_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_factory: Callable[[], nn.Module],
    locations: Optional[List[str]] = None,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    criterion_factory: Optional[Callable[[torch.Tensor], nn.Module]] = None,
    regularizer: Optional[Callable[[nn.Module], torch.Tensor]] = None,
    n_seeds: int = 5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train on the complete development set and evaluate on the independent HPA test set.

    Trains across n_seeds random initializations, applying training-calibrated thresholds.
    Reports individual seeds, mean +- std, and ensembled probability predictions.

    Args:
        train_df: Full development dataframe.
        test_df: Independent HPA test dataframe.
        model_factory: Model constructor.
        n_seeds: Number of random seeds (default: 5).

    Returns:
        Dict with seed_metrics, mean_metrics, std_metrics, and ensemble_metrics.
    """
    if locations is None:
        locations = SUBCELLULAR_LOCATIONS

    emb_cols = get_embedding_columns(train_df)
    x_tr = train_df[emb_cols].to_numpy(dtype=np.float32)
    y_tr = train_df[locations].to_numpy(dtype=np.float32)

    x_test = test_df[emb_cols].to_numpy(dtype=np.float32)
    # Reindex test labels to 11-class space (8 classes present, 3 absent filled with 0)
    present_locs = [loc for loc in locations if loc in test_df.columns]
    y_test_df = test_df[present_locs].reindex(columns=locations, fill_value=0.0)
    y_test = y_test_df.to_numpy(dtype=np.float32)

    seed_metrics: List[Dict[str, Any]] = []
    test_prob_list: List[np.ndarray] = []
    seed_thresholds: List[np.ndarray] = []

    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx
        set_seed(seed)
        train_loader = get_dataloader(x_tr, y_tr, batch_size=batch_size, shuffle=True)

        if criterion_factory is not None:
            class_weights = compute_class_weights(y_tr)
            criterion = criterion_factory(class_weights)
        else:
            criterion = nn.BCEWithLogitsLoss()

        model = model_factory()
        trained_model, _ = train_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            epochs=epochs,
            lr=lr,
            regularizer=regularizer,
            verbose=False,
        )

        tr_probs = predict_probs(trained_model, x_tr)
        tau = optimize_thresholds_mcc(tr_probs, y_tr)
        seed_thresholds.append(tau)

        test_probs = predict_probs(trained_model, x_test)
        test_prob_list.append(test_probs)

        test_preds = apply_thresholds(test_probs, tau)
        metrics = compute_multilabel_metrics(y_test, test_preds, locations=locations)
        seed_metrics.append(metrics)

    # Aggregate mean +- std across individual seeds
    scalar_keys = ["exact_match", "hamming_accuracy", "jaccard", "micro_f1", "macro_f1", "mean_mcc"]
    mean_metrics: Dict[str, float] = {}
    std_metrics: Dict[str, float] = {}
    for k in scalar_keys:
        vals = [sm[k] for sm in seed_metrics]
        mean_metrics[k] = float(np.mean(vals))
        std_metrics[k] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    per_class_mcc_mean: Dict[str, float] = {}
    per_class_mcc_std: Dict[str, float] = {}
    for loc in locations:
        mccs = [sm["per_class"][loc]["mcc"] for sm in seed_metrics]
        per_class_mcc_mean[loc] = float(np.mean(mccs))
        per_class_mcc_std[loc] = float(np.std(mccs, ddof=1)) if len(mccs) > 1 else 0.0

    # Ensemble evaluation (mean of predicted probabilities across 5 seeds)
    ensemble_probs = np.mean(test_prob_list, axis=0)
    ensemble_tau = np.mean(seed_thresholds, axis=0)
    ensemble_preds = apply_thresholds(ensemble_probs, ensemble_tau)
    ensemble_metrics = compute_multilabel_metrics(y_test, ensemble_preds, locations=locations)

    return {
        "n_seeds": n_seeds,
        "seed_metrics": seed_metrics,
        "mean": mean_metrics,
        "std": std_metrics,
        "per_class_mcc_mean": per_class_mcc_mean,
        "per_class_mcc_std": per_class_mcc_std,
        "ensemble_metrics": ensemble_metrics,
        "locations": locations,
    }


def run_nuclear_experiment(
    df: pd.DataFrame,
    model_factory: Callable[[], nn.Module],
    locations: Optional[List[str]] = None,
    test_size: float = 0.2,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    regularizer: Optional[Callable[[nn.Module], torch.Tensor]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train model on 80% of sub-nuclear data and evaluate on 20% holdout set."""
    if locations is None:
        locations = NUCLEAR_LOCATIONS

    emb_cols = get_embedding_columns(df)
    X = df[emb_cols].to_numpy(dtype=np.float32)
    y = df[locations].to_numpy(dtype=np.float32)

    set_seed(seed)
    from sklearn.model_selection import train_test_split
    x_tr, x_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, random_state=seed)

    train_loader = get_dataloader(x_tr, y_tr, batch_size=batch_size, shuffle=True)
    model = model_factory()

    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_x=x_val,
        val_y=y_val,
        criterion=nn.BCEWithLogitsLoss(),
        epochs=epochs,
        lr=lr,
        regularizer=regularizer,
    )

    tr_probs = predict_probs(trained_model, x_tr)
    tau = optimize_thresholds_mcc(tr_probs, y_tr)

    val_probs = predict_probs(trained_model, x_val)
    val_preds = apply_thresholds(val_probs, tau)

    metrics = compute_multilabel_metrics(y_val, val_preds, locations=locations)

    return {
        "metrics": metrics,
        "history": history,
        "thresholds": tau,
        "locations": locations,
    }
