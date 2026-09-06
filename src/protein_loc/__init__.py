"""
protein_loc: Multi-label protein subcellular and sub-nuclear localization prediction.
"""

from protein_loc.data import (
    SUBCELLULAR_LOCATIONS,
    HPA_TEST_LOCATIONS,
    NUCLEAR_LOCATIONS,
    load_subcellular_data,
    load_nuclear_data,
    get_homology_partitions,
    create_kfold_splits,
    create_train_dev_split,
    compute_class_weights,
    smote_per_label,
    smote_union,
    smote_thresholded_label_propagation,
    get_dataloader,
    SubcellularDataset,
)
from protein_loc.models import (
    LogisticRegression,
    MLP,
    get_model,
)
from protein_loc.losses import (
    weighted_bce,
    pos_weighted_bce,
    focal_loss,
    ridge_penalty,
)
from protein_loc.thresholding import (
    optimize_thresholds_mcc,
    apply_thresholds,
    optimize_thresholds_oof,
)
from protein_loc.metrics import (
    compute_multilabel_metrics,
    format_metrics_table,
    format_per_class_table,
)
from protein_loc.training import (
    train_model,
    predict_probs,
    set_seed,
)
from protein_loc.evaluation import (
    run_cross_validation,
    evaluate_hpa_test,
    run_nuclear_experiment,
)
from protein_loc.visualization import (
    plot_cross_validated_mcc_heatmap,
    plot_training_curves,
    plot_error_breakdown_pie,
    plot_per_class_tp_fn_fp,
)

__all__ = [
    "SUBCELLULAR_LOCATIONS",
    "HPA_TEST_LOCATIONS",
    "NUCLEAR_LOCATIONS",
    "load_subcellular_data",
    "load_nuclear_data",
    "get_homology_partitions",
    "create_kfold_splits",
    "create_train_dev_split",
    "compute_class_weights",
    "smote_per_label",
    "smote_union",
    "smote_thresholded_label_propagation",
    "get_dataloader",
    "SubcellularDataset",
    "LogisticRegression",
    "MLP",
    "get_model",
    "weighted_bce",
    "pos_weighted_bce",
    "focal_loss",
    "ridge_penalty",
    "optimize_thresholds_mcc",
    "apply_thresholds",
    "optimize_thresholds_oof",
    "compute_multilabel_metrics",
    "format_metrics_table",
    "format_per_class_table",
    "train_model",
    "predict_probs",
    "set_seed",
    "run_cross_validation",
    "evaluate_hpa_test",
    "run_nuclear_experiment",
    "plot_cross_validated_mcc_heatmap",
    "plot_training_curves",
    "plot_error_breakdown_pie",
    "plot_per_class_tp_fn_fp",
]
