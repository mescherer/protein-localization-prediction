"""
Data loading, preprocessing, partitioning, and PyTorch dataset utilities.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors


SUBCELLULAR_LOCATIONS: List[str] = [
    "Membrane",
    "Cytoplasm",
    "Nucleus",
    "Extracellular",
    "Cell membrane",
    "Mitochondrion",
    "Plastid",
    "Endoplasmic reticulum",
    "Lysosome/Vacuole",
    "Golgi apparatus",
    "Peroxisome",
]

HPA_TEST_LOCATIONS: List[str] = [
    "Cytoplasm",
    "Nucleus",
    "Cell membrane",
    "Mitochondrion",
    "Endoplasmic reticulum",
    "Lysosome/Vacuole",
    "Golgi apparatus",
    "Peroxisome",
]

NUCLEAR_LOCATIONS: List[str] = [
    "Nucleoplasm",
    "Nucleoli",
    "Nuclear Bodies",
    "Nuclear speckles",
    "Nuclear membrane",
    "Fibrillar center",
    "other",
]


class SubcellularDataset(Dataset):
    """PyTorch Dataset for protein embedding features and multi-label targets."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def get_embedding_columns(df: pd.DataFrame) -> List[str]:
    """Return sorted embedding column names starting with 'emb_'."""
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("No embedding columns matching prefix 'emb_' found in dataframe.")
    return sorted(emb_cols, key=lambda col: int(col.split("_")[1]) if col.split("_")[1].isdigit() else col)


def load_subcellular_data(
    csv_path: str,
    locations: Optional[List[str]] = None,
    is_test: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load model-ready subcellular dataset CSV.

    Args:
        csv_path: Path to CSV containing embeddings and label columns.
        locations: List of target label columns. Defaults to SUBCELLULAR_LOCATIONS.
        is_test: Whether this is the HPA test set (which contains 8 of 11 classes).
            If True, missing classes from SUBCELLULAR_LOCATIONS are aligned and zero-filled.

    Returns:
        df: Full dataframe with metadata, labels, and embeddings.
        X: (N, 1280) numpy array of float32 embeddings.
        y: (N, num_locations) numpy array of float32 binary targets.
    """
    if locations is None:
        locations = SUBCELLULAR_LOCATIONS

    df = pd.read_csv(csv_path)
    emb_cols = get_embedding_columns(df)
    X = df[emb_cols].to_numpy(dtype=np.float32)

    if is_test:
        # Check available locations in test set
        present_locs = [loc for loc in locations if loc in df.columns]
        y_df = df[present_locs]
        # Reindex to match full canonical label space with 0 fill
        y_df = y_df.reindex(columns=locations, fill_value=0.0)
        y = y_df.to_numpy(dtype=np.float32)
    else:
        for loc in locations:
            if loc not in df.columns:
                raise KeyError(f"Expected label column '{loc}' not found in {csv_path}.")
        y = df[locations].to_numpy(dtype=np.float32)

    return df, X, y


def load_nuclear_data(
    csv_path: str,
    locations: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load model-ready sub-nuclear dataset CSV.

    Args:
        csv_path: Path to sub-nuclear CSV.
        locations: List of sub-nuclear label columns. Defaults to NUCLEAR_LOCATIONS.

    Returns:
        df: Full dataframe.
        X: (N, 1280) numpy array of float32 embeddings.
        y: (N, 7) numpy array of float32 binary targets.
    """
    if locations is None:
        locations = NUCLEAR_LOCATIONS

    df = pd.read_csv(csv_path)
    emb_cols = get_embedding_columns(df)
    X = df[emb_cols].to_numpy(dtype=np.float32)

    for loc in locations:
        if loc not in df.columns:
            raise KeyError(f"Expected nuclear label '{loc}' not found in {csv_path}.")
    y = df[locations].to_numpy(dtype=np.float32)

    return df, X, y


def get_homology_partitions(
    df: pd.DataFrame,
    partition_col: str = "Partition",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Dynamically construct leave-one-partition-out fold indices.

    Extracts all unique partition IDs from `df[partition_col]` and yields
    (train_indices, val_indices) where each partition acts as validation once.

    Args:
        df: Dataframe with partition column.
        partition_col: Name of partition column.

    Returns:
        List of (train_idx, val_idx) numpy arrays.
    """
    if partition_col not in df.columns:
        raise KeyError(f"Partition column '{partition_col}' not present in dataframe.")

    unique_partitions = sorted(df[partition_col].dropna().unique())
    if len(unique_partitions) < 2:
        raise ValueError(f"At least 2 partitions required for cross-validation, found {unique_partitions}.")

    splits = []
    for val_part in unique_partitions:
        val_mask = (df[partition_col] == val_part).to_numpy()
        train_mask = ~val_mask
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        splits.append((train_idx, val_idx))

    return splits


def create_kfold_splits(
    n_samples: int,
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate deterministic random K-fold splits."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n_samples)))


def create_train_dev_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic 80/20 train/dev split."""
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def compute_class_weights(y_train: np.ndarray) -> torch.Tensor:
    """Compute balancing weights strictly on training set: N_samples / per-class positive count."""
    y = np.asarray(y_train, dtype=np.float64)
    pos_counts = y.sum(axis=0)
    pos_counts = np.maximum(pos_counts, 1.0)
    weights = y.shape[0] / pos_counts
    return torch.tensor(weights, dtype=torch.float32)


def _resolve_smote_target_count(y: np.ndarray, target_count: Optional[int] = None) -> int:
    """Resolve the synthetic positive target count for SMOTE-based oversampling."""
    if target_count is None:
        target_count = int(np.max(y.sum(axis=0)))
    return max(1, int(target_count))


def _get_minority_labels(y: np.ndarray, minority_labels: Optional[List[int]] = None) -> List[int]:
    """Return the minority label indices under a one-vs-rest positive-count definition."""
    y = np.asarray(y)
    positive_counts = y.sum(axis=0)
    majority_count = int(np.max(positive_counts))
    minority_idx = np.where(positive_counts < majority_count)[0].tolist()
    if minority_labels is not None:
        minority_idx = [idx for idx in minority_labels if idx in minority_idx]
    return minority_idx


def smote_per_label(
    X: np.ndarray,
    y: np.ndarray,
    target_count: Optional[int] = None,
    minority_labels: Optional[List[int]] = None,
    random_state: int = 42,
    return_metadata: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Oversample minority labels independently with standard one-vs-rest SMOTE.

    Synthetic samples inherit only the triggering minority label and set all other
    labels to 0. This provides a direct per-label SMOTE baseline for comparison.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    target_count = _resolve_smote_target_count(y, target_count)
    minority_idx = _get_minority_labels(y, minority_labels)

    if not minority_idx:
        if return_metadata:
            return X.copy(), y.copy(), {"minority_labels": [], "generated_samples": 0}
        return X.copy(), y.copy()

    synthetic_X_parts = []
    synthetic_y_parts = []
    metadata = {"minority_labels": minority_idx, "generated_samples": 0}

    for label_idx in minority_idx:
        binary_y = y[:, label_idx].astype(int)
        if np.all(binary_y == 1) or np.all(binary_y == 0):
            continue

        smote = SMOTE(sampling_strategy={1: target_count}, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, binary_y)
        n_original = X.shape[0]
        synthetic_mask = np.arange(X_resampled.shape[0]) >= n_original
        synthetic_X = X_resampled[synthetic_mask]
        if synthetic_X.size == 0:
            continue

        synthetic_Y = np.zeros((synthetic_X.shape[0], y.shape[1]), dtype=np.int8)
        synthetic_Y[:, label_idx] = 1
        synthetic_X_parts.append(synthetic_X)
        synthetic_y_parts.append(synthetic_Y)
        metadata["generated_samples"] += synthetic_X.shape[0]

    if not synthetic_X_parts:
        if return_metadata:
            return X.copy(), y.copy(), metadata
        return X.copy(), y.copy()

    X_aug = np.vstack([X.copy(), *synthetic_X_parts])
    y_aug = np.vstack([y.copy(), *synthetic_y_parts])

    if return_metadata:
        return X_aug, y_aug, metadata
    return X_aug, y_aug


def smote_union(
    X: np.ndarray,
    y: np.ndarray,
    target_count: Optional[int] = None,
    minority_labels: Optional[List[int]] = None,
    random_state: int = 42,
    return_metadata: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Oversample minority labels with union labels based on parent-class membership.

    For each synthetic sample, the label vector is the union of labels carried by the
    pair of minority-class samples used to construct the interpolation. This provides
    a second baseline that preserves multi-label structure without conditional
    propagation.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    target_count = _resolve_smote_target_count(y, target_count)
    minority_idx = _get_minority_labels(y, minority_labels)

    if not minority_idx:
        if return_metadata:
            return X.copy(), y.copy(), {"minority_labels": [], "generated_samples": 0}
        return X.copy(), y.copy()

    synthetic_X_parts = []
    synthetic_y_parts = []
    metadata = {"minority_labels": minority_idx, "generated_samples": 0}

    for label_idx in minority_idx:
        positive_idx = np.where(y[:, label_idx] == 1)[0]
        if len(positive_idx) < 2:
            continue

        n_needed = max(0, target_count - int(positive_idx.size))
        if n_needed <= 0:
            continue

        positive_X = X[positive_idx]
        neighbor_model = NearestNeighbors(n_neighbors=min(5, len(positive_X)), metric="euclidean")
        neighbor_model.fit(positive_X)

        synthetic_X = np.empty((n_needed, X.shape[1]), dtype=np.float64)
        synthetic_Y = np.zeros((n_needed, y.shape[1]), dtype=np.int8)

        for i in range(n_needed):
            base_pos = rng.integers(0, len(positive_idx))
            base_sample = positive_X[base_pos]
            neighbor_indices = neighbor_model.kneighbors(base_sample.reshape(1, -1), return_distance=False)[0]
            neighbor_indices = [idx for idx in neighbor_indices if idx != base_pos]
            if not neighbor_indices:
                neighbor_indices = [base_pos]
            neighbor_pos = neighbor_indices[rng.integers(0, len(neighbor_indices))]
            neighbor_sample = positive_X[neighbor_pos]
            alpha = rng.random()
            synthetic_sample = base_sample + alpha * (neighbor_sample - base_sample)
            synthetic_X[i] = synthetic_sample

            parent_union = np.logical_or(y[positive_idx[base_pos]], y[positive_idx[neighbor_pos]]).astype(np.int8)
            synthetic_Y[i] = parent_union

        synthetic_X_parts.append(synthetic_X)
        synthetic_y_parts.append(synthetic_Y)
        metadata["generated_samples"] += synthetic_X.shape[0]

    if not synthetic_X_parts:
        if return_metadata:
            return X.copy(), y.copy(), metadata
        return X.copy(), y.copy()

    X_aug = np.vstack([X.copy(), *synthetic_X_parts])
    y_aug = np.vstack([y.copy(), *synthetic_y_parts])

    if return_metadata:
        return X_aug, y_aug, metadata
    return X_aug, y_aug


def smote_thresholded_label_propagation(
    X: np.ndarray,
    y: np.ndarray,
    co_occurrence: np.ndarray,
    threshold: float = 0.5,
    target_count: Optional[int] = None,
    minority_labels: Optional[List[int]] = None,
    random_state: int = 42,
    return_metadata: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Oversample minority labels with conditional label propagation using a co-occurrence matrix.

    Synthetic samples inherit the triggering label plus any other labels j for which
    co_occurrence[label_idx, j] >= threshold. If multiple minority-label campaigns
    generate the same synthetic point, the function raises an informative error rather
    than silently deduplicating the sample.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    co_occurrence = np.asarray(co_occurrence, dtype=np.float64)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    if co_occurrence.shape != (y.shape[1], y.shape[1]):
        raise ValueError(
            f"co_occurrence must have shape ({y.shape[1]}, {y.shape[1]}), got {co_occurrence.shape}."
        )

    target_count = _resolve_smote_target_count(y, target_count)
    minority_idx = _get_minority_labels(y, minority_labels)

    if not minority_idx:
        if return_metadata:
            return X.copy(), y.copy(), {"minority_labels": [], "generated_samples": 0, "duplicate_count": 0}
        return X.copy(), y.copy()

    synthetic_X_parts = []
    synthetic_y_parts = []
    metadata = {"minority_labels": minority_idx, "generated_samples": 0, "duplicate_count": 0}
    seen_points = {}

    for label_idx in minority_idx:
        binary_y = y[:, label_idx].astype(int)
        if np.all(binary_y == 1) or np.all(binary_y == 0):
            continue

        smote = SMOTE(sampling_strategy={1: target_count}, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, binary_y)
        n_original = X.shape[0]
        synthetic_mask = np.arange(X_resampled.shape[0]) >= n_original
        synthetic_X = X_resampled[synthetic_mask]
        if synthetic_X.size == 0:
            continue

        synthetic_Y = np.zeros((synthetic_X.shape[0], y.shape[1]), dtype=np.int8)
        synthetic_Y[:, label_idx] = 1
        for other_label in range(y.shape[1]):
            if other_label == label_idx:
                continue
            synthetic_Y[:, other_label] = (co_occurrence[label_idx, other_label] >= threshold).astype(np.int8)

        for i in range(synthetic_X.shape[0]):
            key = tuple(np.round(synthetic_X[i], 12).tolist())
            if key in seen_points:
                metadata["duplicate_count"] += 1
                seen_points[key].append(label_idx)
            else:
                seen_points[key] = [label_idx]

        synthetic_X_parts.append(synthetic_X)
        synthetic_y_parts.append(synthetic_Y)
        metadata["generated_samples"] += synthetic_X.shape[0]

    if metadata["duplicate_count"] > 0:
        raise ValueError(
            "Detected duplicate synthetic points across minority-label campaigns. "
            "This requires a handling decision before proceeding."
        )

    if not synthetic_X_parts:
        if return_metadata:
            return X.copy(), y.copy(), metadata
        return X.copy(), y.copy()

    X_aug = np.vstack([X.copy(), *synthetic_X_parts])
    y_aug = np.vstack([y.copy(), *synthetic_y_parts])

    if return_metadata:
        return X_aug, y_aug, metadata
    return X_aug, y_aug


def get_dataloader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Construct PyTorch DataLoader from feature and target numpy arrays."""
    dataset = SubcellularDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
