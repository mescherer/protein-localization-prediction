"""
Data loading, preprocessing, partitioning, and PyTorch dataset utilities.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split


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


def get_dataloader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Construct PyTorch DataLoader from feature and target numpy arrays."""
    dataset = SubcellularDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
