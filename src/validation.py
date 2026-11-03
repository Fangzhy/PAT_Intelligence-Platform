"""Regression metrics and auditable whole-batch cross-validation partitions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def regression_metrics(reference, prediction):
    reference, prediction = np.asarray(reference, dtype=float).reshape(-1), np.asarray(prediction, dtype=float).reshape(-1)
    if not len(reference) or reference.shape != prediction.shape or not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError("Reference and prediction must be nonempty, equal-length finite vectors.")
    residual = prediction - reference
    denominator = float(np.sum((reference - reference.mean()) ** 2))
    return {
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAE": float(np.mean(np.abs(residual))),
        "R2": float(1 - np.sum(residual**2) / denominator) if len(reference) >= 2 and denominator > 0 else np.nan,
        "Bias": float(residual.mean()),
    }


@dataclass
class BatchFold:
    number: int
    training_positions: np.ndarray
    validation_positions: np.ndarray
    training_batches: tuple[str, ...]
    validation_batches: tuple[str, ...]


def batch_folds(groups, n_splits=3):
    groups = np.asarray(groups, dtype=str)
    if len(groups) == 0 or (groups == "").any() or len(np.unique(groups)) < n_splits or n_splits < 2:
        raise ValueError("Cross-validation requires nonempty batch IDs and at least n_splits distinct training batches.")
    folds = []
    for number, (training, validation) in enumerate(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups), start=1):
        training_batches, validation_batches = tuple(sorted(set(groups[training]))), tuple(sorted(set(groups[validation])))
        if set(training_batches) & set(validation_batches):
            raise ValueError("Cross-validation batches overlap.")
        if len(training) < 2:
            raise ValueError("Every cross-validation training fold needs at least two assay rows.")
        folds.append(BatchFold(number, training, validation, training_batches, validation_batches))
    return folds


def fold_manifest(folds, row_ids):
    return pd.DataFrame([{
        "fold": fold.number, "training_batches": ", ".join(fold.training_batches),
        "validation_batches": ", ".join(fold.validation_batches),
        "training_rows": len(fold.training_positions), "validation_rows": len(fold.validation_positions),
        "training_row_ids": ";".join(np.asarray(row_ids)[fold.training_positions]),
        "validation_row_ids": ";".join(np.asarray(row_ids)[fold.validation_positions]),
    } for fold in folds])
