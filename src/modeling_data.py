"""Assay-labeled, common-row modeling cohorts and immutable batch partitions."""

from dataclasses import dataclass, field
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from src.data_integration import IntegrationResult, SENSOR_RANGES
from src.preprocessing import validate_axis
from src.demo_data import source_provenance
from src.input_policy import make_input_policy


TARGETS = {"Glucose": "glucose_reference_g_l", "Lactate": "lactate_reference_g_l"}
SENSOR_COLUMNS = tuple(SENSOR_RANGES)


@dataclass
class ModelingDataset:
    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame
    axis: np.ndarray
    target: str
    cohort_id: str
    excluded: pd.DataFrame
    data_provenance: dict = field(default_factory=dict)
    input_policy: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BatchPartition:
    training_batches: tuple[str, ...]
    holdout_batches: tuple[str, ...]

    def validate(self, batches):
        observed = set(batches)
        training, holdout = set(self.training_batches), set(self.holdout_batches)
        if len(training) != len(self.training_batches) or len(holdout) != len(self.holdout_batches):
            raise ValueError("Batch partitions must not contain duplicate IDs.")
        if len(training) < 3 or not holdout or training & holdout or training | holdout != observed:
            raise ValueError("Use at least three training batches and disjoint held-out batches covering the cohort exactly.")

    def masks(self, metadata):
        self.validate(metadata["batch_id"].unique())
        return metadata["batch_id"].isin(self.training_batches).to_numpy(), metadata["batch_id"].isin(self.holdout_batches).to_numpy()


def fixed_batch_partition(batches, holdout_count=2):
    """Reserve the last sorted batch IDs, identically for every feature/model."""
    batches = sorted(set(batches))
    if not isinstance(holdout_count, int) or holdout_count < 1 or len(batches) < holdout_count + 3:
        raise ValueError("Need at least three training batches plus the requested holdout batches.")
    partition = BatchPartition(tuple(batches[:-holdout_count]), tuple(batches[-holdout_count:]))
    partition.validate(batches)
    return partition


def modeling_dataset(result: IntegrationResult, target="Glucose") -> ModelingDataset:
    """Use actual reference assays and one complete common cohort for all features.

    No simulated truth, elapsed time, scenario label, reference-result timestamp,
    or arbitrary extra column becomes an input. All inputs are Raman intensities
    and five declared sensors. Exclusion is explicit, not imputation.
    """
    if target not in TARGETS:
        raise ValueError("Prediction target must be Glucose or Lactate.")
    pairs = result.reference_pairs.copy().reset_index(drop=True)
    columns = [column for column in pairs if column.startswith("raman_") and column != "raman_source_row"]
    axis = validate_axis([float(column.removeprefix("raman_")) for column in columns], uniform=True)
    if len(axis) < 11:
        raise ValueError("Modeling requires at least eleven uniformly spaced Raman channels.")
    if pairs.empty:
        raise ValueError("No eligible offline assay pairs are available for modeling.")
    spectral = pairs[columns].to_numpy(dtype=float)
    sensors = pairs[[f"sensor_{column}" for column in SENSOR_COLUMNS]].to_numpy(dtype=float)
    # A secondary assay may be unavailable. All feature/model cases for the
    # chosen response still use one common row cohort; no label is synthesized.
    reference_columns = list(TARGETS.values())
    target_reference = pairs[TARGETS[target]].to_numpy(dtype=float)
    conditions = {
        "No matched spectrum": ~pairs["spectrum_matched"].astype("boolean").fillna(False).to_numpy(dtype=bool),
        "Spectrum reused by multiple assays": pairs["spectrum_reused"].astype("boolean").fillna(False).to_numpy(dtype=bool),
        "Missing or non-finite Raman intensity": ~np.isfinite(spectral).all(axis=1),
        "Missing or non-finite sensor input": ~np.isfinite(sensors).all(axis=1),
        "Sensor range flag": pairs["sensor_range_flag"].astype("boolean").fillna(False).to_numpy(dtype=bool),
        "Future sensor reading": pairs["sensor_time_delta_minutes"].gt(0).to_numpy(),
        "Invalid or missing target reference": ~np.isfinite(target_reference) | (target_reference < 0),
    }
    reasons = ["; ".join(reason for reason, mask in conditions.items() if mask[index]) for index in range(len(pairs))]
    accepted = np.array([not reason for reason in reasons])
    excluded = pairs.loc[~accepted, ["batch_id", "assay_sample_id", "spectrum_sample_id"]].copy()
    excluded["reason"] = np.array(reasons)[~accepted]
    if not accepted.any():
        raise ValueError("No complete, unambiguous assay/spectral/sensor rows remain. Review Data Integration.")
    selected = pairs.loc[accepted].copy()
    row_ids = pd.Index(selected["batch_id"].astype(str) + "::" + selected["assay_sample_id"].astype(str), name="row_id")
    if not row_ids.is_unique:
        raise ValueError("Assay row IDs must be unique within the modeling cohort.")
    canonical_columns = [f"raman_{value:.12g}" for value in axis]
    X = pd.DataFrame(spectral[accepted], columns=canonical_columns, index=row_ids)
    X[list(SENSOR_COLUMNS)] = sensors[accepted]
    metadata_columns = ["batch_id", "assay_sample_id", "spectrum_sample_id", "collection_timestamp", "result_timestamp", "collection_time_delta_minutes"]
    metadata = selected[metadata_columns].copy()
    metadata.index = row_ids
    starts = pd.to_datetime(metadata["batch_id"].map(result.processed.batches.loc[result.processed.batches["_eligible"]].set_index("batch_id")["start_timestamp"]), utc=True)
    metadata["elapsed_time_h"] = (metadata["collection_timestamp"] - starts).dt.total_seconds() / 3600
    # Hash both available assay columns and the accepted common rows, never one
    # feature recipe. Fully labeled targets share the same cohort identity.
    fingerprint = sha256()
    for frame in (X, metadata, selected[reference_columns].set_axis(row_ids)):
        fingerprint.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
        fingerprint.update(json.dumps(list(frame.columns)).encode())
    fingerprint.update(axis.tobytes())
    y = pd.Series(selected[TARGETS[target]].to_numpy(dtype=float), index=row_ids, name=TARGETS[target])
    return ModelingDataset(X, y, metadata, axis, target, fingerprint.hexdigest(), excluded.reset_index(drop=True),
        source_provenance(result.processed.batches), make_input_policy(result.config, result.sensor_ranges))
