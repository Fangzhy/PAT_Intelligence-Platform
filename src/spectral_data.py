"""Adapt Phase 1 or prepared Phase 2 Raman sources without mixing in truth."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_generation import SyntheticDataset
from src.data_integration import IntegrationResult
from src.preprocessing import validate_axis


@dataclass
class RamanCollection:
    intensities: np.ndarray
    wavenumbers_cm1: np.ndarray
    metadata: pd.DataFrame
    excluded: pd.DataFrame


def raman_from_simulation(dataset: SyntheticDataset) -> RamanCollection:
    metadata = dataset.spectral_metadata.copy().reset_index(drop=True)
    metadata = metadata.merge(dataset.batches[["batch_id", "scenario"]], on="batch_id", how="left", validate="many_to_one")
    values = dataset.spectra.loc[metadata["sample_id"]].to_numpy(copy=True)
    return RamanCollection(values, dataset.wavenumbers_cm1.copy(), metadata, pd.DataFrame(columns=["sample_id", "batch_id", "reason"]))


def raman_from_integration(result: IntegrationResult) -> RamanCollection:
    """Use only eligible, finite Raman rows; report exclusions instead of filling."""
    frame = result.processed.raman.copy()
    columns = [column for column in frame if column.startswith("raman_")]
    axis = validate_axis([float(column.removeprefix("raman_")) for column in columns])
    values = frame[columns].to_numpy(dtype=float, copy=True)
    finite = np.isfinite(values).all(axis=1)
    eligible = frame["_eligible"].to_numpy(dtype=bool)
    accepted = finite & eligible
    excluded = frame.loc[~accepted, ["sample_id", "batch_id", "_source_row"]].copy()
    excluded["reason"] = ["Invalid or ambiguous identity/time" if not valid else "Missing or non-finite Raman intensity" for valid in eligible[~accepted]]
    metadata = frame.loc[accepted, ["sample_id", "batch_id", "measurement_timestamp"]].reset_index(drop=True)
    batches = result.processed.batches
    batches = batches[batches["_eligible"]].copy()
    if "scenario" not in batches:
        batches["scenario"] = "Imported source"
    metadata = metadata.merge(batches[["batch_id", "start_timestamp", "scenario"]], on="batch_id", how="left", validate="many_to_one")
    metadata["elapsed_time_h"] = (metadata["measurement_timestamp"] - metadata["start_timestamp"]).dt.total_seconds() / 3600
    return RamanCollection(values[accepted], axis, metadata.drop(columns="start_timestamp"), excluded.reset_index(drop=True))
