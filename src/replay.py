"""Deterministic replay clock, causal source views, and per-observation audit rows."""

from dataclasses import asdict, dataclass, field
from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from src.modeling_data import SENSOR_COLUMNS, TARGETS
from src.monitoring import alarm_state, input_quality_errors, predict_observation


@dataclass
class ReplayBatch:
    inputs: pd.DataFrame
    metadata: pd.DataFrame
    axis: np.ndarray
    assays: pd.DataFrame


def replay_batch(dataset, batch_id):
    """Align observations by sample ID and batch, without reading simulated truth."""
    metadata = dataset.spectral_metadata.loc[dataset.spectral_metadata.batch_id == batch_id].copy()
    if metadata.empty:
        raise ValueError("Choose an existing synthetic batch.")
    metadata = metadata.sort_values("elapsed_time_h").reset_index(drop=True)
    if not metadata.sample_id.is_unique or metadata.measurement_timestamp.isna().any() or metadata.measurement_timestamp.duplicated().any():
        raise ValueError("Replay requires unique sample IDs and valid unique measurement times.")
    sensors = dataset.sensors.loc[dataset.sensors.batch_id == batch_id].set_index("sample_id")
    if not sensors.index.is_unique:
        raise ValueError("Sensor sample IDs are ambiguous within the batch.")
    ordered_sensors = sensors.reindex(metadata.sample_id)
    values = dataset.spectra.loc[metadata.sample_id].to_numpy(copy=True)
    columns = [f"raman_{value:.12g}" for value in dataset.wavenumbers_cm1]
    inputs = pd.DataFrame(values, columns=columns, index=pd.Index(metadata.sample_id, name="sample_id"))
    inputs[list(SENSOR_COLUMNS)] = ordered_sensors[list(SENSOR_COLUMNS)].to_numpy()
    metadata["sensor_timestamp"] = ordered_sensors.measurement_timestamp.to_numpy()
    assays = dataset.offline_assays.loc[dataset.offline_assays.batch_id == batch_id].copy()
    return ReplayBatch(inputs, metadata, dataset.wavenumbers_cm1.copy(), assays)


def replay_fault_example(batch):
    """A copy with missing pH, impossible DO, a missing channel, and baseline shift."""
    inputs = batch.inputs.copy(deep=True)
    spectral_columns = [column for column in inputs if column.startswith("raman_")]
    if len(inputs) > 3:
        inputs.iloc[3, inputs.columns.get_loc("ph")] = np.nan
    if len(inputs) > 7:
        inputs.iloc[7, inputs.columns.get_loc("dissolved_oxygen_pct")] = 145.0
    if len(inputs) > 10:
        inputs.iloc[10, inputs.columns.get_loc(spectral_columns[0])] = np.nan
    if len(inputs) > 12:
        inputs.iloc[12, inputs.columns.get_indexer(spectral_columns)] += 2.0
    return ReplayBatch(inputs, batch.metadata.copy(deep=True), batch.axis.copy(), batch.assays.copy(deep=True))


@dataclass
class ReplayClock:
    cursor: int = -1
    running: bool = False
    rate: float = 1.0
    credit: float = 0.0
    last_tick: float | None = None

    def _validate(self, total_rows):
        if total_rows < 1 or not np.isfinite(self.rate) or self.rate <= 0:
            raise ValueError("Replay needs observations and a finite positive observation rate.")
        if self.cursor >= total_rows:
            raise ValueError("Replay position exceeds this batch; reset before changing data.")

    def start(self, now, total_rows):
        self._validate(total_rows)
        if not np.isfinite(now):
            raise ValueError("Replay time must be a finite monotonic-clock value.")
        if self.running:
            return
        if self.cursor < 0:
            self.cursor = 0
        self.running = self.cursor < total_rows - 1
        self.last_tick = float(now) if self.running else None

    def pause(self):
        self.running = False
        self.last_tick = None

    def tick(self, now, total_rows):
        self._validate(total_rows)
        if not np.isfinite(now):
            raise ValueError("Replay time must be a finite monotonic-clock value.")
        if not self.running:
            return self.cursor
        if self.last_tick is None:
            self.last_tick = float(now)
            return self.cursor
        if float(now) < self.last_tick:
            return self.cursor
        elapsed = float(now) - self.last_tick
        self.last_tick = float(now)
        self.credit += elapsed * self.rate
        steps = int(np.floor(self.credit + 1e-10))
        self.credit = max(0.0, self.credit - steps)
        self.cursor = min(total_rows - 1, self.cursor + steps)
        if self.cursor == total_rows - 1:
            self.pause()
            self.credit = 0.0
        return self.cursor

    def set_rate(self, rate, now, total_rows):
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("Observation rate must be finite and positive.")
        self.tick(now, total_rows)
        self.rate = float(rate)

    def step(self, total_rows):
        self._validate(total_rows)
        if self.running:
            raise ValueError("Pause replay before stepping manually.")
        self.cursor = min(total_rows - 1, self.cursor + 1)
        self.credit = 0.0
        return self.cursor

    def reset(self):
        self.cursor, self.credit, self.last_tick, self.running = -1, 0.0, None, False


def known_assays(batch, cutoff):
    """Reveal reference values at result time, retaining their original sample time."""
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("Reference availability requires a timezone-aware replay cutoff.")
    assays = batch.assays
    valid = assays.result_timestamp.notna() & assays.collection_timestamp.notna() & (assays.result_timestamp >= assays.collection_timestamp)
    return assays.loc[valid & (assays.result_timestamp <= cutoff)].copy()


def observation_record(batch, position, models, settings, mode="Replay"):
    metadata = batch.metadata.iloc[position]
    inputs = batch.inputs.iloc[[position]]
    quality = input_quality_errors(inputs, batch.inputs.columns, metadata.measurement_timestamp,
        metadata.sensor_timestamp, settings.max_sensor_age_minutes)
    record = {"sample_id": metadata.sample_id, "batch_id": metadata.batch_id,
        "elapsed_time_h": metadata.elapsed_time_h, "measurement_timestamp": metadata.measurement_timestamp,
        "sensor_timestamp": metadata.sensor_timestamp, "mode": mode, "quality_flags": "; ".join(quality)}
    for column in SENSOR_COLUMNS:
        value = inputs.iloc[0].get(column, np.nan)
        record[column] = value
        record[f"{column}_alarm"] = alarm_state(value, settings.sensor_limits[column])
    for target in TARGETS:
        result = predict_observation(models.get(target), target, inputs, batch.axis,
            metadata.measurement_timestamp, metadata.sensor_timestamp, settings)
        prefix = target.lower()
        record.update({
            f"{prefix}_prediction_g_l": result.prediction_g_l,
            f"{prefix}_status": result.status, f"{prefix}_flags": "; ".join(result.flags),
            f"{prefix}_model_version": result.model_version,
            f"{prefix}_validation_status": result.validation_status,
            f"{prefix}_max_abs_z": result.max_abs_z,
            f"{prefix}_prediction_time_utc": result.prediction_time_utc,
            f"{prefix}_latency_ms": result.latency_ms,
            f"{prefix}_alarm": alarm_state(result.prediction_g_l, settings.prediction_limits[target]) if result.status == "ok" else "Not assessed",
        })
    return record


@dataclass
class MonitoringSession:
    context_id: str
    clock: ReplayClock = field(default_factory=ReplayClock)
    records: dict = field(default_factory=dict)

    def history(self, batch, until_position, models, settings, mode="Replay"):
        """Infer each revealed observation once, preserving gaps and provenance."""
        if until_position < 0:
            return pd.DataFrame()
        if until_position >= len(batch.inputs):
            raise ValueError("History position exceeds the recorded batch.")
        for position in range(until_position + 1):
            if position not in self.records:
                self.records[position] = observation_record(batch, position, models, settings, mode)
        return pd.DataFrame([self.records[position] for position in range(until_position + 1)])

    def reset(self):
        self.clock.reset()
        self.records.clear()


def monitoring_report_zip(history, available_assays, settings, context):
    """Export revealed history and currently available references, never future data."""
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("observations.csv", history.to_csv(index=False))
        archive.writestr("available_assays.csv", available_assays.to_csv(index=False))
        manifest = {"phase": 5, "context": context, "settings": asdict(settings),
            "prediction_policy": "Saved pipeline only. Invalid/out-of-domain estimates are gaps, not imputed values.",
            "reference_policy": "Only results available at the displayed replay cutoff are exported.",
            "limitations": "Advisory synthetic replay. No equipment control, manufacturing validation, or calibrated multivariate domain limits."}
        archive.writestr("replay_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    return output.getvalue()
