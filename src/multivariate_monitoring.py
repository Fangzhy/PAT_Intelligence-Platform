"""Reference-only PCA process monitoring with separate empirical limit batches.

Sensors are pooled over elapsed time. Limits are descriptive observation-level
quantiles, not calibrated batch false-alarm probabilities for autocorrelated data.
Scenario labels select known nominal references and annotate evaluation only.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
from numbers import Integral
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data_integration import SENSOR_RANGES


SENSOR_COLUMNS = tuple(SENSOR_RANGES)
KEY_COLUMNS = ("sample_id", "batch_id", "elapsed_time_h", "measurement_timestamp")
NOMINAL_SCENARIO = "Nominal simulation"
LIMIT_METHOD = "Separate nominal batches; per-statistic empirical quantile (higher)"
LIMITATIONS = (
    "Synthetic educational monitoring; no manufacturing validation or equipment control.",
    "Pooled batch trajectories: no time alignment, phase-specific centering, or dynamic PCA.",
    "Time-correlated observations and few independent batches: quantile is not a guaranteed false-alarm probability.",
    "The two charts and repeated observations have no family-wise or batch-level error control.",
    "Scenario labels are simulation design, not anomaly predictions or product-quality labels.",
    "Changing components or limits after reviewing evaluation results makes further comparison exploratory.",
)


@dataclass(frozen=True)
class ReferencePartition:
    fit_batches: tuple[str, ...]
    limit_batches: tuple[str, ...]
    evaluation_batches: tuple[str, ...]


@dataclass
class ProcessMonitoringModel:
    scaler: StandardScaler
    pca: PCA
    t2_limit: float
    spe_limit: float
    quantile: float
    partition: ReferencePartition
    reference_audit: pd.DataFrame
    model_id: str


def _batch_metadata(batches):
    required = {"batch_id", "scenario", "disturbance_start_h"}
    if not required.issubset(batches.columns):
        raise ValueError("Batch metadata needs batch_id, scenario, and disturbance_start_h.")
    if batches.empty or batches.batch_id.isna().any() or batches.batch_id.duplicated().any():
        raise ValueError("Batch metadata needs unique, nonmissing batch IDs.")
    return batches.set_index("batch_id")


def default_reference_partition(batches):
    """Reserve entire nominal batches for fitting, limits, and evaluation."""
    metadata = _batch_metadata(batches)
    nominal = sorted(metadata.index[metadata.scenario.eq(NOMINAL_SCENARIO)])
    if len(nominal) < 3:
        raise ValueError("Generate at least three nominal batches on Overview to reserve separate fitting, limit, and evaluation batches.")
    fit, limits = tuple(nominal[:-2]), (nominal[-2],)
    evaluation = tuple(batch for batch in metadata.index if batch not in fit + limits)
    return ReferencePartition(fit, limits, evaluation)


def validate_reference_partition(partition, batches):
    metadata = _batch_metadata(batches)
    groups = (partition.fit_batches, partition.limit_batches, partition.evaluation_batches)
    if any(not group for group in groups):
        raise ValueError("Reserve nonempty fitting, limit-calibration, and evaluation batches.")
    flattened = [batch for group in groups for batch in group]
    if len(set(flattened)) != len(flattened):
        raise ValueError("Fitting, limit-calibration, and evaluation batches must be disjoint, without duplicates.")
    if set(flattened) != set(metadata.index):
        raise ValueError("Assign every known batch exactly once; unknown batch IDs are not allowed.")
    reference = list(partition.fit_batches + partition.limit_batches)
    if not metadata.loc[reference, "scenario"].eq(NOMINAL_SCENARIO).all():
        raise ValueError("Only known nominal batches may fit the PCA model or its limits.")
    if not metadata.loc[list(partition.evaluation_batches), "scenario"].eq(NOMINAL_SCENARIO).any():
        raise ValueError("Reserve at least one nominal batch for independent comparison.")


def process_observations(sensors):
    """Copy and screen sensor rows; retain invalid observations without imputation.

    Extra columns (including latent concentrations or scenario labels) are never
    PCA inputs. Index position and sample ID remain explicit through projection.
    """
    required = KEY_COLUMNS + SENSOR_COLUMNS
    if not set(required).issubset(sensors.columns) or sensors.columns.duplicated().any():
        raise ValueError("Supply unique columns for all sensor variables and sample/batch/time keys.")
    frame = sensors[list(required)].copy(deep=True).reset_index(drop=True)
    issues = [[] for _ in range(len(frame))]

    def flag(mask, message):
        for position in np.flatnonzero(np.asarray(mask)):
            issues[position].append(message)

    flag(frame.sample_id.isna() | frame.batch_id.isna(), "Missing sample or batch ID")
    flag(frame.sample_id.duplicated(keep=False), "Duplicate sample ID")
    elapsed = pd.to_numeric(frame.elapsed_time_h, errors="coerce")
    flag(~np.isfinite(elapsed) | (elapsed < 0), "Invalid elapsed time")
    frame["elapsed_time_h"] = elapsed
    timestamps = pd.to_datetime(frame.measurement_timestamp, errors="coerce", utc=True)
    flag(timestamps.isna(), "Invalid measurement timestamp")
    frame["measurement_timestamp"] = timestamps
    for column, (lower, upper) in SENSOR_RANGES.items():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        finite = np.isfinite(frame[column])
        flag(~finite, f"Missing/nonfinite {column}")
        flag(finite & ~frame[column].between(lower, upper), f"Implausible {column}")
    frame["quality_flags"] = ["; ".join(messages) for messages in issues]
    frame["status"] = np.where(frame.quality_flags.eq(""), "scored", "invalid_input")
    return frame


def pca_monitoring_statistics(scaler, pca, values):
    """T² uses retained sample-variance eigenvalues; Q is a sum, never an RMSE.

    Q and its additive per-sensor contributions are in standardized squared units.
    This operation projects using saved statistics and performs no fitting.
    """
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(SENSOR_COLUMNS) or not np.isfinite(matrix).all():
        raise ValueError("PCA monitoring needs a finite matrix of the five sensor columns in schema order.")
    if not len(matrix):
        return np.empty((0, pca.n_components_)), np.empty(0), np.empty(0), np.empty_like(matrix)
    scaled = scaler.transform(matrix)
    scores = pca.transform(scaled)
    residual = scaled - pca.inverse_transform(scores)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        t2 = np.sum(scores**2 / pca.explained_variance_, axis=1)
        contributions = residual**2
        spe = contributions.sum(axis=1)
    if not np.isfinite(scores).all() or not np.isfinite(t2).all() or not np.isfinite(spe).all():
        raise ValueError("PCA projection produced nonfinite monitoring statistics.")
    return scores, t2, spe, contributions


def fit_process_monitor(sensors, batches, *, partition=None, n_components=2, quantile=0.99):
    """Fit scale/PCA on fitting batches; obtain limits from separate nominal batches.

    No evaluation sensor value, label, or score determines fitting or thresholds.
    Invalid reference rows are explicitly audited and excluded, never repaired.
    """
    partition = partition or default_reference_partition(batches)
    validate_reference_partition(partition, batches)
    if isinstance(n_components, bool) or not isinstance(n_components, Integral) or not 1 <= n_components < len(SENSOR_COLUMNS):
        raise ValueError("Retain one to four PCA components, leaving a residual subspace for SPE.")
    if isinstance(quantile, bool) or not np.isfinite(quantile) or not 0.5 <= quantile < 1:
        raise ValueError("The empirical limit quantile must be finite and in [0.5, 1).")
    frame = process_observations(sensors)
    if not set(frame.batch_id.dropna()).issubset(set(batches.batch_id)):
        raise ValueError("Sensor rows contain unknown batch IDs.")
    valid = frame.status.eq("scored")
    reference_ids = partition.fit_batches + partition.limit_batches
    if any(not (valid & frame.batch_id.eq(batch)).any() for batch in reference_ids):
        raise ValueError("Every fitting and limit-calibration batch needs valid sensor rows.")
    fit = frame[valid & frame.batch_id.isin(partition.fit_batches)]
    calibration = frame[valid & frame.batch_id.isin(partition.limit_batches)]
    if len(fit) <= n_components or len(calibration) < 2:
        raise ValueError("Need more fitting rows than components and at least two valid limit-calibration rows.")
    values = fit[list(SENSOR_COLUMNS)].to_numpy(dtype=float)
    scaler = StandardScaler().fit(values)
    constant = scaler.var_ <= np.finfo(float).eps * np.maximum(1.0, scaler.mean_**2)
    if constant.any():
        raise ValueError("Constant reference sensors cannot be autoscaled: " + ", ".join(np.array(SENSOR_COLUMNS)[constant]))
    pca = PCA(n_components=n_components, svd_solver="full").fit(scaler.transform(values))
    if pca.explained_variance_[-1] <= pca.explained_variance_[0] * 1e-12:
        raise ValueError("Retained PCA eigenvalues are numerically singular; choose fewer components.")
    _, t2, spe, _ = pca_monitoring_statistics(scaler, pca, calibration[list(SENSOR_COLUMNS)].to_numpy())
    t2_limit = float(np.quantile(t2, quantile, method="higher"))
    spe_limit = float(np.quantile(spe, quantile, method="higher"))
    if min(t2_limit, spe_limit) <= 1e-12:
        raise ValueError("Reference calibration gives a degenerate zero monitoring limit; use more representative reference variation or fewer components.")
    audit = frame[frame.batch_id.isin(reference_ids)][list(KEY_COLUMNS) + ["status", "quality_flags"]].copy()
    audit["role"] = np.where(audit.batch_id.isin(partition.fit_batches), "PCA fitting", "Limit calibration")
    identity = sha256(json.dumps({"partition": asdict(partition), "components": int(n_components), "quantile": float(quantile), "method": LIMIT_METHOD}, sort_keys=True).encode())
    reference = frame[frame.batch_id.isin(reference_ids)].sort_values(["batch_id", "sample_id"])
    identity.update(pd.util.hash_pandas_object(reference, index=False).to_numpy().tobytes())
    return ProcessMonitoringModel(scaler, pca, t2_limit, spe_limit, float(quantile), partition, audit, f"process-pca-{identity.hexdigest()[:12]}")


def score_process_monitor(model, sensors):
    """Project all rows, leaving nullable flags and NaN scores for invalid inputs."""
    frame = process_observations(sensors)
    roles = {batch: role for group, role in ((model.partition.fit_batches, "PCA fitting"), (model.partition.limit_batches, "Limit calibration")) for batch in group}
    frame["role"] = frame.batch_id.map(roles).fillna("Evaluation")
    valid = frame.status.eq("scored")
    columns = [f"PC{index + 1}" for index in range(model.pca.n_components_)]
    for column in columns + ["t2", "spe", "anomaly_ratio"]:
        frame[column] = np.nan
    for column in ("t2_alarm", "spe_alarm", "alarm"):
        frame[column] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    scores, t2, spe, _ = pca_monitoring_statistics(model.scaler, model.pca, frame.loc[valid, list(SENSOR_COLUMNS)].to_numpy())
    frame.loc[valid, columns] = scores
    frame.loc[valid, "t2"] = t2
    frame.loc[valid, "spe"] = spe
    frame.loc[valid, "anomaly_ratio"] = np.maximum(t2 / model.t2_limit, spe / model.spe_limit)
    frame.loc[valid, "t2_alarm"] = t2 > model.t2_limit
    frame.loc[valid, "spe_alarm"] = spe > model.spe_limit
    frame.loc[valid, "alarm"] = (t2 > model.t2_limit) | (spe > model.spe_limit)
    return frame


def summarize_monitoring_batches(scored, batches):
    """Report exceedances among scored rows, with invalid counts kept separate."""
    metadata = _batch_metadata(batches)
    records = []
    for batch_id, group in scored.groupby("batch_id", sort=False):
        valid = group.status.eq("scored")
        alarms = group.alarm.fillna(False)
        count = int(valid.sum())
        onset = metadata.loc[batch_id, "disturbance_start_h"]
        post = valid & (group.elapsed_time_h > onset) if pd.notna(onset) else pd.Series(False, index=group.index)
        records.append({
            "batch_id": batch_id, "role": group.role.iloc[0], "scenario": metadata.loc[batch_id, "scenario"],
            "observations": len(group), "scored_rows": count, "invalid_rows": int((~valid).sum()),
            "flagged_rows": int(alarms.sum()), "flagged_fraction": float(alarms.sum() / count) if count else np.nan,
            "first_alarm_elapsed_h": float(group.loc[alarms, "elapsed_time_h"].min()) if alarms.any() else np.nan,
            "disturbance_start_h": onset,
            "post_disturbance_flagged_fraction": float(alarms[post].mean()) if post.any() else np.nan,
            "review_state": "Unscorable" if not count else "Flagged" if alarms.any() else "No exceedance; incomplete" if (~valid).any() else "No exceedance",
        })
    return pd.DataFrame(records)


def spe_sensor_contributions(model, observation):
    """Squared residual attribution; does not establish a disturbance's cause."""
    screened = process_observations(observation)
    if len(screened) != 1 or screened.status.ne("scored").any():
        raise ValueError("Choose one valid observation for SPE contribution inspection.")
    values = screened[list(SENSOR_COLUMNS)].to_numpy()
    _, _, _, contributions = pca_monitoring_statistics(model.scaler, model.pca, values)
    return pd.DataFrame({"sensor": SENSOR_COLUMNS, "spe_contribution": contributions[0]})


def monitoring_model_manifest(model):
    """JSON-safe provenance and the actual fitted scaling/PCA parameters."""
    return {
        "model_id": model.model_id, "features": list(SENSOR_COLUMNS), "partition": asdict(model.partition),
        "n_components": int(model.pca.n_components_), "quantile": model.quantile,
        "limit_method": LIMIT_METHOD, "t2_limit": model.t2_limit, "spe_limit": model.spe_limit,
        "scaler_mean": model.scaler.mean_.tolist(), "scaler_scale": model.scaler.scale_.tolist(),
        "pca_mean": model.pca.mean_.tolist(), "components": model.pca.components_.tolist(),
        "eigenvalues_ddof1": model.pca.explained_variance_.tolist(),
        "explained_variance_ratio": model.pca.explained_variance_ratio_.tolist(),
        "alarm_rule": "T2 > T2 limit OR SPE > SPE limit; equality is within limits; invalid input is unscorable",
        "anomaly_ratio": "max(T2 / T2_limit, SPE / SPE_limit); not a probability",
        "limitations": list(LIMITATIONS),
    }


def multivariate_report_zip(model, scored, batches, *, context=None):
    """Export the full historical comparison, reference audit, and calibration."""
    buffer = BytesIO()
    manifest = {**monitoring_model_manifest(model), "context": context or {}, "scope": "Full historical batch comparison; not a causal streaming report"}
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, allow_nan=False))
        archive.writestr("observations.csv", scored.to_csv(index=False))
        archive.writestr("batch_summary.csv", summarize_monitoring_batches(scored, batches).to_csv(index=False))
        archive.writestr("reference_audit.csv", model.reference_audit.to_csv(index=False))
        loadings = pd.DataFrame(model.pca.components_.T, index=SENSOR_COLUMNS, columns=[f"PC{index + 1}" for index in range(model.pca.n_components_)])
        archive.writestr("loadings.csv", loadings.rename_axis("sensor").to_csv())
        archive.writestr("README.txt", "Phase 6: PCA statistical process monitoring\n\n" + "\n".join(LIMITATIONS))
    return buffer.getvalue()
