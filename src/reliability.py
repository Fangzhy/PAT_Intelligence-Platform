"""Frozen performance evidence, causal assay residuals, and descriptive drift.

Distribution shift is a review signal, not proof of a sensor fault, concept
drift, or loss of predictive performance. Thresholds are teaching assumptions.
"""

from dataclasses import asdict, dataclass, field
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from src.modeling_data import SENSOR_COLUMNS, TARGETS
from src.modeling_storage import _json_text, study_manifest
from src.monitoring import MonitoringSettings, predict_observation
from src.replay import ReplayBatch, replay_fault_example
from src.validation import regression_metrics


RELIABILITY_LIMITATIONS = (
    "Synthetic educational evaluation, not manufacturing validation or CQA/CPP evidence.",
    "Grouped CV was used for selection; its performance is optimistic. Independent test results are frozen.",
    "Historical operational review may overlap training batches and is not a new independent test.",
    "Input/prediction shift does not by itself establish sensor drift, concept drift, or model degradation.",
    "Drift thresholds are illustrative effect-size screens, without calibrated alarm probabilities.",
    "Sparse assays, temporal correlation, process phase, and assay error limit interpretation.",
    "SPE/T² monitoring is on Multivariate_Monitoring; these distribution screens are separate.",
)
FAULT_SCENARIOS = (
    "Unmodified observations", "pH offset +0.10 after midpoint",
    "Raman baseline +0.05 a.u. after midpoint", "Assay bias +1 g/L after midpoint",
    "Missing, implausible, and out-of-domain inputs",
)
HISTORY_COLUMNS = ["sample_id", "batch_id", "elapsed_time_h", "measurement_timestamp",
                   "prediction_g_l", "prediction_status", "prediction_flags", "model_version",
                   "prediction_time_utc", "max_abs_z"]


@dataclass(frozen=True)
class ReliabilitySettings:
    residual_limit_g_l: float = 0.5
    bias_limit_g_l: float = 0.25
    drift_limit: float = 1.0
    min_drift_rows: int = 5
    rolling_pairs: int = 3

    def __post_init__(self):
        for name in ("residual_limit_g_l", "bias_limit_g_l", "drift_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("min_drift_rows", "rolling_pairs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2.")


def _cutoff(value):
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("Review requires a valid timezone-aware cutoff.")
    return timestamp.tz_convert("UTC")


def frozen_performance(study, evaluation=None):
    """Read the CV-selected winner's saved numbers, never evaluate or reselect."""
    winner = study.results[study.winner_id]
    records = []
    for role, metrics, rows in (
        ("Training (in-sample)", winner.training_metrics, len(study.training_reference)),
        ("Grouped CV (RMSECV; selection)", winner.cv_metrics, len(study.training_reference)),
    ):
        records.append({"evidence": role, "rows": rows, "RMSE_g_l": metrics["RMSE"], "MAE_g_l": metrics["MAE"], "R2": metrics["R2"], "Bias_g_l": metrics["Bias"]})
    if evaluation is not None:
        if evaluation["study_id"] != study.study_id or evaluation["winner_id"] != study.winner_id:
            raise ValueError("Frozen evaluation must belong to this CV-selected study.")
        metrics = evaluation["metrics"].set_index("case_id").loc[study.winner_id]
        rows = evaluation["predictions"].case_id.eq(study.winner_id).sum()
        records.append({"evidence": "Independent batches (RMSEP; frozen)", "rows": int(rows), "RMSE_g_l": metrics["RMSEP"], "MAE_g_l": metrics["MAE"], "R2": metrics["R2"], "Bias_g_l": metrics["Bias"]})
    return pd.DataFrame(records)


def input_drift_features(inputs):
    """Observed sensors plus raw spectral level indicators, never assay labels.

    Edge mean is a baseline proxy, confounded by true signal/scatter. Missing
    channels make spectral summaries unknown rather than silently averaging a
    different subset. These are distinct from the model's learned feature map.
    """
    frame = pd.DataFrame(index=inputs.index)
    for column in SENSOR_COLUMNS:
        frame[column] = pd.to_numeric(inputs[column], errors="coerce") if column in inputs else np.nan
    spectral_columns = [column for column in inputs if column.startswith("raman_")]
    if spectral_columns:
        spectra = inputs[spectral_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        complete = np.isfinite(spectra).all(axis=1)
        edge = max(1, len(spectral_columns) // 10)
        frame["raman_mean_au"] = np.where(complete, spectra.mean(axis=1), np.nan)
        frame["raman_edge_mean_au"] = np.where(complete, np.c_[spectra[:, :edge], spectra[:, -edge:]].mean(axis=1), np.nan)
    else:
        frame["raman_mean_au"] = np.nan
        frame["raman_edge_mean_au"] = np.nan
    return frame


@dataclass
class ReliabilityReference:
    input_features: pd.DataFrame
    predictions_g_l: np.ndarray
    oof_residuals_g_l: np.ndarray
    training_row_ids: tuple[str, ...]


def build_reliability_reference(study, cohort):
    """Use only original training rows and the already fitted winner pipeline."""
    if study.cohort_id != cohort.cohort_id or study.target != cohort.target or tuple(cohort.X.columns) != study.input_columns:
        raise ValueError("Reliability references must match the original study cohort, target, and schema.")
    training, _ = study.partition.masks(cohort.metadata)
    X = cohort.X.loc[training]
    if not X.index.equals(study.training_reference.index):
        raise ValueError("Training row identities differ from the stored study.")
    winner = study.results[study.winner_id]
    predictions = np.asarray(winner.estimator.predict(X), dtype=float).reshape(-1)
    if predictions.shape != (len(X),) or not np.isfinite(predictions).all():
        raise ValueError("Training-reference predictions are invalid.")
    return ReliabilityReference(input_drift_features(X), predictions[predictions >= 0],
        winner.oof_prediction - study.training_reference.to_numpy(), tuple(X.index))


def distribution_shift(reference, current, settings):
    """Per-column W1 / training SD, signed mean shift, and explicit sample counts.

    No hypothesis-test p-values, refitting, or calibrated false-alarm claims.
    Constant reference columns use an explicit change screen instead of / zero.
    """
    records = []
    for column in reference:
        baseline = pd.to_numeric(reference[column], errors="coerce").to_numpy(dtype=float)
        values = pd.to_numeric(current[column], errors="coerce").to_numpy(dtype=float) if column in current else np.full(len(current), np.nan)
        baseline = baseline[np.isfinite(baseline)]
        finite = values[np.isfinite(values)]
        record = {"feature": column, "reference_rows": len(baseline), "current_rows": len(finite),
                  "invalid_current_rows": len(values) - len(finite), "reference_mean": np.nan, "current_mean": np.nan,
                  "reference_sd": np.nan, "mean_shift_sd": np.nan, "wasserstein": np.nan,
                  "wasserstein_sd": np.nan, "status": "insufficient_data", "shift_flag": pd.NA}
        if len(baseline) >= settings.min_drift_rows and len(finite) >= settings.min_drift_rows:
            mean, sd = float(baseline.mean()), float(baseline.std(ddof=0))
            distance = float(wasserstein_distance(baseline, finite))
            record.update(reference_mean=mean, current_mean=float(finite.mean()), reference_sd=sd, wasserstein=distance)
            if sd <= 1e-12 * max(1, abs(mean)):
                changed = not np.isclose(finite, mean, rtol=1e-6, atol=1e-8).all()
                record.update(status="constant_reference_changed" if changed else "constant_reference_unchanged", shift_flag=changed)
            else:
                record.update(mean_shift_sd=float((finite.mean() - mean) / sd), wasserstein_sd=distance / sd,
                              status="assessed", shift_flag=distance / sd > settings.drift_limit)
        records.append(record)
    return pd.DataFrame(records).astype({"shift_flag": "boolean"})


def input_distribution_shift(reference, inputs, settings, *, compatible_axes=True):
    """Do not interpret spectral summary distances across incompatible axes."""
    current = input_drift_features(inputs)
    spectral = [column for column in current if column.startswith("raman_")]
    if not compatible_axes:
        current[spectral] = np.nan
    result = distribution_shift(reference, current, settings)
    if not compatible_axes:
        result.loc[result.feature.isin(spectral), "status"] = "axis_mismatch"
    return result


@dataclass
class ReliabilityReview:
    context_id: str
    records: dict = field(default_factory=dict)

    def history(self, batch, cutoff, bundle, settings: MonitoringSettings):
        """Infer each revealed observation once; backward review hides the future."""
        cutoff = _cutoff(cutoff)
        visible = batch.metadata[batch.metadata.measurement_timestamp <= cutoff]
        if not visible.sample_id.is_unique:
            raise ValueError("Review needs unique observation sample IDs.")
        rows = []
        for position in np.flatnonzero((batch.metadata.measurement_timestamp <= cutoff).to_numpy()):
            metadata = batch.metadata.iloc[position]
            identity = (metadata.batch_id, metadata.sample_id)
            if identity not in self.records:
                prediction = predict_observation(bundle, bundle.target, batch.inputs.iloc[[position]], batch.axis,
                    metadata.measurement_timestamp, metadata.sensor_timestamp, settings)
                self.records[identity] = {"sample_id": metadata.sample_id, "batch_id": metadata.batch_id,
                    "elapsed_time_h": metadata.elapsed_time_h, "measurement_timestamp": metadata.measurement_timestamp,
                    "prediction_g_l": prediction.prediction_g_l, "prediction_status": prediction.status,
                    "prediction_flags": "; ".join(prediction.flags), "model_version": prediction.model_version,
                    "prediction_time_utc": prediction.prediction_time_utc, "max_abs_z": prediction.max_abs_z}
            rows.append(self.records[identity])
        return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def available_reference_residuals(history, assays, target, cutoff, settings):
    """Exact same-batch/sample pairing; assay values arrive only at result time.

    No nearest-time substitution or ground-truth concentration is used. Rolling
    bias follows result availability order, separately within each batch.
    """
    cutoff = _cutoff(cutoff)
    if target not in TARGETS:
        raise ValueError("Choose a supported assay target.")
    if history.duplicated(["batch_id", "sample_id"]).any():
        raise ValueError("Prediction sample IDs must be unique within batch.")
    required = {"sample_id", "batch_id", "collection_timestamp", "result_timestamp", TARGETS[target]}
    if not required.issubset(assays.columns):
        raise ValueError("Assays need sample/batch IDs, collection/result timestamps, and the target reference.")
    available = assays.loc[pd.to_datetime(assays.result_timestamp, utc=True, errors="coerce") <= cutoff].copy()
    available["collection_timestamp"] = pd.to_datetime(available.collection_timestamp, utc=True, errors="coerce")
    available["result_timestamp"] = pd.to_datetime(available.result_timestamp, utc=True, errors="coerce")
    available["reference_g_l"] = pd.to_numeric(available[TARGETS[target]], errors="coerce")
    ambiguous = available.duplicated(["batch_id", "sample_id"], keep=False)
    available["ambiguous_assay"] = ambiguous
    rows = available.merge(history, on=["batch_id", "sample_id"], how="left", validate="many_to_one", suffixes=("_assay", ""))
    rows["prediction_g_l"] = pd.to_numeric(rows.prediction_g_l, errors="coerce")
    rows["pair_status"] = "assessed"
    unavailable = rows.prediction_status.ne("ok") | ~np.isfinite(rows.prediction_g_l)
    rows.loc[unavailable, "pair_status"] = "prediction_unavailable"
    mismatch = rows.measurement_timestamp.notna() & (rows.measurement_timestamp != rows.collection_timestamp)
    rows.loc[mismatch, "pair_status"] = "sample_time_mismatch"
    invalid = (rows.ambiguous_assay | rows.collection_timestamp.isna() | (rows.result_timestamp < rows.collection_timestamp)
               | ~np.isfinite(rows.reference_g_l) | (rows.reference_g_l < 0))
    rows.loc[invalid, "pair_status"] = "invalid_reference"
    valid = rows.pair_status.eq("assessed")
    rows["residual_g_l"] = np.where(valid, rows.prediction_g_l - rows.reference_g_l, np.nan)
    rows["residual_alarm"] = pd.Series(pd.NA, index=rows.index, dtype="boolean")
    rows.loc[valid, "residual_alarm"] = rows.loc[valid, "residual_g_l"].abs() > settings.residual_limit_g_l
    rows["rolling_bias_g_l"] = np.nan
    rows["bias_alarm"] = pd.Series(pd.NA, index=rows.index, dtype="boolean")
    for _, group in rows[valid].sort_values(["result_timestamp", "collection_timestamp", "sample_id"]).groupby("batch_id", sort=False):
        rolling = group.residual_g_l.rolling(settings.rolling_pairs, min_periods=settings.rolling_pairs).mean()
        rows.loc[group.index, "rolling_bias_g_l"] = rolling
        assessed = rolling.notna()
        rows.loc[rolling.index[assessed], "bias_alarm"] = rolling[assessed].abs() > settings.bias_limit_g_l
    return rows.sort_values(["result_timestamp", "batch_id", "sample_id"]).reset_index(drop=True)


def residual_metrics(rows, *, by_batch=False):
    """Report coverage with every metric; no unscored reference is counted normal."""
    groups = rows.groupby("batch_id", sort=False) if by_batch else [("All available", rows)]
    records = []
    for batch_id, group in groups:
        valid = group.pair_status.eq("assessed")
        count = int(valid.sum())
        metrics = regression_metrics(group.loc[valid, "reference_g_l"], group.loc[valid, "prediction_g_l"]) if count else dict.fromkeys(("RMSE", "MAE", "R2", "Bias"), np.nan)
        records.append({"batch_id": batch_id, "available_assays": len(group), "assessed_pairs": count,
            "unassessed_pairs": len(group) - count, "status": "assessed" if count else "not_assessed", **metrics})
    return pd.DataFrame(records)


def reliability_fault_example(batch, scenario, target):
    """Copy-only prespecified perturbations; assay bias is not model degradation."""
    if scenario not in FAULT_SCENARIOS or target not in TARGETS:
        raise ValueError("Choose a supported reliability scenario and target.")
    if scenario == FAULT_SCENARIOS[-1]:
        return replay_fault_example(batch)
    result = ReplayBatch(batch.inputs.copy(deep=True), batch.metadata.copy(deep=True), batch.axis.copy(), batch.assays.copy(deep=True))
    midpoint = float(result.metadata.elapsed_time_h.max()) / 2
    late = result.metadata.elapsed_time_h.to_numpy() >= midpoint
    if scenario == FAULT_SCENARIOS[1]:
        result.inputs.loc[late, "ph"] += 0.10
    elif scenario == FAULT_SCENARIOS[2]:
        columns = [column for column in result.inputs if column.startswith("raman_")]
        result.inputs.loc[late, columns] += 0.05
    elif scenario == FAULT_SCENARIOS[3]:
        result.assays.loc[result.assays.collection_elapsed_time_h >= midpoint, TARGETS[target]] += 1.0
    return result


def reliability_report_zip(study, evaluation, history, residuals, input_shift, prediction_shift, settings, context):
    """Frozen evidence and revealed-only review are explicitly separate sections."""
    output = BytesIO()
    manifest = {"phase": 7, "study": study_manifest(study), "review_context": context,
        "review_settings": asdict(settings), "limitations": RELIABILITY_LIMITATIONS,
        "residual_definition": "prediction minus offline assay, paired by exact batch/sample/time after result availability",
        "drift_definition": "Empirical Wasserstein-1 distance / original training SD; descriptive, not a probability",
        "report_scope": "Frozen historical validation evidence is separate from cutoff-filtered operational review"}
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", _json_text(manifest))
        archive.writestr("frozen_validation/performance.csv", frozen_performance(study, evaluation).to_csv(index=False))
        archive.writestr("frozen_validation/cv_folds.csv", study.folds.to_csv(index=False))
        if evaluation is not None:
            batches = evaluation["batch_metrics"]
            archive.writestr("frozen_validation/batch_metrics.csv", batches[batches.case_id.eq(study.winner_id)].to_csv(index=False))
        for name, frame in (("observations", history), ("available_assay_residuals", residuals),
                            ("input_shift", input_shift), ("prediction_shift", prediction_shift),
                            ("reference_metrics", residual_metrics(residuals))):
            archive.writestr(f"review/{name}.csv", frame.to_csv(index=False))
        archive.writestr("README.txt", "Only review/ contains cutoff-filtered current observations/assays. Frozen validation is a previously completed historical evaluation.\n\n" + "\n".join(RELIABILITY_LIMITATIONS))
    return output.getvalue()
