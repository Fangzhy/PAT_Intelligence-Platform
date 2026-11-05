"""Advisory alarms and guarded inference using fitted, saved soft-sensor pipelines."""

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import pandas as pd

from src.data_integration import SENSOR_RANGES
from src.model_registry import ModelBundle
from src.modeling_data import SENSOR_COLUMNS, TARGETS
from src.prediction_reliability import assess_prediction_reliability
from src.input_policy import read_input_policy


DEFAULT_SENSOR_LIMITS = {
    "temperature_c": (36.5, 37.5), "ph": (6.8, 7.2),
    "dissolved_oxygen_pct": (30.0, 80.0), "feed_rate_l_h": (0.0, 0.008),
    "agitation_rpm": (150.0, 450.0),
}
DEFAULT_PREDICTION_LIMITS = {"Glucose": (1.0, 15.0), "Lactate": (0.0, 5.0)}


@dataclass
class MonitoringSettings:
    domain_z_limit: float = 6.0
    max_sensor_age_minutes: float = 30.0
    sensor_limits: dict = field(default_factory=lambda: dict(DEFAULT_SENSOR_LIMITS))
    prediction_limits: dict = field(default_factory=lambda: dict(DEFAULT_PREDICTION_LIMITS))
    input_sensor_ranges: dict | None = None

    def __post_init__(self):
        self.sensor_limits = dict(self.sensor_limits)
        self.prediction_limits = dict(self.prediction_limits)
        if self.input_sensor_ranges is not None:
            ranges = self.input_sensor_ranges
            if set(ranges) != set(SENSOR_RANGES) or any(len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1] for bounds in ranges.values()):
                raise ValueError("Supply valid saved plausibility ranges for all sensors.")
            self.input_sensor_ranges = {key: tuple(bounds) for key, bounds in ranges.items()}
        if not np.isfinite(self.domain_z_limit) or self.domain_z_limit <= 0:
            raise ValueError("The model-domain threshold must be finite and positive.")
        if not np.isfinite(self.max_sensor_age_minutes) or self.max_sensor_age_minutes < 0:
            raise ValueError("Maximum sensor age must be finite and nonnegative.")
        for name, limits, required in (("sensor", self.sensor_limits, set(SENSOR_COLUMNS)), ("prediction", self.prediction_limits, set(TARGETS))):
            if set(limits) != required:
                raise ValueError(f"Supply limits for every {name} variable.")
            for variable, bounds in limits.items():
                try:
                    bounds = tuple(float(value) for value in bounds)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Enter finite numeric lower and upper limits for {variable}.") from error
                if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                    raise ValueError(f"Invalid monitoring limits for {variable}.")
                limits[variable] = bounds
        self.sensor_limits = {key: tuple(value) for key, value in self.sensor_limits.items()}
        self.prediction_limits = {key: tuple(value) for key, value in self.prediction_limits.items()}


def alarm_state(value, bounds):
    """Inclusive limits: exactly on a bound is within; missing is never normal."""
    try:
        value = float(value)
    except (ValueError, TypeError):
        return "Invalid/missing"
    if not np.isfinite(value):
        return "Invalid/missing"
    if value < bounds[0]:
        return "Low"
    if value > bounds[1]:
        return "High"
    return "Within limits"


def _utc(value):
    try:
        timestamp = pd.Timestamp(value)
        return timestamp.tz_convert("UTC") if timestamp.tzinfo is not None else pd.NaT
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def input_quality_errors(row, required_columns, measurement_time, sensor_time, max_sensor_age_minutes, sensor_ranges=None):
    """Validate one observed row; timestamps are compared on the replay clock."""
    issues = []
    missing = set(required_columns) - set(row.columns)
    if missing:
        issues.append(f"Missing inputs: {', '.join(sorted(missing))}")
    else:
        try:
            if not np.isfinite(row[list(required_columns)].to_numpy(dtype=float)).all():
                issues.append("Missing, nonnumeric, or non-finite model input")
        except (ValueError, TypeError):
            issues.append("Nonnumeric model input")
    for column, bounds in (SENSOR_RANGES if sensor_ranges is None else sensor_ranges).items():
        if column in row:
            status = alarm_state(row.iloc[0][column], bounds)
            if status != "Within limits":
                issues.append(f"Sensor plausibility failure: {column}")
    measured, sensed = _utc(measurement_time), _utc(sensor_time)
    if pd.isna(measured) or pd.isna(sensed):
        issues.append("Missing, invalid, or timezone-naive measurement timestamp")
    else:
        age = (measured - sensed).total_seconds() / 60
        if age < 0:
            issues.append("Sensor observation is later than the spectrum")
        elif age > max_sensor_age_minutes:
            issues.append("Sensor observation is too old for this spectrum")
    return tuple(issues)


def feature_domain_check(features, scaler, threshold):
    """Training-mean/scale screen; not a calibrated multivariate domain test."""
    values = np.asarray(features, dtype=float)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("The domain threshold must be finite and positive.")
    if values.ndim != 2 or len(values) != 1 or not np.isfinite(values).all():
        raise ValueError("Domain checking needs one finite engineered feature row.")
    scaled = np.asarray(scaler.transform(features), dtype=float)
    if not np.isfinite(scaled).all():
        raise ValueError("Standardized model features are non-finite.")
    maximum = float(np.abs(scaled).max())
    constant = np.asarray(scaler.var_) <= np.finfo(float).eps**2 * np.maximum(1, np.asarray(scaler.mean_)**2)
    constant_changed = constant & (np.abs(values[0] - scaler.mean_) > 1e-8 + 1e-6 * np.abs(scaler.mean_))
    return maximum > threshold or bool(constant_changed.any()), maximum, int(constant_changed.sum())


@dataclass
class PredictionResult:
    target: str
    prediction_g_l: float
    status: str
    flags: tuple[str, ...]
    model_version: str
    validation_status: str
    max_abs_z: float
    prediction_time_utc: object
    latency_ms: float
    reliability: dict = field(default_factory=dict)


def predict_observation(bundle: ModelBundle | None, target, row, axis,
                        measurement_time, sensor_time, settings: MonitoringSettings):
    """No fitting, fallback values, target substitution, clipping, or auto control."""
    if target not in TARGETS or not isinstance(row, pd.DataFrame) or len(row) != 1:
        raise ValueError("Supply one DataFrame row and a supported concentration target.")
    started = perf_counter()
    maximum = np.nan
    diagnostics = {}
    version = bundle.version if bundle else "Unavailable"
    validation = bundle.metadata.get("validation_status", "Not independently validated") if bundle else "No model loaded"

    def result(status, flags=(), value=np.nan):
        return PredictionResult(target, float(value), status, tuple(flags), version, validation, maximum,
            pd.Timestamp.now(tz="UTC") if bundle is not None else pd.NaT, (perf_counter() - started) * 1000, diagnostics)

    if bundle is None:
        return result("model_unavailable", ["No saved model selected; no concentration is synthesized"])
    if bundle.target != target:
        return result("invalid_input", ["Model target does not match the requested concentration"])
    expected_axis = np.asarray(bundle.metadata["spectral_axis_cm1"], dtype=float)
    supplied_axis = np.asarray(axis, dtype=float)
    if expected_axis.shape != supplied_axis.shape or not np.allclose(expected_axis, supplied_axis, rtol=1e-10, atol=1e-10):
        return result("axis_mismatch", ["Replay spectral axis differs from the fitted model; no resampling is performed"])
    required = bundle.metadata["input_columns"]
    tolerance, ranges = settings.max_sensor_age_minutes, settings.input_sensor_ranges
    if bundle.metadata.get("input_policy"):
        try:
            policy, ranges = read_input_policy(bundle.metadata["input_policy"])
            tolerance = policy.sensor_tolerance_minutes
        except (ValueError, KeyError, TypeError) as error:
            return result("invalid_input_policy", [str(error)])
    quality = input_quality_errors(row, required, measurement_time, sensor_time, tolerance, ranges)
    if quality:
        return result("invalid_input", quality)
    ordered = row[list(required)].copy()
    try:
        pipeline = bundle.pipeline
        features = pipeline.named_steps["features"].transform(ordered)
        outside, maximum, constant_changes = feature_domain_check(features, pipeline.named_steps["scale"], settings.domain_z_limit)
        if outside:
            flags = [f"Engineered features outside the training mean/scale screen: max |z| = {maximum:.4g}"]
            if constant_changes:
                flags.append(f"{constant_changes} training-constant feature(s) changed")
            return result("out_of_domain", flags)
        prediction = np.asarray(pipeline.predict(ordered), dtype=float).reshape(-1)
        if prediction.shape != (1,) or not np.isfinite(prediction[0]) or prediction[0] < 0:
            return result("invalid_prediction", ["Non-finite, negative, or invalid-shape concentration estimate; not clipped"])
        diagnostics = assess_prediction_reliability(pipeline, ordered, prediction).iloc[0].to_dict()
        flags = []
        if diagnostics["applicability_status"] == "outside_training_limits":
            flags.append("PLS input applicability exceeds training empirical score-distance or spectral-residual limits")
        if diagnostics["range_status"] == "outside_declared_range":
            flags.append("Concentration estimate is outside the saved declared demo range; advisory alarm limits do not override this gate")
        if flags:
            return result("outside_reliability_limits", flags)
        return result("ok", value=prediction[0])
    except (ValueError, TypeError, FloatingPointError) as error:
        return result("inference_error", [str(error)])
