"""Allowlisted, aggregate evidence and offline explanations; no fitting or I/O."""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import numpy as np
import pandas as pd

from src.feature_engineering import FEATURE_SETS
from src.modeling import MODELS
from src.modeling_data import SENSOR_COLUMNS, TARGETS
from src.monitoring import alarm_state
from src.prediction_reliability import study_range_performance


PROMPT_VERSION = "pat-explanations-v2"
LIMITATIONS = {
    "educational": "Educational evidence does not establish manufacturing validation or a CPP/CQA relationship.",
    "selection": "Grouped CV was used for selection and can be optimistic after tuning.",
    "missing_test": "Independent-test metrics were not provided; this does not establish whether other testing exists.",
    "frozen_test": "Independent synthetic-batch metrics are already recorded; this explanation does not evaluate or change the frozen test.",
    "historical": "Historical operational review is separate from independent validation and may overlap calibration batches.",
    "shift": "Distribution-shift flags do not establish sensor drift, root cause, or model degradation.",
    "gaps": "Rejected predictions remain unavailable; no concentration is inferred to fill a gap.",
    "spectral": "Spectral transformations and peaks do not identify molecules or establish concentration prediction accuracy.",
    "spectral_pca": "Exploratory PCA variance and reconstruction residuals describe processed spectral variation, not predictive accuracy or calibrated alarms.",
    "spectral_residual": "Raw-minus-smoothed residuals can contain genuine peaks; they are not a validated instrument-noise estimate.",
    "spectral_fft": "Fourier frequencies describe variation along Raman shift, not temporal Hz. Wavelet thresholding can remove weak peaks; padded coefficient norms are not conserved physical energy.",
    "monitoring": "This is a captured historical monitoring snapshot, not a live freshness statement or a new predictive validation.",
    "advisory": "Advisory limits and input-quality/domain screens are distinct. A missing value or rejected estimate is not a normal process observation.",
    "empirical": "Empirical per-chart quantiles do not guarantee a joint false-alarm rate under correlated, repeatedly monitored batch observations.",
    "prediction_reliability": "Declared concentration ranges and PLS empirical applicability checks are demonstration policies, not validated method ranges, prediction intervals, or manufacturing validation. Eligible-only errors require coverage; unassessed checks do not establish reliability. Python checks determine eligibility.",
    "contributions": "T-squared/SPE exceedances and residual contributions are review signals, not confirmed physical causes.",
}
CHECKS = {
    "batch_split": "Review whole-batch separation and selection optimism before interpreting model comparisons.",
    "residuals": "Inspect the displayed residual plots and coverage alongside aggregate errors.",
    "preprocessing": "Review the fitted feature recipe and training-only preprocessing.",
    "assays": "Check reference quality and laboratory result availability before interpreting residuals.",
    "domain": "Inspect quality/domain flags and missing predictions before interpreting distribution changes.",
    "phase": "Compare similar process phases and available observations when reviewing distribution changes.",
    "spectra": "Compare raw and transformed views and check whether useful broad or narrow spectral features were removed.",
    "axes": "Check the displayed Raman, spatial-frequency, and coefficient axes and their units before interpretation.",
    "reference": "Review the reference-batch scope and preprocessing used to define the PCA coordinates.",
    "freshness": "Review source health, receipt time, freshness, and prediction eligibility on the live monitoring cards.",
    "limits": "Inspect the advisory thresholds and input-quality flags together; do not treat an alarm as an equipment instruction.",
    "calibration": "Review separate PCA fitting, limit-calibration, and evaluation batches before interpreting chart exceedances.",
    "contributions": "Inspect standardized residual contributions alongside the observed sensor trends without assigning a root cause.",
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ResultSnapshot:
    """The private context contributes only its hash to the outbound payload."""
    serialized: str

    @property
    def payload(self):
        return json.loads(self.serialized)

    @property
    def identity(self):
        return self.payload["snapshot_id"]

    @property
    def requires_consent(self):
        return self.payload["source_kind"] == "prepared_sources"


class _Facts:
    def __init__(self):
        self.rows = []

    def add(self, label, value, scope, unit=""):
        if isinstance(value, bool):
            pass
        elif value is not None:
            value = float(value)
            value = value if math.isfinite(value) else None
        self.rows.append({"id": f"E{len(self.rows) + 1:03d}", "label": label,
                          "value": value, "unit": unit, "scope": scope})


def _finish(page, target, source_kind, facts, limitations, checks, private_context):
    targets = {"Predictive Modeling": ("Glucose", "Lactate"), "Model Reliability": ("Glucose", "Lactate"),
               "Spectral Analytics": ("Raman spectra",), "Process Monitoring": ("Glucose and Lactate",),
               "Multivariate Monitoring": ("Observed process sensors",)}
    if target not in targets.get(page, ()) or source_kind not in ("synthetic", "prepared_sources"):
        raise ValueError("Unsupported explanation target or source.")
    payload = {"page": page, "target": target, "source_kind": source_kind,
               "facts": facts.rows, "limitations": limitations, "review_checks": checks,
               "protocol": PROMPT_VERSION}
    payload["snapshot_id"] = sha256((_json(payload) + _json(private_context)).encode()).hexdigest()
    serialized = _json(payload)
    if len(serialized.encode()) > 24000 or len(facts.rows) > 100:
        raise ValueError("Explanation summary exceeds its bounded input size.")
    return ResultSnapshot(serialized)


def modeling_snapshot(study, evaluation, *, inspected_case=None, source_kind="synthetic"):
    """Read saved CV/test records only. Never accepts a raw modeling dataset."""
    if evaluation is not None and (evaluation["study_id"] != study.study_id or evaluation["winner_id"] != study.winner_id):
        raise ValueError("Evaluation does not belong to this frozen study.")
    facts = _Facts()
    facts.add("Training assay pairs", len(study.training_reference), "training", "pairs")
    facts.add("Training batches", len(study.partition.training_batches), "training", "batches")
    facts.add("Successful comparison experiments", len(study.comparison), "grouped_cv", "experiments")
    facts.add("Training-only CV folds", study.config.n_splits, "grouped_cv", "folds")
    # Bounded comparison: winner, inspected case, then the leading CV cases.
    chosen = list(dict.fromkeys([study.winner_id, inspected_case or study.winner_id,
                                *study.comparison.case_id.tolist()[:4]]))
    facts.add("Experiments included in this bounded summary", len(chosen), "grouped_cv", "experiments")
    for case_id in chosen:
        result = study.results[case_id]
        label = f"{MODELS[result.case.model]} / {FEATURE_SETS[result.case.feature_set]}"
        role = "CV-selected" if case_id == study.winner_id else "Inspected" if case_id == inspected_case else "Comparison"
        for key, name, unit in (("RMSE", "RMSECV", "g/L"), ("MAE", "CV MAE", "g/L"), ("R2", "CV R-squared", "dimensionless")):
            facts.add(f"{role}: {label} — {name}", result.cv_metrics[key], "grouped_cv", unit)
        if result.case.model == "pls":
            facts.add(f"{role}: fitted PLS latent variables", result.best_params["n_components"], "training", "components")
    limitations = ["educational", "selection", "missing_test" if evaluation is None else "frozen_test"]
    policy = getattr(study.results[study.winner_id].estimator, "prediction_policy_", None)
    if policy is not None:
        limitations.append("prediction_reliability")
        for bounds, label in ((policy.declared_range, "Declared demo concentration range"), (policy.decision_range, "Decision range")):
            facts.add(f"{label} lower", bounds[0], "saved_policy", "g/L")
            facts.add(f"{label} upper", bounds[1], "saved_policy", "g/L")
        facts.add("Spectral applicability assessed for saved model", policy.applicability_scope != "not_assessed", "saved_policy", "boolean")
        table = study_range_performance(study, evaluation)
        for _, band in table[table.reference_band.eq("Decision range")].iterrows():
            scope = "independent_test" if band.evidence == "Independent batches (frozen)" else "grouped_cv"
            for column, label, unit in (("reference_rows", "reference pairs", "pairs"),
                ("eligible_rows", "eligible pairs under new checks", "pairs"),
                ("assessed_rows", "pairs with both checks assessed", "pairs"),
                ("all_RMSE", "all-row RMSE", "g/L"), ("all_Bias", "all-row bias", "g/L")):
                facts.add(f"Winner decision range {label}", band[column], scope, unit)
    if evaluation is not None:
        winner = evaluation["metrics"].set_index("case_id").loc[study.winner_id]
        for key, label, unit in (("RMSEP", "Frozen winner independent-batch RMSEP", "g/L"),
                                 ("MAE", "Frozen winner independent-batch MAE", "g/L"),
                                 ("R2", "Frozen winner independent-batch R-squared", "dimensionless")):
            facts.add(label, winner[key], "independent_test", unit)
    return _finish("Predictive Modeling", study.target, source_kind, facts, limitations,
                   ["batch_split", "residuals", "preprocessing"],
                   {"study": study.study_id, "cohort": study.cohort_id, "inspected": inspected_case})


def reliability_snapshot(study, evaluation, history, residuals, metrics, input_shift,
                         prediction_shift, rules, *, context, source_kind="synthetic"):
    """Aggregate only the caller's already cutoff-filtered review tables."""
    cutoff = pd.Timestamp(context["cutoff_utc"])
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("Reliability summaries need a timezone-aware review cutoff.")
    for table, column in ((history, "measurement_timestamp"), (residuals, "result_timestamp")):
        times = pd.to_datetime(table[column], utc=True, errors="coerce")
        if times.isna().any() or (times > cutoff).any():
            raise ValueError("Summary tables contain invalid or future observations/results.")
    base = modeling_snapshot(study, evaluation, source_kind=source_kind)
    facts = _Facts()
    facts.rows = base.payload["facts"]
    facts.add("Revealed observations", len(history), "review", "observations")
    facts.add("Unavailable predictions", int(history.prediction_status.ne("ok").sum()), "review", "observations")
    facts.add("Available assays", len(residuals), "review", "assays")
    facts.add("Assessed reference pairs", metrics.assessed_pairs.iloc[0], "review", "pairs")
    for key in ("RMSE", "MAE", "Bias"):
        facts.add(f"Available assessed pairs — {key}", metrics[key].iloc[0], "review", "g/L")
    facts.add("Absolute residual review threshold", rules.residual_limit_g_l, "review", "g/L")
    facts.add("Absolute rolling bias review threshold", rules.bias_limit_g_l, "review", "g/L")
    facts.add("Residual review flags", int(residuals.residual_alarm.fillna(False).sum()), "review", "pairs")
    facts.add("Rolling bias review flags", int(residuals.bias_alarm.fillna(False).sum()), "review", "pairs")
    for table, label in ((input_shift, "Input"), (prediction_shift, "Prediction")):
        facts.add(f"{label} distribution-shift flags", int(table.shift_flag.fillna(False).sum()), "review", "features")
        facts.add(f"{label} distribution screens unassessed", int(table.shift_flag.isna().sum()), "review", "features")
    return _finish("Model Reliability", study.target, source_kind, facts,
                   base.payload["limitations"] + ["historical", "shift", "gaps"],
                   ["assays", "residuals", "domain", "phase"], context)


def acquired_reliability_snapshot(review, bundle, *, source_kind="prepared_sources"):
    """Bounded numeric facts from received-only diagnostics; identifiers stay private."""
    facts = _Facts()
    history, pairs, metrics = review["history"], review["residuals"], review["metrics"]
    facts.add("Received prediction observations in review window", len(history), "operational_review", "observations")
    facts.add("Unavailable predictions in review window", int(history.prediction_status.ne("ok").sum()), "operational_review", "observations")
    facts.add("Currently eligible predictions in review window", int(history.current_eligible.sum()), "captured_current_status", "observations")
    facts.add("Available received assay references", len(pairs), "operational_review", "assays")
    facts.add("Assessed operational reference pairs", metrics.assessed_pairs.iloc[0], "operational_review", "pairs")
    for name in ("RMSE", "MAE", "Bias", "R2"):
        facts.add(f"Operational assessed-pair {name}", metrics[name].iloc[0], "operational_review", "dimensionless" if name == "R2" else "g/L")
    for name, label in (("residual_alarm", "Residual exceedances"), ("bias_alarm", "Rolling bias exceedances")):
        facts.add(label, int(pairs[name].fillna(False).sum()), "operational_review", "pairs")
    facts.add("Matching training baseline available", review["context"]["baseline_available"], "training_reference", "boolean")
    for name, label in (("input_shift", "Input"), ("prediction_shift", "Prediction")):
        facts.add(f"{label} distribution shift flags", int(review[name].shift_flag.fillna(False).sum()), "operational_review", "features")
    for name, scope in (("cv_metrics", "grouped_cv"), ("holdout_metrics", "independent_test")):
        for metric, value in (bundle.metadata.get(name) or {}).items():
            if metric in ("RMSE", "RMSEP", "MAE", "Bias", "R2"):
                facts.add(f"Recorded {scope} {metric}", value, scope, "dimensionless" if metric == "R2" else "g/L")
    return _finish("Model Reliability", bundle.target, source_kind, facts,
        ["educational", "selection", "shift", "gaps", "monitoring", "prediction_reliability",
         "frozen_test" if bundle.metadata.get("holdout_metrics") else "missing_test"],
        ["assays", "domain", "phase", "freshness"], review["context"])


def spectral_snapshot(axis, signal, baseline, smoothed, fourier_features, *, selected_count,
                      reference_count, peak_count, settings, pca_variance=None,
                      reconstruction_rmse=None, context, source_kind="synthetic"):
    """Scalar descriptions of already computed exploration; no new transforms/fits."""
    axis, signal = np.asarray(axis), np.asarray(signal)
    facts = _Facts()
    for label, value, unit in (("Complete selected spectra", selected_count, "spectra"),
        ("Reference spectra", reference_count, "spectra"), ("Selected Raman channels", len(axis), "channels"),
        ("Raman region lower bound", axis[0], "cm⁻¹"), ("Raman region upper bound", axis[-1], "cm⁻¹"),
        ("Raman channel spacing", axis[1] - axis[0], "cm⁻¹"),
        ("Selected spectrum mean intensity", np.mean(signal), "a.u."),
        ("Selected spectrum minimum intensity", np.min(signal), "a.u."),
        ("Selected spectrum maximum intensity", np.max(signal), "a.u."),
        ("Mean estimated AsLS baseline", np.mean(baseline), "a.u."),
        ("Raw minus SG-smoothed residual SD", np.std(signal - smoothed, ddof=1), "a.u."),
        ("Detected local maxima at selected prominence", peak_count, "peaks"),
        ("Peak prominence setting", settings["prominence"], "a.u."),
        ("SG window", settings["window_length"], "channels"),
        ("Baseline smoothness setting", settings["smoothness"], "dimensionless"),
        ("Baseline asymmetry setting", settings["asymmetry"], "dimensionless"),
        ("FFT Hann window enabled", settings["fft_window"] == "hann", "boolean"),
        ("FFT constant detrending enabled", settings["fft_detrend"] == "constant", "boolean"),
        ("FFT linear detrending enabled", settings["fft_detrend"] == "linear", "boolean"),
        ("Fourier pass-band lower bound", settings["band"][0], "cycles per cm⁻¹"),
        ("Fourier pass-band upper bound", settings["band"][1], "cycles per cm⁻¹"),
        ("Retained Fourier bins", fourier_features["selected_bins"], "bins"),
        ("Retained fraction of analyzed mean square", fourier_features["fraction_of_analyzed_mean_square"], "fraction"),
        ("Wavelet decomposition levels", settings["wavelet_level"], "levels"),
        ("Wavelet soft threshold", settings["wavelet_threshold"], "a.u.")):
        facts.add(label, value, "spectral_exploration", unit)
    if pca_variance is not None:
        facts.add("Fitted exploratory PCA components", len(pca_variance), "exploratory_reference", "components")
        facts.add("Cumulative retained reference variance", np.sum(pca_variance), "exploratory_reference", "fraction")
        facts.add("Mean reconstruction RMSE across all selected spectra", np.mean(reconstruction_rmse), "exploratory_selected", "PCA input units")
    else:
        facts.add("Exploratory PCA reference variance", None, "exploratory_reference", "fraction")
    return _finish("Spectral Analytics", "Raman spectra", source_kind, facts,
                   ["educational", "spectral", "spectral_pca", "spectral_residual", "spectral_fft"],
                   ["spectra", "axes", "reference"], {"view": context, "settings": settings})


_SENSOR_LABELS = {"temperature_c": ("Temperature", "°C"), "ph": ("pH", "dimensionless"),
    "dissolved_oxygen_pct": ("Dissolved oxygen", "% air saturation"),
    "feed_rate_l_h": ("Feed flow", "L/h"), "agitation_rpm": ("Agitation", "rpm")}


def process_snapshot(history, sensors, assays, settings, *, cutoff, context, current_eligible=True,
                     pending=0, health_flag_count=0, future_tolerance_seconds=0,
                     source_kind="synthetic"):
    """Explain received/revealed rows only, withholding ineligible current cards."""
    cutoff = pd.Timestamp(cutoff)
    if pd.isna(cutoff) or cutoff.tzinfo is None or not np.isfinite(future_tolerance_seconds) or future_tolerance_seconds < 0:
        raise ValueError("Process summaries need a valid capture clock and future tolerance.")
    for table, column, tolerance in ((history, "measurement_timestamp", future_tolerance_seconds),
        (sensors, "measurement_timestamp", future_tolerance_seconds), (assays, "result_timestamp", 0)):
        if table.empty:
            continue
        times = pd.to_datetime(table[column], utc=True, errors="coerce")
        if times.isna().any() or (times > cutoff + pd.Timedelta(seconds=tolerance)).any():
            raise ValueError("Process summary contains invalid or future observations/results.")
    if history.empty and sensors.empty:
        return None
    facts = _Facts()
    for label, value, unit in (("Received or revealed prediction observations", len(history), "observations"),
        ("Received or revealed sensor observations", len(sensors), "observations"),
        ("Available assay results at capture", len(assays), "assays"),
        ("Spectra waiting for inputs", pending, "spectra"),
        ("Source health/freshness review flags", health_flag_count, "flags")):
        facts.add(label, value, "captured_monitoring", unit)
    for table, column, label in ((history, "quality_flags", "Prediction observations with quality flags"),
        (sensors, "sensor_quality_flags" if "sensor_quality_flags" in sensors else "quality_flags", "Sensor observations with quality flags")):
        facts.add(label, int(table[column].fillna("").ne("").sum()) if column in table else 0,
                  "captured_monitoring", "observations")
    if not sensors.empty:
        latest_sensor = sensors.iloc[-1]
        for column in SENSOR_COLUMNS:
            label, unit = _SENSOR_LABELS[column]
            value = pd.to_numeric(latest_sensor[column], errors="coerce")
            facts.add(f"Latest received {label} observation at capture", value, "captured_monitoring", unit)
            facts.add(f"{label} advisory lower bound", settings.sensor_limits[column][0], "advisory_rule", unit)
            facts.add(f"{label} advisory upper bound", settings.sensor_limits[column][1], "advisory_rule", unit)
        flagged = sum(alarm_state(latest_sensor[column], settings.sensor_limits[column]) in ("Low", "High") for column in SENSOR_COLUMNS)
        facts.add("Latest sensor advisory exceedances", flagged, "captured_monitoring", "sensors")
    for target in TARGETS:
        prefix = target.lower()
        status = history[f"{prefix}_status"] if not history.empty else pd.Series(dtype=str)
        predictions = pd.to_numeric(history[f"{prefix}_prediction_g_l"], errors="coerce") if not history.empty else pd.Series(dtype=float)
        valid = status.eq("ok") & np.isfinite(predictions)
        latest_ok = bool(current_eligible and len(valid) and valid.iloc[-1])
        value = predictions.iloc[-1] if latest_ok else None
        facts.add(f"{target} displayed estimate at capture", value, "captured_monitoring", "g/L")
        facts.add(f"{target} eligible estimate at capture", latest_ok, "captured_monitoring", "boolean")
        facts.add(f"{target} accepted historical estimates", int(valid.sum()), "captured_history", "observations")
        facts.add(f"{target} unavailable historical estimates", len(status) - int(valid.sum()), "captured_history", "observations")
        facts.add(f"{target} advisory lower bound", settings.prediction_limits[target][0], "advisory_rule", "g/L")
        facts.add(f"{target} advisory upper bound", settings.prediction_limits[target][1], "advisory_rule", "g/L")
        for column, label, outside in (("applicability_status", "spectral applicability rejections", "outside_training_limits"),
                                       ("range_status", "declared-range rejections", "outside_declared_range")):
            field = f"{prefix}_{column}"
            if field in history:
                facts.add(f"{target} {label}", int(history[field].eq(outside).sum()), "captured_history", "observations")
                facts.add(f"{target} {column} unassessed", int(history[field].eq("not_assessed").sum()), "captured_history", "observations")
    return _finish("Process Monitoring", "Glucose and Lactate", source_kind, facts,
                   ["educational", "monitoring", "advisory", "gaps", "prediction_reliability"], ["freshness", "limits", "assays", "domain"],
                   {"view": context, "cutoff": cutoff.isoformat(), "eligible": bool(current_eligible)})


def multivariate_snapshot(model, view, observation, contributions, *, context):
    """Read existing PCA scores/limits and optional residual contributions."""
    facts = _Facts()
    for label, value, unit in (("Retained process PCA components", model.pca.n_components_, "components"),
        ("Cumulative retained reference variance", model.pca.explained_variance_ratio_.sum(), "fraction"),
        ("Empirical per-chart calibration quantile", model.quantile, "fraction"),
        ("T-squared empirical limit", model.t2_limit, "dimensionless"),
        ("SPE empirical limit", model.spe_limit, "squared standardized residual"),
        ("PCA fitting batches", len(model.partition.fit_batches), "batches"),
        ("Limit-calibration batches", len(model.partition.limit_batches), "batches"),
        ("Selected chart observations", len(view), "observations")):
        facts.add(label, value, "historical_multivariate", unit)
    scored = view.status.eq("scored")
    facts.add("Selected chart scored observations", int(scored.sum()), "historical_multivariate", "observations")
    facts.add("Selected chart unscorable observations", int((~scored).sum()), "historical_multivariate", "observations")
    for column, label in (("t2_alarm", "T-squared"), ("spe_alarm", "SPE"), ("alarm", "Either-chart")):
        flags = view.loc[scored, column].fillna(False)
        facts.add(f"Selected chart {label} exceedances", int(flags.sum()), "historical_multivariate", "observations")
    valid = len(observation) == 1 and observation.status.iloc[0] == "scored"
    for column, label, unit in (("t2", "T-squared", "dimensionless"), ("spe", "SPE", "squared standardized residual")):
        facts.add(f"Separately inspected observation {label}", observation[column].iloc[0] if valid else None, "inspection", unit)
    if valid and contributions is not None:
        for column in SENSOR_COLUMNS:
            rows = contributions.loc[contributions.sensor.eq(column), "spe_contribution"]
            facts.add(f"Inspected {_SENSOR_LABELS[column][0]} SPE contribution", rows.iloc[0] if len(rows) == 1 else None,
                      "inspection", "squared standardized residual")
    return _finish("Multivariate Monitoring", "Observed process sensors", "synthetic", facts,
                   ["educational", "historical", "empirical", "contributions", "advisory"],
                   ["calibration", "contributions", "phase", "domain"], {"model": model.model_id, "view": context})


def template_sections(snapshot):
    payload = snapshot.payload
    return {"observations": [],
            "interpretation": "This local summary uses the recorded evidence shown below. No AI interpretation was generated.",
            "limitations": payload["limitations"], "review_checks": payload["review_checks"]}


def template_text(snapshot):
    """Values are formatted by Python, including explicit unassessed gaps."""
    payload = snapshot.payload
    lines = [f"{payload['page']} — {payload['target']}", "Local template summary; not AI-generated."]
    for fact in payload["facts"]:
        value = "Unavailable / not assessed" if fact["value"] is None else f"{fact['value']:.6g}"
        lines.append(f"{fact['id']}: {fact['label']}: {value} {fact['unit']}.".strip())
    lines.extend(LIMITATIONS[name] for name in payload["limitations"])
    lines.extend(CHECKS[name] for name in payload["review_checks"])
    return "\n".join(lines)


def demo_snapshot(page="Model Reliability"):
    """Small non-process-data example for explicitly invoked live checks."""
    facts = _Facts()
    if page == "Spectral Analytics":
        facts.add("Selected Raman channels", 71, "spectral_exploration", "channels")
        facts.add("Raw minus SG-smoothed residual SD", 0.02, "spectral_exploration", "a.u.")
        facts.add("Retained fraction of analyzed mean square", 0.6, "spectral_exploration", "fraction")
        facts.add("Cumulative retained reference variance", 0.88, "exploratory_reference", "fraction")
        return _finish(page, "Raman spectra", "synthetic", facts,
            ["educational", "spectral", "spectral_pca", "spectral_residual", "spectral_fft"],
            ["spectra", "axes", "reference"], {"example": "connection-check"})
    if page == "Process Monitoring":
        facts.add("Glucose displayed estimate at capture", 3.2, "captured_monitoring", "g/L")
        facts.add("Lactate displayed estimate at capture", None, "captured_monitoring", "g/L")
        facts.add("Source health/freshness review flags", 1, "captured_monitoring", "flags")
        return _finish(page, "Glucose and Lactate", "synthetic", facts,
            ["educational", "monitoring", "advisory", "gaps"], ["freshness", "limits", "domain"],
            {"example": "connection-check"})
    if page == "Multivariate Monitoring":
        facts.add("Empirical per-chart calibration quantile", 0.99, "historical_multivariate", "fraction")
        facts.add("Inspected T-squared", 4.8, "inspection", "dimensionless")
        facts.add("T-squared empirical limit", 6.5, "historical_multivariate", "dimensionless")
        facts.add("Inspected SPE", 1.2, "inspection", "squared standardized residual")
        facts.add("SPE empirical limit", 0.8, "historical_multivariate", "squared standardized residual")
        return _finish(page, "Observed process sensors", "synthetic", facts,
            ["educational", "historical", "empirical", "contributions", "advisory"],
            ["calibration", "contributions", "domain"], {"example": "connection-check"})
    if page != "Model Reliability":
        raise ValueError("Choose a supported demonstration page.")
    facts.add("Available glucose prediction", 3.2, "review", "g/L")
    facts.add("Grouped CV RMSE", 0.45, "grouped_cv", "g/L")
    facts.add("Out-of-domain prediction", None, "review", "g/L")
    return _finish("Model Reliability", "Glucose", "synthetic", facts,
                   ["educational", "selection", "missing_test", "gaps"],
                   ["domain", "assays"], {"example": "connection-check"})
