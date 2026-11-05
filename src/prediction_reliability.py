"""Training-only prediction policy and descriptive PLS input applicability.

Empirical limits are educational screens, not SIMCA DModX, prediction intervals,
or a validated false-alarm probability. No labels enter inference checks.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.validation import regression_metrics


RELIABILITY_PROTOCOL = "prediction-reliability-v1"
DIAGNOSTIC_COLUMNS = ["applicability_status", "range_status", "score_distance",
                      "score_limit", "spectral_residual_rms", "residual_limit", "reliability_eligible"]


def checked_bounds(bounds, name):
    if bounds is None:
        return None
    try:
        values = tuple(float(value) for value in bounds)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} needs two finite nonnegative bounds.") from error
    if len(values) != 2 or not np.isfinite(values).all() or values[0] < 0 or values[0] >= values[1]:
        raise ValueError(f"{name} needs increasing finite nonnegative bounds.")
    return values


@dataclass
class PredictionPolicy:
    declared_range: tuple
    decision_range: tuple
    range_basis: str
    training_reference_range: tuple
    applicability_scope: str
    score_mean: object = None
    score_precision: object = None
    score_limit: float = np.nan
    residual_limit: float = np.nan
    quantile: float = 0.99
    protocol: str = RELIABILITY_PROTOCOL

    def summary(self):
        return {name: getattr(self, name) for name in (
            "protocol", "declared_range", "decision_range", "range_basis", "training_reference_range",
            "applicability_scope", "score_limit", "residual_limit", "quantile")}


def _pls_coordinates(pipeline, X):
    features = pipeline.named_steps["features"].transform(X)
    scaled = pipeline.named_steps["scale"].transform(features)
    model = pipeline.named_steps["model"]
    scores = np.asarray(model.transform(scaled), dtype=float)
    reconstructed = model.inverse_transform(scores)
    residual = np.sqrt(np.mean((scaled - reconstructed) ** 2, axis=1))
    if not np.isfinite(scores).all() or not np.isfinite(residual).all():
        raise ValueError("PLS applicability coordinates must be finite.")
    return scores, residual


def fit_prediction_policy(pipeline, X, y, declared_range=None, decision_range=None):
    """Called only on each fold's training rows or the final training partition."""
    reference = np.asarray(y, dtype=float).reshape(-1)
    if len(reference) != len(X) or len(X) < 3 or not np.isfinite(reference).all() or reference.min() < 0 or np.ptp(reference) <= 0:
        raise ValueError("Prediction policy needs varying nonnegative training references.")
    observed = (float(reference.min()), float(reference.max()))
    declared = checked_bounds(declared_range, "Declared concentration range") or observed
    decision = checked_bounds(decision_range, "Decision range") or declared
    # Study configuration validates nesting against the final training extent.
    # A fold's observed extent can be narrower; its checks never use fold labels.
    model = pipeline.named_steps["model"]
    recipe = pipeline.named_steps["features"].feature_set
    spectral = hasattr(model, "x_scores_") and recipe != "sensors"
    scope = ("PLS spectra + sensors" if recipe == "combined" else "PLS engineered spectral inputs") if spectral else "not_assessed"
    policy = PredictionPolicy(declared, decision,
        "User-declared demo range" if declared_range is not None else "Training-reference extent (provisional demo range)", observed, scope)
    if spectral:
        scores, residuals = _pls_coordinates(pipeline, X)
        policy.score_mean = scores.mean(axis=0)
        covariance = np.atleast_2d(np.cov(scores, rowvar=False))
        policy.score_precision = np.linalg.pinv(covariance, hermitian=True)
        centered = scores - policy.score_mean
        distances = np.maximum(0, np.einsum("ij,jk,ik->i", centered, policy.score_precision, centered))
        policy.score_limit = max(float(np.quantile(distances, policy.quantile, method="higher")), 1e-10)
        policy.residual_limit = max(float(np.quantile(residuals, policy.quantile, method="higher")), 1e-8)
    pipeline.prediction_policy_ = policy
    return policy


def assess_prediction_reliability(pipeline, X, prediction):
    """Shared batch/online checks. Unsupported/legacy applicability stays explicit."""
    values = np.asarray(prediction, dtype=float).reshape(-1)
    if len(values) != len(X):
        raise ValueError("Reliability needs one prediction per input row.")
    rows = pd.DataFrame(index=X.index)
    rows["applicability_status"] = "not_assessed"
    rows["range_status"] = "not_assessed"
    for name in ("score_distance", "score_limit", "spectral_residual_rms", "residual_limit"):
        rows[name] = np.nan
    valid = np.isfinite(values) & (values >= 0)
    policy = getattr(pipeline, "prediction_policy_", None)
    if policy is not None:
        if policy.protocol != RELIABILITY_PROTOCOL:
            raise ValueError("Unsupported prediction reliability policy.")
        within = (values >= policy.declared_range[0]) & (values <= policy.declared_range[1]) & valid
        rows["range_status"] = np.where(within, "within_declared_range", "outside_declared_range")
        if policy.applicability_scope != "not_assessed":
            scores, residual = _pls_coordinates(pipeline, X)
            centered = scores - policy.score_mean
            distance = np.maximum(0, np.einsum("ij,jk,ik->i", centered, policy.score_precision, centered))
            rows["score_distance"], rows["spectral_residual_rms"] = distance, residual
            rows["score_limit"], rows["residual_limit"] = policy.score_limit, policy.residual_limit
            passed = (distance <= policy.score_limit + 1e-10) & (residual <= policy.residual_limit + 1e-10)
            rows["applicability_status"] = np.where(passed, "within_training_limits", "outside_training_limits")
    rows["reliability_eligible"] = (valid & rows.applicability_status.ne("outside_training_limits")
                                    & rows.range_status.ne("outside_declared_range"))
    return rows


def range_performance(rows, policy):
    """Keep all-row errors alongside accepted-only errors and rejection coverage.

    Membership uses reference concentration; eligibility uses inputs/predictions.
    Absent diagnostic evidence is never interpreted as assessed coverage.
    """
    if policy is None:
        return pd.DataFrame()
    reference = pd.to_numeric(rows.reference_g_l, errors="coerce")
    prediction = pd.to_numeric(rows.prediction_g_l, errors="coerce")
    lower, upper = policy.decision_range
    groups = [("All references", np.ones(len(rows), dtype=bool)),
              ("Declared range", reference.between(*policy.declared_range)),
              ("Below decision range", reference < lower),
              ("Decision range", reference.between(lower, upper)),
              ("Above decision range", reference > upper)]
    records = []
    for label, mask in groups:
        subset = rows.loc[mask]
        finite = np.isfinite(reference.loc[mask]) & np.isfinite(prediction.loc[mask])
        assessed = (subset.applicability_status.ne("not_assessed") & subset.range_status.ne("not_assessed")) if "applicability_status" in subset else pd.Series(False, index=subset.index)
        eligible = subset.reliability_eligible.fillna(False).astype(bool) if "reliability_eligible" in subset else pd.Series(False, index=subset.index)
        record = {"reference_band": label, "reference_rows": len(subset), "finite_predictions": int(finite.sum()),
                  "assessed_rows": int(assessed.sum()), "eligible_rows": int(eligible.sum()),
                  "rejected_rows": len(subset) - int(eligible.sum()) if "reliability_eligible" in subset else 0,
                  "eligible_fraction": float(eligible.mean()) if len(subset) and "reliability_eligible" in subset else np.nan}
        for name, use in (("all", finite), ("eligible", finite & eligible)):
            metrics = regression_metrics(reference.loc[mask].loc[use], prediction.loc[mask].loc[use]) if use.any() else dict.fromkeys(("RMSE", "MAE", "Bias", "R2"), np.nan)
            record.update({f"{name}_{key}": value for key, value in metrics.items()})
        records.append(record)
    return pd.DataFrame(records)


def study_range_performance(study, evaluation=None, case_id=None):
    case_id = case_id or study.winner_id
    result = study.results[case_id]
    policy = getattr(result.estimator, "prediction_policy_", None)
    if policy is None:
        return pd.DataFrame()
    cv = pd.DataFrame({"reference_g_l": study.training_reference, "prediction_g_l": result.oof_prediction})
    diagnostics = getattr(result, "oof_reliability", None)
    if diagnostics is not None:
        cv = cv.join(diagnostics)
    tables = [range_performance(cv, policy).assign(evidence="Grouped CV (selection; fold-local checks)")]
    if evaluation is not None:
        if evaluation["study_id"] != study.study_id or evaluation["winner_id"] != study.winner_id:
            raise ValueError("Range evidence must match the frozen study.")
        test = evaluation["predictions"].loc[lambda frame: frame.case_id.eq(case_id)]
        tables.append(range_performance(test, policy).assign(evidence="Independent batches (frozen)"))
    return pd.concat(tables, ignore_index=True)


def reliability_record(result, prefix=""):
    """Small diagnostic fields shared by replay, acquisition and review exports."""
    return {f"{prefix}{key}": result.reliability.get(key, "not_assessed" if key.endswith("status") else np.nan)
            for key in DIAGNOSTIC_COLUMNS}
