"""Shared deterministic prediction-reliability panels; no training or network I/O."""

import numpy as np
import pandas as pd
import streamlit as st

from src.prediction_reliability import study_range_performance


def render_model_policy(pipeline, *, key):
    policy = getattr(pipeline, "prediction_policy_", None)
    if policy is None:
        st.info("Prediction reliability: not assessed for this older saved model. Its frozen evaluation remains unchanged. Use a genuinely new cohort to prepare a model with the new checks.")
        return
    low, high = policy.declared_range
    decision_low, decision_high = policy.decision_range
    st.write(f"Declared concentration range: **{low:.4g}–{high:.4g} g/L** · Decision range: **{decision_low:.4g}–{decision_high:.4g} g/L**")
    st.caption(f"{policy.range_basis}. This is a demonstration policy, not an experimentally validated method range. Advisory high/low alarm limits are separate.")
    if policy.applicability_scope == "not_assessed":
        st.info("Spectral applicability is not assessed for this model. PLS with spectral inputs is required; a passed concentration-range check does not establish spectral applicability.")
    else:
        st.dataframe(pd.DataFrame([{"scope": policy.applicability_scope,
            "score_distance_limit": policy.score_limit, "spectral_residual_RMS_limit": policy.residual_limit,
            "training_quantile": policy.quantile}]), hide_index=True, width="stretch", key=f"{key}_policy")
        st.caption("PLS score distance is a squared Mahalanobis distance. Residual RMS measures reconstruction error in standardized engineered-input units. Limits use training empirical quantiles; they are not SIMCA DModX, prediction intervals, or a guaranteed false-alarm rate. Spectra + sensors models use their joint PLS input space.")


def render_study_reliability(study, evaluation=None, *, case_id=None, key):
    st.subheader("Prediction reliability")
    case_id = case_id or study.winner_id
    render_model_policy(study.results[case_id].estimator, key=key)
    table = study_range_performance(study, evaluation, case_id)
    if not table.empty:
        st.dataframe(table, hide_index=True, width="stretch", key=f"{key}_range_performance")
        st.caption("Bands use assay reference concentration, with inclusive decision-range bounds. All-row errors retain rejected observations; eligible-only errors must be read with coverage. Eligibility here covers finite nonnegative estimates and the new applicability/range checks; online quality, freshness and configurable maximum-|z| guards can reject additional observations. assessed_rows counts rows with both new checks available. Empty bands have no accuracy estimate. CV checks use fold-training references and limits; CV remains selection evidence. Frozen test results never select settings or the winner.")


def render_observation_reliability(latest, *, targets, key, prefixed=True, current=True):
    st.subheader("Prediction reliability")
    records = []
    for target in targets:
        prefix = f"{target.lower()}_" if prefixed else ""
        record = {"target": target,
            "applicability": latest.get(f"{prefix}applicability_status", "not_assessed"),
            "declared_range": latest.get(f"{prefix}range_status", "not_assessed"),
            "score_distance": latest.get(f"{prefix}score_distance", np.nan),
            "score_limit": latest.get(f"{prefix}score_limit", np.nan),
            "spectral_residual_RMS": latest.get(f"{prefix}spectral_residual_rms", np.nan),
            "residual_limit": latest.get(f"{prefix}residual_limit", np.nan)}
        records.append(record)
    st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch", key=f"{key}_latest_reliability")
    st.caption("Checks describe this recorded inference attempt. Unassessed checks do not establish reliability; rejected estimates remain unavailable. Model eligibility is determined by Python checks.")
    if not current:
        st.warning("These are historical inference diagnostics. Current stream freshness or source health prevents using this observation as a current prediction.")
