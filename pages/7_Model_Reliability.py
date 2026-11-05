"""Phase 7: frozen model evidence and causal historical reliability review."""

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app_data import load_dataset, load_monitoring_dataset
from src.demo_data import demo_provenance
from src.demo_data_ui import render_demo_provenance
from src.data_generation import SimulationConfig
from src.data_integration import integrate_sources, sources_from_dataset
from src.modeling_data import TARGETS, modeling_dataset
from src.modeling_storage import DEFAULT_STATE_ROOT
from src.monitoring import MonitoringSettings
from src.prediction_reliability_ui import render_study_reliability, render_observation_reliability
from src.navigation import pause_monitoring_on_navigation
from src.reliability import (
    FAULT_SCENARIOS, RELIABILITY_LIMITATIONS, ReliabilityReview, ReliabilitySettings,
    available_reference_residuals, build_reliability_reference, distribution_shift,
    frozen_performance, input_distribution_shift, reliability_fault_example,
    reliability_report_zip, residual_metrics,
)
from src.reliability_storage import (
    DEFAULT_RELIABILITY_ROOT, load_reliability_evidence, prepare_reliability_demo,
    reliability_demo_config,
)
from src.replay import replay_batch
from src.llm_bootstrap import refresh_legacy_llm_modules

refresh_legacy_llm_modules()
from src.llm_results import modeling_snapshot, reliability_snapshot
from src.llm_ui import render_explanation
from src.ui_labels import workflow_label


SOURCES = ["Separate reliability demo", "Saved Phase 4 synthetic study", "Saved Phase 4 integration study"]


@st.cache_data(max_entries=8, show_spinner=False)
def synthetic_cohort(config, target):
    return modeling_dataset(integrate_sources(sources_from_dataset(load_dataset(config))), target)


def render_learning():
    st.latex(r"e_i=\hat y_i-y_i^{assay},\quad RMSE=\sqrt{\frac{1}{n}\sum e_i^2},\quad Bias=\frac{1}{n}\sum e_i")
    st.write("Training errors describe a fitted model on its own data. Grouped CV predicts withheld training batches but is optimistic after selection. RMSEP describes the saved independent batch test, whose winner was selected before test disclosure. This page reads that result without retesting or changing it.")
    st.write("Example: predictions [2, 3] and assays [1, 3] give residuals [1, 0], bias 0.5 g/L, MAE 0.5 g/L, RMSE √0.5 ≈ 0.707 g/L, and R² = 0.5. A result reported at 6 h cannot trigger a residual alarm at 4 h, even if its sample was collected at 0 h.")
    st.latex(r"D_j=\frac{W_1(F_j^{train},F_j^{review})}{s_j^{train}},\qquad \Delta\mu_j=\frac{\bar x_j^{review}-\bar x_j^{train}}{s_j^{train}}")
    st.write("Wasserstein distance measures the average movement needed to match two empirical distributions. For [0, 2] versus [1, 3], W₁ = 1 and the reference population SD is 1, so D = 1. The threshold is an editable effect-size assumption, not a p-value or calibrated probability. Constant references use an explicit changed/unchanged check.")
    st.markdown("""
- **Sensor drift:** a possible measurement offset; an observed distribution shift can also reflect a real process change.
- **Spectral baseline drift:** a background change; raw edge means are only a proxy and also respond to signal/scatter.
- **Distribution shift:** changed inputs or predictions; it does not prove prediction error increased.
- **Prediction bias:** nonzero mean prediction-minus-assay error; assay bias is one possible explanation.
- **Model degradation:** worsened predictive behavior under suitable independent assessment; a drift flag alone cannot establish it.

Prepare the separate demo, review through the latest assay, then compare each controlled fault scenario. Move time backward to confirm future assays and their residuals disappear. The rolling bias requires the configured number of assessed pairs and follows result-arrival order within a batch.
""")
    for limitation in RELIABILITY_LIMITATIONS:
        st.caption(limitation)
    st.write("The equations, implementation walkthrough, examples, tests, and three interview questions are in docs/phase_7_learning_guide.md. Open Model_Deployment from the sidebar to inspect saved pipelines, run batch inference, and export release bundles.")
    st.markdown("References: [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html) · [SciPy Wasserstein distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)")


def main():
    st.set_page_config(page_title="PAT | Model Reliability", page_icon="🔎", layout="wide")
    views = ["Historical operational review", "Acquired data reliability"]
    requested = st.session_state.pop("reliability_mode_requested", None)
    if requested in views:
        st.session_state["reliability_mode_preference"] = requested
        st.session_state.pop("reliability_mode", None)
    default = views[1] if st.session_state.get("stream_monitor_workspace") is not None else views[0]
    mode = st.radio("Reliability data source", views, index=views.index(st.session_state.get("reliability_mode_preference", default)), key="reliability_mode")
    st.session_state["reliability_mode_preference"] = mode
    if mode == views[0]:
        pause_monitoring_on_navigation()
    else:
        replay = st.session_state.get("monitoring_session")
        if replay is not None:
            replay.clock.pause()
    st.caption("PAT INTELLIGENCE PLATFORM")
    st.title("Review model evidence. Follow reliability over time.")
    if mode == views[1]:
        from src.acquired_reliability_ui import render_acquired_reliability
        st.info("Shared acquired-data review · Frozen model evidence remains separate · Advisory only · No manufacturing validation")
        render_acquired_reliability()
        return
    st.info("Synthetic historical review · Frozen independent-batch evidence · Advisory only · No manufacturing validation")
    overview = st.session_state.get("simulation_config", SimulationConfig())
    preferences = st.session_state.get("reliability_preferences", {})
    context = json.dumps(asdict(overview), sort_keys=True)
    if st.session_state.get("reliability_data_context") != context:
        for key in ("reliability_review", "reliability_clock", "reliability_batch", "reliability_scenario"):
            st.session_state.pop(key, None)
        st.session_state["reliability_data_context"] = context
    with st.sidebar:
        st.header("Frozen study evidence")
        source = st.selectbox("Study source", SOURCES, index=SOURCES.index(preferences.get("source", SOURCES[0])), format_func=workflow_label, key="reliability_source")
        target = st.selectbox("Reliability target", list(TARGETS), index=list(TARGETS).index(preferences.get("target", "Glucose")), key="reliability_target")
    demo_root = Path(st.session_state.get("reliability_state_root", DEFAULT_RELIABILITY_ROOT))
    root = Path(st.session_state.get("modeling_state_root", DEFAULT_STATE_ROOT))
    try:
        if source == SOURCES[0]:
            demo_config = reliability_demo_config(overview)
            cohort = synthetic_cohort(demo_config, target)
            root = demo_root / target.lower()
            st.caption(f"Separate demo calibration: seed {demo_config.seed}, {demo_config.n_batches} batches. Fixed raw-Raman PLS recipe; training-only grouped CV chooses its component count. B007/B008 are the default independent test. Existing studies are preserved.")
            if st.button("Prepare / reuse frozen reliability demo", key="prepare_reliability_demo"):
                with st.spinner("Preparing or reusing a separate, frozen CV/test study…"):
                    prepare_reliability_demo(overview, target, demo_root)
        elif source == SOURCES[1]:
            cohort = synthetic_cohort(replace(overview, n_batches=max(8, overview.n_batches)), target)
        else:
            integration = st.session_state.get("integration_result")
            if integration is None:
                st.warning("Run Data_Integration first, then select the same target/source used for the saved Predictive_Modeling study.")
                return
            cohort = modeling_dataset(integration, target)
        study, evaluation, bundle = load_reliability_evidence(cohort, root)
    except (ValueError, OSError, KeyError) as error:
        st.error(f"Study evidence is unavailable: {error}")
        return
    st.session_state["reliability_preferences"] = {**preferences, "source": source, "target": target}
    if study is None:
        st.write("No matching saved study exists. Prepare the separate demo above, or train and finalize the matching study on Predictive Modeling. Ordinary page visits never train or reveal a holdout.")
        with st.expander("Information", expanded=True):
            render_learning()
        return
    render_study_reliability(study, evaluation, key="review_prediction_reliability")
    st.subheader("Recorded performance of the CV-selected model")
    st.dataframe(frozen_performance(study, evaluation), hide_index=True, width="stretch", key="reliability_performance")
    st.caption(f"Study {study.study_id[:12]} · Cohort {study.cohort_id[:12]} · Target {target} (g/L) · Winner {study.winner_id} · Training batches: {', '.join(study.partition.training_batches)} · Test batches: {', '.join(study.partition.holdout_batches)}")
    st.caption("CV was used for hyperparameter/model selection and may be optimistic. The independent test is read from its completed saved record; this page does not reselect a winner or reopen Predictive_Modeling tuning.")
    if evaluation is None or bundle is None:
        st.warning("The study has no completed independent test. Finalize it explicitly on Predictive Modeling; this page will not claim or evaluate its holdout.")
        render_explanation(modeling_snapshot(study, None,
            source_kind="prepared_sources" if source == SOURCES[2] else "synthetic"), key="reliability")
        return
    batches = evaluation["batch_metrics"]
    st.dataframe(batches[batches.case_id.eq(study.winner_id)], hide_index=True, width="stretch", key="reliability_frozen_batch_metrics")
    st.caption(f"Saved pipeline version: {bundle.version} · Artifact identity: {bundle.metadata['artifact_identity'][:12]} · scikit-learn {bundle.metadata['versions']['scikit_learn']}")
    try:
        reference = build_reliability_reference(study, cohort)
    except ValueError as error:
        st.error(str(error))
        return

    st.subheader("Historical operational review — separate from the frozen test")
    dataset = load_monitoring_dataset(overview)
    monitoring_provenance = demo_provenance(overview, monitoring=True)
    render_demo_provenance(monitoring_provenance)
    with st.sidebar:
        st.header("Review clock and rules")
        batch_ids = dataset.batches.batch_id.tolist()
        remembered = preferences.get("batch", "MON-B004" if "MON-B004" in batch_ids else batch_ids[0])
        selected_batch = st.selectbox("Review batch", batch_ids, index=batch_ids.index(remembered) if remembered in batch_ids else 0, key="reliability_batch")
        scenario = st.selectbox("Controlled reliability example", list(FAULT_SCENARIOS), index=list(FAULT_SCENARIOS).index(preferences.get("scenario", FAULT_SCENARIOS[0])), key="reliability_scenario")
        residual_limit = st.number_input("Absolute residual limit (g/L)", min_value=0.01, value=preferences.get("residual_limit", 0.5), step=0.05, key="reliability_residual_limit")
        bias_limit = st.number_input("Absolute rolling bias limit (g/L)", min_value=0.01, value=preferences.get("bias_limit", 0.25), step=0.05, key="reliability_bias_limit")
        rolling_pairs = st.number_input("Pairs in rolling bias", min_value=2, max_value=20, value=preferences.get("rolling_pairs", 3), step=1, key="reliability_rolling_pairs")
        drift_limit = st.number_input("Distribution distance / training SD limit", min_value=0.1, value=preferences.get("drift_limit", 1.0), step=0.1, key="reliability_drift_limit")
        domain_limit = st.number_input("Model input maximum |z|", min_value=0.5, max_value=100.0, value=preferences.get("domain_limit", 6.0), step=0.5, key="reliability_domain_limit")
        st.caption("Rules are illustrative. Residual/bias equality is within limits. Drift requires five finite observations per distribution. Changing review rules does not alter frozen validation.")
    rules = ReliabilitySettings(residual_limit, bias_limit, drift_limit, rolling_pairs=int(rolling_pairs))
    inference_rules = MonitoringSettings(domain_z_limit=domain_limit)
    batch = reliability_fault_example(replay_batch(dataset, selected_batch), scenario, target)
    if scenario != FAULT_SCENARIOS[0]:
        st.warning(f"Controlled copy: {scenario}. Original observations and frozen test evidence remain unchanged. Assay bias demonstrates a reference fault, not a changed model.")
    start = batch.metadata.measurement_timestamp.iloc[0]
    latest = max(batch.metadata.measurement_timestamp.max(), batch.assays.result_timestamp.max())
    maximum = float(np.ceil((latest - start).total_seconds() / 3600 * 2) / 2)
    signature = sha256(json.dumps({"study": study.study_id, "artifact": bundle.metadata["artifact_stamp"], "overview": asdict(overview), "data_provenance": monitoring_provenance, "batch": selected_batch, "scenario": scenario, "domain": domain_limit}, sort_keys=True).encode()).hexdigest()
    review = st.session_state.get("reliability_review")
    if review is None or review.context_id != signature:
        review = ReliabilityReview(signature)
        st.session_state["reliability_review"] = review
        st.session_state.pop("reliability_clock", None)
    if st.button("Review through latest assay", key="reliability_latest"):
        st.session_state["reliability_clock"] = maximum
    elapsed = st.slider("Review elapsed time (h)", 0.0, maximum, step=0.5, key="reliability_clock")
    cutoff = start + pd.Timedelta(hours=elapsed)
    history = review.history(batch, cutoff, bundle, inference_rules)
    if not history.empty:
        render_observation_reliability(history.iloc[-1], targets=[target], prefixed=False, key="review")
    residuals = available_reference_residuals(history, batch.assays, target, cutoff, rules)
    inputs = batch.inputs.loc[history.sample_id]
    compatible_axes = batch.axis.shape == study.axis.shape and np.allclose(batch.axis, study.axis, rtol=1e-10, atol=1e-10)
    input_shift = input_distribution_shift(reference.input_features, inputs, rules, compatible_axes=compatible_axes)
    prediction_shift = distribution_shift(pd.DataFrame({"prediction_g_l": reference.predictions_g_l}),
        pd.DataFrame({"prediction_g_l": history.loc[history.prediction_status.eq("ok"), "prediction_g_l"]}), rules)
    st.session_state["reliability_preferences"] = {"source": source, "target": target, "batch": selected_batch, "scenario": scenario,
        "residual_limit": residual_limit, "bias_limit": bias_limit, "rolling_pairs": int(rolling_pairs), "drift_limit": drift_limit, "domain_limit": domain_limit}
    metrics = residual_metrics(residuals)
    cards = st.columns(4)
    cards[0].metric("Revealed observations", len(history))
    cards[1].metric("Available assays", len(residuals))
    cards[2].metric("Assessed reference pairs", int(metrics.assessed_pairs.iloc[0]))
    cards[3].metric("Residual exceedances", int(residuals.residual_alarm.sum()))
    st.caption(f"References available through {cutoff} · Inference uses the saved pipeline · Separate monitoring demo seed and MON- identities · Operational review does not reopen or replace the frozen independent test")
    invalid = history.prediction_status.ne("ok")
    if invalid.any():
        st.warning(f"{int(invalid.sum())} revealed predictions are unavailable. Inspect missing/implausible input, domain, axis, or prediction flags. No reference or simulation truth is substituted.")
    if residuals.residual_alarm.fillna(False).any() or residuals.bias_alarm.fillna(False).any():
        st.warning("Available assay residuals or rolling bias exceed configured limits. Review model inputs, assay quality, and process context; this does not establish the cause.")
    if input_shift.shift_flag.fillna(False).any() or prediction_shift.shift_flag.fillna(False).any():
        st.warning("An input or prediction distribution exceeds the illustrative shift screen. Shift alone does not establish model degradation.")
    prediction_tab, drift_tab, audit_tab, learn_tab = st.tabs(["Assay residuals & bias", "Input & prediction shift", "Provenance & report", "Information"])
    with prediction_tab:
        figure = go.Figure(go.Scatter(x=history.measurement_timestamp, y=history.prediction_g_l, mode="lines+markers", name="Saved-pipeline estimate", connectgaps=False))
        figure.add_trace(go.Scatter(x=residuals.collection_timestamp, y=residuals.reference_g_l, mode="markers", marker=dict(symbol="diamond", size=9), name="Assay (result now available)"))
        figure.update_layout(xaxis_title="Collection / measurement time (UTC)", yaxis_title=f"{target} (g/L)")
        st.plotly_chart(figure, width="stretch", key="reliability_prediction_chart")
        errors = go.Figure(go.Scatter(x=residuals.collection_timestamp, y=residuals.residual_g_l, mode="lines+markers", connectgaps=False, name="Prediction − assay"))
        for bound in (-residual_limit, residual_limit):
            errors.add_hline(y=bound, line_dash="dash")
        errors.update_layout(xaxis_title="Assay collection time (UTC)", yaxis_title="Prediction − assay residual (g/L)")
        st.plotly_chart(errors, width="stretch", key="reliability_residual_chart")
        bias = go.Figure(go.Scatter(x=residuals.result_timestamp, y=residuals.rolling_bias_g_l, mode="lines+markers", connectgaps=False, name="Rolling bias"))
        for bound in (-bias_limit, bias_limit):
            bias.add_hline(y=bound, line_dash="dot")
        bias.update_layout(xaxis_title="Laboratory result available time (UTC)", yaxis_title=f"Rolling bias over {rolling_pairs} assessed pairs (g/L)")
        st.plotly_chart(bias, width="stretch", key="reliability_bias_chart")
        st.dataframe(metrics, hide_index=True, width="stretch", key="reliability_reference_metrics")
        st.dataframe(residual_metrics(residuals, by_batch=True), hide_index=True, width="stretch")
        st.dataframe(residuals, hide_index=True, width="stretch", key="reliability_residuals")
        st.caption("Metrics use only assessed, exact sample/time pairs. Invalid/unavailable pairs remain explicit. R² is undefined for fewer than two varying references. No result from after the review cutoff contributes to these residuals or rolling bias.")
    with drift_tab:
        st.dataframe(input_shift, hide_index=True, width="stretch", key="reliability_input_shift")
        st.dataframe(prediction_shift, hide_index=True, width="stretch", key="reliability_prediction_shift")
        st.caption("References are the original training assay-paired inputs and fitted predictions, not evaluation batches. Input screens use finite observed values and report missing counts; they do not replace plausibility checks. Raman mean and edge mean are baseline indicators, confounded by chemistry and scatter. No spectral drift comparison is valid when model and review axes differ.")
        if not compatible_axes:
            st.warning("Spectral axes differ: Raman distribution summaries are not directly comparable; inference reports axis mismatch.")
        st.caption("Prediction shift excludes rejected predictions; input rejection can change the remaining distribution. Early/partial-batch windows differ naturally from whole-batch training data. These effects can create review flags without model failure. Drift rows with too few observations stay unassessed.")
    with audit_tab:
        st.dataframe(history, hide_index=True, width="stretch", key="reliability_history")
        st.dataframe(study.folds, hide_index=True, width="stretch")
        context = {"source": source, "model_version": bundle.version, "artifact_identity": bundle.metadata["artifact_identity"],
            "data_provenance": monitoring_provenance,
            "artifact_stamp": bundle.metadata["artifact_stamp"], "review_simulation": asdict(overview), "batch_id": selected_batch,
            "scenario": scenario, "cutoff_utc": cutoff.isoformat(), "inference_rules": asdict(inference_rules),
            "training_reference_row_ids": reference.training_row_ids}
        st.download_button("Download model reliability report (ZIP)", reliability_report_zip(study, evaluation, history, residuals, input_shift, prediction_shift, rules, context), file_name="pat_model_reliability.zip", mime="application/zip", key="reliability_report")
        st.caption("The ZIP separates previously frozen historical validation evidence from review/ tables restricted to this clock. Moving time backward hides future observations and reference results in the review export. Rules, source, model version, training partition, and quality flags are recorded.")
    with learn_tab:
        render_learning()
    render_explanation(reliability_snapshot(study, evaluation, history, residuals, metrics,
        input_shift, prediction_shift, rules,
        context={"review": signature, "cutoff_utc": cutoff.isoformat(), "rules": asdict(rules)},
        source_kind="prepared_sources" if source == SOURCES[2] else "synthetic"), key="reliability")


if __name__ == "__main__":
    main()
