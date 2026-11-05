"""Phase 6: historical PCA monitoring with independent reference/limit batches."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app_data import load_dataset
from src.data_generation import SimulationConfig
from src.multivariate_monitoring import (
    LIMITATIONS, NOMINAL_SCENARIO, SENSOR_COLUMNS, ReferencePartition,
    default_reference_partition, fit_process_monitor, monitoring_model_manifest,
    multivariate_report_zip, score_process_monitor, spe_sensor_contributions,
    summarize_monitoring_batches,
)
from src.navigation import pause_monitoring_on_navigation
from src.multivariate_storage import DEFAULT_PCA_ROOT, save_pca_reference, list_pca_references, load_pca_reference
from src.multivariate_stream_ui import render_acquired_pca
from src.llm_bootstrap import refresh_legacy_llm_modules

refresh_legacy_llm_modules()
from src.llm_results import multivariate_snapshot
from src.llm_ui import render_explanation


SENSOR_LABELS = {
    "temperature_c": "Temperature (°C)", "ph": "pH (dimensionless)",
    "dissolved_oxygen_pct": "Dissolved oxygen (% air saturation)",
    "feed_rate_l_h": "Feed flow (L/h)", "agitation_rpm": "Agitation (rpm)",
}


def process_chart(frame, batches, column, label, *, limit=None):
    """Preserve invalid gaps and mark designed disturbance onset separately."""
    figure = go.Figure()
    for batch_id, group in frame.groupby("batch_id", sort=False):
        group = group.sort_values("elapsed_time_h")
        figure.add_trace(go.Scatter(x=group.elapsed_time_h, y=group[column], mode="lines+markers", name=batch_id, connectgaps=False))
        if column in ("t2", "spe", "anomaly_ratio"):
            alarm_column = {"t2": "t2_alarm", "spe": "spe_alarm", "anomaly_ratio": "alarm"}[column]
            flagged = group[group[alarm_column].fillna(False)]
            figure.add_trace(go.Scatter(x=flagged.elapsed_time_h, y=flagged[column], mode="markers", marker=dict(symbol="x", size=10, color="#C2415C"), name=f"{batch_id}: exceedance", showlegend=False))
        onset = batches.loc[batches.batch_id.eq(batch_id), "disturbance_start_h"].iloc[0]
        if pd.notna(onset):
            figure.add_vline(x=float(onset), line_dash="dot", opacity=0.5, annotation_text=f"{batch_id}: designed disturbance")
    if limit is not None:
        figure.add_hline(y=limit, line_dash="dash", line_color="#D97706", annotation_text="Reference empirical limit")
    figure.update_layout(xaxis_title="Elapsed process time (h)", yaxis_title=label, height=340, margin=dict(l=15, r=15, t=35, b=15), legend=dict(orientation="h"))
    return figure


def render_learning():
    st.subheader("Two complementary distances")
    st.latex(r"z_j=\frac{x_j-\mu_j^{fit}}{s_j^{fit}},\quad t=(z-\bar z)P,\quad T^2=\sum_{k=1}^{A}\frac{t_k^2}{\lambda_k},\quad Q=\|z-\hat z\|^2")
    st.write("PCA captures correlated sensor variation. T² measures an extreme position within the retained pattern. SPE/Q measures variation outside it. Sensor units differ, so the five observed sensor variables are autoscaled using fitting batches only. The eigenvalues use sample variance (n−1); the scaler uses population SD (n).")
    st.write("Numerical example: scores (2, 1) with eigenvalues (4, 0.25) give T² = 1 + 4 = 5. A residual (0.3, −0.4, 0, 0, 0) gives Q = 0.25 in standardized squared units. For limits 4 and 0.5, the anomaly ratio is max(5/4, 0.25/0.5) = 1.25; only T² exceeds its limit.")
    st.markdown("""
1. Fit the default reference model and compare the held-out nominal and disturbed batches.
2. Inspect T², Q, and process trends near the designed disturbance onset. Compare early nominal exceedances with post-disturbance exceedances.
3. Inspect PCA loadings, then one observation's squared residual contributions. Contributions describe the model residual, not a proven physical cause.
4. Try another component count or quantile and recalibrate. Once evaluation results influence a choice, subsequent results are exploratory.

The upper limit for each chart is a separate nominal batch's empirical quantile using the `higher` order statistic. An observation flags when **either statistic is strictly above its limit**. Equality is within limits. The anomaly ratio is `max(T² / limit, Q / limit)`; it is not a probability.
""")
    for limitation in LIMITATIONS:
        st.caption(limitation)
    st.write("Read docs/phase_6_learning_guide.md for equations, code explanations, tests, assumptions, and three interview questions with sample answers. Open Model_Reliability from the sidebar to review saved model evidence, residuals, bias, and distribution shifts.")
    st.markdown("References: [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) · [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) · [NumPy quantiles](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html) · [NIST multivariate charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc341.htm)")


def save_reference_control(model, config, root):
    if st.button("Save fitted historical PCA reference", key="mspc_save_reference"):
        try:
            reference_config = SimulationConfig(**st.session_state.get("mspc_reference_config", asdict(config)))
            identity = save_pca_reference(model, reference_config, root)
            st.success(f"Historical PCA reference saved: {identity[:12]}. Incoming observations do not update it.")
        except (OSError, ValueError) as error:
            st.error(f"PCA reference could not be saved: {error}")


def main():
    st.set_page_config(page_title="PAT | Multivariate Monitoring", page_icon="📊", layout="wide")
    views = ["Historical reference comparison", "Acquired sensor monitoring"]
    requested = st.session_state.pop("mspc_mode_requested", None)
    if requested in views:
        st.session_state["mspc_mode_preference"] = requested
        st.session_state.pop("mspc_mode", None)
    default_view = views[1] if st.session_state.get("stream_monitor_workspace") is not None else views[0]
    mode = st.radio("Multivariate data source", views, index=views.index(st.session_state.get("mspc_mode_preference", default_view)), key="mspc_mode")
    st.session_state["mspc_mode_preference"] = mode
    if mode == views[0]:
        pause_monitoring_on_navigation()
    else:
        replay = st.session_state.get("monitoring_session")
        if replay is not None:
            replay.clock.pause()
    st.caption("PAT INTELLIGENCE PLATFORM")
    st.title("Monitor the joint process pattern.")
    st.write("Reference-fitted PCA, Hotelling’s T², and SPE/Q for temperature, pH, dissolved oxygen, feed flow, and agitation.")
    st.info("Historical reference calibration and acquired sensor monitoring · Advisory only · Empirical limits · No manufacturing validation")
    st.caption("Fit the PCA reference using historical development batches. Acquired mode applies that fitted scaler, PCA and limits to the same received sensor collection used by Process Monitoring.")
    config = st.session_state.get("simulation_config", SimulationConfig())
    context = json.dumps(asdict(config), sort_keys=True)
    if st.session_state.get("mspc_data_context") != context:
        for key in ("mspc_model", "mspc_reference_config", "mspc_loaded_reference", "mspc_fit_batches", "mspc_limit_batches", "mspc_components", "mspc_quantile", "mspc_batches", "mspc_inspect_batch", "mspc_inspect_sample"):
            st.session_state.pop(key, None)
        st.session_state["mspc_data_context"] = context
    dataset = load_dataset(config)
    root = Path(st.session_state.get("multivariate_state_root", DEFAULT_PCA_ROOT))
    with st.expander("Saved historical PCA references"):
        references = list_pca_references(root)
        if references:
            identity = st.selectbox("Saved PCA reference", list(references), format_func=lambda value: references[value]["reference"]["model_id"] + " · " + value[:8], key="mspc_saved_reference")
            if st.button("Load saved PCA reference", key="mspc_load_reference"):
                try:
                    loaded, metadata = load_pca_reference(root, identity)
                    st.session_state["mspc_model"] = loaded
                    st.session_state["mspc_loaded_reference"] = metadata
                    st.session_state["mspc_reference_config"] = metadata["historical_config"]
                    for key in ("mspc_fit_batches", "mspc_limit_batches", "mspc_components", "mspc_quantile"):
                        st.session_state.pop(key, None)
                except (ValueError, OSError, KeyError) as error:
                    st.error(f"Saved PCA reference is unavailable: {error}")
        else:
            st.caption("No saved PCA reference found. Fit historical reference data first, then save it explicitly below.")
    if mode == views[1] and st.session_state.get("mspc_model") is not None:
        model = st.session_state["mspc_model"]
        st.caption(f"Historical fitted reference: {model.model_id}. Acquired mode never refits its scaler, components or limits.")
        save_reference_control(model, config, root)
        render_acquired_pca(model)
        return
    if mode == views[1]:
        stream = st.session_state.get("stream_monitoring_session")
        if stream is not None:
            stream.pause()
        st.info("Prepare or load a historical PCA reference first. Shared acquisition is paused during reference setup; its received records and cursor are retained.")
    try:
        default = default_reference_partition(dataset.batches)
    except ValueError as error:
        st.warning(str(error))
        st.write("Use at least four batches to include a deliberately disturbed batch. The default six-batch configuration provides three fitting batches, one limit batch, and two evaluation batches.")
        with st.expander("Information"):
            render_learning()
        return
    nominal = dataset.batches.loc[dataset.batches.scenario.eq(NOMINAL_SCENARIO), "batch_id"].tolist()
    saved_model = st.session_state.get("mspc_model")
    saved_partition = saved_model.partition if saved_model else default
    if not set(saved_partition.fit_batches + saved_partition.limit_batches).issubset(nominal):
        saved_partition = default
    with st.sidebar:
        st.header("Reference calibration")
        st.caption(f"Overview: seed {config.seed}, {config.n_batches} batches, {config.duration_h:g} h. Inputs are observed sensors only.")
        with st.form("mspc_reference_form"):
            fit_batches = st.multiselect("PCA fitting batches", nominal, default=list(saved_partition.fit_batches), key="mspc_fit_batches")
            limit_batches = st.multiselect("Limit-calibration batches", nominal, default=list(saved_partition.limit_batches), key="mspc_limit_batches")
            components = st.slider("Retained PCA components", 1, 4, int(saved_model.pca.n_components_) if saved_model else 2, key="mspc_components")
            quantiles = [0.95, 0.975, 0.99]
            quantile = st.selectbox("Per-chart empirical quantile", quantiles, index=quantiles.index(saved_model.quantile) if saved_model else 2, key="mspc_quantile")
            submitted = st.form_submit_button("Fit reference monitoring model", type="primary", key="mspc_fit")
        st.caption("The remaining batches are evaluation only. Fitting and limit batches must be disjoint and nominal. Reserve one nominal evaluation batch. Controls take effect when you fit.")
    partition = ReferencePartition(tuple(fit_batches), tuple(limit_batches), tuple(batch for batch in dataset.batches.batch_id if batch not in fit_batches + limit_batches))
    if submitted:
        st.session_state.pop("mspc_model", None)
        st.session_state.pop("mspc_batches", None)
        try:
            st.session_state["mspc_model"] = fit_process_monitor(dataset.sensors, dataset.batches, partition=partition, n_components=components, quantile=quantile)
            st.session_state["mspc_reference_config"] = asdict(config)
        except (ValueError, TypeError, FloatingPointError) as error:
            st.error(f"Reference calibration could not complete: {error}")
    model = st.session_state.get("mspc_model")
    if model is None:
        st.write("Select nominal reference batches and click **Fit reference monitoring model**. No calibration or evaluation result is created before this action.")
        with st.expander("Information", expanded=True):
            render_learning()
        render_explanation(None, key="multivariate", empty_message="Fit a reference monitoring model to explain its recorded limits and chart statistics.")
        return

    save_reference_control(model, config, root)
    if mode == views[1]:
        render_acquired_pca(model)
        return

    scored = score_process_monitor(model, dataset.sensors)
    summary = summarize_monitoring_batches(scored, dataset.batches)
    evaluation = scored[scored.role.eq("Evaluation")]
    manifest = monitoring_model_manifest(model)
    st.caption(f"{model.model_id} · {model.pca.n_components_} components · Reference empirical quantile {model.quantile:g} · Chart controls never refit the model")
    cards = st.columns(4)
    cards[0].metric("PCA fitting observations", int(((model.reference_audit.role == "PCA fitting") & (model.reference_audit.status == "scored")).sum()))
    cards[1].metric("Limit-calibration observations", int(((model.reference_audit.role == "Limit calibration") & (model.reference_audit.status == "scored")).sum()))
    cards[2].metric("Evaluation observations scored", int(evaluation.status.eq("scored").sum()))
    cards[3].metric("Evaluation exceedances", int(evaluation.alarm.sum()))
    st.dataframe(summary, hide_index=True, width="stretch", key="mspc_batch_summary")
    st.caption("Scenario and disturbance onset are simulation annotations, never PCA features. Fractions use scored rows only. A batch flagged at any observation is a review signal, not proof of a process failure; nominal batches can also flag. Invalid observations are counted separately.")
    if evaluation.status.ne("scored").any():
        st.warning("Some evaluation inputs are unscorable. Their statistics are gaps and their alarm state is unknown.")
    st.warning("Empirical limits pool the whole time-varying batch trajectory. Few batches, temporal correlation, two charts, and repeated testing prevent interpreting the quantile as a guaranteed false-alarm rate.")
    selected = st.multiselect("Batches in comparison charts", summary.batch_id.tolist(), default=[batch for batch in model.partition.evaluation_batches if batch in summary.batch_id.tolist()], key="mspc_batches")
    view = scored[scored.batch_id.isin(selected)]
    monitor_tab, pca_tab, sources_tab, learn_tab = st.tabs(["Control charts & process trends", "PCA pattern & residuals", "Calibration & observations", "Information"])
    with monitor_tab:
        if view.empty:
            st.info("Select at least one batch to display control charts.")
        else:
            st.plotly_chart(process_chart(view, dataset.batches, "t2", "Hotelling’s T² (dimensionless)", limit=model.t2_limit), width="stretch", key="mspc_t2_chart")
            st.plotly_chart(process_chart(view, dataset.batches, "spe", "SPE/Q (standardized squared units)", limit=model.spe_limit), width="stretch", key="mspc_spe_chart")
            st.plotly_chart(process_chart(view, dataset.batches, "anomaly_ratio", "Maximum statistic / limit (dimensionless)", limit=1), width="stretch", key="mspc_ratio_chart")
            st.caption(f"T² limit = {model.t2_limit:.5g} · Q limit = {model.spe_limit:.5g}. Crosses mark exceedances. Dotted vertical lines mark designed disturbance onset, independently of detector alarms.")
            variable = st.selectbox("Observed process variable", list(SENSOR_COLUMNS), format_func=SENSOR_LABELS.get, key="mspc_sensor")
            st.plotly_chart(process_chart(view, dataset.batches, variable, SENSOR_LABELS[variable]), width="stretch", key="mspc_process_chart")
    with pca_tab:
        valid = view[view.status.eq("scored")].copy()
        if not valid.empty:
            valid["review"] = np.where(valid.alarm, "Exceedance", "Within limits")
            y = "PC2" if model.pca.n_components_ > 1 else "elapsed_time_h"
            figure = px.scatter(valid, x="PC1", y=y, color="batch_id", symbol="review", hover_data=["sample_id", "elapsed_time_h", "role"], labels={"PC1": "PC1 score (standardized coordinates)", "PC2": "PC2 score (standardized coordinates)", "elapsed_time_h": "Elapsed process time (h)"})
            st.plotly_chart(figure, width="stretch", key="mspc_scores_chart")
        explained = pd.DataFrame({"component": [f"PC{i + 1}" for i in range(model.pca.n_components_)], "eigenvalue": model.pca.explained_variance_, "explained_fraction": model.pca.explained_variance_ratio_, "cumulative_fraction": np.cumsum(model.pca.explained_variance_ratio_)})
        st.dataframe(explained, hide_index=True, width="stretch", key="mspc_variance")
        loadings = pd.DataFrame(model.pca.components_.T, index=pd.Index(SENSOR_COLUMNS, name="sensor"), columns=explained.component)
        st.dataframe(loadings.reset_index(), hide_index=True, width="stretch", key="mspc_loadings")
        st.caption("Loadings are unit-length eigenvectors in standardized sensor coordinates. Signs can reverse without changing T² or Q. These components were fitted only on the PCA fitting batches.")
        default_batch = next((batch for batch in model.partition.evaluation_batches if batch in summary.batch_id.tolist()), summary.batch_id.iloc[0])
        batch = st.selectbox("Residual inspection batch", summary.batch_id.tolist(), index=summary.batch_id.tolist().index(default_batch), key="mspc_inspect_batch")
        observations = scored[scored.batch_id.eq(batch)]
        sample = st.selectbox("Residual inspection sample", observations.sample_id.tolist(), key="mspc_inspect_sample")
        observation = observations[observations.sample_id.eq(sample)]
        contributions = None
        if observation.status.eq("scored").all():
            contributions = spe_sensor_contributions(model, observation)
            st.plotly_chart(px.bar(contributions, x="sensor", y="spe_contribution", labels={"spe_contribution": "Squared standardized residual", "sensor": "Observed sensor"}), width="stretch", key="mspc_contributions_chart")
            st.caption(f"Contributions sum to Q = {observation.spe.iloc[0]:.5g}. A large contribution suggests what to inspect; it does not establish a physical cause.")
        else:
            st.warning("This observation is unscorable: " + observation.quality_flags.iloc[0])
    with sources_tab:
        st.json(manifest)
        st.dataframe(model.reference_audit, hide_index=True, width="stretch", key="mspc_reference_audit")
        st.dataframe(view, hide_index=True, width="stretch", key="mspc_observations")
        st.caption("Raw measurements remain unchanged. Missing, nonfinite, duplicate, or implausible rows are audited without interpolation. Neither simulated glucose/lactate truth nor scenario labels enter the five-column PCA input matrix.")
        st.download_button("Download multivariate monitoring report (ZIP)", multivariate_report_zip(model, scored, dataset.batches, context={"simulation": asdict(config)}), file_name="pat_multivariate_monitoring.zip", mime="application/zip", key="mspc_report")
        st.caption("The download includes all batches and reference calibration, regardless of chart selection. This is a full historical comparison, separate from the revealed-only Process_Monitoring replay export.")
    with learn_tab:
        render_learning()
    render_explanation(multivariate_snapshot(model, view, observation, contributions,
        context={"simulation": asdict(config), "charts": selected, "inspection_batch": batch, "inspection_sample": sample}),
        key="multivariate")


if __name__ == "__main__":
    main()
