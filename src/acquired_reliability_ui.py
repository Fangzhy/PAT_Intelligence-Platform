"""Acquired reliability view over the shared monitoring workspace."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.acquired_reliability import acquired_reliability_review, acquired_reliability_report_zip
from src.acquired_reliability_storage import (DEFAULT_ACQUIRED_RELIABILITY_ROOT,
    load_acquired_reference, prepare_acquired_reference)
from src.data_generation import SimulationConfig, generate_dataset
from src.data_integration import integrate_sources, sources_from_dataset
from src.demo_data_ui import render_received_provenance
from src.llm_results import acquired_reliability_snapshot
from src.llm_ui import render_captured_explanation
from src.modeling_data import modeling_dataset
from src.prediction_reliability_ui import render_model_policy, render_observation_reliability
from src.reliability import ReliabilitySettings


def _control(action):
    session = st.session_state.get("stream_monitoring_session")
    if session is None:
        return
    if action == "pause":
        session.pause()
    elif action == "start":
        session.acquisition.running = True
    else:
        st.session_state["reliability_poll_shared"] = True


def _prepare_baseline(bundle, source, overview, root):
    if source == "Model demo calibration":
        config = bundle.metadata.get("calibration_config")
        if config is None:
            raise ValueError("This model has no separate demo calibration configuration.")
        integration = integrate_sources(sources_from_dataset(generate_dataset(SimulationConfig(**config))))
    elif source == "Historical synthetic development cohort":
        integration = integrate_sources(sources_from_dataset(generate_dataset(replace(overview, n_batches=max(8, overview.n_batches)))))
    else:
        integration = st.session_state.get("integration_result")
        if integration is None:
            raise ValueError("Restore the model's original prepared development sources on Data Integration first.")
    return prepare_acquired_reference(bundle, modeling_dataset(integration, bundle.target), root)


def _line(frame, x, y, label, limits=()):
    figure = go.Figure(go.Scatter(x=frame[x], y=frame[y], mode="lines+markers", connectgaps=False, name=label))
    for limit in limits:
        figure.add_hline(y=limit, line_dash="dash")
    figure.update_layout(xaxis_title=x.replace("_", " "), yaxis_title=label, height=320)
    return figure


@st.fragment(run_every=0.5)
def render_acquired_reliability():
    session = st.session_state.get("stream_monitoring_session")
    workspace = st.session_state.get("stream_monitor_workspace")
    if session is None or workspace is None or workspace["signature"] != session.context_id:
        st.info("Configure Data Acquisition and select a saved monitoring model first. This page reads the same collection and creates no source connection.")
        if st.button("Open Data Acquisition", key="reliability_open_acquisition"):
            st.switch_page("pages/9_Test_Acquisition.py")
        return
    models = workspace["models"]
    targets = [target for target, bundle in models.items() if bundle is not None]
    if not targets:
        session.pause()
        st.info("Shared acquisition is paused; received records and cursor are retained. Select a saved glucose or lactate model in Data Acquisition / Process Monitoring, then resume collection. Model reliability needs a recorded prediction model.")
        return
    target = st.selectbox("Active monitoring target", targets, key="acquired_reliability_target")
    bundle = models[target]
    controls = st.columns(3)
    controls[0].button("Start / resume shared acquisition", key="reliability_shared_start", disabled=session.acquisition.running or session.acquisition.complete, on_click=_control, args=("start",))
    controls[1].button("Pause shared acquisition", key="reliability_shared_pause", disabled=not session.acquisition.running, on_click=_control, args=("pause",))
    controls[2].button("Poll shared source once", key="reliability_shared_poll", disabled=session.acquisition.running or session.acquisition.complete, on_click=_control, args=("poll",))
    forced = st.session_state.pop("reliability_poll_shared", False)
    if forced or session.acquisition.running:
        adapter = st.session_state.get("stream_monitor_adapter")
        if adapter is None:
            session.pause()
            st.error("Shared adapter unavailable. Restore the source on Data Acquisition.")
            return
        session.poll_once(adapter, models, workspace["monitoring"], workspace["rules"], limit=workspace["limit"])
    now = pd.Timestamp.now(tz="UTC")
    st.caption(f"Shared source: {workspace['context']['source']} · Cursor: {session.acquisition.cursor} · As of {now}")
    st.caption(f"{target} · Model {bundle.version} · Artifact {bundle.metadata['artifact_identity'][:12]} · Study {bundle.metadata['study_id'][:12]}")
    render_received_provenance(session.acquisition.snapshot().batches)
    _, health = session.health(workspace["rules"], now=now)
    for flag in health:
        st.warning(flag)
    st.subheader("Recorded historical model evidence")
    frozen = []
    for field, label in (("cv_metrics", "Grouped CV (selection)"), ("holdout_metrics", "Independent test (frozen)")):
        metrics = bundle.metadata.get(field)
        if metrics:
            frozen.append({"evidence": label, **{key: value for key, value in metrics.items() if key in ("RMSE", "RMSEP", "MAE", "Bias", "R2")}})
    st.dataframe(pd.DataFrame(frozen), hide_index=True, width="stretch", key="acquired_reliability_frozen")
    st.caption(bundle.metadata["validation_status"])
    if not bundle.metadata.get("holdout_metrics"):
        st.warning("No matching completed independent-test metrics are recorded for this active model. Operational errors do not replace independent validation.")
    render_model_policy(bundle.pipeline, key="acquired_reliability_policy")
    root = Path(st.session_state.get("acquired_reliability_state_root", DEFAULT_ACQUIRED_RELIABILITY_ROOT))
    try:
        reference = load_acquired_reference(bundle, root)
    except (ValueError, OSError, KeyError) as error:
        st.error(f"Training reliability reference unavailable: {error}")
        reference = None
    with st.expander("Original training reliability baseline"):
        if reference is None:
            st.caption("Drift checks need the exact model's original training cohort. Preparing a baseline uses the saved pipeline on historical training rows once; it never trains or evaluates the holdout.")
            options = ["Model demo calibration", "Historical synthetic development cohort", "Latest prepared development sources"]
            source = st.selectbox("Baseline development source", options, index=0 if bundle.metadata.get("calibration_config") else 1, key="acquired_reliability_baseline_source")
            if st.button("Prepare / save matching training baseline", key="acquired_reliability_prepare"):
                try:
                    reference = _prepare_baseline(bundle, source, st.session_state.get("simulation_config", SimulationConfig()), root)
                    st.success("Matching original training baseline saved. Incoming data and validation artifacts are unchanged.")
                except (ValueError, OSError, KeyError) as error:
                    st.error(str(error))
        else:
            st.caption(f"Saved training-only baseline · {len(reference['training_row_ids'])} rows · {reference['sha256'][:12]}")
    raw = session.history()
    if raw.empty:
        st.info("No prediction observations received. Poll or start the shared acquisition.")
        return
    batches = sorted(raw.batch_id.unique())
    batch = st.selectbox("Acquired reliability batch", batches, key="acquired_reliability_batch")
    window = st.number_input("Review lookback (process hours)", min_value=1.0, max_value=720.0, value=24.0, key="acquired_reliability_window")
    with st.expander("Operational review rules"):
        residual = st.number_input("Acquired absolute residual limit (g/L)", min_value=0.01, value=0.5, key="acquired_reliability_residual_limit")
        bias = st.number_input("Acquired absolute rolling bias limit (g/L)", min_value=0.01, value=0.25, key="acquired_reliability_bias_limit")
        pairs = st.number_input("Acquired pairs in rolling bias", min_value=2, max_value=20, value=3, key="acquired_reliability_pairs")
        drift = st.number_input("Acquired distribution distance / training SD limit", min_value=0.1, value=1.0, key="acquired_reliability_drift_limit")
        st.caption("At least five finite observations per distribution. Thresholds are illustrative; equality is within limits. These rules never change prediction eligibility or frozen validation.")
    rules = ReliabilitySettings(residual, bias, drift, rolling_pairs=int(pairs))
    review = acquired_reliability_review(session, bundle, rules, workspace["rules"], reference, now=now, batch_id=batch, window_hours=window)
    history, assays, metrics = review["history"], review["residuals"], review["metrics"]
    cards = st.columns(4)
    cards[0].metric("Received prediction observations", len(history))
    cards[1].metric("Unavailable predictions", int(history.prediction_status.ne("ok").sum()))
    cards[2].metric("Available assay references", len(assays))
    cards[3].metric("Assessed operational pairs", int(metrics.assessed_pairs.iloc[0]))
    st.caption(f"Review window: {review['context']['observed_from_utc']} through {review['context']['observed_through_utc']}. Lookback ends at the latest received measurement. Assay availability is assessed at the current UTC clock.")
    if not history.empty:
        latest = history.sort_values(["measurement_timestamp", "spectrum_event_id"]).iloc[-1]
        st.metric("Latest current prediction availability", "Eligible" if latest.current_eligible else "Unavailable")
        render_observation_reliability(latest, targets=[target], prefixed=False, key="acquired_reliability", current=bool(latest.current_eligible))
    if not workspace["rules"].enforce_freshness:
        st.warning("Wall-clock freshness is disabled for historical testing. Recorded scores are available; current prediction status remains unavailable.")
    if not int(metrics.assessed_pairs.iloc[0]):
        st.info("Prediction accuracy not yet assessed. A valid matching received laboratory assay is required.")
    if assays.residual_alarm.fillna(False).any() or assays.bias_alarm.fillna(False).any():
        st.warning("Available assay residuals or rolling bias exceed the review rules. Inspect input quality, reference quality and process context; this does not establish the cause.")
    if any(review[name].shift_flag.fillna(False).any() for name in ("input_shift", "prediction_shift")):
        st.warning("An input or prediction distribution exceeds the illustrative shift screen. Shift alone does not establish model degradation.")
    residual_tab, drift_tab, audit = st.tabs(["Acquired assay residuals & bias", "Acquired input & prediction shift", "Acquired reliability audit & export"])
    with residual_tab:
        st.plotly_chart(_line(history, "measurement_timestamp", "prediction_g_l", f"{target} prediction (g/L)"), width="stretch", key="acquired_reliability_prediction_chart")
        st.plotly_chart(_line(assays, "collection_timestamp", "residual_g_l", "Prediction minus assay (g/L)", (-residual, residual)), width="stretch", key="acquired_reliability_residual_chart")
        st.plotly_chart(_line(assays, "pair_available_at_utc", "rolling_bias_g_l", "Rolling bias (g/L)", (-bias, bias)), width="stretch", key="acquired_reliability_bias_chart")
        st.dataframe(metrics, hide_index=True, width="stretch", key="acquired_reliability_metrics")
        st.dataframe(assays, hide_index=True, width="stretch", key="acquired_reliability_residuals")
        st.caption("Exact batch/sample/collection-time matching. Pair availability is the latest laboratory result, receipt and prediction assessment time. Invalid, conflicting and unmatched pairs remain unassessed. Rolling bias follows pair availability order within this batch/window.")
    with drift_tab:
        if reference is None:
            st.info("Training baseline unavailable: input and prediction drift are not assessed.")
        st.dataframe(review["input_shift"], hide_index=True, width="stretch", key="acquired_reliability_input_shift")
        st.dataframe(review["prediction_shift"], hide_index=True, width="stretch", key="acquired_reliability_prediction_shift")
        st.caption(f"{review['context']['input_drift_rows']} aligned, non-invalidated observations supply raw sensor/spectral indicators. Prediction shift uses accepted estimates only. Spectral axis mismatches leave spectral indicators unknown. No learned transformer is refitted.")
        st.caption("Partial-batch windows and process phase differ naturally from whole-batch training data. Raw spectral mean/edge indicators are confounded by chemistry and scatter. Distribution shift alone does not establish model degradation.")
    with audit:
        st.dataframe(history, hide_index=True, width="stretch", key="acquired_reliability_history")
        st.dataframe(session.acquisition.ledger(), hide_index=True, width="stretch", key="acquired_reliability_delivery_audit")
        st.download_button("Download acquired model reliability report (ZIP)", acquired_reliability_report_zip(review, session, bundle, reference), file_name="pat_acquired_model_reliability.zip", mime="application/zip", key="acquired_reliability_report")
        st.caption("Shared cursor/history are retained. Model-specific preprocessing stays in its saved pipeline; PCA retains its separate reference. Operational review never replaces the frozen independent test.")
    source_kind = "synthetic" if workspace["context"]["source"] in ("Fresh in-app test stream", "Historical in-app test source", "Local HTTP test source") else "prepared_sources"
    snapshot = acquired_reliability_snapshot(review, bundle, source_kind=source_kind)
    render_captured_explanation(snapshot, key="acquired_reliability", context_id=session.context_id + target, cutoff=now)
