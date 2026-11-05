"""Acquired PCA view over the existing shared monitoring workspace."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.demo_data_ui import render_received_provenance
from src.multivariate_stream import score_acquired_sensors, acquired_sensor_contributions, acquired_pca_report_zip


def _chart(frame, column, title, limit):
    figure = go.Figure()
    for batch, rows in frame.groupby("batch_id", sort=False):
        figure.add_trace(go.Scatter(x=rows.measurement_timestamp, y=rows[column], mode="lines+markers", name=batch, connectgaps=False))
    figure.add_hline(y=limit, line_dash="dash", annotation_text="Saved historical reference limit")
    figure.update_layout(xaxis_title="Sensor measurement time (UTC)", yaxis_title=title, height=330)
    return figure


def _control(action):
    session = st.session_state.get("stream_monitoring_session")
    if session is None:
        return
    if action == "pause":
        session.pause()
    elif action == "start":
        session.acquisition.running = True
    elif action == "poll":
        st.session_state["mspc_poll_shared"] = True


@st.fragment(run_every=0.5)
def render_acquired_pca(model):
    session = st.session_state.get("stream_monitoring_session")
    workspace = st.session_state.get("stream_monitor_workspace")
    if session is None or workspace is None or workspace["signature"] != session.context_id:
        st.info("Configure a monitoring source on Data Acquisition, then poll observations. This view uses that same collection; it does not create another source connection.")
        if st.button("Open Data Acquisition", key="mspc_open_acquisition"):
            st.switch_page("pages/9_Test_Acquisition.py")
        return
    controls = st.columns(3)
    controls[0].button("Start / resume shared acquisition", key="mspc_shared_start", disabled=session.acquisition.running or session.acquisition.complete, on_click=_control, args=("start",))
    controls[1].button("Pause shared acquisition", key="mspc_shared_pause", disabled=not session.acquisition.running, on_click=_control, args=("pause",))
    controls[2].button("Poll shared source once", key="mspc_shared_poll", disabled=session.acquisition.running or session.acquisition.complete, on_click=_control, args=("poll",))
    forced = st.session_state.pop("mspc_poll_shared", False)
    if forced or session.acquisition.running:
        adapter = st.session_state.get("stream_monitor_adapter")
        if adapter is None:
            session.pause()
            st.error("The shared source adapter is unavailable. Configure it again on Data Acquisition.")
            return
        session.poll_once(adapter, workspace["models"], workspace["monitoring"], workspace["rules"], limit=workspace["limit"])
    now = pd.Timestamp.now(tz="UTC")
    rules, monitoring = workspace["rules"], workspace["monitoring"]
    scored = score_acquired_sensors(model, session, rules, monitoring, now=now)
    st.caption(f"Shared source: {workspace['context']['source']} · Cursor: {session.acquisition.cursor} · PCA reference: {model.model_id} · As of {now}")
    render_received_provenance(session.acquisition.snapshot().batches)
    _, health = session.health(rules, now=now)
    if session.acquisition.last_error:
        st.error(session.acquisition.last_error)
    for flag in health:
        st.warning(flag)
    if not rules.enforce_freshness:
        st.warning("Wall-clock freshness is disabled for historical testing. Historical PCA scores do not establish current process freshness.")
    cards = st.columns(4)
    cards[0].metric("Received sensor observations", len(scored))
    cards[1].metric("PCA-scored observations", int(scored.status.eq("scored").sum()))
    cards[2].metric("Unscorable observations", int(scored.status.ne("scored").sum()))
    cards[3].metric("Historical PCA exceedances", int(scored.alarm.fillna(False).sum()))
    if scored.empty:
        st.info("No sensor observations have been received. Poll the shared source to begin.")
        return
    selected = st.selectbox("Acquired PCA batch", ["All batches", *sorted(scored.batch_id.unique())], key="mspc_acquired_batch")
    view = scored if selected == "All batches" else scored[scored.batch_id.eq(selected)]
    latest = view.sort_values(["measurement_timestamp", "sensor_event_id"]).iloc[-1]
    current = bool(latest.current_eligible)
    latest_cards = st.columns(3)
    latest_cards[0].metric("Latest sensor PCA availability", "Eligible" if current else "Unavailable")
    latest_cards[1].metric("Latest current T²", f"{latest.t2:.4g}" if current else "Unavailable")
    latest_cards[2].metric("Latest current SPE", f"{latest.spe:.4g}" if current else "Unavailable")
    latest_cards[0].caption("Exceeds an empirical limit" if current and bool(latest.alarm) else "Within empirical limits" if current else "Not assessed as current")
    st.caption(f"Latest sensor: {latest.sample_id} · Freshness: {latest.freshness_status} · Status: {latest.status}")
    st.caption("Historical valid scores remain on charts after later staleness. Current cards require a valid score, current freshness and healthy source. Invalid/conflicted and stale-on-arrival rows remain gaps with unknown alarm states.")
    st.caption("Empirical per-chart limits do not guarantee a false-alarm probability for repeatedly observed, correlated batch data.")
    charts, contributions_tab, audit = st.tabs(["Acquired PCA control charts", "Acquired SPE contributions", "Acquired quality & export"])
    with charts:
        st.plotly_chart(_chart(view, "t2", "Hotelling’s T² (dimensionless)", model.t2_limit), width="stretch", key="mspc_acquired_t2")
        st.plotly_chart(_chart(view, "spe", "SPE/Q (standardized squared units)", model.spe_limit), width="stretch", key="mspc_acquired_spe")
        st.plotly_chart(_chart(view, "anomaly_ratio", "Maximum statistic / limit", 1.0), width="stretch", key="mspc_acquired_ratio")
        st.caption("The scaler, PCA components, eigenvalues and chart limits are the historical fitted reference. Incoming observations do not recalibrate them. These projections are operational diagnostics, not a new independent validation.")
    with contributions_tab:
        valid = view.loc[view.status.eq("scored")]
        if valid.empty:
            st.info("No valid scored observation is available for SPE contributions.")
        else:
            event = st.selectbox("Acquired PCA observation", valid.sensor_event_id.tolist(), key="mspc_acquired_event")
            contributions = acquired_sensor_contributions(model, scored, event)
            st.dataframe(contributions, hide_index=True, width="stretch", key="mspc_acquired_contributions")
            st.plotly_chart(px.bar(contributions, x="sensor", y="spe_contribution"), width="stretch", key="mspc_acquired_contributions_chart")
            observation = valid[valid.sensor_event_id.eq(event)].iloc[0]
            st.caption(f"Contributions sum to historical Q = {observation.spe:.5g}. They do not identify a root cause or establish current freshness.")
    with audit:
        st.dataframe(view, hide_index=True, width="stretch", key="mspc_acquired_observations")
        st.dataframe(session.acquisition.ledger(), hide_index=True, width="stretch", key="mspc_acquired_delivery_audit")
        st.download_button("Download acquired PCA monitoring report (ZIP)", acquired_pca_report_zip(model, session, scored, rules, monitoring, now=now), file_name="pat_acquired_pca_monitoring.zip", mime="application/zip", key="mspc_acquired_report")
        st.caption("The export includes received sensor history and flags; the original historical reference audit is in a separate folder. Prediction models keep their own saved preprocessing. No second collector is created.")
