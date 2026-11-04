"""Acquired-stream presentation; the pure processing engine is stream_monitoring."""

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.acquisition import AcquisitionError, InMemoryTestAdapter, LocalHTTPAdapter, test_source_records
from src.app_data import load_dataset
from src.data_integration import SourceTables, sources_from_dataset
from src.model_registry import DEFAULT_MONITORING_MODELS, discover_models, load_model, prepare_demo_models
from src.modeling_data import SENSOR_COLUMNS, TARGETS
from src.modeling_storage import DEFAULT_STATE_ROOT
from src.monitoring import MonitoringSettings, alarm_state
from src.stream_monitoring import AcquiredStreamSession, StreamSettings, acquired_stream_report_zip, freshness_errors
from src.test_stream_source import PacedTestAdapter
from src.runtime import public_demo
from src.llm_bootstrap import refresh_legacy_llm_modules

refresh_legacy_llm_modules()
from src.llm_results import process_snapshot
from src.llm_ui import render_captured_explanation
from src.ui_labels import workflow_label


SOURCES = ["Fresh in-app test stream", "Historical in-app test source", "Local HTTP test source"]
SENSOR_LABELS = {"temperature_c": "Temperature (°C)", "ph": "pH (dimensionless)",
    "dissolved_oxygen_pct": "Dissolved oxygen (% air saturation)", "feed_rate_l_h": "Feed flow (L/h)", "agitation_rpm": "Agitation (rpm)"}


@st.cache_data(max_entries=4, show_spinner=False)
def fixture_records(config, batch, faults):
    sources = sources_from_dataset(load_dataset(config))
    selected = SourceTables(**{name: frame[frame.batch_id.eq(batch)].copy() for name, frame in sources.as_dict().items()})
    return test_source_records(selected, faults=faults)


@st.cache_resource(max_entries=8, show_spinner=False)
def _model(reference):
    return load_model(reference)


def _make_adapter(context):
    if context["source"] == SOURCES[2]:
        return LocalHTTPAdapter(context["url"])
    from src.data_generation import SimulationConfig
    records = fixture_records(SimulationConfig(**context["simulation"]), context["batch"], context["faults"])
    return PacedTestAdapter(records, seconds_per_hour=context["pace"]) if context["source"] == SOURCES[0] else InMemoryTestAdapter(records)


def _action(action):
    session = st.session_state["stream_monitoring_session"]
    if action == "reset":
        st.session_state["stream_monitor_reset"] = True
    elif action == "start":
        session.acquisition.running = True
    elif action == "pause":
        session.pause()
    elif action == "poll":
        st.session_state["stream_monitor_poll"] = True


def _chart(frame, column, label, *, limits=None, assays=None, reference_column=None):
    figure = go.Figure()
    if not frame.empty:
        for batch, group in frame.groupby("batch_id", sort=False):
            group = group.sort_values("measurement_timestamp")
            figure.add_trace(go.Scatter(x=group.measurement_timestamp, y=pd.to_numeric(group[column], errors="coerce"), mode="lines+markers", name=str(batch), connectgaps=False))
    if assays is not None and not assays.empty:
        figure.add_trace(go.Scatter(x=assays.collection_timestamp, y=assays[reference_column], mode="markers", name="Available offline assay", marker=dict(symbol="diamond", size=9)))
    for limit in limits or ():
        figure.add_hline(y=limit, line_dash="dot")
    figure.update_layout(xaxis_title="Source measurement / collection time (UTC)", yaxis_title=label, height=330)
    return figure


@st.fragment(run_every=0.5)
def render_stream(context, models, monitoring, rules, limit, signature):
    session = st.session_state["stream_monitoring_session"]
    if st.session_state.pop("stream_monitor_reset", False):
        session = AcquiredStreamSession(signature)
        st.session_state["stream_monitoring_session"] = session
        st.session_state["stream_monitor_adapter"] = _make_adapter(context)
    if st.session_state.pop("stream_monitor_poll", False) or session.acquisition.running:
        session.poll_once(st.session_state["stream_monitor_adapter"], models, monitoring, rules, limit=limit)
    controls = st.columns(4)
    controls[0].button("Start acquired stream", key="stream_start", disabled=session.acquisition.running or session.acquisition.complete, on_click=_action, args=("start",))
    controls[1].button("Pause acquired stream", key="stream_pause", disabled=not session.acquisition.running, on_click=_action, args=("pause",))
    controls[2].button("Poll stream once", key="stream_poll", disabled=session.acquisition.running or session.acquisition.complete, on_click=_action, args=("poll",))
    controls[3].button("Reset acquired stream", key="stream_reset", on_click=_action, args=("reset",))
    now = pd.Timestamp.now(tz="UTC")
    history, sensors, assays = session.history(), session.sensor_history(rules, now=now), session.available_assays(now=now)
    status, health_flags = session.health(rules, now=now)
    metrics = st.columns(4)
    metrics[0].metric("Stream state", status)
    metrics[1].metric("Received deliveries", session.acquisition.cursor)
    metrics[2].metric("Spectra assessed", len(session.records))
    metrics[3].metric("Spectra waiting for inputs", len(session.pending))
    st.caption(f"Source: {session.acquisition.source_id or 'Not yet read'} · Last receipt: {session.acquisition.last_success_utc or 'None'} · Inference uses received records only; no fitting occurs here")
    if session.acquisition.last_error:
        st.error(session.acquisition.last_error)
    for flag in health_flags:
        st.warning(flag)
    if session.pending:
        st.info("Some spectra are waiting for sensor/batch inputs. Only same-batch sensor times at or before the spectrum can match. Expiry or end of source produces a flagged gap.")
    if not history.empty and history.quality_flags.ne("").any():
        st.warning("Some observations have input quality failures or pending alignment. Inspect the prediction history; gaps are not filled.")
    batch_ids = sorted(set(sensors.batch_id) | (set(history.batch_id) if not history.empty else set()))
    selected = st.selectbox("Stream batch view", ["All batches", *batch_ids], key="stream_batch_view")
    sensor_view = sensors if selected == "All batches" else sensors[sensors.batch_id.eq(selected)]
    prediction_view = history if selected == "All batches" or history.empty else history[history.batch_id.eq(selected)]
    assay_view = assays if selected == "All batches" else assays[assays.batch_id.eq(selected)]
    if not sensor_view.empty:
        latest_sensor = sensor_view.iloc[-1]
        st.caption(f"Latest sensor sample: {latest_sensor.sample_id} · Batch {latest_sensor.batch_id} · {latest_sensor.measurement_timestamp}")
        cards = st.columns(5)
        for index, column in enumerate(SENSOR_COLUMNS):
            value = pd.to_numeric(pd.Series([latest_sensor[column]]), errors="coerce").iloc[0]
            cards[index].metric(SENSOR_LABELS[column], f"{value:.4g}" if np.isfinite(value) else "Unavailable")
            cards[index].caption(alarm_state(value, monitoring.sensor_limits[column]))
        if latest_sensor.sensor_quality_flags:
            st.warning(latest_sensor.sensor_quality_flags)
    current_ok, fresh_flags = False, []
    if not prediction_view.empty:
        latest = prediction_view.iloc[-1]
        fresh_flags = freshness_errors(latest.measurement_timestamp, now, rules, "Latest spectrum")
        fresh_flags.extend(freshness_errors(latest.sensor_timestamp, now, rules, "Matched sensor"))
        current_ok = not health_flags and not fresh_flags and not session.acquisition.last_error
        cards = st.columns(2)
        for index, target in enumerate(TARGETS):
            prefix = target.lower()
            value = latest[f"{prefix}_prediction_g_l"]
            show = current_ok and latest[f"{prefix}_status"] == "ok" and np.isfinite(value)
            cards[index].metric(f"Stream predicted {target.lower()} (g/L)", f"{value:.3f}" if show else "Unavailable")
            cards[index].caption(f"{latest[f'{prefix}_status']} · {latest[f'{prefix}_model_version']} · Inference {latest[f'{prefix}_prediction_time_utc']}")
            if latest[f"{prefix}_flags"]:
                cards[index].warning(latest[f"{prefix}_flags"])
        for flag in fresh_flags:
            st.warning(flag)
    process_tab, predictions_tab, audit_tab = st.tabs(["Acquired sensors", "Online predictions", "Quality & delivery audit"])
    with process_tab:
        variable = st.selectbox("Acquired stream sensor", list(SENSOR_COLUMNS), format_func=SENSOR_LABELS.get, key="stream_sensor")
        st.plotly_chart(_chart(sensor_view, variable, SENSOR_LABELS[variable], limits=monitoring.sensor_limits[variable]), width="stretch", key="stream_sensor_chart")
        st.dataframe(sensor_view, hide_index=True, width="stretch")
    with predictions_tab:
        for target in TARGETS:
            if not prediction_view.empty:
                st.plotly_chart(_chart(prediction_view, f"{target.lower()}_prediction_g_l", f"Predicted {target} (g/L)", limits=monitoring.prediction_limits[target], assays=assay_view, reference_column=TARGETS[target]), width="stretch", key=f"stream_{target.lower()}_chart")
        st.dataframe(prediction_view, hide_index=True, width="stretch", key="stream_prediction_history")
        st.dataframe(assay_view, hide_index=True, width="stretch")
        st.caption("One frozen inference attempt per spectrum/context. Missing, stale, implausible, mismatched-axis, and out-of-domain inputs produce gaps. Later identity conflicts withdraw published estimates while retaining original estimates and a revision audit. Offline assays never enter model inputs.")
    with audit_tab:
        st.dataframe(session.acquisition.ledger(), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(session.invalidations), hide_index=True, width="stretch")
        st.download_button("Download acquired stream monitoring report (ZIP)", acquired_stream_report_zip(session, models, monitoring, rules, context, now=now), file_name="pat_acquired_stream_monitoring.zip", mime="application/zip", key="stream_report")
    st.caption("Monitoring owns this consumer's cursor/history. Leaving this page pauses polling. Historical predictions remain in the audit; current cards are withheld after a failed read, data silence, or freshness failure. Source/model/rule changes begin a fresh paused context.")
    try:
        snapshot = process_snapshot(prediction_view, sensor_view, assay_view, monitoring, cutoff=now,
            context={"view": signature, "batch_view": selected, "invalidations": len(session.invalidations)},
            current_eligible=current_ok, pending=len(session.pending),
            health_flag_count=len(health_flags) + len(fresh_flags),
            future_tolerance_seconds=rules.future_tolerance_seconds,
            source_kind="prepared_sources" if context["source"] == SOURCES[2] else "synthetic")
    except ValueError:
        snapshot = None
        st.info("An explanation snapshot cannot be assembled from these timestamps. Review the source clock and delivery audit.")
    capture_context = sha256(f"{signature}:{selected}:{len(session.invalidations)}".encode()).hexdigest()
    render_captured_explanation(snapshot, key="stream", context_id=capture_context, cutoff=now)


def render_acquired_stream_page(config):
    st.subheader("Acquisition → online quality checks → saved pipeline → predictions")
    preferences = st.session_state.get("stream_monitor_preferences", {})
    inherited = st.session_state.get("acquisition_preferences", {})
    batch_ids = load_dataset(config).batches.batch_id.tolist()
    with st.sidebar:
        st.header("Acquired stream source")
        allowed = SOURCES[:2] if public_demo() else SOURCES
        remembered_source = preferences.get("source", SOURCES[0])
        if remembered_source not in allowed:
            st.session_state.pop("stream_source", None)
            remembered_source = allowed[0]
        source = st.selectbox("Stream source", allowed, index=allowed.index(remembered_source), key="stream_source")
        remembered_batch = preferences.get("batch", batch_ids[0])
        batch = remembered_batch if remembered_batch in batch_ids else batch_ids[0]
        faults, pace = preferences.get("faults", False), preferences.get("pace", 1.0)
        url = preferences.get("url", inherited.get("url", "http://127.0.0.1:8765"))
        http_channels, http_duration = preferences.get("http_channels", 71), preferences.get("http_duration", 24)
        if source == SOURCES[2]:
            url = st.text_input("Stream HTTP test URL", value=url, key="stream_url")
            with st.expander("HTTP demo model configuration"):
                http_channels = st.number_input("HTTP demo Raman channels", min_value=32, max_value=2000, value=http_channels, step=1, key="stream_http_channels")
                http_duration = st.selectbox("HTTP demo calibration duration (h)", [24, 72, 144, 240], index=[24, 72, 144, 240].index(http_duration), key="stream_http_duration")
        else:
            batch = st.selectbox("Stream fixture batch", batch_ids, index=batch_ids.index(batch), key="stream_fixture_batch")
            faults = st.checkbox("Stream fixture delivery faults", value=faults, key="stream_faults")
            if source == SOURCES[0]:
                pace = st.number_input("Seconds per original process hour", min_value=0.05, max_value=60.0, value=pace, step=0.25, key="stream_pace")
        limit = st.selectbox("Stream records per poll", [5, 10, 25, 50, 100], index=[5, 10, 25, 50, 100].index(preferences.get("limit", 10)), key="stream_limit")
    fresh = source != SOURCES[1]
    st.badge("Fresh timestamp test stream" if source == SOURCES[0] else "Historical test data" if source == SOURCES[1] else "HTTP test stream", color="orange")
    if source == SOURCES[0]:
        st.write("The in-app producer releases synthetic records on a fresh UTC clock. Process time is accelerated; the model input measurements are unchanged.")
    elif source == SOURCES[2]:
        st.code(r".\.venv\Scripts\python.exe -m src.test_acquisition_source --port 8765 --stream --duration 24 --channels 71", language="powershell")
        st.caption("Match demo calibration channels/duration to your test source. The default command emits fresh UTC timestamps. A historical HTTP source requires disabling wall-clock freshness explicitly.")
    calibration = replace(config, n_wavenumbers=int(http_channels), duration_h=float(http_duration), sample_interval_h=1.0) if source == SOURCES[2] else config
    expected_axis = np.linspace(400, 1800, calibration.n_wavenumbers)
    modeling_root = Path(st.session_state.get("modeling_state_root", DEFAULT_STATE_ROOT))
    model_root = Path(st.session_state.get("monitoring_models_root", DEFAULT_MONITORING_MODELS))
    with st.expander("Saved pipelines for online inference", expanded=True):
        st.write("Select saved glucose/lactate pipelines. Test-model preparation is explicit, uses separate calibration data and grouped CV, and does not occur during polling.")
        if st.button("Prepare / reuse stream demo models", key="prepare_stream_models"):
            try:
                prepared = prepare_demo_models(calibration, model_root)
            except (ValueError, OSError) as error:
                st.error(f"Test calibration could not complete: {error}")
                return
            preferences["models"] = {reference.target: f"{reference.version} · {reference.source}" for reference in prepared}
            for key in ("stream_glucose_model", "stream_lactate_model"):
                st.session_state.pop(key, None)
        references, issues = discover_models(modeling_root, model_root)
        if issues:
            st.warning("Some saved models could not be discovered.")
            st.dataframe(pd.DataFrame(issues), hide_index=True, width="stretch")
        models, chosen = {}, {}
        for target in TARGETS:
            options = {"No saved model": None} | {f"{reference.version} · {reference.source}": reference for reference in references if reference.target == target}
            labels, default = list(options), "No saved model"
            for label, reference in options.items():
                if reference is not None:
                    try:
                        axis = np.asarray(json.loads(Path(reference.metadata_path).read_text(encoding="utf-8"))["spectral_axis_cm1"])
                        if axis.shape == expected_axis.shape and np.allclose(axis, expected_axis):
                            default = label
                            break
                    except (OSError, ValueError, KeyError):
                        continue
            remembered = preferences.get("models", {}).get(target)
            selected = st.selectbox(f"Stream {target.lower()} pipeline", labels, index=labels.index(remembered) if remembered in options else labels.index(default), format_func=workflow_label, key=f"stream_{target.lower()}_model")
            chosen[target] = selected
            try:
                models[target] = _model(options[selected]) if options[selected] else None
            except ValueError as error:
                models[target] = None
                st.warning(str(error))
        if not any(models.values()):
            st.info("Direct sensor monitoring works without a model. Predicted concentrations remain unavailable until a compatible saved pipeline is selected.")
    with st.expander("Online alignment, freshness, and model rules"):
        wait = st.number_input("Wait for missing inputs (seconds)", 0.0, 60.0, preferences.get("wait", 2.0), step=0.5, key="stream_wait")
        if preferences.get("source", source) != source:
            st.session_state.pop("stream_enforce", None)
        enforce = st.checkbox("Enforce wall-clock input freshness", value=preferences.get("enforce", fresh) if preferences.get("source", source) == source else fresh, key="stream_enforce")
        max_age = st.number_input("Maximum input wall-clock age (seconds)", 1.0, 86400.0, preferences.get("max_age", 300.0), step=10.0, key="stream_max_age")
        silence = st.number_input("No-new-data warning after (seconds)", 1.0, 3600.0, preferences.get("silence", 10.0), step=1.0, key="stream_silence")
        sensor_age = st.number_input("Maximum backward sensor age (minutes)", 0.0, 240.0, preferences.get("sensor_age", 30.0), step=5.0, key="stream_sensor_age")
        domain = st.number_input("Stream engineered-feature maximum |z|", 0.5, 100.0, preferences.get("domain", 6.0), step=0.5, key="stream_domain")
        st.caption("Both the spectrum and its matched sensor must pass freshness checks. Future clock tolerance is two seconds. Advisory sensor/concentration limits use the existing monitoring defaults.")
    rules = StreamSettings(wait, enforce, max_age, silence_seconds=silence)
    monitoring = MonitoringSettings(domain_z_limit=domain, max_sensor_age_minutes=sensor_age)
    if not enforce:
        st.warning("Wall-clock freshness is disabled for historical testing. This view does not establish current process freshness.")
    context = {"source": source, "url": url if source == SOURCES[2] else None, "simulation": asdict(config), "batch": batch, "faults": faults, "pace": pace}
    signature = sha256(json.dumps({**context, "models": {target: {"version": model.version, "stamp": model.metadata.get("artifact_stamp")} if model else None for target, model in models.items()}, "monitoring": asdict(monitoring), "stream": asdict(rules)}, sort_keys=True).encode()).hexdigest()
    session = st.session_state.get("stream_monitoring_session")
    if session is None or session.context_id != signature or "stream_monitor_adapter" not in st.session_state:
        if session is not None:
            session.pause()
        try:
            adapter = _make_adapter(context)
        except AcquisitionError as error:
            st.error(str(error))
            return
        st.session_state["stream_monitoring_session"] = AcquiredStreamSession(signature)
        st.session_state["stream_monitor_adapter"] = adapter
    st.session_state["stream_monitor_preferences"] = {"source": source, "batch": batch, "faults": faults, "url": url, "pace": pace, "limit": limit,
        "http_channels": int(http_channels), "http_duration": http_duration, "models": chosen, "wait": wait, "enforce": enforce,
        "max_age": max_age, "silence": silence, "sensor_age": sensor_age, "domain": domain}
    render_stream(context, models, monitoring, rules, limit, signature)
    with st.expander("Learn acquired-stream monitoring"):
        st.latex(r"t_{sensor}\leq t_{spectrum},\quad t_{spectrum}-t_{sensor}\leq\tau,\quad \hat y=f_{saved}(x)")
        st.write("Example: a spectrum at 10:00 can use a received 09:59 sensor within tolerance; it cannot use a 10:01 sensor. A delayed valid sensor can satisfy a pending spectrum during its wait window. After finalization, the original decision is retained. Later conflicting inputs withdraw its publication with an audit record.")
        st.write("Freshness compares source timestamps with current UTC; sensor alignment compares the two measurement timestamps. Both differ from receipt/inference time. Assays arrive independently and are not needed to infer a concentration. Read docs/acquired_stream_monitoring_learning_guide.md for the workflow, tests, and examples.")
