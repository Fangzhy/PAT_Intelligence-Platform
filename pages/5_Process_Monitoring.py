"""Phase 5: nonblocking synthetic replay with saved-pipeline inference and alerts."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_data import load_dataset
from src.data_generation import SimulationConfig
from src.model_registry import DEFAULT_MONITORING_MODELS, discover_models, load_model, prepare_demo_models
from src.modeling_data import SENSOR_COLUMNS, TARGETS
from src.modeling_storage import DEFAULT_STATE_ROOT
from src.monitoring import DEFAULT_PREDICTION_LIMITS, DEFAULT_SENSOR_LIMITS, MonitoringSettings
from src.replay import MonitoringSession, known_assays, monitoring_report_zip, replay_batch, replay_fault_example


SENSOR_LABELS = {
    "temperature_c": ("Temperature", "°C"), "ph": ("pH", "dimensionless"),
    "dissolved_oxygen_pct": ("Dissolved oxygen", "% air saturation"),
    "feed_rate_l_h": ("Feed flow", "L/h"), "agitation_rpm": ("Agitation", "rpm"),
}


@st.cache_resource(max_entries=8, show_spinner=False)
def cached_model(reference):
    """Reference stamps invalidate the cache when a local artifact changes."""
    return load_model(reference)


def _limits_editor(title, defaults, saved, key):
    st.write(title)
    frame = pd.DataFrame([{"variable": variable, "unit": SENSOR_LABELS[variable][1] if variable in SENSOR_LABELS else "g/L", "lower": bounds[0], "upper": bounds[1]} for variable, bounds in (saved or defaults).items()])
    edited = st.data_editor(frame, hide_index=True, disabled=["variable", "unit"], width="stretch", key=key)
    return {row.variable: (row.lower, row.upper) for row in edited.itertuples()}


def _sensor_chart(history, variable, settings):
    label, unit = SENSOR_LABELS[variable]
    chart = go.Figure(go.Scatter(x=history.elapsed_time_h, y=pd.to_numeric(history[variable], errors="coerce"), mode="lines+markers", name="Observed sensor", connectgaps=False))
    lower, upper = settings.sensor_limits[variable]
    chart.add_hline(y=lower, line_dash="dash", annotation_text="Lower advisory limit")
    chart.add_hline(y=upper, line_dash="dash", annotation_text="Upper advisory limit")
    chart.update_layout(xaxis_title="Elapsed process time (h)", yaxis_title=f"{label} ({unit})", height=340)
    return chart


def _prediction_chart(history, assays, target, settings):
    prefix = target.lower()
    chart = go.Figure(go.Scatter(x=history.elapsed_time_h, y=history[f"{prefix}_prediction_g_l"], mode="lines+markers", name=f"{target}: model estimate", connectgaps=False))
    if not assays.empty:
        chart.add_trace(go.Scatter(x=assays.collection_elapsed_time_h, y=assays[TARGETS[target]], mode="markers", marker=dict(symbol="diamond", size=9), name="Offline assay (result now available)"))
    for value in settings.prediction_limits[target]:
        chart.add_hline(y=value, line_dash="dot")
    chart.update_layout(xaxis_title="Elapsed process time (h)", yaxis_title=f"{target} (g/L)", height=320, legend=dict(orientation="h"))
    return chart


@st.fragment(run_every=0.5)
def render_monitor(batch, models, settings, mode, review_elapsed, context):
    """Timer refresh only processes newly revealed rows; no sleep or fit loop."""
    session = st.session_state["monitoring_session"]
    clock = session.clock
    total = len(batch.inputs)
    now = monotonic()
    if mode == "Replay":
        clock.tick(now, total)
    else:
        clock.pause()
    controls = st.columns(4)
    if controls[0].button("Start / resume", key="replay_start", disabled=mode != "Replay" or clock.running or clock.cursor >= total - 1):
        clock.start(now, total)
    if controls[1].button("Pause", key="replay_pause", disabled=mode != "Replay" or not clock.running):
        clock.pause()
    if controls[2].button("Advance one observation", key="replay_step", disabled=mode != "Replay" or clock.running or clock.cursor >= total - 1):
        clock.step(total)
    if controls[3].button("Reset replay", key="replay_reset", disabled=mode != "Replay"):
        session.reset()

    start = batch.metadata.measurement_timestamp.iloc[0] - pd.Timedelta(hours=batch.metadata.elapsed_time_h.iloc[0])
    if mode == "Historical review":
        position = int(np.searchsorted(batch.metadata.elapsed_time_h.to_numpy(), review_elapsed, side="right") - 1)
        cutoff = start + pd.Timedelta(hours=review_elapsed)
        status = "Historical review"
    else:
        position = clock.cursor
        cutoff = batch.metadata.iloc[position].measurement_timestamp if position >= 0 else start - pd.Timedelta(nanoseconds=1)
        status = "Complete" if position == total - 1 else "Running" if clock.running else "Paused"
    history = session.history(batch, position, models, settings, mode=mode)
    assays = known_assays(batch, cutoff)
    metrics = st.columns(3)
    metrics[0].metric("Replay state", status)
    metrics[1].metric("Revealed observations", len(history))
    metrics[2].metric("Displayed elapsed time (h)", f"{(cutoff - start).total_seconds() / 3600:g}" if position >= 0 else "—")
    st.progress((position + 1) / total)
    st.caption("Replay clock advances only while running. Speed means observations per real second. Navigating to another page pauses replay; changing batch, models, mode, or monitoring rules starts a fresh paused context.")
    if history.empty:
        st.write("Start replay or advance one observation to view measurements. No concentration is synthesized before a model inference succeeds.")
        return

    latest = history.iloc[-1]
    st.caption(f"Sample {latest.sample_id} · Simulated measurement time: {latest.measurement_timestamp} · References available through {cutoff}")
    sensor_cards = st.columns(5)
    for index, column in enumerate(SENSOR_COLUMNS):
        label, unit = SENSOR_LABELS[column]
        value = pd.to_numeric(pd.Series([latest[column]]), errors="coerce").iloc[0]
        sensor_cards[index].metric(label, f"{value:.4g} {unit}" if np.isfinite(value) else "Unavailable")
    prediction_cards = st.columns(2)
    for index, target in enumerate(TARGETS):
        prefix = target.lower()
        value = latest[f"{prefix}_prediction_g_l"]
        prediction_cards[index].metric(f"Predicted {target.lower()}", f"{value:.3f} g/L" if np.isfinite(value) else "Unavailable")
        prediction_cards[index].caption(f"Status: {latest[f'{prefix}_status']} · Model: {latest[f'{prefix}_model_version']}")
        if latest[f"{prefix}_status"] not in ("ok", "model_unavailable"):
            prediction_cards[index].warning(latest[f"{prefix}_flags"])
        prediction_cards[index].caption(f"Inference attempt (UTC): {latest[f'{prefix}_prediction_time_utc']} · {latest[f'{prefix}_latency_ms']:.2f} ms")

    process_tab, soft_tab, history_tab, sources_tab = st.tabs(["Process trends & alerts", "Soft sensors", "Revealed history", "Available laboratory references"])
    with process_tab:
        if latest.quality_flags:
            st.error(f"Input quality flags: {latest.quality_flags}")
        alarms = [{"variable": column, "value": latest[column], "unit": SENSOR_LABELS[column][1], "state": latest[f"{column}_alarm"]} for column in SENSOR_COLUMNS]
        alarms.extend({"variable": f"Predicted {target}", "value": latest[f"{target.lower()}_prediction_g_l"], "unit": "g/L", "state": latest[f"{target.lower()}_alarm"]} for target in TARGETS)
        alarms = pd.DataFrame(alarms)
        if alarms.state.isin(["High", "Low"]).any():
            st.warning("Advisory monitoring limits exceeded. Review the process trend and prediction quality; no equipment action is performed.")
        st.dataframe(alarms, hide_index=True, width="stretch")
        variable = st.selectbox("Sensor trend", list(SENSOR_COLUMNS), format_func=lambda name: SENSOR_LABELS[name][0], key="monitor_sensor_trend")
        st.plotly_chart(_sensor_chart(history, variable, settings), width="stretch", key="monitor_sensor_plot")
        st.caption("Advisory limits are inclusive and distinct from the wider input plausibility checks. A limit violation does not by itself establish a CPP/CQA relationship.")
    with soft_tab:
        for target in TARGETS:
            st.plotly_chart(_prediction_chart(history, assays, target, settings), width="stretch", key=f"monitor_{target.lower()}_plot")
        st.caption("Lines contain only saved-pipeline predictions. Invalid/out-of-domain observations are gaps, never carried-forward estimates. Diamonds are delayed offline assays revealed at result time and plotted at collection time. There is no simulated ground-truth trace.")
        st.dataframe(pd.DataFrame([{"target": target, "model_version": latest[f"{target.lower()}_model_version"], "validation_status": latest[f"{target.lower()}_validation_status"], "max_abs_z": latest[f"{target.lower()}_max_abs_z"]} for target in TARGETS]), hide_index=True, width="stretch")
        st.caption("The training-mean/scale domain screen checks individual engineered features, including changes to training-constant features. It is not a calibrated probability, PCA monitoring limit, or guarantee of reliable prediction.")
    with history_tab:
        st.dataframe(history, hide_index=True, width="stretch")
        st.caption("Each revealed observation is inferred once in this context, retaining data time, inference time, model version, quality, and alarm states. Rules and models are frozen for this history.")
    with sources_tab:
        st.dataframe(assays, hide_index=True, width="stretch")
        st.caption("Only results whose reporting time is at or before the displayed clock are available. Historical review can extend past batch end to reveal the final delayed assays.")
    export_context = {**context, "cutoff_utc": str(cutoff), "mode": mode, "revealed_rows": len(history)}
    st.download_button("Download revealed monitoring report (ZIP)", monitoring_report_zip(history, assays, settings, export_context), file_name="pat_monitoring_replay.zip", mime="application/zip", key="monitor_report")


def main():
    st.set_page_config(page_title="Vaccine PAT | Process Monitoring", page_icon="⏱️", layout="wide")
    st.caption("VACCINE PAT INTELLIGENCE PLATFORM · PHASE 5")
    st.title("Replay the process. Monitor real model estimates.")
    st.write("Synthetic time-ordered replay with direct sensors, saved glucose/lactate pipelines, input validation, and advisory monitoring limits.")
    st.info("Educational replay · Advisory only · No physical equipment control · Monitored process variables are not confirmed CPPs without process-specific CQA evidence.")
    config = st.session_state.get("simulation_config", SimulationConfig())
    dataset = load_dataset(config)
    preferences = st.session_state.get("monitoring_preferences", {})
    with st.sidebar:
        st.header("Replay setup")
        st.caption(f"Overview configuration: seed {config.seed}, {config.n_batches} batches, {config.duration_h:g} h. No equipment connection is used.")
        batches = dataset.batches.batch_id.tolist()
        selected_batch = st.selectbox("Replay batch", batches, index=batches.index(preferences.get("batch")) if preferences.get("batch") in batches else 0, key="monitor_batch")
        modes = ["Replay", "Historical review"]
        mode = st.radio("Monitoring mode", modes, index=modes.index(preferences.get("mode", "Replay")), key="monitor_mode")
        rate = st.selectbox("Observations per second", [0.5, 1.0, 2.0, 5.0, 10.0], index=[0.5, 1.0, 2.0, 5.0, 10.0].index(preferences.get("rate", 1.0)), key="monitor_rate")
        faults = st.checkbox("Inject known replay faults", value=preferences.get("faults", False), key="monitor_faults")
    batch = replay_batch(dataset, selected_batch)
    if faults:
        batch = replay_fault_example(batch)
        st.warning("Fault copy: missing pH at index 3, DO = 145% at index 7, a missing Raman channel at index 10, and +2 a.u. Raman baseline at index 12 when those observations exist. Original source data remains unchanged.")
    modeling_root = Path(st.session_state.get("modeling_state_root", DEFAULT_STATE_ROOT))
    monitoring_root = Path(st.session_state.get("monitoring_models_root", DEFAULT_MONITORING_MODELS))
    with st.expander("Saved models and separate demo calibration", expanded=True):
        st.write("Select finalized Phase 4 pipelines or prepare separate demo soft sensors. Preparation is an explicit training action on another seed; ordinary replay only loads and applies saved pipelines. Demo models have grouped CV only, with no independent-test or manufacturing-validation claim.")
        if st.button("Prepare / reuse saved demo models", key="prepare_monitor_models"):
            try:
                with st.spinner("Preparing glucose/lactate calibration pipelines on separate data…"):
                    prepared = prepare_demo_models(config, monitoring_root)
                for key in ("monitor_glucose_model", "monitor_lactate_model"):
                    st.session_state.pop(key, None)
                saved_preferences = dict(st.session_state.get("monitoring_preferences", {}))
                saved_preferences["models"] = {reference.target: f"{reference.version} · {reference.source}" for reference in prepared}
                st.session_state["monitoring_preferences"] = saved_preferences
                st.rerun()
            except (ValueError, OSError) as error:
                st.error(f"Demo calibration could not complete: {error}")
        references, issues = discover_models(modeling_root, monitoring_root)
        if issues:
            st.warning("Some local model artifacts are unavailable. Their entries are excluded below.")
            st.dataframe(pd.DataFrame(issues), hide_index=True, width="stretch")
        models, chosen = {}, {}
        for target in TARGETS:
            options = {"No saved model": None}
            options.update({f"{reference.version} · {reference.source}": reference for reference in references if reference.target == target})
            labels = list(options)
            remembered = preferences.get("models", {}).get(target)
            default_label = "No saved model"
            for label, candidate in options.items():
                if candidate is None:
                    continue
                try:
                    axis = np.asarray(json.loads(Path(candidate.metadata_path).read_text(encoding="utf-8"))["spectral_axis_cm1"], dtype=float)
                    if axis.shape == batch.axis.shape and np.allclose(axis, batch.axis, rtol=1e-10, atol=1e-10):
                        default_label = label
                        break
                except (ValueError, KeyError, OSError):
                    continue
            selected = st.selectbox(f"{target} pipeline", labels, index=labels.index(remembered) if remembered in options else labels.index(default_label), key=f"monitor_{target.lower()}_model")
            reference = options[selected]
            chosen[target] = selected
            if reference is None:
                models[target] = None
            else:
                try:
                    models[target] = cached_model(reference)
                except ValueError as error:
                    models[target] = None
                    st.warning(f"{target} pipeline unavailable: {error}")
        if not any(models.values()):
            st.info("No usable saved models are selected. Direct sensor replay still works; glucose/lactate predictions remain unavailable until saved models are loaded.")
    with st.expander("Monitoring rules — illustrative editable limits"):
        domain_limit = st.number_input("Maximum engineered-feature |z|", min_value=0.5, max_value=100.0, value=preferences.get("domain_limit", 6.0), step=0.5, key="monitor_domain")
        max_age = st.number_input("Maximum sensor age relative to spectrum (minutes)", min_value=0.0, max_value=240.0, value=preferences.get("max_age", 30.0), step=5.0, key="monitor_age")
        sensor_limits = _limits_editor("Process sensor advisory limits", DEFAULT_SENSOR_LIMITS, preferences.get("sensor_limits"), "monitor_sensor_limits")
        prediction_limits = _limits_editor("Predicted concentration advisory limits (g/L)", DEFAULT_PREDICTION_LIMITS, preferences.get("prediction_limits"), "monitor_prediction_limits")
        st.caption("Limits are teaching assumptions, not validated operating specifications. Missing/invalid input and domain failures suppress estimates; advisory limit crossings produce review warnings without automatic equipment action.")
    try:
        settings = MonitoringSettings(domain_limit, max_age, sensor_limits, prediction_limits)
    except (ValueError, TypeError) as error:
        st.error(str(error))
        return
    start = batch.metadata.measurement_timestamp.iloc[0]
    result_elapsed = (batch.assays.result_timestamp - start).dt.total_seconds() / 3600
    review_max = float(np.ceil(max(float(batch.metadata.elapsed_time_h.max()), float(result_elapsed.max())) * 2) / 2)
    review_elapsed = st.slider("Historical review elapsed time (h)", 0.0, review_max, 0.0, step=0.5, key="monitor_review_time") if mode == "Historical review" else 0.0
    context = {"simulation": asdict(config), "batch_id": selected_batch, "fault_example": faults, "mode": mode,
        "models": {target: model.version if model else "Unavailable" for target, model in models.items()},
        "artifacts": {target: {"identity": model.metadata["artifact_identity"], "stamp": model.metadata["artifact_stamp"]} if model else None for target, model in models.items()}}
    signature = sha256(json.dumps({**context, "rules": asdict(settings)}, sort_keys=True).encode()).hexdigest()
    session = st.session_state.get("monitoring_session")
    if session is None or session.context_id != signature:
        session = MonitoringSession(signature)
        st.session_state["monitoring_session"] = session
    session.clock.set_rate(rate, monotonic(), len(batch.inputs))
    st.session_state["monitoring_preferences"] = {"batch": selected_batch, "mode": mode, "rate": rate,
        "faults": faults, "models": chosen, "domain_limit": domain_limit, "max_age": max_age,
        "sensor_limits": sensor_limits, "prediction_limits": prediction_limits}
    render_monitor(batch, models, settings, mode, review_elapsed, context)
    with st.expander("Learn Phase 5"):
        st.latex(r"\hat y_t=f_{saved}(x_t),\qquad z_j=\frac{\phi_j(x_t)-\mu_j^{train}}{s_j^{train}}")
        st.write("The saved feature transformation φ and estimator are reused without fitting. A maximum |z| above the configured threshold or a change to a training-constant feature causes a domain warning and a gap. This individual-feature screen does not detect every multivariate shift and is not a calibrated alarm probability.")
        st.write("Example: a feature with training mean 10 and SD 2 has z = 3 when observed at 16. A concentration exactly on an advisory bound is within limits; missing input is never interpreted as normal. Sample collection time and laboratory result time remain separate from actual inference time.")
        st.markdown("""
1. Prepare separate saved demo models, then advance a few observations.
2. Start, pause, change speed, resume, and reset; inspect the versioned history.
3. Inject known faults and use historical review to inspect indices 3, 7, 10, and 12.
4. Move historical time backward and forward; only revealed observations and available assay results are displayed.
5. Compare nominal and deliberately disturbed batches without calling their scenario labels model predictions.

Read `docs/phase_5_learning_guide.md` for equations, code explanations, limitations, tests, and three interview questions with sample answers. Phase 6 requires your explicit instruction.
""")
        st.markdown("Runtime references: [Streamlit fragments](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment) · [Session State](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state) · [Training scaler statistics](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)")


if __name__ == "__main__":
    main()
