"""Optional extension: read-only acquisition adapters against test sources."""

from dataclasses import asdict
from hashlib import sha256
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.acquisition import (
    AcquisitionError, AcquisitionSession, InMemoryTestAdapter, LocalHTTPAdapter,
    acquisition_report_zip, test_source_records,
)
from src.app_data import load_dataset
from src.data_generation import SimulationConfig
from src.data_integration import SENSOR_RANGES, SourceTables, sources_from_dataset
from src.runtime import apply_page_runtime, public_demo


MODES = ["In-app test source", "Local HTTP test source"]
SENSOR_LABELS = {
    "temperature_c": "Temperature (°C)", "ph": "pH (dimensionless)",
    "dissolved_oxygen_pct": "Dissolved oxygen (% air saturation)",
    "feed_rate_l_h": "Feed flow (L/h)", "agitation_rpm": "Agitation (rpm)",
}


@st.cache_data(max_entries=4, show_spinner=False)
def fixture_records(config, batch_id, faults):
    sources = sources_from_dataset(load_dataset(config))
    selected = SourceTables(**{name: frame[frame.batch_id.eq(batch_id)].copy() for name, frame in sources.as_dict().items()})
    return test_source_records(selected, faults=faults)


def acquisition_action(action):
    """Apply controls before rendering, so button availability reflects new state."""
    session = st.session_state["acquisition_session"]
    if action == "reset":
        st.session_state["acquisition_session"] = AcquisitionSession(session.context_id)
    elif action == "start":
        session.running = True
    elif action in ("pause", "publish"):
        session.running = False
    if action in ("poll", "publish"):
        st.session_state[f"acq_{action}_requested"] = True


@st.fragment(run_every=1.0)
def render_acquisition(adapter, context, limit):
    session = st.session_state["acquisition_session"]
    step = st.session_state.pop("acq_poll_requested", False)
    publish = st.session_state.pop("acq_publish_requested", False)
    if step or session.running:
        session.poll_once(adapter, limit)
    snapshot, ledger = session.snapshot(), session.ledger()
    ready = not snapshot.raman.empty and not snapshot.batches.empty
    controls = st.columns(5)
    controls[0].button("Connect / start polling", key="acq_start", disabled=session.running or session.complete, on_click=acquisition_action, args=("start",))
    controls[1].button("Pause polling", key="acq_pause", disabled=not session.running, on_click=acquisition_action, args=("pause",))
    controls[2].button("Poll once", key="acq_poll", disabled=session.running or session.complete, on_click=acquisition_action, args=("poll",))
    controls[3].button("Reset collection", key="acq_reset", on_click=acquisition_action, args=("reset",))
    controls[4].button("Send snapshot to Data Integration", key="acq_publish", disabled=not ready, on_click=acquisition_action, args=("publish",))
    if publish:
        st.session_state["acquisition_snapshot"] = {"sources": snapshot, "source_id": session.source_id,
            "cursor": session.cursor, "last_success_utc": session.last_success_utc, "context": dict(context)}
        st.session_state["acquisition_handoff_requested"] = True
        st.success("Snapshot saved. Open Data Integration in the sidebar; Acquired test snapshot will be selected. Further acquisition does not change that snapshot until you send another one.")
    status = "Read failed — paused" if session.last_error else "Complete" if session.complete else "Polling" if session.running else "Paused"
    metrics = st.columns(4)
    metrics[0].metric("Acquisition state", status)
    metrics[1].metric("Deliveries received", session.cursor)
    metrics[2].metric("Accepted unique events", len(session.accepted))
    metrics[3].metric("Quarantined deliveries", int(ledger.status.eq("quarantined").sum()) if not ledger.empty else 0)
    if session.last_error:
        st.error(session.last_error)
        st.caption("The failed read did not advance the cursor or remove collected observations. Fix/restart the test source, then poll again. A different source snapshot requires Reset collection.")
    st.caption(f"Source identity: {session.source_id or 'Not yet read'} · Cursor: {session.cursor} · Last successful receipt (UTC): {session.last_success_utc or 'None'}")
    st.caption("Polling is explicit and runs once per second while started. Each poll reads a bounded page of test records; source time cadence is defined by the producer. Source timestamps and laboratory result availability remain separate from actual receipt time. Navigation pauses polling.")
    counts = pd.DataFrame([{"source": name, "collected_rows": len(frame)} for name, frame in snapshot.as_dict().items()])
    st.dataframe(counts, hide_index=True, width="stretch")
    if not ledger.empty:
        duplicates = int(ledger.status.eq("duplicate").sum())
        flagged = ledger["flags"].ne("")
        st.caption(f"Identical duplicate deliveries suppressed: {duplicates}. Schema acceptance does not mean measurement quality is approved.")
        if flagged.any():
            st.warning("Some received records have quality flags or are quarantined. Review the delivery ledger and the Data Integration quality report.")
    observations_tab, audit_tab = st.tabs(["Collected observations", "Delivery audit & export"])
    with observations_tab:
        if not snapshot.sensors.empty:
            variable = st.selectbox("Acquired sensor variable", list(SENSOR_RANGES), format_func=SENSOR_LABELS.get, key="acq_sensor")
            chart = snapshot.sensors.copy()
            chart[variable] = pd.to_numeric(chart[variable], errors="coerce")
            chart["measurement_timestamp"] = pd.to_datetime(chart.measurement_timestamp, utc=True)
            figure = px.line(chart, x="measurement_timestamp", y=variable, color="batch_id", markers=True,
                labels={"measurement_timestamp": "Source measurement time (UTC)", variable: SENSOR_LABELS[variable], "batch_id": "Batch"})
            figure.update_traces(connectgaps=False)
            st.plotly_chart(figure, width="stretch", key="acq_sensor_plot")
        source = st.selectbox("Collected source table", list(snapshot.as_dict()), key="acq_table")
        st.dataframe(snapshot.as_dict()[source], hide_index=True, width="stretch")
        st.caption("Only polled records are visible. Assays enter the test stream at their laboratory result time. Missing numeric cells remain missing; no concentration estimate is inferred on this page.")
    with audit_tab:
        only_flags = st.checkbox("Only show duplicates, quarantines, or quality flags", value=True, key="acq_only_flags")
        view = ledger[ledger.status.ne("accepted") | ledger["flags"].ne("")] if only_flags and not ledger.empty else ledger
        st.dataframe(view, hide_index=True, width="stretch")
        st.caption("Every delivery retains its original JSON and receipt time. Conflicting payloads for one event ID quarantine that ID, including an earlier accepted copy. Distinct event IDs with duplicate sample identities are retained for Phase 2's ambiguity checks.")
        st.download_button("Download acquired snapshot & audit (ZIP)", acquisition_report_zip(session, context), file_name="pat_test_acquisition.zip", mime="application/zip", key="acq_report")


def main():
    st.set_page_config(page_title="Vaccine PAT | Data Acquisition", page_icon="📥", layout="wide")
    apply_page_runtime()
    # Other pages pause acquisition through navigation.py; this page pauses replay.
    replay = st.session_state.get("monitoring_session")
    if replay is not None:
        replay.clock.pause()
    stream = st.session_state.get("stream_monitoring_session")
    if stream is not None:
        stream.pause()
    st.caption("VACCINE PAT INTELLIGENCE PLATFORM · OPTIONAL ACQUISITION PROTOTYPE")
    st.title("Data Acquisition")
    st.badge("Test mode", icon=":material/science:", color="orange")
    st.write("Collect records from an in-app fixture or a local HTTP test source, then send a snapshot to Data Integration.")
    st.info("Read-only test sources · Simulator preserved · Source and receipt timestamps are separate · No real process connection or equipment control")
    config = st.session_state.get("simulation_config", SimulationConfig())
    batches = load_dataset(config).batches.batch_id.tolist()
    preferences = st.session_state.get("acquisition_preferences", {})
    with st.sidebar:
        st.header("Test source settings")
        with st.form("acquisition_setup"):
            allowed = MODES[:1] if public_demo() else MODES
            remembered_mode = preferences.get("mode", MODES[0])
            if remembered_mode not in allowed:
                st.session_state.pop("acq_mode", None)
                remembered_mode = allowed[0]
            mode = st.selectbox("Acquisition source", allowed, index=allowed.index(remembered_mode), key="acq_mode")
            remembered = preferences.get("batch", batches[0])
            batch = st.selectbox("In-app fixture batch", batches, index=batches.index(remembered) if remembered in batches else 0, key="acq_batch")
            faults = st.checkbox("Include test delivery faults", value=preferences.get("faults", False), key="acq_faults")
            url = preferences.get("url", "http://127.0.0.1:8765")
            if not public_demo():
                url = st.text_input("Local test source URL", value=url, key="acq_url")
            limit = st.selectbox("Records per poll", [5, 10, 25, 50, 100], index=[5, 10, 25, 50, 100].index(preferences.get("limit", 10)), key="acq_limit")
            st.form_submit_button("Apply source settings", key="acq_apply")
        st.caption("Fixture batch/fault settings apply to the in-app source. Configure the HTTP server independently. Source changes create a new paused collection; page size preserves the cursor.")
    st.session_state["acquisition_preferences"] = {"mode": mode, "batch": batch, "faults": faults, "url": url, "limit": limit}
    if st.button("Use this source in Process Monitoring", key="acq_open_monitoring"):
        previous = st.session_state.get("stream_monitor_preferences", {})
        st.session_state["stream_monitor_preferences"] = {**previous,
            "source": "Historical in-app test source" if mode == MODES[0] else "Local HTTP test source",
            "batch": batch, "faults": faults, "url": url, "limit": limit,
            "enforce": False if mode == MODES[0] else True}
        st.session_state["monitoring_mode_requested"] = "Acquired stream"
        st.switch_page("pages/5_Process_Monitoring.py")
    context = {"mode": mode, "url": url if mode == MODES[1] else None,
        "simulation": asdict(config) if mode == MODES[0] else None,
        "fixture_batch": batch if mode == MODES[0] else None, "faults": faults if mode == MODES[0] else None}
    signature = sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    session = st.session_state.get("acquisition_session")
    if session is None or session.context_id != signature:
        st.session_state["acquisition_session"] = AcquisitionSession(signature)
    try:
        adapter = InMemoryTestAdapter(fixture_records(config, batch, faults)) if mode == MODES[0] else LocalHTTPAdapter(url)
    except AcquisitionError as error:
        st.session_state["acquisition_session"].running = False
        st.error(str(error))
        return
    render_acquisition(adapter, context, limit)
    if public_demo():
        st.caption("The public demo uses in-app sources. Local HTTP test servers are available when running this project on your computer.")
    else:
        with st.expander("Run the standalone HTTP test source", expanded=False):
            st.write("Run this command in another terminal at the project root, select Local HTTP test source above, apply settings, then Poll once or start polling:")
            st.code(r".\.venv\Scripts\python.exe -m src.test_acquisition_source --port 8765", language="powershell")
            st.write("The server binds to 127.0.0.1 and serves a fixed historical synthetic batch. Add --faults for duplicate delivery, invalid timestamp/schema, and missing pH examples. Ctrl+C stops the server. Restarting with identical settings preserves the snapshot ID; changed settings require resetting the collector.")
    with st.expander("Learn acquisition adapters"):
        st.write("AcquisitionAdapter defines poll(cursor, limit). The two adapters return the same JSON page contract. AcquisitionSession owns cursor, deduplication, quarantine, receipt times, and the collection buffer; it does not know simulator equations or fit models. A future approved connector would translate its own source into this contract.")
        st.write("Example: a page delivering sequences 0–9 moves the cursor from 0 to 10. Retrying after a failed read requests 10 again. A repeated event ID with the identical payload creates an audit row but no duplicate source row. An assay collected at 0 h is delivered only at its later result time.")
        st.write("The collection is session-local, bounded at 10,000 deliveries, and pauses on navigation or read failure. It is not a durable historian, exactly-once messaging system, validated live integration, or OPC UA connector. Read docs/acquisition_adapter_learning_guide.md for the protocol, examples, limitations, tests, and interview questions.")


if __name__ == "__main__":
    main()
