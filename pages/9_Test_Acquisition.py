"""Collect new monitoring inputs; share the collector with Process Monitoring."""

import streamlit as st

from src.data_generation import SimulationConfig
from src.runtime import apply_page_runtime
from src.stream_monitoring_ui import render_acquired_stream_page


def main():
    st.set_page_config(page_title="PAT | Data Acquisition", page_icon="📥", layout="wide")
    apply_page_runtime()
    replay = st.session_state.get("monitoring_session")
    if replay is not None:
        replay.clock.pause()
    st.caption("PAT INTELLIGENCE PLATFORM · MONITORING")
    st.title("Data Acquisition")
    st.badge("Test mode", icon=":material/science:", color="orange")
    st.session_state["monitoring_mode_requested"] = "Acquired stream"
    st.write("Receive new monitoring data from the separate synthetic population or a compatible HTTP source. The saved model rules process the received observations.")
    st.info("Monitoring inputs only · Same collection and cursor in Process Monitoring · No training or equipment control on this page")
    if st.button("Open Process Monitoring", key="acq_open_monitoring"):
        st.session_state["monitoring_mode_requested"] = "Acquired stream"
        st.switch_page("pages/5_Process_Monitoring.py")
    render_acquired_stream_page(st.session_state.get("simulation_config", SimulationConfig()), acquisition_view=True)


if __name__ == "__main__":
    main()
