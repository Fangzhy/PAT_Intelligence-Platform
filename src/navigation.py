"""Page navigation behavior, independent of cached data-loading modules."""

import streamlit as st


def pause_monitoring_on_navigation():
    """Freeze replay and test acquisition while visiting other analysis pages."""
    session = st.session_state.get("monitoring_session")
    if session is not None:
        session.clock.pause()
    acquisition = st.session_state.get("acquisition_session")
    if acquisition is not None:
        acquisition.running = False
