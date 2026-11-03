"""Page navigation behavior, independent of cached data-loading modules."""

import streamlit as st


def pause_monitoring_on_navigation():
    """Freeze replay on other pages instead of catching up unseen time on return."""
    session = st.session_state.get("monitoring_session")
    if session is not None:
        session.clock.pause()
