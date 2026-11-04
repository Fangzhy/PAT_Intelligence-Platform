"""Explicit local/public demo configuration and visitor-scoped temporary state."""

import os
from pathlib import Path
from tempfile import gettempdir
from uuid import uuid4

from src.data_generation import SimulationConfig


def deployment_mode():
    mode = os.environ.get("PAT_DEPLOYMENT_MODE")
    if mode is None:
        import streamlit as st
        try:
            mode = st.secrets.get("deployment", {}).get("mode", "local")
        except FileNotFoundError:
            mode = "local"
    if mode not in ("local", "public_demo"):
        raise ValueError("PAT_DEPLOYMENT_MODE must be local or public_demo.")
    return mode


def public_demo():
    return deployment_mode() == "public_demo"


def initialize_runtime(state):
    """Set roots before pages load; public visitors never share mutable studies.

    Temporary storage is deliberately not a durable registry. Paths are derived
    from an internally generated UUID, never a query parameter or uploaded path.
    """
    if not public_demo():
        return
    session_id = state.setdefault("public_demo_session_id", uuid4().hex)
    if not isinstance(session_id, str) or len(session_id) != 32 or any(char not in "0123456789abcdef" for char in session_id):
        raise ValueError("Invalid public demo session identity.")
    root = Path(gettempdir()) / "vaccine-pat-demo" / session_id
    for key, suffix in {
        "modeling_state_root": "modeling", "monitoring_models_root": "monitoring/models",
        "reliability_state_root": "reliability", "deployment_state_root": "releases",
    }.items():
        state[key] = str(root / suffix)
    state.setdefault("simulation_config", SimulationConfig(n_batches=6, duration_h=72, n_wavenumbers=71))
    config = state["simulation_config"]
    state["simulation_config"] = SimulationConfig(n_batches=min(config.n_batches, 8), duration_h=72 if config.duration_h not in (72, 144) else config.duration_h,
        sample_interval_h=config.sample_interval_h if config.sample_interval_h in (1.0, 2.0) else 1.0,
        n_wavenumbers=min(config.n_wavenumbers, 71), seed=config.seed)


def apply_page_runtime():
    """Also support directly testing/launching an individual page."""
    import streamlit as st
    initialize_runtime(st.session_state)
