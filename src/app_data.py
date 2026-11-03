"""Cached synthetic data shared across application pages."""

import streamlit as st

from src.data_generation import SimulationConfig, SyntheticDataset, generate_dataset
# Compatibility export for callers that already use this module. Pages import
# navigation directly so an older cached app_data module cannot block startup.
from src.navigation import pause_monitoring_on_navigation


@st.cache_data(show_spinner="Simulating independent bioreactor batches…", max_entries=4)
def load_dataset(config: SimulationConfig) -> SyntheticDataset:
    """Reuse the same dataset when navigating or changing an exploration widget."""
    return generate_dataset(config)
