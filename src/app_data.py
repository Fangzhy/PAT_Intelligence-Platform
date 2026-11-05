"""Cached synthetic data shared across application pages."""

import streamlit as st

from src.data_generation import SimulationConfig, SyntheticDataset
from src.demo_data import generate_development_dataset, generate_monitoring_dataset
# Compatibility export for callers that already use this module. Pages import
# navigation directly so an older cached app_data module cannot block startup.
from src.navigation import pause_monitoring_on_navigation


@st.cache_data(show_spinner="Simulating independent bioreactor batches…", max_entries=4)
def load_dataset(config: SimulationConfig) -> SyntheticDataset:
    """Reuse the same dataset when navigating or changing an exploration widget."""
    return generate_development_dataset(config)


@st.cache_data(show_spinner="Simulating a separate monitoring demo population…", max_entries=4)
def load_monitoring_dataset(development_config: SimulationConfig) -> SyntheticDataset:
    """Distinct seed/IDs; the original development configuration is unchanged."""
    return generate_monitoring_dataset(development_config)
