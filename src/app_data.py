"""Cached synthetic data shared by the Overview and integration pages."""

import streamlit as st

from src.data_generation import SimulationConfig, SyntheticDataset, generate_dataset


@st.cache_data(show_spinner="Simulating independent bioreactor batches…", max_entries=4)
def load_dataset(config: SimulationConfig) -> SyntheticDataset:
    """Reuse the same dataset when navigating or changing an exploration widget."""
    return generate_dataset(config)
