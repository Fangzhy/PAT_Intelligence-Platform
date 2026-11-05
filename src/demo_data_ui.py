"""Small provenance captions for historical and monitoring data sources."""

import streamlit as st
from src.demo_data import source_provenance


def render_demo_provenance(provenance):
    role = provenance["data_role"].replace("_", " ")
    st.caption(f"Data role: {role} · Generation seed: {provenance['generation_seed']} · Development seed: {provenance['development_seed']} · Dataset: {provenance['data_source_id'][:12]}")


def render_received_provenance(batches):
    provenance = source_provenance(batches)
    with st.expander("Received data provenance"):
        if not provenance["sources"]:
            st.caption("No source provenance has been received. External source seeds and roles are unknown unless declared by the producer.")
        else:
            st.json(provenance)
            st.caption("Provenance is declared by the source. Receipt time and a fresh timestamp do not establish that measurements are new or independent.")
