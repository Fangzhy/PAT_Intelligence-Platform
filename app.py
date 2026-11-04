"""Application entrypoint: order pages by the PAT workflow, independent of filenames."""

import streamlit as st
from src.runtime import initialize_runtime, public_demo
from src.llm_bootstrap import refresh_legacy_llm_modules


def main():
    st.set_page_config(page_title="Vaccine PAT Intelligence", page_icon="🧪", layout="wide")
    refresh_legacy_llm_modules()
    initialize_runtime(st.session_state)
    if public_demo():
        st.sidebar.caption("PUBLIC DEMO · Temporary session storage · Synthetic data")
    pages = {
        "": [st.Page("pages/1_Overview.py", title="Overview", icon="🧪", default=True)],
        "Data": [
            st.Page("pages/9_Test_Acquisition.py", title="Data Acquisition", icon="📥", url_path="Data_Acquisition"),
            st.Page("pages/2_Data_Integration.py", title="Data Integration & Quality", icon="🔗", url_path="Data_Integration"),
        ],
        "Analytics": [
            st.Page("pages/3_Spectral_Analytics.py", title="Spectral Analytics", icon="🔬"),
            st.Page("pages/4_Predictive_Modeling.py", title="Predictive Modeling", icon="🧠"),
        ],
        "Monitoring": [
            st.Page("pages/5_Process_Monitoring.py", title="Process Monitoring", icon="⏱️"),
            st.Page("pages/6_Multivariate_Monitoring.py", title="Multivariate Monitoring", icon="📊"),
            st.Page("pages/7_Model_Reliability.py", title="Model Reliability", icon="🔎"),
        ],
        "Lifecycle": [
            st.Page("pages/8_Model_Deployment.py", title="Deployment & Reporting", icon="📦"),
        ],
    }
    st.navigation(pages, position="sidebar", expanded=True).run()


if __name__ == "__main__":
    main()
