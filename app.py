"""Application entrypoint: order pages by the PAT workflow, independent of filenames."""

import streamlit as st


def main():
    st.set_page_config(page_title="Vaccine PAT Intelligence", page_icon="🧪", layout="wide")
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
    }
    st.navigation(pages, position="sidebar", expanded=True).run()


if __name__ == "__main__":
    main()
