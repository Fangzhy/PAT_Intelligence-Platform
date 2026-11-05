"""Workflow navigation, visible test mode, and the integration acquisition shortcut."""

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_sidebar_groups_follow_the_data_workflow_and_register_each_page(monkeypatch):
    original = st.navigation
    rendered_groups = []

    def navigation(pages, **kwargs):
        rendered_groups.append({group: [page.title for page in members] for group, members in pages.items()})
        return original(pages, **kwargs)

    monkeypatch.setattr(st, "navigation", navigation)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert rendered_groups == [{
        "": ["Overview"],
        "Data": ["Data Integration & Quality"],
        "Analytics": ["Spectral Analytics", "Predictive Modeling"],
        "Monitoring": ["Data Acquisition", "Process Monitoring", "Multivariate Monitoring", "Model Reliability"],
        "Lifecycle": ["Deployment & Reporting"],
    }]
    registered = list(app._registered_pages.values())
    assert [page["page_name"] for page in registered][:3] == ["Overview", "Data Integration & Quality", "Spectral Analytics"]
    assert len({page["script_path"] for page in registered}) == 9
    assert registered[0]["url_pathname"] == ""
    assert next(page for page in registered if page["page_name"] == "Data Acquisition")["url_pathname"] == "Data_Acquisition"


def test_monitoring_acquisition_keeps_development_source_separate():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    simulation = app.session_state["simulation_config"]
    app.switch_page("pages/2_Data_Integration.py").run()
    app.radio(key="integration_source").set_value("Upload CSV sources").run()
    assert app.radio(key="integration_source").options == ["Synthetic dataset", "Upload CSV sources"]
    app.switch_page("pages/9_Test_Acquisition.py").run()
    assert not app.exception
    assert app.title[0].value == "Data Acquisition"
    assert any("Test mode" in markdown.value for markdown in app.markdown)
    assert app.metric[1].value == "0"
    # AppTest retains an explicit switch_page selection across runs; follow the
    # destination reached by the shortcut as the browser's location would.
    app.switch_page("pages/9_Test_Acquisition.py")
    app.button(key="acq_poll").click().run()
    cursor = app.session_state["acquisition_session"].cursor
    assert not any(button.key == "acq_publish" for button in app.button)
    app.switch_page("pages/2_Data_Integration.py").run()
    assert not app.exception
    assert app.radio(key="integration_source").value == "Upload CSV sources"
    app.switch_page("pages/9_Test_Acquisition.py").run()
    assert not app.exception and app.metric[1].value == str(cursor)
    assert not app.session_state["acquisition_session"].running
    assert app.session_state["simulation_config"] == simulation
