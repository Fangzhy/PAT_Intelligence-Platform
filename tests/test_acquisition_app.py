"""Acquisition controls, test-source handoff, navigation, and HTTP errors."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.data_generation import SimulationConfig


ROOT = Path(__file__).resolve().parents[1]
PAGE = "pages/9_Test_Acquisition.py"
CONFIG = SimulationConfig(n_batches=2, duration_h=24, n_wavenumbers=32)


def page():
    app = AppTest.from_file(str(ROOT / PAGE), default_timeout=60)
    app.session_state["simulation_config"] = CONFIG
    app.run()
    return app.selectbox(key="stream_source").select("Historical in-app test source").run()


def test_initial_load_does_not_poll_and_single_poll_reset_work():
    app = page()
    assert not app.exception
    assert [metric.value for metric in app.metric[:3]] == ["Paused", "0", "0"]
    assert not any(button.key == "acq_publish" for button in app.button)
    app.button(key="acq_poll").click().run()
    assert not app.exception
    assert app.metric[1].value == "10"
    assert len(app.session_state["acquisition_session"].snapshot().sensors) >= 1
    app.run()
    assert app.metric[1].value == "10"
    app.button(key="acq_reset").click().run()
    assert not app.exception
    assert app.metric[1].value == "0"


def test_start_pause_source_changes_and_page_size_preserve_expected_state():
    app = page()
    app.button(key="acq_start").click().run()
    assert not app.exception
    assert app.session_state["acquisition_session"].running
    app.button(key="acq_pause").click().run()
    assert not app.session_state["acquisition_session"].running
    cursor = app.session_state["acquisition_session"].cursor
    app.selectbox(key="stream_limit").select(25)
    app.run()
    assert app.session_state["acquisition_session"].cursor == cursor
    app.selectbox(key="stream_fixture_batch").select("MON-B002")
    app.run()
    assert not app.exception
    assert app.metric[1].value == "0"
    assert not app.session_state["acquisition_session"].running


def test_fault_collection_has_duplicates_quarantine_and_missing_ph():
    app = page()
    app.checkbox(key="stream_faults").check()
    app.selectbox(key="stream_limit").select(100)
    app.run()
    app.button(key="acq_poll").click().run()
    assert not app.exception
    assert app.metric[0].value == "Complete"
    assert app.session_state["acquisition_session"].ledger().status.eq("quarantined").sum() == 2
    assert app.session_state["acquisition_session"].ledger().status.eq("duplicate").sum() == 1
    assert app.session_state["acquisition_session"].snapshot().sensors.ph.isna().sum() == 1
    assert app.button(key="acq_start").disabled


def test_http_source_is_explicit_and_connection_error_is_visible(monkeypatch):
    import src.acquisition as acquisition
    def disconnected(*args, **kwargs):
        raise acquisition.AcquisitionError("Test source disconnected")
    monkeypatch.setattr(acquisition.LocalHTTPAdapter, "poll", disconnected)
    app = page()
    app.selectbox(key="stream_source").select("Local HTTP test source")
    app.run()
    assert not app.exception and not app.error
    app.button(key="acq_start").click().run()
    assert not app.exception
    assert app.metric[0].value == "Read failed — paused"
    assert app.metric[1].value == "0"
    assert any("disconnected" in error.value for error in app.error)


def test_invalid_test_url_is_rejected_without_a_network_read():
    app = page()
    app.selectbox(key="stream_source").select("Local HTTP test source").run()
    app.text_input(key="stream_url").set_value("http://example.com:8765")
    app.run()
    assert not app.exception
    assert any("loopback" in error.value for error in app.error)


def test_collected_inputs_are_available_in_process_monitoring_without_development_handoff(tmp_path):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    app.session_state["modeling_state_root"] = str(tmp_path / "models")
    app.session_state["monitoring_models_root"] = str(tmp_path / "demos")
    app.run()
    app.switch_page(PAGE).run()
    app.selectbox(key="stream_source").select("Historical in-app test source").run()
    app.button(key="acq_poll").click().run()
    session = app.session_state["stream_monitoring_session"]
    cursor = session.acquisition.cursor
    history = session.history().copy()
    app.switch_page("pages/5_Process_Monitoring.py").run()
    assert not app.exception
    assert app.radio(key="monitor_mode").value == "Acquired stream"
    assert app.session_state["stream_monitoring_session"] is session
    assert app.session_state["acquisition_session"].cursor == cursor
    assert session.history().equals(history)
    assert any("sensor_matched" in frame.value for frame in app.dataframe)
    app.switch_page("pages/2_Data_Integration.py").run()
    assert app.radio(key="integration_source").value == "Synthetic dataset"
    assert app.metric[0].value == "870"


@pytest.mark.parametrize("destination", ["pages/1_Overview.py", "pages/6_Multivariate_Monitoring.py", "pages/7_Model_Reliability.py"])
def test_navigation_pauses_acquisition_and_preserves_its_cursor(destination, tmp_path):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    app.session_state["modeling_state_root"] = str(tmp_path / "models")
    app.session_state["monitoring_models_root"] = str(tmp_path / "monitoring")
    app.session_state["reliability_state_root"] = str(tmp_path / "reliability")
    app.run()
    app.switch_page(PAGE).run()
    app.button(key="acq_start").click().run()
    session = app.session_state["acquisition_session"]
    assert session.running
    cursor = session.cursor
    app.switch_page(destination).run()
    assert not app.exception
    assert not app.session_state["acquisition_session"].running
    app.switch_page(PAGE).run()
    assert not app.exception and app.metric[1].value == str(cursor)


def test_entering_acquisition_pauses_process_replay(tmp_path):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    app.session_state["modeling_state_root"] = str(tmp_path / "models")
    app.session_state["monitoring_models_root"] = str(tmp_path / "monitoring")
    app.run()
    app.switch_page("pages/5_Process_Monitoring.py").run()
    app.button(key="replay_start").click().run()
    assert app.session_state["monitoring_session"].clock.running
    app.switch_page(PAGE).run()
    assert not app.exception
    assert not app.session_state["monitoring_session"].clock.running


def test_development_integration_has_no_monitoring_snapshot_source():
    app = AppTest.from_file(str(ROOT / "pages/2_Data_Integration.py"), default_timeout=60).run()
    assert not app.exception
    assert app.radio(key="integration_source").options == ["Synthetic dataset", "Upload CSV sources"]
