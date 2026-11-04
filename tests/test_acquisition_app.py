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
    return app.run()


def test_initial_load_does_not_poll_and_single_poll_reset_work():
    app = page()
    assert not app.exception
    assert [metric.value for metric in app.metric[:3]] == ["Paused", "0", "0"]
    assert app.button(key="acq_publish").disabled
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
    app.selectbox(key="acq_limit").select(25)
    app.button(key="acq_apply").click().run()
    assert app.session_state["acquisition_session"].cursor == cursor
    app.selectbox(key="acq_batch").select("B002")
    app.button(key="acq_apply").click().run()
    assert not app.exception
    assert app.metric[1].value == "0"
    assert not app.session_state["acquisition_session"].running


def test_fault_collection_has_duplicates_quarantine_and_missing_ph():
    app = page()
    app.checkbox(key="acq_faults").check()
    app.selectbox(key="acq_limit").select(100)
    app.button(key="acq_apply").click().run()
    app.button(key="acq_poll").click().run()
    assert not app.exception
    assert app.metric[0].value == "Complete"
    assert app.metric[3].value == "2"
    assert app.session_state["acquisition_session"].ledger().status.eq("duplicate").sum() == 1
    assert app.session_state["acquisition_session"].snapshot().sensors.ph.isna().sum() == 1
    assert app.button(key="acq_start").disabled


def test_http_source_is_explicit_and_connection_error_is_visible(monkeypatch):
    import src.acquisition as acquisition
    def disconnected(*args, **kwargs):
        raise acquisition.AcquisitionError("Test source disconnected")
    monkeypatch.setattr(acquisition.LocalHTTPAdapter, "poll", disconnected)
    app = page()
    app.selectbox(key="acq_mode").select("Local HTTP test source")
    app.button(key="acq_apply").click().run()
    assert not app.exception and not app.error
    app.button(key="acq_start").click().run()
    assert not app.exception
    assert app.metric[0].value == "Read failed — paused"
    assert app.metric[1].value == "0"
    assert any("disconnected" in error.value for error in app.error)


def test_invalid_test_url_is_rejected_without_a_network_read():
    app = page()
    app.selectbox(key="acq_mode").select("Local HTTP test source")
    app.text_input(key="acq_url").set_value("http://example.com:8765")
    app.button(key="acq_apply").click().run()
    assert not app.exception
    assert any("loopback" in error.value for error in app.error)


def test_explicit_snapshot_handoff_opens_existing_integration_and_spectral_workflow():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    app.switch_page(PAGE).run()
    app.selectbox(key="acq_limit").select(100)
    app.button(key="acq_apply").click().run()
    app.button(key="acq_poll").click().run()
    assert not app.exception
    count = len(app.session_state["acquisition_session"].snapshot().raman)
    app.button(key="acq_publish").click().run()
    assert not app.exception
    published = app.session_state["acquisition_snapshot"]
    assert len(published["sources"].raman) == count
    app.button(key="acq_poll").click().run()
    assert len(app.session_state["acquisition_snapshot"]["sources"].raman) == count
    app.switch_page("pages/2_Data_Integration.py").run()
    assert not app.exception
    assert app.radio(key="integration_source").value == "Acquired test snapshot"
    assert app.metric[0].value == str(count)
    assert len(app.session_state["integration_result"].raw.raman) == count
    app.switch_page("pages/3_Spectral_Analytics.py").run()
    assert not app.exception


@pytest.mark.parametrize("destination", ["pages/1_Overview.py", "pages/5_Process_Monitoring.py", "pages/6_Multivariate_Monitoring.py", "pages/7_Model_Reliability.py"])
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


def test_missing_acquired_snapshot_has_guidance():
    app = AppTest.from_file(str(ROOT / "pages/2_Data_Integration.py"), default_timeout=60).run()
    app.radio(key="integration_source").set_value("Acquired test snapshot").run()
    assert not app.exception
    assert any("send a snapshot" in warning.value for warning in app.warning)
