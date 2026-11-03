"""Exercise the actual Streamlit page and its main exploration interactions."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_overview_loads_and_updates_batch_and_variable_selections():
    app = AppTest.from_file(str(APP_PATH), default_timeout=45).run()
    assert not app.exception
    assert app.metric[0].value == "6"
    assert app.metric[1].value == "870"
    assert app.metric[2].value == "281"
    assert len(app.tabs) == 4

    app.multiselect(key="batch_filter").set_value(["B004"]).run()
    app.selectbox(key="sensor_variable").select("Dissolved oxygen (% air saturation)").run()
    app.selectbox(key="truth_variable").select("Lactate (g/L)").run()
    app.slider(key="raman_sample").set_value(100).run()
    assert not app.exception
    assert app.metric[0].value == "1"
    assert app.metric[1].value == "145"
    assert app.dataframe[0].value["batch_id"].tolist() == ["B004"]
    assert any("B004-S00100" in caption.value for caption in app.caption)


def test_empty_batch_selection_shows_guidance():
    app = AppTest.from_file(str(APP_PATH), default_timeout=45).run()
    app.multiselect(key="batch_filter").set_value([]).run()
    assert not app.exception
    assert len(app.warning) == 1
    assert "Select at least one batch" in app.warning[0].value


def test_simulation_form_regenerates_a_different_sized_dataset():
    app = AppTest.from_file(str(APP_PATH), default_timeout=45).run()
    app.slider(key="n_batches").set_value(3)
    app.selectbox(key="duration").select(72)
    app.selectbox(key="interval").select(2.0)
    app.number_input(key="seed").set_value(17)
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value == "3"
    assert app.metric[1].value == "111"
    assert len(app.multiselect(key="batch_filter").options) == 3


def test_pages_start_with_legacy_cached_app_data_missing_navigation_helper(monkeypatch, tmp_path):
    """Reproduce a long-running server retaining the pre-Phase-5 module exports."""
    import src.app_data as cached_data
    from src.replay import MonitoringSession

    monkeypatch.delattr(cached_data, "pause_monitoring_on_navigation", raising=False)
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    session = MonitoringSession("legacy-module-regression")
    session.clock.start(0, 20)
    app.session_state["monitoring_session"] = session
    app.session_state["modeling_state_root"] = str(tmp_path / "models")
    app.run()
    assert not app.exception
    assert not app.session_state["monitoring_session"].clock.running
    for page in (
        "pages/2_Data_Integration.py",
        "pages/3_Spectral_Analytics.py",
        "pages/4_Predictive_Modeling.py",
        "pages/6_Multivariate_Monitoring.py",
        "pages/7_Model_Reliability.py",
        "pages/9_Test_Acquisition.py",
    ):
        app.switch_page(page).run()
        assert not app.exception
