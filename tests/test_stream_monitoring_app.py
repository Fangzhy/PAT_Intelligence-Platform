"""Acquired-stream controls, saved inference, source handoff, and replay regressions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.data_generation import SimulationConfig
from src.model_registry import prepare_demo_models


ROOT = Path(__file__).resolve().parents[1]
PAGE = "pages/5_Process_Monitoring.py"
CONFIG = SimulationConfig(n_batches=2, duration_h=24, n_wavenumbers=32)


@pytest.fixture(scope="module")
def model_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("stream-ui-models")
    prepare_demo_models(CONFIG, root)
    return root


def page(tmp_path, model_root=None, *, historical=False):
    app = AppTest.from_file(str(ROOT / PAGE), default_timeout=90)
    app.session_state["simulation_config"] = CONFIG
    app.session_state["modeling_state_root"] = str(tmp_path / "phase4")
    app.session_state["monitoring_models_root"] = str(model_root or tmp_path / "models")
    app.run()
    app.radio(key="monitor_mode").set_value("Acquired stream").run()
    if historical:
        app.selectbox(key="stream_source").select("Historical in-app test source").run()
        app.selectbox(key="stream_limit").select(100).run()
    return app


def history(app):
    return next(frame.value for frame in app.dataframe if "glucose_status" in frame.value)


def test_acquired_stream_does_not_fit_until_explicit_preparation_and_reuses_models(tmp_path, monkeypatch):
    import src.model_registry as registry
    calls = []
    original = registry.train_study
    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(registry, "train_study", record)
    app = page(tmp_path)
    assert not app.exception and calls == [] and app.metric[1].value == "0"
    assert not list(tmp_path.iterdir())
    app.button(key="stream_poll").click().run()
    assert not app.exception
    assert history(app).glucose_status.iloc[0] == "model_unavailable"
    app.button(key="prepare_stream_models").click().run()
    assert not app.exception and calls == [1, 1]
    assert app.selectbox(key="stream_glucose_model").value != "No saved model"
    app.button(key="stream_poll").click().run()
    assert not app.exception and history(app).glucose_status.iloc[0] == "ok"
    assert np.isfinite(history(app).glucose_prediction_g_l.iloc[0])
    app.button(key="prepare_stream_models").click().run()
    assert not app.exception and calls == [1, 1]


def test_historical_stream_inference_has_frozen_history_and_rule_changes_reset(tmp_path, model_root):
    app = page(tmp_path, model_root, historical=True)
    assert not app.checkbox(key="stream_enforce").value
    app.button(key="stream_poll").click().run()
    assert not app.exception
    assert [metric.value for metric in app.metric[:4]] == ["Complete", "53", "25", "0"]
    before = history(app).copy()
    assert before.glucose_status.eq("ok").all()
    app.selectbox(key="stream_sensor").select("ph").run()
    assert not app.exception
    pd.testing.assert_frame_equal(history(app), before)
    app.number_input(key="stream_domain").set_value(7.0).run()
    assert not app.exception and app.metric[1].value == "0"


def test_historical_source_freshness_failure_and_fresh_source_default(tmp_path, model_root):
    app = page(tmp_path, model_root, historical=True)
    app.checkbox(key="stream_enforce").check().run()
    app.button(key="stream_poll").click().run()
    assert not app.exception and history(app).glucose_status.eq("stale_input").all()
    assert history(app).glucose_prediction_g_l.isna().all()
    app.checkbox(key="stream_enforce").uncheck().run()
    app.selectbox(key="stream_source").select("Fresh in-app test stream").run()
    assert not app.exception and app.checkbox(key="stream_enforce").value


def test_stale_current_cards_are_withheld_while_historical_predictions_remain(tmp_path, model_root):
    app = page(tmp_path, model_root, historical=True)
    app.button(key="stream_poll").click().run()
    assert not app.exception
    before = history(app).glucose_prediction_g_l.to_numpy(copy=True)
    app.session_state["stream_monitoring_session"].last_new_data_utc = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=20)
    app.run()
    assert not app.exception
    assert next(metric.value for metric in app.metric if metric.label == "Stream predicted glucose (g/L)") == "Unavailable"
    np.testing.assert_array_equal(history(app).glucose_prediction_g_l, before)
    assert any("No recent new source records" in warning.value for warning in app.warning)


def test_invalid_http_configuration_never_reuses_previous_adapter(tmp_path):
    app = page(tmp_path)
    app.button(key="stream_poll").click().run()
    app.selectbox(key="stream_source").select("Local HTTP test source").run()
    app.text_input(key="stream_url").set_value("http://example.com:8765").run()
    assert not app.exception and any("loopback" in error.value for error in app.error)
    app.run()
    assert not app.exception and any("loopback" in error.value for error in app.error)
    assert not app.metric


def test_http_failure_pauses_with_unchanged_cursor_and_visible_error(tmp_path, monkeypatch):
    import src.acquisition as acquisition
    def fail(*args, **kwargs):
        raise acquisition.AcquisitionError("HTTP source disconnected")
    monkeypatch.setattr(acquisition.LocalHTTPAdapter, "poll", fail)
    app = page(tmp_path)
    app.selectbox(key="stream_source").select("Local HTTP test source").run()
    app.button(key="stream_start").click().run()
    assert not app.exception and app.metric[0].value == "Read failed — paused"
    assert app.metric[1].value == "0" and not app.session_state["stream_monitoring_session"].acquisition.running
    assert any("disconnected" in error.value for error in app.error)


def test_start_pause_reset_and_mode_switch_preserve_separate_replay(tmp_path):
    app = page(tmp_path)
    app.button(key="stream_start").click().run()
    assert not app.exception and app.session_state["stream_monitoring_session"].acquisition.running
    app.button(key="stream_pause").click().run()
    assert not app.session_state["stream_monitoring_session"].acquisition.running
    cursor = app.session_state["stream_monitoring_session"].acquisition.cursor
    app.radio(key="monitor_mode").set_value("Replay").run()
    assert not app.exception and app.metric[1].value == "0"
    app.button(key="replay_step").click().run()
    assert app.metric[1].value == "1"
    app.radio(key="monitor_mode").set_value("Acquired stream").run()
    assert not app.exception and app.metric[1].value == str(cursor)
    app.button(key="stream_reset").click().run()
    assert not app.exception and app.metric[1].value == "0"


def test_navigation_pauses_stream_and_restores_its_consumer_history(tmp_path):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    app.session_state["modeling_state_root"] = str(tmp_path / "phase4")
    app.session_state["monitoring_models_root"] = str(tmp_path / "models")
    app.run()
    app.switch_page(PAGE).run()
    app.radio(key="monitor_mode").set_value("Acquired stream").run()
    app.button(key="stream_start").click().run()
    cursor = app.session_state["stream_monitoring_session"].acquisition.cursor
    assert app.session_state["stream_monitoring_session"].acquisition.running
    app.switch_page("pages/9_Test_Acquisition.py").run()
    assert not app.exception and not app.session_state["stream_monitoring_session"].acquisition.running
    app.switch_page("pages/2_Data_Integration.py").run()
    assert not app.exception
    app.switch_page(PAGE).run()
    assert not app.exception and app.radio(key="monitor_mode").value == "Acquired stream"
    assert app.metric[1].value == str(cursor)


def test_data_acquisition_source_handoff_configures_monitoring_without_copying_cursor(tmp_path):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    app.session_state["modeling_state_root"] = str(tmp_path / "phase4")
    app.session_state["monitoring_models_root"] = str(tmp_path / "models")
    app.run()
    app.switch_page("pages/9_Test_Acquisition.py").run()
    app.button(key="acq_poll").click().run()
    assert app.session_state["acquisition_session"].cursor == 10
    app.button(key="acq_open_monitoring").click().run()
    assert not app.exception and app.radio(key="monitor_mode").value == "Acquired stream"
    assert app.selectbox(key="stream_source").value == "Historical in-app test source"
    assert not app.checkbox(key="stream_enforce").value
    assert app.metric[1].value == "0"
    # Follow the programmatic redirect in AppTest's persistent page selection.
    app.switch_page(PAGE)
    app.button(key="stream_poll").click().run()
    assert not app.exception and int(app.metric[1].value) > 0
    assert app.session_state["acquisition_session"].cursor == 10
