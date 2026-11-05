"""Replay controls, explicit model preparation, timing views, and phase navigation."""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.data_generation import SimulationConfig
from src.model_registry import prepare_demo_models


ROOT = Path(__file__).resolve().parents[1]
PAGE = "pages/5_Process_Monitoring.py"
CONFIG = SimulationConfig(n_batches=3, duration_h=72, n_wavenumbers=71, seed=331)


def page(tmp_path, models_root=None):
    app = AppTest.from_file(str(ROOT / PAGE), default_timeout=90)
    app.session_state["simulation_config"] = CONFIG
    app.session_state["modeling_state_root"] = str(tmp_path / "phase4")
    app.session_state["monitoring_models_root"] = str(models_root or tmp_path / "models")
    return app.run()


def history_table(app):
    return next(frame.value for frame in app.dataframe if "glucose_status" in frame.value)


@pytest.fixture(scope="module")
def models_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("monitor-ui-models")
    prepare_demo_models(CONFIG, root)
    return root


def test_missing_models_still_allow_sensor_replay_without_fake_estimates(tmp_path):
    app = page(tmp_path)
    assert not app.exception
    assert app.metric[1].value == "0"
    assert not list(tmp_path.iterdir())
    app.button(key="replay_step").click().run()
    assert not app.exception
    assert app.metric[1].value == "1"
    assert history_table(app).glucose_status.eq("model_unavailable").all()
    assert history_table(app).glucose_prediction_g_l.isna().all()
    app.button(key="replay_step").click().run()
    assert app.metric[1].value == "2"
    app.button(key="replay_reset").click().run()
    assert app.metric[1].value == "0"


def test_model_preparation_is_explicit_and_replay_never_retrains(tmp_path, monkeypatch):
    import src.model_registry as registry
    calls = []
    original = registry.train_study

    def recording_training(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(registry, "train_study", recording_training)
    app = page(tmp_path)
    assert calls == []
    app.button(key="prepare_monitor_models").click().run()
    assert not app.exception
    assert len(calls) == 2
    assert app.selectbox(key="monitor_glucose_model").value != "No saved model"
    assert app.selectbox(key="monitor_lactate_model").value != "No saved model"
    app.button(key="replay_step").click().run()
    app.run()
    assert not app.exception
    assert len(calls) == 2
    assert history_table(app).glucose_model_version.str.startswith("phase5-demo-glucose").all()
    assert not (tmp_path / "phase4").exists()


def test_start_pause_resume_speed_and_rule_changes_are_visible(tmp_path, models_root):
    app = page(tmp_path, models_root)
    app.button(key="replay_start").click().run()
    assert not app.exception
    assert app.session_state["monitoring_session"].clock.running
    app.button(key="replay_pause").click().run()
    assert not app.session_state["monitoring_session"].clock.running
    app.run()
    count = int(app.metric[1].value)
    app.selectbox(key="monitor_rate").select(5.0).run()
    assert int(app.metric[1].value) == count
    app.button(key="replay_start").click().run()
    assert app.session_state["monitoring_session"].clock.running
    app.button(key="replay_pause").click().run()
    app.number_input(key="monitor_domain").set_value(7.0).run()
    assert not app.exception
    assert app.metric[1].value == "0"
    assert not app.session_state["monitoring_session"].clock.running


def test_fault_history_has_gaps_and_backward_review_hides_future_rows(tmp_path, models_root):
    app = page(tmp_path, models_root)
    app.checkbox(key="monitor_faults").check().run()
    app.radio(key="monitor_mode").set_value("Historical review").run()
    app.slider(key="monitor_review_time").set_value(12.0).run()
    assert not app.exception
    history = history_table(app)
    assert len(history) == 13
    assert history.iloc[3].glucose_status == "invalid_input"
    assert history.iloc[7].glucose_status == "invalid_input"
    assert history.iloc[10].glucose_status == "invalid_input"
    assert history.iloc[12].glucose_status == "out_of_domain"
    assert np.isnan(history.iloc[3].glucose_prediction_g_l)
    assays = next(frame.value for frame in app.dataframe if "result_timestamp" in frame.value)
    assert len(assays) == 1
    app.slider(key="monitor_review_time").set_value(0.0).run()
    assert not app.exception
    assert len(history_table(app)) == 1
    assert next(frame.value for frame in app.dataframe if "result_timestamp" in frame.value).empty


def test_nominal_history_contains_actual_model_estimates_and_postbatch_assays(tmp_path, models_root):
    app = page(tmp_path, models_root)
    app.radio(key="monitor_mode").set_value("Historical review").run()
    app.slider(key="monitor_review_time").set_value(24.0).run()
    assert not app.exception
    history = history_table(app)
    assert np.isfinite(history.iloc[-1].glucose_prediction_g_l)
    assert history.iloc[-1].lactate_status == "outside_reliability_limits"
    assert not history.iloc[-1].lactate_reliability_eligible
    assert np.isnan(history.iloc[-1].lactate_prediction_g_l)
    assert history.iloc[-1].lactate_flags
    accepted = history.lactate_status.eq("ok")
    assert accepted.any() and np.isfinite(history.loc[accepted, "lactate_prediction_g_l"]).all()
    app.slider(key="monitor_review_time").set_value(app.slider(key="monitor_review_time").max).run()
    assert not app.exception
    assert len(history_table(app)) == 73
    assert len(next(frame.value for frame in app.dataframe if "result_timestamp" in frame.value)) == 4


def test_navigation_pauses_replay_and_preserves_earlier_page_features(tmp_path):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    app.session_state["modeling_state_root"] = str(tmp_path / "phase4")
    app.session_state["monitoring_models_root"] = str(tmp_path / "models")
    app.run()
    app.switch_page(PAGE).run()
    app.button(key="replay_start").click().run()
    assert app.session_state["monitoring_session"].clock.running
    app.switch_page("pages/3_Spectral_Analytics.py").run()
    assert not app.exception
    assert not app.session_state["monitoring_session"].clock.running
    assert app.metric[0].value == "870"
    app.switch_page("pages/4_Predictive_Modeling.py").run()
    assert not app.exception
    assert app.metric[1].value == "42"
    app.switch_page(PAGE).run()
    assert not app.exception
    assert not app.session_state["monitoring_session"].clock.running
    assert int(app.metric[1].value) >= 1
