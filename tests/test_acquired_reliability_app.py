"""Shared navigation, explicit baseline preparation and historical regression."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from streamlit.testing.v1 import AppTest

from src.data_generation import SimulationConfig
from src.model_registry import prepare_demo_models


ROOT = Path(__file__).resolve().parents[1]
PAGE = "pages/7_Model_Reliability.py"
CONFIG = SimulationConfig(n_batches=6, duration_h=72, n_wavenumbers=32)


@pytest.fixture(scope="module")
def model_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("acquired-reliability-ui-models")
    prepare_demo_models(CONFIG, root)
    return root


def app_page(tmp_path, model_root):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    app.session_state["simulation_config"] = CONFIG
    app.session_state["modeling_state_root"] = str(tmp_path / "modeling")
    app.session_state["monitoring_models_root"] = str(model_root)
    app.session_state["multivariate_state_root"] = str(tmp_path / "pca")
    app.session_state["acquired_reliability_state_root"] = str(tmp_path / "references")
    app.session_state["reliability_state_root"] = str(tmp_path / "historical")
    return app.run()


def history(app):
    return next(frame.value for frame in app.dataframe if "prediction_status" in frame.value and "current_eligible" in frame.value and "pair_status" not in frame.value)


def test_all_monitoring_pages_share_cursor_and_reliability_never_reinfers(tmp_path, model_root, monkeypatch):
    app = app_page(tmp_path, model_root)
    app.switch_page("pages/6_Multivariate_Monitoring.py").run()
    app.button(key="mspc_fit").click().run()
    assert not app.exception
    pca = app.session_state["mspc_model"]
    means = pca.scaler.mean_.copy()
    app.switch_page("pages/9_Test_Acquisition.py").run()
    app.selectbox(key="stream_source").select("Historical in-app test source").run()
    app.button(key="acq_poll").click().run()
    shared = app.session_state["stream_monitoring_session"]
    cursor = shared.acquisition.cursor
    before = shared.history().copy(deep=True)
    def forbidden(*args, **kwargs):
        pytest.fail("Page visits must not fit or repeat inference")
    with monkeypatch.context() as patch:
        patch.setattr(Pipeline, "predict", forbidden)
        patch.setattr(StandardScaler, "fit", forbidden)
        app.switch_page(PAGE).run()
        assert not app.exception
        assert app.radio(key="reliability_mode").value == "Acquired data reliability"
        assert shared is app.session_state["stream_monitoring_session"]
        assert shared.acquisition.cursor == cursor
        pd.testing.assert_frame_equal(shared.history(), before)
        assert history(app).prediction_g_l.equals(before.glucose_prediction_g_l)
        assert not (tmp_path / "references").exists()
        app.switch_page("pages/6_Multivariate_Monitoring.py").run()
        app.switch_page("pages/5_Process_Monitoring.py").run()
        assert not app.exception and shared.acquisition.cursor == cursor
    app.switch_page(PAGE).run()
    app.button(key="reliability_shared_poll").click().run()
    assert not app.exception and shared.acquisition.cursor > cursor
    np.testing.assert_array_equal(pca.scaler.mean_, means)


def test_explicit_matching_baseline_persists_and_rule_changes_preserve_models(tmp_path, model_root, monkeypatch):
    app = app_page(tmp_path, model_root)
    artifacts = {str(path): path.read_bytes() for path in model_root.rglob("*") if path.is_file()}
    app.switch_page("pages/9_Test_Acquisition.py").run()
    app.selectbox(key="stream_source").select("Historical in-app test source").run()
    app.selectbox(key="stream_limit").select(100).run()
    app.button(key="acq_poll").click().run()
    app.switch_page(PAGE).run()
    assert not app.exception and any("Training baseline unavailable" in item.value for item in app.info)
    shared = app.session_state["stream_monitoring_session"]
    cursor = shared.acquisition.cursor
    app.button(key="acquired_reliability_prepare").click().run()
    assert not app.exception
    saved = list((tmp_path / "references").glob("*/reference.json"))
    assert len(saved) == 1
    baseline = saved[0].read_bytes()
    before = history(app).copy(deep=True)
    monkeypatch.setattr(Pipeline, "predict", lambda *args, **kwargs: pytest.fail("Rule changes must not infer"))
    app.number_input(key="acquired_reliability_residual_limit").set_value(0.1).run()
    assert not app.exception and shared.acquisition.cursor == cursor
    pd.testing.assert_frame_equal(history(app), before)
    assert saved[0].read_bytes() == baseline
    assert artifacts == {str(path): path.read_bytes() for path in model_root.rglob("*") if path.is_file()}
    assert not (tmp_path / "historical").exists()
    app.selectbox(key="acquired_reliability_target").select("Lactate").run()
    assert not app.exception and "lactate" in history(app).model_version.iloc[0].lower()
    assert any("Training baseline unavailable" in item.value for item in app.info)


def test_unconfigured_acquired_mode_does_not_create_collector(tmp_path, model_root):
    app = app_page(tmp_path, model_root)
    app.switch_page(PAGE).run()
    app.radio(key="reliability_mode").set_value("Acquired data reliability").run()
    assert not app.exception
    assert "stream_monitoring_session" not in app.session_state
    assert any("Configure Data Acquisition" in item.value for item in app.info)
    assert not list(tmp_path.iterdir())


def test_read_failure_withholds_current_status_and_retains_diagnostics(tmp_path, model_root):
    app = app_page(tmp_path, model_root)
    app.switch_page("pages/9_Test_Acquisition.py").run()
    app.button(key="acq_poll").click().run()
    app.switch_page(PAGE).run()
    assert not app.exception
    shared = app.session_state["stream_monitoring_session"]
    cursor = shared.acquisition.cursor
    before = history(app).prediction_g_l.copy()
    shared.acquisition.last_error = "API unavailable"
    app.run()
    assert not app.exception and shared.acquisition.cursor == cursor
    assert not history(app).current_eligible.any()
    pd.testing.assert_series_equal(history(app).prediction_g_l, before)
    assert any("API unavailable" in item.value for item in app.warning)
