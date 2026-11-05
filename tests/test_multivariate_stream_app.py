"""Acquisition/PCA/soft-sensor navigation, explicit persistence and source health."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit.testing.v1 import AppTest

from src.data_generation import SimulationConfig
from src.multivariate_storage import list_pca_references


ROOT = Path(__file__).resolve().parents[1]
PAGE = "pages/6_Multivariate_Monitoring.py"
CONFIG = SimulationConfig(n_batches=6, duration_h=72, n_wavenumbers=32)


def app_page(tmp_path):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    app.session_state["simulation_config"] = CONFIG
    app.session_state["modeling_state_root"] = str(tmp_path / "modeling")
    app.session_state["monitoring_models_root"] = str(tmp_path / "models")
    app.session_state["multivariate_state_root"] = str(tmp_path / "pca")
    return app.run()


def acquired_table(app):
    return next(frame.value for frame in app.dataframe if "pca_model_id" in frame.value)


def test_one_shared_cursor_and_saved_reference_across_all_three_monitoring_pages(tmp_path, monkeypatch):
    app = app_page(tmp_path)
    app.switch_page(PAGE).run()
    app.button(key="mspc_fit").click().run()
    assert not app.exception
    model = app.session_state["mspc_model"]
    mean, components = model.scaler.mean_.copy(), model.pca.components_.copy()
    limits = (model.t2_limit, model.spe_limit)
    app.button(key="mspc_save_reference").click().run()
    assert not app.exception
    saved = list_pca_references(tmp_path / "pca")
    assert len(saved) == 1
    artifacts = {str(path): path.read_bytes() for path in (tmp_path / "pca").rglob("*") if path.is_file()}
    monkeypatch.setattr(PCA, "fit", lambda *args, **kwargs: pytest.fail("No PCA refit during acquisition"))
    monkeypatch.setattr(StandardScaler, "fit", lambda *args, **kwargs: pytest.fail("No scaler refit during acquisition"))
    app.switch_page("pages/9_Test_Acquisition.py").run()
    app.selectbox(key="stream_source").select("Historical in-app test source").run()
    app.button(key="acq_poll").click().run()
    shared = app.session_state["stream_monitoring_session"]
    cursor = shared.acquisition.cursor
    history = shared.history().copy()
    app.switch_page(PAGE).run()
    assert not app.exception
    assert app.radio(key="mspc_mode").value == "Acquired sensor monitoring"
    assert app.session_state["stream_monitoring_session"] is shared
    assert shared.acquisition.cursor == cursor
    scored = acquired_table(app)
    assert scored.status.eq("scored").all()
    assert scored.batch_id.eq("MON-B001").all()
    contributions = next(frame.value for frame in app.dataframe if "spe_contribution" in frame.value)
    assert contributions.spe_contribution.sum() == pytest.approx(scored.spe.iloc[0])
    app.button(key="mspc_shared_poll").click().run()
    assert not app.exception and shared.acquisition.cursor > cursor
    advanced_cursor = shared.acquisition.cursor
    app.switch_page("pages/5_Process_Monitoring.py").run()
    assert not app.exception
    assert shared.acquisition.cursor == advanced_cursor
    assert app.session_state["stream_monitoring_session"] is shared
    assert shared.history().iloc[:len(history)].equals(history)
    np.testing.assert_array_equal(model.scaler.mean_, mean)
    np.testing.assert_array_equal(model.pca.components_, components)
    assert (model.t2_limit, model.spe_limit) == limits
    assert artifacts == {str(path): path.read_bytes() for path in (tmp_path / "pca").rglob("*") if path.is_file()}


def test_explicit_saved_reference_load_in_new_session_does_not_refit(tmp_path, monkeypatch):
    app = app_page(tmp_path)
    app.switch_page(PAGE).run()
    app.button(key="mspc_fit").click().run()
    original_id = app.session_state["mspc_model"].model_id
    app.button(key="mspc_save_reference").click().run()
    monkeypatch.setattr(PCA, "fit", lambda *args, **kwargs: pytest.fail("Loading must not fit PCA"))
    fresh = app_page(tmp_path)
    fresh.switch_page(PAGE).run()
    assert not fresh.exception
    assert "mspc_model" not in fresh.session_state
    fresh.button(key="mspc_load_reference").click().run()
    assert not fresh.exception
    assert fresh.session_state["mspc_model"].model_id == original_id


def test_unconfigured_acquired_mode_never_creates_or_polls_another_collector(tmp_path, monkeypatch):
    app = app_page(tmp_path)
    app.switch_page(PAGE).run()
    app.button(key="mspc_fit").click().run()
    from src.acquisition import LocalHTTPAdapter
    monkeypatch.setattr(LocalHTTPAdapter, "poll", lambda *args, **kwargs: pytest.fail("PCA view must not create a source"))
    app.radio(key="mspc_mode").set_value("Acquired sensor monitoring").run()
    assert not app.exception
    assert any("Configure a monitoring source" in message.value for message in app.info)
    assert "stream_monitoring_session" not in app.session_state
    assert not (tmp_path / "pca").exists()


def test_source_failure_withholds_current_pca_cards_and_retains_history(tmp_path):
    app = app_page(tmp_path)
    app.switch_page(PAGE).run()
    app.button(key="mspc_fit").click().run()
    app.switch_page("pages/9_Test_Acquisition.py").run()
    app.button(key="acq_poll").click().run()
    app.switch_page(PAGE).run()
    assert not app.exception
    assert next(metric.value for metric in app.metric if metric.label == "Latest current T²") != "Unavailable"
    before = acquired_table(app).t2.to_numpy(copy=True)
    shared = app.session_state["stream_monitoring_session"]
    cursor = shared.acquisition.cursor
    shared.acquisition.last_error = "API unavailable"
    app.run()
    assert not app.exception
    assert next(metric.value for metric in app.metric if metric.label == "Latest current T²") == "Unavailable"
    np.testing.assert_array_equal(acquired_table(app).t2, before)
    assert shared.acquisition.cursor == cursor
