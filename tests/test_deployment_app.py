"""Actual Streamlit deployment flow and public visitor isolation."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.data_generation import SimulationConfig
from src.deployment_check import deployment_preflight
from src.model_registry import prepare_demo_models
from src.runtime import deployment_mode, initialize_runtime


ROOT = Path(__file__).resolve().parents[1]


def deployment_page(tmp_path, models=None):
    app = AppTest.from_file(str(ROOT / "pages/8_Model_Deployment.py"), default_timeout=90)
    app.session_state["modeling_state_root"] = str(tmp_path / "studies")
    app.session_state["monitoring_models_root"] = str(models or tmp_path / "models")
    app.session_state["reliability_state_root"] = str(tmp_path / "reliability")
    app.session_state["deployment_state_root"] = str(tmp_path / "releases")
    app.session_state["simulation_config"] = SimulationConfig(duration_h=72, n_wavenumbers=71)
    return app.run()


def test_missing_model_page_does_not_train_or_write(tmp_path, monkeypatch):
    import src.model_registry as registry
    monkeypatch.setattr(registry, "train_study", lambda *a, **k: pytest.fail("No render-time fit"))
    app = deployment_page(tmp_path)
    assert not app.exception
    assert any("No trusted saved models" in info.value for info in app.info)
    assert list(tmp_path.iterdir()) == []


def test_prediction_release_and_rule_change_flow(tmp_path):
    models = tmp_path / "models"
    prepare_demo_models(SimulationConfig(duration_h=72, n_wavenumbers=71), models)
    app = deployment_page(tmp_path, models)
    assert not app.exception
    app.button(key="deployment_predict").click().run()
    assert not app.exception
    result = app.session_state["deployment_prediction_result"]["frame"]
    assert len(result) == 5 and result.model_version.nunique() == 1
    assert len(app.get("download_button")) >= 4
    app.number_input(key="deployment_domain").set_value(7.0).run()
    assert not app.exception
    assert not any(getattr(table, "key", None) == "deployment_predictions" for table in app.dataframe)
    app.button(key="deployment_release").click().run()
    assert not app.exception
    folder = Path(app.session_state["deployment_release_result"]["folder"])
    assert (folder / "release_manifest.json").exists()


def test_demo_preparation_is_explicit_and_reuses_saved_pipelines(tmp_path, monkeypatch):
    app = deployment_page(tmp_path)
    app.button(key="deployment_prepare").click().run()
    assert not app.exception
    assert len(app.selectbox(key="deployment_model").options) == 2
    import src.model_registry as registry
    monkeypatch.setattr(registry, "train_study", lambda *a, **kw: pytest.fail("Existing demo should be reused"))
    app.run()
    app.button(key="deployment_prepare").click().run()
    assert not app.exception


def test_local_runtime_does_not_override_existing_roots(monkeypatch):
    monkeypatch.setenv("PAT_DEPLOYMENT_MODE", "local")
    state = {"modeling_state_root": "my-studies"}
    initialize_runtime(state)
    assert state == {"modeling_state_root": "my-studies"}


def test_public_runtime_isolated_stable_and_bounded(monkeypatch):
    monkeypatch.setenv("PAT_DEPLOYMENT_MODE", "public_demo")
    first, second = {}, {}
    initialize_runtime(first)
    initialize_runtime(second)
    assert first["modeling_state_root"] != second["modeling_state_root"]
    before = dict(first)
    initialize_runtime(first)
    assert first == before
    assert not Path(first["modeling_state_root"]).exists()
    first["simulation_config"] = SimulationConfig(n_batches=12, n_wavenumbers=281)
    initialize_runtime(first)
    assert first["simulation_config"].n_batches == 8
    assert first["simulation_config"].n_wavenumbers == 71
    with pytest.raises(ValueError):
        initialize_runtime({"public_demo_session_id": "../escape"})


def test_unknown_mode_fails_explicitly(monkeypatch):
    monkeypatch.setenv("PAT_DEPLOYMENT_MODE", "unknown")
    with pytest.raises(ValueError):
        deployment_mode()


def test_public_app_visits_all_pages_without_model_training(monkeypatch):
    monkeypatch.setenv("PAT_DEPLOYMENT_MODE", "public_demo")
    import src.model_registry as registry
    monkeypatch.setattr(registry, "train_study", lambda *a, **kw: pytest.fail("No implicit model preparation"))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not app.exception
    assert app.session_state["simulation_config"].n_wavenumbers == 71
    for page in sorted((ROOT / "pages").glob("*.py")):
        app.switch_page("pages/" + page.name).run()
        assert not app.exception, page.name
    assert "Local HTTP test source" not in app.selectbox(key="acq_mode").options
    app.switch_page("pages/5_Process_Monitoring.py").run()
    assert app.radio(key="monitor_mode").value == "Replay"
    app.radio(key="monitor_mode").set_value("Acquired stream").run()
    assert not app.exception
    assert "Local HTTP test source" not in app.selectbox(key="stream_source").options
    assert not Path(app.session_state["modeling_state_root"]).exists()


def test_preflight_detects_missing_files_and_pins(tmp_path):
    assert deployment_preflight()["ready"]
    (tmp_path / "requirements.txt").write_text("nonexistent-pat-package==1.0\n")
    result = deployment_preflight(tmp_path)
    assert not result["ready"]
    assert not result["dependencies"][0]["ok"]
