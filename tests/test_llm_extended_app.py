"""Manual explanations on spectral, replay/stream, and PCA monitoring pages."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.data_generation import SimulationConfig, generate_dataset
from src.data_integration import integrate_sources, sources_from_dataset
from test_llm_app import fake_ai


ROOT = Path(__file__).resolve().parents[1]
CONFIG = SimulationConfig(n_batches=6, duration_h=24, n_wavenumbers=32)


def page(name, tmp_path=None):
    app = AppTest.from_file(str(ROOT / "pages" / name), default_timeout=90)
    app.session_state["simulation_config"] = CONFIG
    if tmp_path is not None:
        app.session_state["modeling_state_root"] = str(tmp_path / "studies")
        app.session_state["monitoring_models_root"] = str(tmp_path / "models")
    return app.run()


def has_explanation_download(app, scope):
    return any(button.key == f"ai_{scope}_download" for button in app.get("download_button"))


def test_spectral_explanation_is_manual_and_reference_changes_hide_old_pca_text(fake_ai):
    app = page("3_Spectral_Analytics.py")
    assert not app.exception and fake_ai == []
    app.button(key="ai_spectral_explain").click().run()
    assert not app.exception and len(fake_ai) == 1
    assert fake_ai[-1].payload["target"] == "Raman spectra"
    assert any(fact["label"] == "Cumulative retained reference variance" for fact in fake_ai[-1].payload["facts"])
    app.multiselect(key="spectral_references").set_value([]).run()
    assert not app.exception and not has_explanation_download(app, "spectral")
    app.button(key="ai_spectral_explain").click().run()
    assert len(fake_ai) == 2
    variance = next(fact for fact in fake_ai[-1].payload["facts"] if fact["label"] == "Exploratory PCA reference variance")
    assert variance["value"] is None


def test_prepared_spectral_summary_requires_current_snapshot_consent(fake_ai):
    app = page("3_Spectral_Analytics.py")
    app.session_state["integration_result"] = integrate_sources(sources_from_dataset(generate_dataset(CONFIG)))
    app.radio(key="spectral_source").set_value("Latest Phase 2 prepared Raman").run()
    assert not app.exception and app.button(key="ai_spectral_explain").disabled
    app.checkbox(key="ai_spectral_consent").check().run()
    app.button(key="ai_spectral_explain").click().run()
    assert not app.exception and len(fake_ai) == 1 and fake_ai[-1].requires_consent


def test_multivariate_explanation_uses_actual_model_and_selected_view_without_refitting(fake_ai, monkeypatch):
    app = page("6_Multivariate_Monitoring.py")
    assert fake_ai == [] and not app.exception
    app.button(key="mspc_fit").click().run()
    before = app.session_state["mspc_model"].model_id
    import src.multivariate_monitoring as monitoring
    monkeypatch.setattr(monitoring, "fit_process_monitor", lambda *a, **kw: pytest.fail("Explanation refit PCA"))
    app.button(key="ai_multivariate_explain").click().run()
    assert not app.exception and len(fake_ai) == 1
    assert app.session_state["mspc_model"].model_id == before
    assert fake_ai[-1].payload["target"] == "Observed process sensors"
    app.multiselect(key="mspc_batches").set_value(["B001"]).run()
    assert not app.exception and not has_explanation_download(app, "multivariate")
    app.button(key="ai_multivariate_explain").click().run()
    count = next(fact["value"] for fact in fake_ai[-1].payload["facts"] if fact["label"] == "Selected chart observations")
    assert count == 25 and len(fake_ai) == 2


def test_replay_capture_is_explicit_preserved_as_historical_and_cleared_by_reset(tmp_path, fake_ai):
    app = page("5_Process_Monitoring.py", tmp_path)
    assert not app.exception and app.button(key="ai_process_capture_button").disabled
    app.button(key="replay_step").click().run()
    app.button(key="ai_process_capture_button").click().run()
    assert not app.exception and fake_ai == []
    app.button(key="ai_process_explain").click().run()
    assert not app.exception and len(fake_ai) == 1
    assert app.session_state["monitoring_session"].clock.cursor == 0
    app.button(key="replay_step").click().run()
    assert len(fake_ai) == 1
    assert any("Monitoring has changed since capture" in item.value for item in app.warning)
    assert any("Captured historical snapshot" in item.value for item in app.caption)
    app.button(key="ai_process_capture_button").click().run()
    assert not has_explanation_download(app, "process")
    app.button(key="ai_process_explain").click().run()
    assert len(fake_ai) == 2
    app.button(key="replay_reset").click().run()
    assert not app.exception and "ai_process_capture" not in app.session_state
    assert not has_explanation_download(app, "process")


def test_backward_historical_review_hides_capture_and_future_assay_summary(tmp_path, fake_ai):
    app = page("5_Process_Monitoring.py", tmp_path)
    app.radio(key="monitor_mode").set_value("Historical review").run()
    app.slider(key="monitor_review_time").set_value(24.0).run()
    app.button(key="ai_process_capture_button").click().run()
    app.button(key="ai_process_explain").click().run()
    assert not app.exception and len(fake_ai) == 1
    app.slider(key="monitor_review_time").set_value(0.0).run()
    assert not app.exception and "ai_process_capture" not in app.session_state
    assert not has_explanation_download(app, "process")
    app.button(key="ai_process_capture_button").click().run()
    app.button(key="ai_process_explain").click().run()
    count = next(fact["value"] for fact in fake_ai[-1].payload["facts"] if fact["label"] == "Available assay results at capture")
    assert count == 0


def test_acquired_stream_polling_does_not_call_ai_and_invalidations_clear_capture(tmp_path, fake_ai):
    app = page("5_Process_Monitoring.py", tmp_path)
    app.radio(key="monitor_mode").set_value("Acquired stream").run()
    app.selectbox(key="stream_source").select("Historical in-app test source").run()
    app.selectbox(key="stream_limit").select(100).run()
    app.button(key="stream_poll").click().run()
    assert not app.exception and fake_ai == []
    app.button(key="ai_stream_capture_button").click().run()
    app.button(key="ai_stream_explain").click().run()
    assert not app.exception and len(fake_ai) == 1
    app.run()
    assert len(fake_ai) == 1
    # Simulate the engine's revision signal without inventing another source
    # delivery. The stream engine's actual conflict/invalidation is tested
    # separately in test_stream_monitoring.py.
    app.session_state["stream_monitoring_session"].invalidations.append({"reason": "test revision"})
    app.run()
    assert not app.exception and "ai_stream_capture" not in app.session_state
    assert not has_explanation_download(app, "stream")


def test_source_or_rule_change_clears_replay_capture(tmp_path, fake_ai):
    app = page("5_Process_Monitoring.py", tmp_path)
    app.button(key="replay_step").click().run()
    app.button(key="ai_process_capture_button").click().run()
    app.button(key="ai_process_explain").click().run()
    app.selectbox(key="monitor_batch").select("MON-B002").run()
    assert not app.exception and "ai_process_capture" not in app.session_state
    assert len(fake_ai) == 1
