"""Explicit reference calibration, chart controls, and cross-phase regressions."""

from pathlib import Path
import json

from streamlit.testing.v1 import AppTest

from src.data_generation import SimulationConfig


ROOT = Path(__file__).resolve().parents[1]
PAGE = "pages/6_Multivariate_Monitoring.py"
CONFIG = SimulationConfig(n_batches=6, duration_h=72, n_wavenumbers=32)


def page():
    app = AppTest.from_file(str(ROOT / PAGE), default_timeout=60)
    app.session_state["simulation_config"] = CONFIG
    return app.run()


def table(app, key):
    columns = {
        "mspc_batch_summary": {"flagged_fraction", "scenario"},
        "mspc_loadings": {"sensor", "PC1"},
        "mspc_observations": {"quality_flags", "t2"},
    }[key]
    return next(frame.value for frame in app.dataframe if columns.issubset(frame.value.columns))


def test_startup_does_not_fit_until_requested_and_default_calibration_is_separate(monkeypatch):
    import src.multivariate_monitoring as monitoring
    calls = []
    original = monitoring.fit_process_monitor
    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(monitoring, "fit_process_monitor", record)
    app = page()
    assert not app.exception
    assert calls == []
    assert not app.metric
    app.button(key="mspc_fit").click().run()
    assert not app.exception
    assert calls == [1]
    assert [metric.value for metric in app.metric[:3]] == ["219", "73", "146"]
    summary = table(app, "mspc_batch_summary").set_index("batch_id")
    assert summary.loc["B004", "role"] == "Evaluation"
    assert summary.loc["B006", "role"] == "Evaluation"
    model_id = app.session_state["mspc_model"].model_id
    app.selectbox(key="mspc_sensor").select("ph").run()
    app.selectbox(key="mspc_inspect_batch").select("B004").run()
    app.selectbox(key="mspc_inspect_sample").select("B004-S00072").run()
    assert not app.exception
    assert calls == [1]
    assert app.session_state["mspc_model"].model_id == model_id
    assert any("Contributions sum to Q" in caption.value for caption in app.caption)


def test_invalid_reference_selection_never_reuses_a_stale_model():
    app = page()
    app.button(key="mspc_fit").click().run()
    assert not app.exception
    app.multiselect(key="mspc_limit_batches").set_value(["B001"])
    app.button(key="mspc_fit").click().run()
    assert not app.exception
    assert any("disjoint" in error.value for error in app.error)
    assert not app.metric
    assert "mspc_model" not in app.session_state


def test_one_component_and_empty_chart_selection_work():
    app = page()
    app.slider(key="mspc_components").set_value(1)
    app.selectbox(key="mspc_quantile").select(0.95)
    app.button(key="mspc_fit").click().run()
    assert not app.exception
    assert app.session_state["mspc_model"].pca.n_components_ == 1
    assert "PC2" not in table(app, "mspc_loadings")
    app.multiselect(key="mspc_batches").set_value([]).run()
    assert not app.exception
    assert table(app, "mspc_observations").empty
    assert any("Select at least one batch" in info.value for info in app.info)


def test_insufficient_batches_and_configuration_change_require_new_reference_fit():
    app = page()
    app.button(key="mspc_fit").click().run()
    app.session_state["simulation_config"] = SimulationConfig(n_batches=2, duration_h=72, n_wavenumbers=32)
    app.run()
    assert not app.exception
    assert any("three nominal" in warning.value for warning in app.warning)
    assert "mspc_model" not in app.session_state
    app.session_state["simulation_config"] = CONFIG
    app.run()
    assert not app.exception and not app.metric
    app.button(key="mspc_fit").click().run()
    assert not app.exception


def test_navigation_pauses_replay_and_preserves_calibration_and_earlier_phases(tmp_path):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    app.session_state["modeling_state_root"] = str(tmp_path / "phase4")
    app.session_state["monitoring_models_root"] = str(tmp_path / "models")
    app.run()
    app.switch_page("pages/5_Process_Monitoring.py").run()
    app.button(key="replay_start").click().run()
    assert app.session_state["monitoring_session"].clock.running
    app.switch_page(PAGE).run()
    assert not app.exception
    assert not app.session_state["monitoring_session"].clock.running
    app.button(key="mspc_fit").click().run()
    assert not app.exception
    model_id = app.session_state["mspc_model"].model_id
    app.switch_page("pages/3_Spectral_Analytics.py").run()
    assert not app.exception and app.metric[0].value == "870"
    app.switch_page("pages/4_Predictive_Modeling.py").run()
    assert not app.exception and app.metric[1].value == "42"
    app.switch_page(PAGE).run()
    assert not app.exception
    assert app.session_state["mspc_model"].model_id == model_id
    app.switch_page("pages/2_Data_Integration.py").run()
    assert not app.exception


def test_control_charts_use_correct_units_limits_and_disturbance_annotations():
    app = page()
    app.button(key="mspc_fit").click().run()
    assert not app.exception
    figures = [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]
    assert figures[0]["layout"]["yaxis"]["title"]["text"] == "Hotelling’s T² (dimensionless)"
    assert figures[1]["layout"]["yaxis"]["title"]["text"] == "SPE/Q (standardized squared units)"
    assert figures[0]["layout"]["xaxis"]["title"]["text"] == "Elapsed process time (h)"
    assert any(shape.get("y0") == app.session_state["mspc_model"].t2_limit for shape in figures[0]["layout"]["shapes"])
    assert any("designed disturbance" in item.get("text", "") for item in figures[0]["layout"]["annotations"])


def test_custom_reference_settings_survive_navigation():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app.switch_page(PAGE).run()
    app.multiselect(key="mspc_fit_batches").set_value(["B001", "B002"])
    app.multiselect(key="mspc_limit_batches").set_value(["B003", "B005"])
    app.slider(key="mspc_components").set_value(3)
    app.selectbox(key="mspc_quantile").select(0.975)
    app.button(key="mspc_fit").click().run()
    assert not app.exception
    model_id = app.session_state["mspc_model"].model_id
    app.switch_page("pages/1_Overview.py").run()
    app.switch_page(PAGE).run()
    assert not app.exception
    assert app.session_state["mspc_model"].model_id == model_id
    assert app.multiselect(key="mspc_fit_batches").value == ["B001", "B002"]
    assert app.multiselect(key="mspc_limit_batches").value == ["B003", "B005"]
    assert app.slider(key="mspc_components").value == 3
    assert app.selectbox(key="mspc_quantile").value == 0.975
