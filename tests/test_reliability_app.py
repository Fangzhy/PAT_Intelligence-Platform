"""Explicit demo preparation, causal review controls, frozen studies, navigation."""

import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.data_generation import SimulationConfig, generate_dataset
from src.data_integration import integrate_sources, sources_from_dataset
from src.modeling_storage import evaluate_once, holdout_is_sealed, save_study
from src.reliability import FAULT_SCENARIOS
from src.reliability_storage import prepare_reliability_demo, reliability_demo_config


ROOT = Path(__file__).resolve().parents[1]
PAGE = "pages/7_Model_Reliability.py"
CONFIG = SimulationConfig(n_batches=6, duration_h=72, n_wavenumbers=71)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("reliability-ui")
    return root, prepare_reliability_demo(CONFIG, "Glucose", root)


def page(tmp_path, root=None):
    app = AppTest.from_file(str(ROOT / PAGE), default_timeout=90)
    app.session_state["simulation_config"] = CONFIG
    app.session_state["reliability_state_root"] = str(root or tmp_path / "reliability")
    app.session_state["modeling_state_root"] = str(tmp_path / "phase4")
    return app.run()


def table(app, columns):
    return next(frame.value for frame in app.dataframe if set(columns).issubset(frame.value.columns)
                and not ("prediction_status" in columns and "pair_status" in frame.value.columns))


def test_missing_study_does_not_train_or_create_files_until_explicit_action(tmp_path, monkeypatch):
    import src.reliability_storage as storage
    calls = []
    original = storage.train_study
    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(storage, "train_study", record)
    app = page(tmp_path)
    assert not app.exception and calls == []
    assert not list(tmp_path.iterdir())
    app.button(key="prepare_reliability_demo").click().run()
    assert not app.exception and calls == [1]
    assert len(table(app, ["evidence", "RMSE_g_l"])) == 3
    assert [metric.value for metric in app.metric] == ["1", "0", "0", "0"]
    app.button(key="prepare_reliability_demo").click().run()
    assert not app.exception and calls == [1]
    assert not (tmp_path / "phase4").exists()


def test_backward_review_hides_future_reference_results_and_preserves_inference(tmp_path, prepared):
    app = page(tmp_path, prepared[0])
    assert not app.exception
    app.button(key="reliability_latest").click().run()
    assert not app.exception
    assert [metric.value for metric in app.metric[:3]] == ["73", "4", "4"]
    history = table(app, ["prediction_status", "prediction_flags"])
    inferred = history.prediction_time_utc.iloc[0]
    app.slider(key="reliability_clock").set_value(0.0).run()
    assert not app.exception
    assert [metric.value for metric in app.metric[:3]] == ["1", "0", "0"]
    assert table(app, ["pair_status", "residual_alarm"]).empty
    assert table(app, ["prediction_status", "prediction_flags"]).prediction_time_utc.iloc[0] == inferred


def test_rule_changes_never_retest_and_residual_bias_fault_is_distinguished(tmp_path, prepared, monkeypatch):
    import src.reliability_storage as storage
    def fail(*args, **kwargs):
        raise AssertionError("Review attempted to train or evaluate")
    monkeypatch.setattr(storage, "train_study", fail)
    monkeypatch.setattr(storage, "evaluate_once", fail)
    app = page(tmp_path, prepared[0])
    app.button(key="reliability_latest").click().run()
    frozen = table(app, ["evidence", "RMSE_g_l"]).copy()
    clean = table(app, ["pair_status", "residual_alarm"]).copy()
    app.number_input(key="reliability_residual_limit").set_value(0.1).run()
    assert not app.exception and app.metric[0].value == "73"
    app.selectbox(key="reliability_scenario").select(FAULT_SCENARIOS[3]).run()
    assert app.metric[0].value == "1"
    app.button(key="reliability_latest").click().run()
    assert not app.exception
    changed = table(app, ["pair_status", "residual_alarm"])
    late = changed.collection_elapsed_time_h >= 36
    np.testing.assert_allclose(changed.loc[late, "residual_g_l"] - clean.loc[late, "residual_g_l"], -1)
    assert table(app, ["evidence", "RMSE_g_l"]).equals(frozen)
    assert any("reference fault" in warning.value for warning in app.warning)


def test_fault_inputs_surface_missing_and_domain_gaps_without_fake_predictions(tmp_path, prepared):
    app = page(tmp_path, prepared[0])
    app.selectbox(key="reliability_scenario").select(FAULT_SCENARIOS[-1]).run()
    app.button(key="reliability_latest").click().run()
    assert not app.exception
    history = table(app, ["prediction_status", "prediction_flags"])
    assert history.prediction_status.iloc[3] == "invalid_input"
    assert history.prediction_status.iloc[12] == "out_of_domain"
    assert history.prediction_g_l.iloc[[3, 7, 10, 12]].isna().all()
    assert any("revealed predictions are unavailable" in warning.value for warning in app.warning)


def test_plot_units_and_result_arrival_axis_are_explicit(tmp_path, prepared):
    app = page(tmp_path, prepared[0])
    app.button(key="reliability_latest").click().run()
    assert not app.exception
    figures = [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]
    assert figures[0]["layout"]["yaxis"]["title"]["text"] == "Glucose (g/L)"
    assert figures[1]["layout"]["yaxis"]["title"]["text"] == "Prediction − assay residual (g/L)"
    assert figures[2]["layout"]["xaxis"]["title"]["text"] == "Laboratory result available time (UTC)"
    assert {shape["y0"] for shape in figures[1]["layout"]["shapes"]} == {-0.5, 0.5}


def test_unfinished_phase4_study_is_read_without_claiming_holdout(tmp_path, prepared):
    _, (cohort, study, _, _) = prepared
    root = tmp_path / "phase4"
    save_study(study, root)
    app = AppTest.from_file(str(ROOT / PAGE), default_timeout=90)
    app.session_state["simulation_config"] = CONFIG
    app.session_state["modeling_state_root"] = str(root)
    app.session_state["integration_result"] = integrate_sources(sources_from_dataset(generate_dataset(reliability_demo_config(CONFIG))))
    app.session_state["reliability_preferences"] = {"source": "Saved Phase 4 integration study"}
    app.run()
    assert not app.exception
    assert len(table(app, ["evidence", "RMSE_g_l"])) == 2
    assert any("no completed independent test" in warning.value for warning in app.warning)
    assert not holdout_is_sealed(root, cohort.cohort_id)


def test_saved_study_axis_mismatch_blocks_spectral_shift_and_predictions(tmp_path, prepared):
    _, (cohort, study, _, _) = prepared
    root = tmp_path / "phase4"
    save_study(study, root)
    evaluate_once(study, cohort, root)
    app = AppTest.from_file(str(ROOT / PAGE), default_timeout=90)
    app.session_state["simulation_config"] = SimulationConfig(n_batches=6, duration_h=72, n_wavenumbers=32)
    app.session_state["modeling_state_root"] = str(root)
    app.session_state["integration_result"] = integrate_sources(sources_from_dataset(generate_dataset(reliability_demo_config(CONFIG))))
    app.session_state["reliability_preferences"] = {"source": "Saved Phase 4 integration study"}
    app.run()
    app.button(key="reliability_latest").click().run()
    assert not app.exception
    history = table(app, ["prediction_status", "prediction_flags"])
    assert history.prediction_status.eq("axis_mismatch").all()
    shift = next(frame.value for frame in app.dataframe if {"feature", "wasserstein_sd"}.issubset(frame.value) and len(frame.value) == 7)
    assert shift[shift.feature.str.startswith("raman_")].status.eq("axis_mismatch").all()
    assert any("Spectral axes differ" in warning.value for warning in app.warning)


def test_navigation_pauses_replay_and_preserves_phase6_model_and_prior_pages(tmp_path):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    app.session_state["modeling_state_root"] = str(tmp_path / "phase4")
    app.session_state["monitoring_models_root"] = str(tmp_path / "monitoring")
    app.session_state["reliability_state_root"] = str(tmp_path / "reliability")
    app.run()
    app.switch_page("pages/6_Multivariate_Monitoring.py").run()
    app.button(key="mspc_fit").click().run()
    model_id = app.session_state["mspc_model"].model_id
    app.switch_page("pages/5_Process_Monitoring.py").run()
    app.button(key="replay_start").click().run()
    assert app.session_state["monitoring_session"].clock.running
    app.switch_page(PAGE).run()
    assert not app.exception
    assert not app.session_state["monitoring_session"].clock.running
    app.switch_page("pages/6_Multivariate_Monitoring.py").run()
    assert not app.exception and app.session_state["mspc_model"].model_id == model_id
    app.switch_page("pages/4_Predictive_Modeling.py").run()
    assert not app.exception and app.metric[1].value == "42"
    app.switch_page("pages/3_Spectral_Analytics.py").run()
    assert not app.exception and app.metric[0].value == "870"


def test_missing_integration_handoff_has_guidance(tmp_path):
    app = page(tmp_path)
    app.selectbox(key="reliability_source").select("Saved Phase 4 integration study").run()
    assert not app.exception
    assert any("Run Data Integration first" in warning.value for warning in app.warning)
