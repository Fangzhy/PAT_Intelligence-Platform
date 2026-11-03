"""Modeling page actions, persistent result reuse, sealing, and phase navigation."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.data_generation import SimulationConfig
from src.modeling_storage import holdout_is_sealed


ROOT = Path(__file__).resolve().parents[1]
PAGE = "pages/4_Predictive_Modeling.py"
RAW_PLS = "PLS · Raw spectra"
SENSOR_RF = "Random Forest · Five process sensors"
FFT_XGB = "XGBoost · FFT amplitudes and band summaries"


def page(tmp_path, *, small=True):
    app = AppTest.from_file(str(ROOT / PAGE), default_timeout=120)
    app.session_state["modeling_state_root"] = str(tmp_path)
    if small:
        app.session_state["simulation_config"] = SimulationConfig(n_batches=6, duration_h=72, n_wavenumbers=71, seed=824)
    return app.run()


def train_small(app, experiments=None):
    app.multiselect(key="model_experiments").set_value(experiments or [RAW_PLS])
    app.slider(key="model_components").set_value(2)
    app.selectbox(key="model_trees").select(40)
    return app.button(key="train_models").click().run()


def test_page_shows_fixed_cohort_without_training_or_holdout_scores(tmp_path):
    app = page(tmp_path, small=False)
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["56", "42", "14", "6"]
    assert len(app.tabs) == 6
    assert not list(tmp_path.iterdir())
    assert not any("RMSEP" in frame.value for frame in app.dataframe)


def test_training_runs_only_on_button_and_page_reruns_reuse_saved_results(tmp_path, monkeypatch):
    import src.modeling as modeling
    calls = []
    original = modeling.train_study

    def recording_training(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(modeling, "train_study", recording_training)
    app = page(tmp_path)
    assert calls == []
    train_small(app)
    assert not app.exception
    assert calls == [1]
    app.selectbox(key="pls_loading").select("LV1").run()
    assert not app.exception
    assert calls == [1]
    assert not any("RMSEP" in frame.value for frame in app.dataframe)
    reloaded = page(tmp_path)
    assert not reloaded.exception
    assert calls == [1]
    assert any("RMSECV" in frame.value for frame in reloaded.dataframe)


def test_final_evaluation_seals_tuning_and_preserves_cv_selected_model(tmp_path):
    app = page(tmp_path)
    train_small(app, [RAW_PLS, SENSOR_RF, FFT_XGB])
    assert not app.exception
    assert not any("RMSEP" in frame.value for frame in app.dataframe)
    app.button(key="evaluate_holdout").click().run()
    assert not app.exception
    assert app.button(key="train_models").disabled
    comparison = next(frame.value for frame in app.dataframe if "RMSEP" in frame.value and "RMSECV" in frame.value)
    assert len(comparison) == 3
    assert set(comparison.model) == {"pls", "random_forest", "xgboost"}
    cohort_folder = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert holdout_is_sealed(tmp_path, cohort_folder.name)
    assert (cohort_folder / "winner_pipeline.joblib").exists()
    app.run()
    assert not app.exception
    assert app.button(key="train_models").disabled
    assert any("stored results" in info.value for info in app.info)


def test_lactate_can_be_trained_before_holdout_reveal(tmp_path):
    app = page(tmp_path)
    app.selectbox(key="model_target").select("Lactate").run()
    train_small(app)
    assert not app.exception
    assert any("Target: Lactate" in caption.value for caption in app.caption)
    assert any("RMSECV" in frame.value for frame in app.dataframe)


def test_empty_plan_and_missing_phase_two_inputs_have_actionable_messages(tmp_path):
    app = page(tmp_path)
    app.multiselect(key="model_experiments").set_value([])
    app.button(key="train_models").click().run()
    assert not app.exception
    assert any("at least one experiment" in error.value for error in app.error)
    app.radio(key="model_source").set_value("Latest Phase 2 prepared sources").run()
    assert not app.exception
    assert any("Run Data Integration first" in warning.value for warning in app.warning)


def test_phase_two_handoff_and_return_to_earlier_pages_preserve_working_features(tmp_path):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    app.session_state["modeling_state_root"] = str(tmp_path)
    app.run()
    app.switch_page("pages/2_Data_Integration.py").run()
    assert app.metric[0].value == "870"
    app.switch_page(PAGE).run()
    app.radio(key="model_source").set_value("Latest Phase 2 prepared sources").run()
    assert not app.exception
    assert [metric.value for metric in app.metric][:3] == ["42", "28", "14"]
    app.switch_page("pages/3_Spectral_Analytics.py").run()
    assert not app.exception
    assert app.metric[0].value == "870"
    app.switch_page("app.py").run()
    assert not app.exception
    assert app.metric[0].value == "6"
