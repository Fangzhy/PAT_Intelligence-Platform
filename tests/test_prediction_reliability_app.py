"""A single saved policy travels from modeling to monitoring and reliability."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.data_generation import SimulationConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIG = SimulationConfig(n_batches=6, duration_h=72, n_wavenumbers=71, seed=922)


def page(path, tmp_path):
    app = AppTest.from_file(str(ROOT / path), default_timeout=90)
    app.session_state["simulation_config"] = CONFIG
    app.session_state["modeling_state_root"] = str(tmp_path / "models")
    app.session_state["monitoring_models_root"] = str(tmp_path / "demos")
    app.session_state["reliability_state_root"] = str(tmp_path / "reviews")
    return app.run()


def range_table(app):
    return next(item.value for item in app.dataframe if "reference_band" in item.value)


def test_custom_policy_is_saved_frozen_and_shared_between_pages(tmp_path, monkeypatch):
    app = page("pages/4_Predictive_Modeling.py", tmp_path)
    app.multiselect(key="model_experiments").set_value(["PLS · Raw spectra"])
    app.slider(key="model_components").set_value(2)
    app.checkbox(key="model_custom_range").check()
    low = app.number_input(key="model_range_low_Glucose").value
    high = app.number_input(key="model_range_high_Glucose").value
    app.number_input(key="model_decision_low_Glucose").set_value(low + (high - low) / 3)
    app.number_input(key="model_decision_high_Glucose").set_value(low + (high - low) * 2 / 3)
    app.button(key="train_models").click().run()
    assert not app.exception
    assert len(range_table(app)) == 5
    assert set(range_table(app).evidence) == {"Grouped CV (selection; fold-local checks)"}
    app.button(key="evaluate_holdout").click().run()
    assert not app.exception
    expected = range_table(app).copy()
    assert len(expected) == 10
    assert app.checkbox(key="model_custom_range").disabled
    assert app.number_input(key="model_decision_low_Glucose").disabled

    def fail(*args, **kwargs):
        raise AssertionError("Page read attempted fitting or test evaluation")
    monkeypatch.setattr("src.modeling.train_study", fail)
    monkeypatch.setattr("src.modeling_storage.evaluate_holdout", fail)
    review = page("pages/7_Model_Reliability.py", tmp_path)
    review.selectbox(key="reliability_source").select("Saved Phase 4 synthetic study").run()
    assert not review.exception
    pd.testing.assert_frame_equal(range_table(review), expected)
    assert any("applicability" in frame.value for frame in review.dataframe)
    monitor = page("pages/5_Process_Monitoring.py", tmp_path)
    monitor.button(key="replay_step").click().run()
    assert not monitor.exception
    assert any("applicability" in frame.value for frame in monitor.dataframe)
    history = next(frame.value for frame in monitor.dataframe if "glucose_status" in frame.value)
    assert "glucose_applicability_status" in history
    assert history.glucose_range_status.iloc[0] != "not_assessed"


def test_invalid_declaration_does_not_create_a_study_or_reveal_test(tmp_path):
    app = page("pages/4_Predictive_Modeling.py", tmp_path)
    app.multiselect(key="model_experiments").set_value(["PLS · Raw spectra"])
    app.checkbox(key="model_custom_range").check()
    app.number_input(key="model_range_high_Glucose").set_value(100000.0)
    app.button(key="train_models").click().run()
    assert not app.exception
    assert any("training-reference extent" in error.value for error in app.error)
    assert not (tmp_path / "models").exists()
