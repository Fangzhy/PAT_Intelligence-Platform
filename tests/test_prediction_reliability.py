"""Applicability geometry, range boundaries, evidence coverage and frozen artifacts."""

from copy import deepcopy
from dataclasses import replace
from io import BytesIO
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_generation import SimulationConfig, generate_dataset
from src.data_integration import integrate_sources, sources_from_dataset
from src.feature_engineering import SpectralFeatures
from src.llm_results import modeling_snapshot
from src.model_registry import ModelBundle
from src.modeling import ModelCase, TrainingConfig, train_study
from src.modeling_data import SENSOR_COLUMNS, fixed_batch_partition, modeling_dataset
from src.modeling_storage import evaluate_once, holdout_is_sealed, save_study, study_report_zip
from src.monitoring import MonitoringSettings, predict_observation
from src.prediction_reliability import (assess_prediction_reliability, fit_prediction_policy,
    range_performance, study_range_performance)


@pytest.fixture
def spectral_model():
    axis = np.linspace(400, 1800, 21)
    generator = np.random.default_rng(51)
    latent = generator.normal(size=(80, 1))
    X = pd.DataFrame(latent @ np.linspace(0.5, 1.5, 21)[None, :] + generator.normal(0, 0.03, (80, 21)),
        columns=[f"raman_{value:.12g}" for value in axis])
    y = 10 + latent[:, 0]
    pipeline = Pipeline([("features", SpectralFeatures(axis)), ("scale", StandardScaler()),
                         ("model", PLSRegression(n_components=1, scale=False))]).fit(X, y)
    fit_prediction_policy(pipeline, X, y, (5, 15), (9, 11))
    return pipeline, X, axis


def test_score_distance_and_residual_detect_different_spectral_failures(spectral_model):
    pipeline, X, _ = spectral_model
    scaler, model = pipeline.named_steps["scale"], pipeline.named_steps["model"]
    # Excursion along the fitted PLS loading direction: large score distance.
    shifted = scaler.mean_ + scaler.scale_ * (100 * model.x_loadings_[:, 0])
    score_row = pd.DataFrame([shifted], columns=X.columns)
    score = assess_prediction_reliability(pipeline, score_row, [10]).iloc[0]
    assert score.score_distance > score.score_limit
    assert score.applicability_status == "outside_training_limits"
    # Orthogonal to the score rotation: a shape failure with no score excursion.
    vector = np.zeros(X.shape[1]); vector[0] = 1
    rotation = model.x_rotations_[:, 0]
    vector -= rotation * (vector @ rotation) / (rotation @ rotation)
    residual_row = pd.DataFrame([scaler.mean_ + scaler.scale_ * vector * 10], columns=X.columns)
    residual = assess_prediction_reliability(pipeline, residual_row, [10]).iloc[0]
    assert residual.score_distance < residual.score_limit
    assert residual.spectral_residual_rms > residual.residual_limit
    assert not residual.reliability_eligible


def test_range_and_diagnostic_limits_are_inclusive(spectral_model):
    pipeline, X, _ = spectral_model
    row = X.iloc[[10]]
    diagnostics = assess_prediction_reliability(pipeline, row, [10]).iloc[0]
    policy = pipeline.prediction_policy_
    policy.score_limit = diagnostics.score_distance
    policy.residual_limit = diagnostics.spectral_residual_rms
    for value in (5, 15):
        result = assess_prediction_reliability(pipeline, row, [value]).iloc[0]
        assert result.reliability_eligible
        assert result.range_status == "within_declared_range"
    for value in (4.999, 15.001, np.nan, -1):
        result = assess_prediction_reliability(pipeline, row, [value]).iloc[0]
        assert result.range_status == "outside_declared_range"
        assert not result.reliability_eligible


def test_online_reliability_rejections_withhold_estimates_and_never_fit(spectral_model, monkeypatch):
    pipeline, X, axis = spectral_model
    bundle = ModelBundle(pipeline, {"target": "Glucose", "model_version": "test", "spectral_axis_cm1": axis,
                                   "input_columns": X.columns.tolist()})
    row = X.iloc[[10]]
    timestamp = pd.Timestamp("2026-01-01", tz="UTC")
    pipeline.prediction_policy_.declared_range = (0, 0.01)
    monkeypatch.setattr(Pipeline, "fit", lambda *args, **kwargs: pytest.fail("No fitting during inference"))
    result = predict_observation(bundle, "Glucose", row, axis, timestamp, timestamp, MonitoringSettings(domain_z_limit=100))
    assert result.status == "outside_reliability_limits"
    assert np.isnan(result.prediction_g_l)
    assert result.reliability["range_status"] == "outside_declared_range"
    assert any("saved declared" in flag for flag in result.flags)


def test_legacy_models_are_explicitly_unassessed(spectral_model):
    pipeline, X, _ = spectral_model
    del pipeline.prediction_policy_
    legacy = assess_prediction_reliability(pipeline, X.iloc[:2], [5, 6])
    assert legacy.applicability_status.eq("not_assessed").all()
    assert legacy.range_status.eq("not_assessed").all()
    assert not hasattr(pipeline, "prediction_policy_")


@pytest.mark.parametrize("recipe,model", [("raw", LinearRegression()), ("sensors", PLSRegression(n_components=1, scale=False))])
def test_unsupported_spectral_applicability_still_checks_concentration_range(spectral_model, recipe, model):
    _, X, axis = spectral_model
    X = X.copy()
    for column in SENSOR_COLUMNS:
        X[column] = np.random.default_rng(len(column)).normal(size=len(X))
    reference = 10 + X.iloc[:, 0].to_numpy()
    pipeline = Pipeline([("features", SpectralFeatures(axis, recipe)), ("scale", StandardScaler()), ("model", model)]).fit(X, reference)
    policy = fit_prediction_policy(pipeline, X, reference, (5, 15), (9, 11))
    rows = assess_prediction_reliability(pipeline, X.iloc[:2], [10, 20])
    assert policy.applicability_scope == "not_assessed"
    assert rows.applicability_status.eq("not_assessed").all()
    assert rows.range_status.tolist() == ["within_declared_range", "outside_declared_range"]
    assert rows.reliability_eligible.tolist() == [True, False]


def test_range_scorecard_retains_rejected_errors_empty_bands_and_coverage(spectral_model):
    pipeline, _, _ = spectral_model
    rows = pd.DataFrame({"reference_g_l": [9, 10, 11, 14], "prediction_g_l": [9, 14, 11, 14],
        "applicability_status": ["within_training_limits", "outside_training_limits", "within_training_limits", "within_training_limits"],
        "range_status": ["within_declared_range"] * 4, "reliability_eligible": [True, False, True, True]})
    report = range_performance(rows, pipeline.prediction_policy_).set_index("reference_band")
    decision = report.loc["Decision range"]
    assert decision.reference_rows == 3
    assert decision.eligible_rows == 2 and decision.rejected_rows == 1
    assert decision.eligible_fraction == pytest.approx(2 / 3)
    assert decision.all_RMSE == pytest.approx(np.sqrt(16 / 3))
    assert decision.all_Bias == pytest.approx(4 / 3)
    assert decision.eligible_RMSE == 0
    assert report.loc["Below decision range", "reference_rows"] == 0
    assert np.isnan(report.loc["Below decision range", "all_RMSE"])


@pytest.mark.parametrize("changes", [
    {"declared_range": (3, 2)}, {"declared_range": (-1, 5)}, {"decision_range": (1, np.inf)},
    {"declared_range": (1, 5), "decision_range": (0, 2)},
])
def test_invalid_policy_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        TrainingConfig(**changes)


@pytest.fixture(scope="module")
def cohort_and_study():
    data = generate_dataset(SimulationConfig(n_batches=8, duration_h=72, n_wavenumbers=71, seed=623))
    cohort = modeling_dataset(integrate_sources(sources_from_dataset(data)))
    study = train_study(cohort, fixed_batch_partition(cohort.metadata.batch_id),
        [ModelCase("raw", "pls")], TrainingConfig(max_pls_components=2))
    return cohort, study


def test_policy_and_oof_checks_never_use_holdout_labels(cohort_and_study):
    cohort, study = cohort_and_study
    changed = deepcopy(cohort)
    training, held_out = study.partition.masks(cohort.metadata)
    changed.y.loc[held_out] = 99999
    again = train_study(changed, study.partition, study.cases, study.config)
    original = study.results[study.winner_id]
    repeat = again.results[again.winner_id]
    assert study.study_id == again.study_id
    assert original.estimator.prediction_policy_.declared_range == (cohort.y.loc[training].min(), cohort.y.loc[training].max())
    assert original.estimator.prediction_policy_.summary() == repeat.estimator.prediction_policy_.summary()
    pd.testing.assert_frame_equal(original.oof_reliability, repeat.oof_reliability)
    assert original.oof_reliability.index.equals(study.training_reference.index)


def test_declared_range_cannot_claim_unsupported_training_extent(cohort_and_study):
    cohort, study = cohort_and_study
    with pytest.raises(ValueError, match="training-reference extent"):
        train_study(cohort, study.partition, study.cases, replace(study.config, declared_range=(0, 1e6)))


def test_frozen_evidence_serialization_reports_and_llm_facts(cohort_and_study, tmp_path, monkeypatch):
    cohort, study = cohort_and_study
    save_study(study, tmp_path)
    result = evaluate_once(study, cohort, tmp_path)
    assert holdout_is_sealed(tmp_path, cohort.cohort_id)
    assert "applicability_status" in result["predictions"]
    report = study_range_performance(study, result)
    assert len(report) == 10
    assert set(report.evidence) == {"Grouped CV (selection; fold-local checks)", "Independent batches (frozen)"}
    restored = joblib.load(tmp_path / cohort.cohort_id / "winner_pipeline.joblib")
    winner = study.results[study.winner_id].estimator
    pd.testing.assert_frame_equal(assess_prediction_reliability(winner, cohort.X, winner.predict(cohort.X)),
        assess_prediction_reliability(restored, cohort.X, restored.predict(cohort.X)))
    monkeypatch.setattr("src.modeling_storage.evaluate_holdout", lambda *args: pytest.fail("Frozen test must not run twice"))
    repeated = evaluate_once(study, cohort, tmp_path)
    pd.testing.assert_frame_equal(result["predictions"], repeated["predictions"])
    with pytest.raises(ValueError, match="locked"):
        save_study(study, tmp_path)
    with ZipFile(BytesIO(study_report_zip(study, result))) as archive:
        assert "prediction_reliability/range_performance.csv" in archive.namelist()
        assert "applicability_status" in archive.read("holdout_predictions.csv").decode()
    snapshot = modeling_snapshot(study, result).payload
    assert "prediction_reliability" in snapshot["limitations"]
    assert any("decision range all-row RMSE" in fact["label"] for fact in snapshot["facts"])


def test_custom_policy_changes_study_identity_without_test_selection(cohort_and_study):
    cohort, study = cohort_and_study
    low, high = study.results[study.winner_id].estimator.prediction_policy_.declared_range
    configured = train_study(cohort, study.partition, study.cases,
        replace(study.config, declared_range=(low, high), decision_range=(low + (high-low)/3, low + 2*(high-low)/3)))
    assert configured.study_id != study.study_id
    assert configured.winner_id == study.winner_id
    np.testing.assert_array_equal(configured.results[configured.winner_id].oof_prediction,
                                  study.results[study.winner_id].oof_prediction)
