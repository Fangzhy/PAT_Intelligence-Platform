"""Causal residuals, descriptive shift, frozen evidence, and guarded predictions."""

from dataclasses import replace
from io import BytesIO
import json
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from src.data_generation import SimulationConfig, generate_dataset
from src.model_registry import ModelBundle
from src.monitoring import MonitoringSettings
from src.reliability import (
    FAULT_SCENARIOS, HISTORY_COLUMNS, ReliabilityReview, ReliabilitySettings,
    available_reference_residuals, build_reliability_reference, distribution_shift,
    frozen_performance, input_distribution_shift, input_drift_features,
    reliability_fault_example, reliability_report_zip, residual_metrics,
)
from src.reliability_storage import prepare_reliability_demo
from src.replay import replay_batch


CONFIG = SimulationConfig(n_batches=6, duration_h=72, n_wavenumbers=71, seed=42)


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    return prepare_reliability_demo(CONFIG, "Glucose", tmp_path_factory.mktemp("reliability-evidence"))


@pytest.fixture(scope="module")
def batch():
    return replay_batch(generate_dataset(CONFIG), "B001")


def simple_pairs():
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    collection = pd.to_datetime([start, start + pd.Timedelta(hours=24)])
    history = pd.DataFrame({"sample_id": ["S0", "S1"], "batch_id": ["B1"] * 2,
        "measurement_timestamp": collection, "prediction_g_l": [2.0, 3.0], "prediction_status": ["ok"] * 2,
        "model_version": ["test-v1"] * 2})
    assays = pd.DataFrame({"sample_id": ["S0", "S1"], "batch_id": ["B1"] * 2,
        "collection_timestamp": collection, "result_timestamp": collection + pd.Timedelta(hours=6),
        "glucose_reference_g_l": [1.0, 3.0]})
    return history, assays, start


def test_known_residual_metrics_and_rolling_bias_use_prediction_minus_assay():
    history, assays, start = simple_pairs()
    rows = available_reference_residuals(history, assays, "Glucose", start + pd.Timedelta(hours=30), ReliabilitySettings(rolling_pairs=2))
    assert rows.residual_g_l.tolist() == [1, 0]
    scores = residual_metrics(rows).iloc[0]
    assert scores.RMSE == pytest.approx(np.sqrt(0.5))
    assert scores.MAE == 0.5 and scores.Bias == 0.5 and scores.R2 == 0.5
    assert scores.assessed_pairs == 2
    assert pd.isna(rows.bias_alarm.iloc[0])
    assert rows.rolling_bias_g_l.iloc[1] == 0.5 and rows.bias_alarm.iloc[1]


def test_result_time_cutoff_is_inclusive_and_never_reveals_future_labels():
    history, assays, start = simple_pairs()
    before = available_reference_residuals(history, assays, "Glucose", start + pd.Timedelta(hours=5.99), ReliabilitySettings())
    assert before.empty
    exact = available_reference_residuals(history, assays, "Glucose", start + pd.Timedelta(hours=6), ReliabilitySettings())
    assert exact.sample_id.tolist() == ["S0"]
    assert pd.isna(exact.bias_alarm.iloc[0])
    changed = assays.copy()
    changed.loc[1, "glucose_reference_g_l"] = 1e9
    other = available_reference_residuals(history, changed, "Glucose", start + pd.Timedelta(hours=6), ReliabilitySettings())
    pd.testing.assert_frame_equal(exact, other)


def test_residual_limit_equality_is_within_and_invalid_prediction_is_unassessed():
    history, assays, start = simple_pairs()
    history.loc[0, "prediction_g_l"] = 1.5
    history.loc[1, "prediction_status"] = "out_of_domain"
    history.loc[1, "prediction_g_l"] = np.nan
    rows = available_reference_residuals(history, assays, "Glucose", start + pd.Timedelta(days=3), ReliabilitySettings())
    assert not rows.residual_alarm.iloc[0]
    assert pd.isna(rows.residual_alarm.iloc[1])
    assert rows.pair_status.iloc[1] == "prediction_unavailable"
    scores = residual_metrics(rows).iloc[0]
    assert scores.assessed_pairs == 1 and scores.unassessed_pairs == 1
    assert np.isnan(scores.R2)


@pytest.mark.parametrize("fault", ["duplicate", "negative", "nan", "result_before_collection", "missing_collection"])
def test_invalid_assays_are_not_normal_pairs(fault):
    history, assays, start = simple_pairs()
    if fault == "duplicate":
        assays = pd.concat([assays.iloc[:1], assays.iloc[:1]], ignore_index=True)
    elif fault == "negative":
        assays.loc[0, "glucose_reference_g_l"] = -1
    elif fault == "nan":
        assays.loc[0, "glucose_reference_g_l"] = np.nan
    elif fault == "result_before_collection":
        assays.loc[0, "result_timestamp"] = start - pd.Timedelta(hours=1)
    else:
        assays.loc[0, "collection_timestamp"] = pd.NaT
    rows = available_reference_residuals(history, assays, "Glucose", start + pd.Timedelta(days=3), ReliabilitySettings())
    assert rows.pair_status.iloc[0] == "invalid_reference"
    assert pd.isna(rows.residual_alarm.iloc[0])


def test_pairing_uses_exact_batch_sample_and_collection_time():
    history, assays, start = simple_pairs()
    assays.loc[0, "batch_id"] = "different-batch"
    assays.loc[1, "collection_timestamp"] += pd.Timedelta(minutes=1)
    rows = available_reference_residuals(history, assays, "Glucose", start + pd.Timedelta(days=3), ReliabilitySettings())
    assert rows.pair_status.tolist() == ["prediction_unavailable", "sample_time_mismatch"]
    assert residual_metrics(rows).iloc[0].status == "not_assessed"
    assert rows.residual_g_l.isna().all()


def test_rolling_bias_orders_by_result_arrival_instead_of_future_collection_labels():
    history, assays, start = simple_pairs()
    history.loc[2] = ["S2", "B1", start + pd.Timedelta(hours=48), 8.0, "ok", "test-v1"]
    assays.loc[1, "result_timestamp"] = start + pd.Timedelta(hours=80)
    assays.loc[2] = ["S2", "B1", start + pd.Timedelta(hours=48), start + pd.Timedelta(hours=54), 5.0]
    rows = available_reference_residuals(history, assays, "Glucose", start + pd.Timedelta(hours=54), ReliabilitySettings(rolling_pairs=2))
    assert rows.sample_id.tolist() == ["S0", "S2"]
    assert rows.rolling_bias_g_l.iloc[-1] == 2.0  # Only arrived residuals 1 and 3.


@pytest.mark.parametrize("cutoff", ["2025-01-01", pd.NaT])
def test_invalid_or_naive_review_cutoff_is_rejected(cutoff):
    history, assays, _ = simple_pairs()
    with pytest.raises(ValueError, match="timezone-aware"):
        available_reference_residuals(history, assays, "Glucose", cutoff, ReliabilitySettings())


def test_known_wasserstein_mean_shift_and_strict_boundary():
    reference = pd.DataFrame({"x": [0.0, 2.0] * 3})
    settings = ReliabilitySettings()
    row = distribution_shift(reference, reference + 1, settings).iloc[0]
    assert row.wasserstein == 1 and row.reference_sd == 1 and row.mean_shift_sd == 1
    assert row.wasserstein_sd == 1 and not row.shift_flag
    assert distribution_shift(reference, reference + 1.1, settings).shift_flag.iloc[0]
    spread = distribution_shift(pd.DataFrame({"x": [-1.0, 1.0] * 3}), pd.DataFrame({"x": [-3.0, 3.0] * 3}), settings).iloc[0]
    assert spread.mean_shift_sd == 0 and spread.wasserstein_sd == 2 and spread.shift_flag


def test_constant_reference_shift_and_insufficient_counts_are_explicit():
    reference = pd.DataFrame({"x": [3.0] * 6})
    settings = ReliabilitySettings()
    same = distribution_shift(reference, reference, settings).iloc[0]
    assert same.status == "constant_reference_unchanged" and not same.shift_flag
    changed = distribution_shift(reference, reference + 0.1, settings).iloc[0]
    assert changed.status == "constant_reference_changed" and changed.shift_flag
    assert np.isnan(changed.wasserstein_sd)
    missing = reference.copy()
    missing.loc[0:1, "x"] = np.nan
    short = distribution_shift(reference, missing, settings).iloc[0]
    assert short.current_rows == 4 and short.invalid_current_rows == 2
    assert short.status == "insufficient_data" and pd.isna(short.shift_flag)


def test_input_features_exclude_labels_and_preserve_incomplete_spectral_rows(batch):
    inputs = batch.inputs.copy()
    inputs["glucose_g_l"] = 1e6
    inputs["scenario"] = "fake"
    features = input_drift_features(inputs)
    assert set(features) == {"temperature_c", "ph", "dissolved_oxygen_pct", "feed_rate_l_h", "agitation_rpm", "raman_mean_au", "raman_edge_mean_au"}
    inputs.iloc[0, 0] = np.nan
    altered = input_drift_features(inputs)
    assert np.isnan(altered.raman_mean_au.iloc[0]) and np.isnan(altered.raman_edge_mean_au.iloc[0])


def test_incompatible_spectral_axes_suppress_spectral_shift_assessment(batch):
    reference = input_drift_features(batch.inputs)
    rows = input_distribution_shift(reference, batch.inputs + 0.01, ReliabilitySettings(), compatible_axes=False)
    spectral = rows[rows.feature.str.startswith("raman_")]
    assert spectral.status.eq("axis_mismatch").all() and spectral.shift_flag.isna().all()
    assert rows[rows.feature.eq("ph")].status.iloc[0] == "assessed"


def test_training_reference_does_not_use_evaluation_rows_or_labels(evidence):
    cohort, study, _, _ = evidence
    original = build_reliability_reference(study, cohort)
    altered = replace(cohort, X=cohort.X.copy(), y=cohort.y.copy())
    held_out = study.partition.masks(cohort.metadata)[1]
    altered.X.loc[held_out] = 1e9
    altered.y.loc[held_out] = -1e9
    second = build_reliability_reference(study, altered)
    pd.testing.assert_frame_equal(original.input_features, second.input_features)
    np.testing.assert_array_equal(original.predictions_g_l, second.predictions_g_l)
    assert original.training_row_ids == tuple(study.training_reference.index)
    with pytest.raises(ValueError, match="original study"):
        build_reliability_reference(study, replace(cohort, cohort_id="different"))


def test_frozen_evidence_reads_selected_winner_and_preserves_metrics(evidence):
    _, study, evaluation, _ = evidence
    performance = frozen_performance(study, evaluation)
    assert len(performance) == 3
    assert performance.RMSE_g_l.iloc[0] == study.results[study.winner_id].training_metrics["RMSE"]
    assert performance.RMSE_g_l.iloc[-1] == evaluation["metrics"].set_index("case_id").loc[study.winner_id, "RMSEP"]
    assert len(frozen_performance(study)) == 2
    with pytest.raises(ValueError, match="CV-selected"):
        frozen_performance(study, {**evaluation, "study_id": "other"})


def test_review_never_refits_reuses_predictions_and_hides_future(evidence, batch, monkeypatch):
    _, study, _, bundle = evidence
    calls = []
    original = bundle.pipeline.predict
    def predict(values):
        calls.append(1)
        return original(values)
    def fail(*args, **kwargs):
        raise AssertionError("Review attempted training")
    monkeypatch.setattr(bundle.pipeline, "predict", predict)
    monkeypatch.setattr(bundle.pipeline, "fit", fail)
    session = ReliabilityReview("test")
    cutoff = batch.metadata.measurement_timestamp.iloc[4]
    first = session.history(batch, cutoff, bundle, MonitoringSettings(domain_z_limit=100))
    assert len(first) == 5 and len(calls) == 5
    repeated = session.history(batch, cutoff, bundle, MonitoringSettings(domain_z_limit=100))
    pd.testing.assert_frame_equal(first, repeated)
    backward = session.history(batch, batch.metadata.measurement_timestamp.iloc[1], bundle, MonitoringSettings(domain_z_limit=100))
    assert len(backward) == 2 and len(calls) == 5
    assert backward.model_version.eq(bundle.version).all()


@pytest.mark.parametrize("prediction, expected", [(np.nan, "invalid_prediction"), (-1.0, "invalid_prediction"), ([1, 2], "invalid_prediction")])
def test_invalid_saved_pipeline_predictions_remain_gaps(evidence, batch, prediction, expected):
    *_, bundle = evidence
    fake = ModelBundle(SimpleNamespace(named_steps=bundle.pipeline.named_steps, predict=lambda values: prediction), bundle.metadata)
    rows = ReliabilityReview("bad-prediction").history(batch, batch.metadata.measurement_timestamp.iloc[0], fake, MonitoringSettings(domain_z_limit=100))
    assert rows.prediction_status.iloc[0] == expected
    assert np.isnan(rows.prediction_g_l.iloc[0])


def test_quality_and_domain_flags_are_retained_and_original_sources_preserved(evidence, batch):
    *_, bundle = evidence
    fault = reliability_fault_example(batch, FAULT_SCENARIOS[-1], "Glucose")
    original = batch.inputs.copy(deep=True)
    history = ReliabilityReview("faults").history(fault, fault.metadata.measurement_timestamp.iloc[12], bundle, MonitoringSettings())
    assert history.prediction_status.iloc[3] == "invalid_input"
    assert history.prediction_status.iloc[7] == "invalid_input"
    assert history.prediction_status.iloc[10] == "invalid_input"
    assert history.prediction_status.iloc[12] == "out_of_domain"
    assert history.prediction_g_l.iloc[[3, 7, 10, 12]].isna().all()
    pd.testing.assert_frame_equal(batch.inputs, original)


def test_controlled_examples_change_only_their_declared_source(batch):
    original_inputs, original_assays = batch.inputs.copy(), batch.assays.copy()
    midpoint = batch.metadata.elapsed_time_h.max() / 2
    late = batch.metadata.elapsed_time_h.to_numpy() >= midpoint
    ph = reliability_fault_example(batch, FAULT_SCENARIOS[1], "Glucose")
    np.testing.assert_allclose(ph.inputs.loc[late, "ph"] - batch.inputs.loc[late, "ph"], 0.1)
    spectral = reliability_fault_example(batch, FAULT_SCENARIOS[2], "Glucose")
    np.testing.assert_allclose(spectral.inputs.loc[late].iloc[:, :len(batch.axis)] - batch.inputs.loc[late].iloc[:, :len(batch.axis)], 0.05)
    assay = reliability_fault_example(batch, FAULT_SCENARIOS[3], "Glucose")
    pd.testing.assert_frame_equal(assay.inputs, batch.inputs)
    late_assay = assay.assays.collection_elapsed_time_h >= midpoint
    np.testing.assert_allclose(assay.assays.loc[late_assay, "glucose_reference_g_l"] - batch.assays.loc[late_assay, "glucose_reference_g_l"], 1)
    pd.testing.assert_frame_equal(batch.inputs, original_inputs)
    pd.testing.assert_frame_equal(batch.assays, original_assays)


def test_empty_history_and_reference_window_has_unknown_metrics():
    history, assays, start = simple_pairs()
    rows = available_reference_residuals(history.iloc[:0], assays, "Glucose", start, ReliabilitySettings())
    assert rows.empty
    scores = residual_metrics(rows).iloc[0]
    assert scores.status == "not_assessed" and np.isnan(scores.RMSE)


def test_report_separates_frozen_validation_from_only_available_review(evidence, batch):
    cohort, study, evaluation, bundle = evidence
    rules = ReliabilitySettings()
    cutoff = batch.metadata.measurement_timestamp.iloc[8]
    history = ReliabilityReview("report").history(batch, cutoff, bundle, MonitoringSettings())
    residuals = available_reference_residuals(history, batch.assays, "Glucose", cutoff, rules)
    reference = build_reliability_reference(study, cohort)
    shift = distribution_shift(reference.input_features, input_drift_features(batch.inputs.iloc[:9]), rules)
    with ZipFile(BytesIO(reliability_report_zip(study, evaluation, history, residuals, shift, shift.iloc[:0], rules, {"cutoff_utc": cutoff.isoformat(), "model_version": bundle.version}))) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["study"]["cv_selected_winner"] == study.winner_id
        assert manifest["review_context"]["model_version"] == bundle.version
        rows = pd.read_csv(BytesIO(archive.read("review/observations.csv")))
        assert len(rows) == 9 and rows.sample_id.tolist() == history.sample_id.tolist()
        available = pd.read_csv(BytesIO(archive.read("review/available_assay_residuals.csv")))
        assert (pd.to_datetime(available.result_timestamp, utc=True) <= cutoff).all()
        assert len(available) == 1
        assert "frozen_validation/batch_metrics.csv" in archive.namelist()
        assert not {"glucose_g_l", "lactate_g_l"} & set(rows)


@pytest.mark.parametrize("kwargs", [{"residual_limit_g_l": 0}, {"bias_limit_g_l": np.nan}, {"drift_limit": -1}, {"min_drift_rows": 1}, {"rolling_pairs": True}])
def test_invalid_reliability_rules_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ReliabilitySettings(**kwargs)
