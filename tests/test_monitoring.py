"""Known alarm boundaries, input validation, domain guards, and true inference."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler

from src.data_generation import SimulationConfig, generate_dataset
from src.model_registry import load_model, prepare_demo_models
from src.monitoring import MonitoringSettings, alarm_state, feature_domain_check, predict_observation
from src.replay import replay_batch, replay_fault_example


@pytest.fixture(scope="module")
def models_and_batch(tmp_path_factory):
    config = SimulationConfig(n_batches=3, duration_h=72, n_wavenumbers=71, seed=331)
    references = prepare_demo_models(config, tmp_path_factory.mktemp("monitor-models"))
    models = {reference.target: load_model(reference) for reference in references}
    return models, replay_batch(generate_dataset(config), "B001")


@pytest.mark.parametrize("value,expected", [(0.999, "Low"), (1, "Within limits"), (5, "Within limits"), (5.001, "High"), (np.nan, "Invalid/missing"), (np.inf, "Invalid/missing"), ("fault", "Invalid/missing")])
def test_alarm_boundaries_are_inclusive_and_missing_is_never_normal(value, expected):
    assert alarm_state(value, (1, 5)) == expected


def test_domain_threshold_is_inclusive_and_training_constant_changes_are_detected():
    scaler = StandardScaler().fit(np.array([[-1.0, 3.0], [1.0, 3.0]]))
    assert feature_domain_check([[6, 3]], scaler, 6)[0] is False
    assert feature_domain_check([[6.001, 3]], scaler, 6)[0] is True
    outside, maximum, constants = feature_domain_check([[0, 4]], scaler, 6)
    assert outside and constants == 1 and maximum < 6


def test_valid_inference_equals_saved_pipeline_prediction_and_does_not_fit(models_and_batch):
    models, batch = models_and_batch
    bundle = models["Glucose"]
    row = batch.inputs.iloc[[20]]
    metadata = batch.metadata.iloc[20]
    settings = MonitoringSettings(domain_z_limit=100)
    mean_before = bundle.pipeline.named_steps["scale"].mean_.copy()
    result = predict_observation(bundle, "Glucose", row, batch.axis, metadata.measurement_timestamp, metadata.sensor_timestamp, settings)
    assert result.status == "ok"
    assert result.prediction_g_l == pytest.approx(np.asarray(bundle.pipeline.predict(row)).reshape(-1)[0])
    assert result.model_version.startswith("phase5-demo-glucose")
    assert result.prediction_time_utc.tzinfo is not None
    np.testing.assert_array_equal(mean_before, bundle.pipeline.named_steps["scale"].mean_)
    # Ground-truth/assay-looking extra columns are explicitly excluded from inputs.
    with_labels = row.assign(glucose_g_l=99999, glucose_reference_g_l=123456)
    other = predict_observation(bundle, "Glucose", with_labels, batch.axis, metadata.measurement_timestamp, metadata.sensor_timestamp, settings)
    assert other.prediction_g_l == pytest.approx(result.prediction_g_l)


def test_absent_model_does_not_synthesize_a_concentration(models_and_batch):
    _, batch = models_and_batch
    metadata = batch.metadata.iloc[0]
    result = predict_observation(None, "Glucose", batch.inputs.iloc[[0]], batch.axis, metadata.measurement_timestamp, metadata.sensor_timestamp, MonitoringSettings())
    assert result.status == "model_unavailable"
    assert np.isnan(result.prediction_g_l)


@pytest.mark.parametrize("fault_position", [3, 7, 10])
def test_missing_and_implausible_inputs_suppress_prediction(models_and_batch, fault_position):
    models, batch = models_and_batch
    faulty = replay_fault_example(batch)
    metadata = faulty.metadata.iloc[fault_position]
    result = predict_observation(models["Glucose"], "Glucose", faulty.inputs.iloc[[fault_position]], faulty.axis, metadata.measurement_timestamp, metadata.sensor_timestamp, MonitoringSettings())
    assert result.status == "invalid_input"
    assert np.isnan(result.prediction_g_l)


def test_out_of_domain_and_axis_mismatch_suppress_inference(models_and_batch):
    models, batch = models_and_batch
    faulty = replay_fault_example(batch)
    metadata = faulty.metadata.iloc[12]
    result = predict_observation(models["Glucose"], "Glucose", faulty.inputs.iloc[[12]], faulty.axis, metadata.measurement_timestamp, metadata.sensor_timestamp, MonitoringSettings())
    assert result.status == "out_of_domain"
    assert np.isnan(result.prediction_g_l)
    mismatch = predict_observation(models["Glucose"], "Glucose", batch.inputs.iloc[[0]], batch.axis + 1, metadata.measurement_timestamp, metadata.sensor_timestamp, MonitoringSettings())
    assert mismatch.status == "axis_mismatch"


@pytest.mark.parametrize("delta,status", [(1, "invalid_input"), (-31, "invalid_input"), (-30, "ok")])
def test_future_and_stale_sensor_timestamps_are_checked_on_replay_time(models_and_batch, delta, status):
    models, batch = models_and_batch
    metadata = batch.metadata.iloc[20]
    result = predict_observation(models["Glucose"], "Glucose", batch.inputs.iloc[[20]], batch.axis,
        metadata.measurement_timestamp, metadata.measurement_timestamp + pd.Timedelta(minutes=delta), MonitoringSettings(domain_z_limit=100))
    assert result.status == status


@pytest.mark.parametrize("constant", [-1, np.nan])
def test_negative_or_nonfinite_predictions_are_not_clipped_or_replaced(models_and_batch, constant):
    models, batch = models_and_batch
    from copy import deepcopy
    bundle = deepcopy(models["Glucose"])
    dummy = DummyRegressor(strategy="constant", constant=0).fit(np.zeros((2, 71)), [0, 1])
    dummy.predict = lambda X: np.full(len(X), constant)
    bundle.pipeline.steps[-1] = ("model", dummy)
    metadata = batch.metadata.iloc[20]
    result = predict_observation(bundle, "Glucose", batch.inputs.iloc[[20]], batch.axis, metadata.measurement_timestamp, metadata.sensor_timestamp, MonitoringSettings(domain_z_limit=100))
    assert result.status == "invalid_prediction"
    assert np.isnan(result.prediction_g_l)


@pytest.mark.parametrize("changes", [{"domain_z_limit": 0}, {"max_sensor_age_minutes": -1}, {"sensor_limits": {}}, {"prediction_limits": {"Glucose": (2, 1), "Lactate": (0, 1)}}])
def test_invalid_monitoring_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        MonitoringSettings(**changes)


def test_monitoring_limit_normalization_preserves_the_callers_dictionary():
    from src.monitoring import DEFAULT_SENSOR_LIMITS
    limits = {key: list(value) for key, value in DEFAULT_SENSOR_LIMITS.items()}
    original = {key: value.copy() for key, value in limits.items()}
    settings = MonitoringSettings(sensor_limits=limits)
    assert limits == original
    assert isinstance(settings.sensor_limits["ph"], tuple)
