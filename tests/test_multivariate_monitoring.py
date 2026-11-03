"""Known-answer PCA statistics, reference isolation, faults, and disturbances."""

from dataclasses import replace
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data_generation import SimulationConfig, generate_dataset
from src.multivariate_monitoring import (
    SENSOR_COLUMNS, ReferencePartition, default_reference_partition,
    fit_process_monitor, monitoring_model_manifest, multivariate_report_zip,
    pca_monitoring_statistics, process_observations, score_process_monitor,
    spe_sensor_contributions, summarize_monitoring_batches,
    validate_reference_partition,
)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(SimulationConfig(n_batches=6, duration_h=72, n_wavenumbers=32))


@pytest.fixture(scope="module")
def model(dataset):
    return fit_process_monitor(dataset.sensors, dataset.batches)


def test_known_diagonal_covariance_has_exact_t2_q_and_contributions():
    reference = np.vstack([np.eye(5), -np.eye(5)]) * [3, 2, 1, 0.5, 0.25]
    scaler = StandardScaler(with_mean=False, with_std=False).fit(reference)
    pca = PCA(n_components=2, svd_solver="full").fit(reference)
    _, t2, q, contributions = pca_monitoring_statistics(scaler, pca, [[3, 2, 0.3, -0.4, 0]])
    # Sample eigenvalues are 18/9 = 2 and 8/9. T² = 9/2 + 4/(8/9) = 9.
    np.testing.assert_allclose(pca.explained_variance_, [2, 8 / 9])
    np.testing.assert_allclose(t2, [9])
    np.testing.assert_allclose(q, [0.25])
    np.testing.assert_allclose(contributions, [[0, 0, 0.09, 0.16, 0]], atol=1e-28)
    _, along_t2, along_q, _ = pca_monitoring_statistics(scaler, pca, [[3, 0, 0, 0, 0]])
    _, orthogonal_t2, orthogonal_q, _ = pca_monitoring_statistics(scaler, pca, [[0, 0, 3, 0, 0]])
    assert along_t2[0] == pytest.approx(4.5)
    assert along_q[0] == pytest.approx(0)
    assert orthogonal_t2[0] == pytest.approx(0)
    assert orthogonal_q[0] == pytest.approx(9)


def test_default_split_reserves_whole_nominal_batches_and_disturbed_evaluation(dataset):
    partition = default_reference_partition(dataset.batches)
    assert partition == ReferencePartition(("B001", "B002", "B003"), ("B005",), ("B004", "B006"))
    assert not set(partition.fit_batches) & set(partition.limit_batches + partition.evaluation_batches)


@pytest.mark.parametrize("partition, message", [
    (ReferencePartition((), ("B005",), ("B001", "B002", "B003", "B004", "B006")), "nonempty"),
    (ReferencePartition(("B001", "B002", "B003"), ("B003", "B005"), ("B004", "B006")), "disjoint"),
    (ReferencePartition(("B001", "B002", "B003"), ("B005",), ("B004", "B099")), "known batch"),
    (ReferencePartition(("B001", "B002", "B004"), ("B005",), ("B003", "B006")), "nominal"),
    (ReferencePartition(("B001", "B002", "B003", "B006"), ("B005",), ("B004",)), "nominal batch"),
])
def test_partition_rejects_leakage_and_missing_comparison(dataset, partition, message):
    with pytest.raises(ValueError, match=message):
        validate_reference_partition(partition, dataset.batches)


def test_too_few_nominal_batches_has_clear_guidance(dataset):
    with pytest.raises(ValueError, match="at least three nominal"):
        default_reference_partition(dataset.batches.iloc[:2])


def test_scaling_fits_only_fit_batches_and_limits_use_separate_calibration(dataset, model):
    fit = dataset.sensors[dataset.sensors.batch_id.isin(model.partition.fit_batches)][list(SENSOR_COLUMNS)]
    np.testing.assert_allclose(model.scaler.mean_, fit.mean())
    np.testing.assert_allclose(model.scaler.scale_, fit.std(ddof=0))
    calibration = dataset.sensors[dataset.sensors.batch_id.isin(model.partition.limit_batches)]
    projected = score_process_monitor(model, calibration)
    assert model.t2_limit == np.quantile(projected.t2, 0.99, method="higher")
    assert model.spe_limit == np.quantile(projected.spe, 0.99, method="higher")
    assert set(model.reference_audit.batch_id) == {"B001", "B002", "B003", "B005"}


def test_extreme_evaluation_values_and_labels_cannot_change_fitting_or_limits(dataset, model):
    sensors = dataset.sensors.copy(deep=True)
    evaluation = sensors.batch_id.isin(model.partition.evaluation_batches)
    sensors.loc[evaluation, list(SENSOR_COLUMNS)] = np.nan
    sensors["glucose_g_l"] = 1e12
    sensors["scenario"] = "fake label"
    batches = dataset.batches.copy(deep=True)
    batches.loc[batches.batch_id.eq("B004"), "disturbance_start_h"] = 0.0
    altered = fit_process_monitor(sensors, batches, partition=model.partition)
    np.testing.assert_array_equal(altered.scaler.mean_, model.scaler.mean_)
    np.testing.assert_array_equal(altered.pca.components_, model.pca.components_)
    assert altered.t2_limit == model.t2_limit
    assert altered.spe_limit == model.spe_limit
    assert altered.model_id == model.model_id


def test_projection_never_refits_and_is_invariant_to_other_projection_rows(dataset, model, monkeypatch):
    before = monitoring_model_manifest(model)
    def fail(*args, **kwargs):
        raise AssertionError("Projection attempted to refit")
    monkeypatch.setattr(model.scaler, "fit", fail)
    monkeypatch.setattr(model.pca, "fit", fail)
    original = dataset.sensors.copy(deep=True)
    together = score_process_monitor(model, dataset.sensors)
    one = score_process_monitor(model, dataset.sensors.iloc[[4]])
    np.testing.assert_allclose(one[["t2", "spe"]], together.iloc[[4]][["t2", "spe"]], atol=1e-12)
    assert monitoring_model_manifest(model) == before
    pd.testing.assert_frame_equal(dataset.sensors, original)


def test_invalid_observations_keep_identity_and_unknown_alarm_state(dataset, model):
    sensors = dataset.sensors.iloc[:6].copy(deep=True)
    sensors.index = [11, 13, 17, 20, 21, 50]
    sensors.iloc[1, sensors.columns.get_loc("ph")] = np.nan
    sensors.iloc[2, sensors.columns.get_loc("dissolved_oxygen_pct")] = 145
    sensors.iloc[3, sensors.columns.get_loc("agitation_rpm")] = np.inf
    sensors.iloc[4, sensors.columns.get_loc("measurement_timestamp")] = pd.NaT
    original = sensors.copy(deep=True)
    scored = score_process_monitor(model, sensors)
    assert scored.sample_id.tolist() == sensors.sample_id.tolist()
    assert scored.status.tolist() == ["scored", "invalid_input", "invalid_input", "invalid_input", "invalid_input", "scored"]
    assert scored.iloc[1:5][["t2", "spe", "anomaly_ratio", "alarm"]].isna().all().all()
    assert "Implausible dissolved_oxygen_pct" in scored.quality_flags.iloc[2]
    pd.testing.assert_frame_equal(sensors, original)


def test_duplicate_ids_are_all_unscorable_and_missing_schema_is_rejected(dataset, model):
    duplicate = pd.concat([dataset.sensors.iloc[:1]] * 2, ignore_index=True)
    scored = score_process_monitor(model, duplicate)
    assert scored.status.eq("invalid_input").all()
    assert scored.alarm.isna().all()
    with pytest.raises(ValueError, match="sensor variables"):
        score_process_monitor(model, dataset.sensors.drop(columns="ph"))


def test_alarm_equality_is_within_and_ratio_above_one_means_exceedance(dataset, model):
    row = dataset.sensors.iloc[[5]]
    original = score_process_monitor(model, row).iloc[0]
    exact = replace(model, t2_limit=original.t2, spe_limit=original.spe)
    scored = score_process_monitor(exact, row).iloc[0]
    assert not scored.alarm
    assert scored.anomaly_ratio == pytest.approx(1)
    crossed = score_process_monitor(replace(exact, t2_limit=original.t2 * 0.99), row).iloc[0]
    assert crossed.t2_alarm and crossed.alarm
    assert not crossed.spe_alarm
    assert crossed.anomaly_ratio > 1


def test_spe_contributions_sum_to_q_and_invalid_inspection_is_rejected(dataset, model):
    row = dataset.sensors[dataset.sensors.batch_id.eq("B004")].iloc[[-1]]
    contribution = spe_sensor_contributions(model, row)
    assert contribution.spe_contribution.ge(0).all()
    assert contribution.sensor.tolist() == list(SENSOR_COLUMNS)
    assert contribution.spe_contribution.sum() == pytest.approx(score_process_monitor(model, row).spe.iloc[0])
    row = row.copy()
    row.loc[:, "ph"] = np.nan
    with pytest.raises(ValueError, match="valid observation"):
        spe_sensor_contributions(model, row)


def test_designed_disturbance_exceeds_limits_and_summary_keeps_roles(dataset, model):
    scored = score_process_monitor(model, dataset.sensors)
    summary = summarize_monitoring_batches(scored, dataset.batches).set_index("batch_id")
    assert summary.loc["B004", "role"] == "Evaluation"
    assert summary.loc["B006", "role"] == "Evaluation"
    assert summary.loc["B004", "post_disturbance_flagged_fraction"] > 0.75
    assert summary.loc["B004", "flagged_fraction"] > summary.loc["B006", "flagged_fraction"]
    assert summary.loc["B004", "review_state"] == "Flagged"
    assert summary.loc["B004", "disturbance_start_h"] == pytest.approx(46.8)
    assert scored.sample_id.tolist() == dataset.sensors.sample_id.tolist()


def test_unscorable_batch_is_not_reported_as_normal(dataset, model):
    sensors = dataset.sensors.copy()
    sensors.loc[sensors.batch_id.eq("B006"), "ph"] = np.nan
    summary = summarize_monitoring_batches(score_process_monitor(model, sensors), dataset.batches).set_index("batch_id")
    assert summary.loc["B006", "review_state"] == "Unscorable"
    assert summary.loc["B006", "scored_rows"] == 0
    assert summary.loc["B006", "invalid_rows"] == 73
    assert np.isnan(summary.loc["B006", "flagged_fraction"])


def test_reference_quality_exclusions_are_audited_and_empty_reference_rejected(dataset):
    sensors = dataset.sensors.copy()
    sensors.loc[0, "ph"] = np.nan
    model = fit_process_monitor(sensors, dataset.batches)
    assert model.reference_audit.iloc[0].status == "invalid_input"
    assert len(model.reference_audit) == 4 * 73
    sensors.loc[sensors.batch_id.eq("B005"), "ph"] = np.nan
    with pytest.raises(ValueError, match="Every fitting"):
        fit_process_monitor(sensors, dataset.batches)


@pytest.mark.parametrize("components", [True, 0, 5, 1.5])
def test_invalid_component_count_is_rejected(dataset, components):
    with pytest.raises(ValueError, match="components"):
        fit_process_monitor(dataset.sensors, dataset.batches, n_components=components)


@pytest.mark.parametrize("quantile", [True, 0.2, 1.0, np.nan])
def test_invalid_quantile_is_rejected(dataset, quantile):
    with pytest.raises(ValueError, match="quantile"):
        fit_process_monitor(dataset.sensors, dataset.batches, quantile=quantile)


def test_constant_reference_sensor_and_singular_retained_components_are_rejected(dataset):
    sensors = dataset.sensors.copy()
    sensors["temperature_c"] = 37.0
    with pytest.raises(ValueError, match="Constant reference"):
        fit_process_monitor(sensors, dataset.batches)
    # All sensors follow a single exact linear pattern within plausibility bounds.
    t = np.linspace(0, 1, len(sensors))
    for column, offset, slope in zip(SENSOR_COLUMNS, [37, 7, 50, 0.001, 200], [0.1, 0.01, 2, 0.001, 5]):
        sensors[column] = offset + slope * t
    with pytest.raises(ValueError, match="singular"):
        fit_process_monitor(sensors, dataset.batches, n_components=2)
    with pytest.raises(ValueError, match="degenerate"):
        fit_process_monitor(sensors, dataset.batches, n_components=1)


def test_empty_projection_and_numeric_time_strings_are_handled(dataset, model):
    empty = score_process_monitor(model, dataset.sensors.iloc[:0])
    assert empty.empty and "alarm" in empty
    sensors = dataset.sensors.iloc[:2].copy()
    sensors["elapsed_time_h"] = sensors.elapsed_time_h.astype(str)
    screened = process_observations(sensors)
    assert screened.elapsed_time_h.tolist() == [0, 1]


def test_report_contains_reference_provenance_scores_and_reproducible_parameters(dataset, model):
    scored = score_process_monitor(model, dataset.sensors)
    with ZipFile(BytesIO(multivariate_report_zip(model, scored, dataset.batches, context={"seed": 42}))) as archive:
        assert set(archive.namelist()) == {"manifest.json", "observations.csv", "batch_summary.csv", "reference_audit.csv", "loadings.csv", "README.txt"}
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["partition"]["limit_batches"] == ["B005"]
        assert manifest["features"] == list(SENSOR_COLUMNS)
        assert manifest["t2_limit"] == model.t2_limit
        assert manifest["context"] == {"seed": 42}
        assert "not a probability" in manifest["anomaly_ratio"]
        exported = pd.read_csv(BytesIO(archive.read("observations.csv")))
        assert exported.sample_id.tolist() == dataset.sensors.sample_id.tolist()
        assert not {"glucose_g_l", "lactate_g_l", "disturbance_fraction"} & set(exported)
        np.testing.assert_allclose(exported.spe, scored.spe)
        # Reconstruct scores/statistics from the exported numerical parameters.
        values = exported[list(SENSOR_COLUMNS)].to_numpy()
        scaled = (values - manifest["scaler_mean"]) / manifest["scaler_scale"]
        scores = (scaled - manifest["pca_mean"]) @ np.array(manifest["components"]).T
        np.testing.assert_allclose((scores**2 / manifest["eigenvalues_ddof1"]).sum(axis=1), exported.t2)


def test_repeated_calibration_is_reproducible(dataset, model):
    repeated = fit_process_monitor(dataset.sensors, dataset.batches)
    assert monitoring_model_manifest(repeated) == monitoring_model_manifest(model)
