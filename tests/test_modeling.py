"""Batch isolation, pooled metrics, model fitting, sparse labels, and diagnostics."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.data_generation import SimulationConfig, generate_dataset
from src.data_integration import integrate_sources, sources_from_dataset
from src.feature_engineering import SpectralFeatures
from src.modeling import ModelCase, TrainingConfig, evaluate_holdout, pls_diagnostics, train_study
from src.modeling_data import TARGETS, BatchPartition, fixed_batch_partition, modeling_dataset
from src.validation import batch_folds, regression_metrics


@pytest.fixture(scope="module")
def modeling_source():
    return integrate_sources(sources_from_dataset(generate_dataset(SimulationConfig(n_batches=8, duration_h=72, n_wavenumbers=71, seed=713))))


@pytest.fixture(scope="module")
def cohort(modeling_source):
    return modeling_dataset(modeling_source)


@pytest.fixture(scope="module")
def study(cohort):
    cases = [ModelCase("raw", "pls"), ModelCase("summary", "linear"), ModelCase("sensors", "random_forest"), ModelCase("fft", "xgboost")]
    return train_study(cohort, fixed_batch_partition(cohort.metadata.batch_id), cases=cases, config=TrainingConfig(max_pls_components=3, tree_count=10))


def test_metrics_match_known_answer_and_expose_undefined_r2():
    metrics = regression_metrics([1, 2, 3], [2, 2, 2])
    assert metrics["RMSE"] == pytest.approx(np.sqrt(2 / 3))
    assert metrics["MAE"] == pytest.approx(2 / 3)
    assert metrics["Bias"] == 0
    assert metrics["R2"] == 0
    assert np.isnan(regression_metrics([2, 2], [2, 2])["R2"])
    for reference, prediction in (([], []), ([1], [1, 2]), ([1], [np.nan])):
        with pytest.raises(ValueError):
            regression_metrics(reference, prediction)


def test_cohort_uses_sparse_offline_references_and_whitelisted_inputs(modeling_source, cohort):
    assert len(cohort.X) == 8 * 4  # Not 8 * 73 simulated truth observations.
    assert cohort.X.shape[1] == 71 + 5
    assert not set(TARGETS.values()) & set(cohort.X.columns)
    assert not {"glucose_g_l", "batch_id", "elapsed_time_h", "result_timestamp", "scenario"} & set(cohort.X.columns)
    references = modeling_source.reference_pairs.set_index("assay_sample_id")["glucose_reference_g_l"]
    np.testing.assert_array_equal(cohort.y, references.loc[cohort.metadata.assay_sample_id])
    assert cohort.X.index.equals(cohort.y.index)
    lactate = modeling_dataset(modeling_source, "Lactate")
    assert cohort.cohort_id == lactate.cohort_id
    assert cohort.X.index.equals(lactate.X.index)


def test_common_cohort_excludes_missing_sensor_and_spectral_values_without_mutation(modeling_source):
    import copy
    source = copy.deepcopy(modeling_source)
    original = source.reference_pairs.copy(deep=True)
    source.reference_pairs.loc[0, "sensor_ph"] = np.nan
    spectral = next(column for column in source.reference_pairs if column.startswith("raman_") and column != "raman_source_row")
    source.reference_pairs.loc[1, spectral] = np.nan
    source.reference_pairs.loc[2, "sensor_time_delta_minutes"] = 1
    result = modeling_dataset(source)
    assert len(result.X) == len(original) - 3
    assert len(result.excluded) == 3
    assert "Future sensor reading" in result.excluded.iloc[2]["reason"]
    assert np.isnan(source.reference_pairs.loc[0, "sensor_ph"])


def test_glucose_labels_do_not_require_an_available_secondary_lactate_assay(modeling_source):
    import copy
    source = copy.deepcopy(modeling_source)
    source.reference_pairs["lactate_reference_g_l"] = np.nan
    glucose = modeling_dataset(source, "Glucose")
    assert len(glucose.X) == 32
    with pytest.raises(ValueError, match="No complete"):
        modeling_dataset(source, "Lactate")


def test_imported_physical_channel_headers_are_canonicalized(modeling_source):
    import copy
    source = copy.deepcopy(modeling_source)
    source.reference_pairs = source.reference_pairs.rename(columns={"raman_400": "raman_400.0"})
    cohort = modeling_dataset(source)
    assert "raman_400" in cohort.X
    assert "raman_400.0" not in cohort.X
    assert SpectralFeatures(cohort.axis, "raw").fit_transform(cohort.X).shape == (32, 71)


def test_batch_partition_and_cv_have_no_batch_overlap_or_missing_validation_rows(cohort):
    partition = fixed_batch_partition(cohort.metadata.batch_id.to_numpy()[::-1])
    assert partition.training_batches == ("B001", "B002", "B003", "B004", "B005", "B006")
    assert partition.holdout_batches == ("B007", "B008")
    training, test = partition.masks(cohort.metadata)
    assert not np.any(training & test)
    folds = batch_folds(cohort.metadata.loc[training, "batch_id"], 3)
    seen = np.zeros(training.sum(), dtype=int)
    for fold in folds:
        assert not set(fold.training_batches) & set(fold.validation_batches)
        assert not set(fold.training_batches + fold.validation_batches) & set(partition.holdout_batches)
        seen[fold.validation_positions] += 1
    np.testing.assert_array_equal(seen, np.ones(training.sum()))


def test_every_model_trains_and_predicts_finite_values(study, cohort):
    assert len(study.results) == 4
    assert study.failures.empty
    assert set(study.comparison.model) == {"pls", "linear", "random_forest", "xgboost"}
    training, test = study.partition.masks(cohort.metadata)
    for result in study.results.values():
        prediction = np.asarray(result.estimator.predict(cohort.X.loc[test])).reshape(-1)
        assert prediction.shape == (test.sum(),)
        assert np.isfinite(prediction).all()
        assert set(result.estimator.named_steps["features"].fit_row_ids_) == set(cohort.X.index[training])
        assert set(result.estimator.named_steps["features"].fit_row_ids_).isdisjoint(cohort.X.index[test])
        metrics = regression_metrics(cohort.y.loc[training], result.oof_prediction)
        assert result.cv_metrics["RMSE"] == pytest.approx(metrics["RMSE"])


def test_fold_feature_fits_never_see_validation_or_holdout_rows(cohort, monkeypatch):
    observed = []
    original = SpectralFeatures.fit

    def recording_fit(self, X, y=None):
        observed.append(set(X.index))
        return original(self, X, y)

    monkeypatch.setattr(SpectralFeatures, "fit", recording_fit)
    partition = fixed_batch_partition(cohort.metadata.batch_id)
    result = train_study(cohort, partition, cases=[ModelCase("preprocessed", "pls")], config=TrainingConfig(max_pls_components=2, tree_count=5))
    training, held_out = partition.masks(cohort.metadata)
    allowed = [set(cohort.X.index[training])]
    for row_ids in result.folds.training_row_ids:
        allowed.append(set(row_ids.split(";")))
    assert observed
    assert all(rows in allowed for rows in observed)
    assert all(rows.isdisjoint(cohort.X.index[held_out]) for rows in observed)


def test_holdout_label_changes_do_not_change_cv_or_selected_parameters(cohort):
    partition = fixed_batch_partition(cohort.metadata.batch_id)
    cases = [ModelCase("raw", "pls")]
    config = TrainingConfig(max_pls_components=2, tree_count=5)
    first = train_study(cohort, partition, cases, config)
    y = cohort.y.copy()
    y.loc[partition.masks(cohort.metadata)[1]] += 10000
    second = train_study(replace(cohort, y=y), partition, cases, config)
    pd.testing.assert_frame_equal(first.comparison, second.comparison)
    np.testing.assert_array_equal(first.results[first.winner_id].oof_prediction, second.results[second.winner_id].oof_prediction)


def test_noise_diagnostics_are_oof_and_sensor_only_model_ignores_spectral_perturbations(study):
    robustness = study.results["sensors__random_forest"].robustness
    np.testing.assert_allclose(robustness["RMSE"], robustness["RMSE"].iloc[0])
    assert len(robustness) == 4
    assert robustness["RMSE"].iloc[0] == pytest.approx(study.results["sensors__random_forest"].cv_metrics["RMSE"])


def test_feature_stability_importances_and_pls_diagnostics_are_consistent(study):
    for result in study.results.values():
        assert len(result.feature_stability) == len(result.estimator.named_steps["features"].get_feature_names_out())
        assert result.feature_stability.mean_normalized_importance.sum() == pytest.approx(1)
    result = study.results["raw__pls"]
    scores, loadings, coefficients = pls_diagnostics(result, study.training_metadata)
    count = result.best_params["n_components"]
    assert loadings.shape == (71, count)
    assert scores.filter(regex="^LV").shape == (len(study.training_reference), count)
    assert len(coefficients) == 71


def test_holdout_scores_use_same_batches_for_every_experiment_and_keep_cv_winner(study, cohort):
    result = evaluate_holdout(study, cohort)
    assert result["winner_id"] == study.winner_id
    for _, rows in result["predictions"].groupby("case_id"):
        assert set(rows.batch_id) == {"B007", "B008"}
        assert len(rows) == 8
    assert set(result["batch_metrics"].batch_id) == {"B007", "B008"}
    assert np.isfinite(result["metrics"]["RMSEP"]).all()
    with pytest.raises(ValueError, match="cohort"):
        evaluate_holdout(study, replace(cohort, cohort_id="f" * 64))


@pytest.mark.parametrize("changes", [{"n_splits": 1}, {"max_pls_components": 0}, {"tree_count": 0}, {"seed": -1}])
def test_invalid_training_config_is_rejected(changes):
    with pytest.raises(ValueError):
        TrainingConfig(**changes)


def test_invalid_or_overlapping_partitions_and_empty_experiment_sets_are_rejected(cohort):
    with pytest.raises(ValueError):
        BatchPartition(("B001", "B002", "B003"), ("B003",)).validate(["B001", "B002", "B003"])
    with pytest.raises(ValueError):
        fixed_batch_partition(["B001", "B002", "B003"])
    with pytest.raises(ValueError):
        train_study(cohort, fixed_batch_partition(cohort.metadata.batch_id), cases=[])
