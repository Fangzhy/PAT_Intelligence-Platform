"""Durable one-time evaluation, frozen partitions, reports, and trusted inference."""

from dataclasses import replace
from io import BytesIO
import json
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data_generation import SimulationConfig, generate_dataset
from src.data_integration import integrate_sources, sources_from_dataset
from src.modeling import ModelCase, TrainingConfig, train_study
from src.modeling_data import BatchPartition, fixed_batch_partition, modeling_dataset
from src.modeling_storage import (
    evaluate_once, holdout_is_sealed, load_holdout, load_study, save_study, study_report_zip,
)


@pytest.fixture(scope="module")
def storage_study():
    cohort = modeling_dataset(integrate_sources(sources_from_dataset(generate_dataset(SimulationConfig(n_batches=6, duration_h=72, n_wavenumbers=71, seed=812)))))
    study = train_study(cohort, fixed_batch_partition(cohort.metadata.batch_id), [ModelCase("raw", "pls")], TrainingConfig(max_pls_components=2, tree_count=5))
    return study, cohort


def test_evaluation_is_executed_once_and_artifact_reproduces_selected_predictions(storage_study, tmp_path, monkeypatch):
    import src.modeling_storage as storage
    study, cohort = storage_study
    calls = []
    original = storage.evaluate_holdout

    def recording_evaluation(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(storage, "evaluate_holdout", recording_evaluation)
    folder = save_study(study, tmp_path)
    first = evaluate_once(study, cohort, tmp_path)
    second = evaluate_once(load_study(tmp_path, cohort.cohort_id), cohort, tmp_path)
    assert calls == [1]
    pd.testing.assert_frame_equal(first["metrics"], second["metrics"])
    assert holdout_is_sealed(tmp_path, cohort.cohort_id)
    pipeline = joblib.load(folder / "winner_pipeline.joblib")  # App-generated fixture only.
    held_out = study.partition.masks(cohort.metadata)[1]
    prediction = np.asarray(pipeline.predict(cohort.X.loc[held_out])).reshape(-1)
    np.testing.assert_allclose(prediction, first["predictions"].prediction_g_l)
    metadata = json.loads((folder / "winner_metadata.json").read_text())
    assert metadata["selected_case"] == study.winner_id
    assert metadata["partition"]["holdout_batches"] == list(study.partition.holdout_batches)
    with pytest.raises(ValueError, match="locked"):
        save_study(study, tmp_path)


def test_existing_cohort_partition_cannot_change(storage_study, tmp_path):
    study, _ = storage_study
    save_study(study, tmp_path)
    other = replace(study, partition=BatchPartition(("B001", "B002", "B003", "B005"), ("B004", "B006")))
    with pytest.raises(ValueError, match="different fixed"):
        save_study(other, tmp_path)


def test_incomplete_evaluation_keeps_claim_and_does_not_reopen_tuning(storage_study, tmp_path, monkeypatch):
    import src.modeling_storage as storage
    study, cohort = storage_study
    save_study(study, tmp_path)

    def failure(*args):
        raise ValueError("Synthetic evaluation failure")

    monkeypatch.setattr(storage, "evaluate_holdout", failure)
    with pytest.raises(ValueError, match="Synthetic"):
        evaluate_once(study, cohort, tmp_path)
    assert holdout_is_sealed(tmp_path, cohort.cohort_id)
    assert load_holdout(tmp_path, cohort.cohort_id) is None
    with pytest.raises(ValueError, match="already claimed"):
        evaluate_once(study, cohort, tmp_path)
    with pytest.raises(ValueError, match="locked"):
        save_study(study, tmp_path)


def test_cv_report_does_not_export_holdout_labels_and_final_report_adds_them(storage_study, tmp_path):
    study, cohort = storage_study
    with ZipFile(BytesIO(study_report_zip(study))) as archive:
        assert not any(name.startswith("holdout_") for name in archive.namelist())
        prediction = pd.read_csv(BytesIO(archive.read("cv_predictions.csv")))
        assert set(prediction.batch_id) == set(study.partition.training_batches)
        assert {"cv_folds.csv", "study_manifest.json", "cv_selected_pipeline.joblib"}.issubset(archive.namelist())
    save_study(study, tmp_path)
    result = evaluate_once(study, cohort, tmp_path)
    with ZipFile(BytesIO(study_report_zip(study, result))) as archive:
        assert "holdout_metrics.csv" in archive.namelist()
        assert "holdout_predictions.csv" in archive.namelist()


def test_no_unsaved_or_mismatched_study_can_reveal_holdout(storage_study, tmp_path):
    study, cohort = storage_study
    with pytest.raises(ValueError, match="Save this exact"):
        evaluate_once(study, cohort, tmp_path)
    assert not holdout_is_sealed(tmp_path, cohort.cohort_id)
    save_study(study, tmp_path)
    with pytest.raises(ValueError, match="match"):
        evaluate_once(study, replace(cohort, target="Lactate"), tmp_path)
