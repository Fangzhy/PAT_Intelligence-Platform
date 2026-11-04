"""Frozen inference, release integrity, and serialization parity challenges."""

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from src.data_generation import SimulationConfig
from src.deployment import create_release, evaluation_report, example_inputs, inference_schema, predict_frame, release_zip, verify_release
from src.model_registry import load_model, prepare_demo_models
from src.monitoring import MonitoringSettings


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    root = tmp_path_factory.mktemp("release-models")
    reference = prepare_demo_models(SimulationConfig(n_batches=3, duration_h=72, n_wavenumbers=71), root)[0]
    return reference, load_model(reference)


def test_frozen_inference_matches_pipeline_and_does_not_fit(model, monkeypatch):
    _, bundle = model
    inputs = example_inputs(bundle)
    assert inputs.columns.tolist() == inference_schema(bundle)["columns"]
    before = inputs.copy(deep=True)
    expected = bundle.pipeline.predict(inputs[bundle.metadata["input_columns"]])
    monkeypatch.setattr(type(bundle.pipeline), "fit", lambda *a, **kw: pytest.fail("Inference must never fit"))
    actual = predict_frame(bundle, inputs, MonitoringSettings(domain_z_limit=100))
    assert actual.status.eq("ok").all()
    np.testing.assert_allclose(actual.prediction_g_l, np.asarray(expected).ravel())
    assert actual.model_version.eq(bundle.version).all()
    assert actual.prediction_time_utc.notna().all()
    pd.testing.assert_frame_equal(inputs, before)


@pytest.mark.parametrize("damage", ["missing", "extra", "duplicate_header", "empty", "too_many"])
def test_batch_schema_is_strict(model, damage):
    _, bundle = model
    frame = example_inputs(bundle)
    if damage == "missing":
        frame = frame.drop(columns="temperature_c")
    elif damage == "extra":
        frame["Glucose"] = 5
    elif damage == "duplicate_header":
        frame = pd.concat([frame, frame[["ph"]]], axis=1)
    elif damage == "empty":
        frame = frame.iloc[:0]
    else:
        frame = pd.concat([frame] * 201, ignore_index=True)
    with pytest.raises(ValueError):
        predict_frame(bundle, frame)


@pytest.mark.parametrize("column,value", [("temperature_c", np.inf), ("ph", -1), ("raman_400", "bad"),
    ("measurement_timestamp", "2025-01-01"), ("sensor_timestamp", "invalid"), ("sample_id", " ")])
def test_bad_rows_become_gaps(model, column, value):
    _, bundle = model
    frame = example_inputs(bundle).astype(object)
    if column == "raman_400":
        column = next(name for name in frame if name.startswith("raman_"))
    frame.loc[0, column] = value
    result = predict_frame(bundle, frame, MonitoringSettings(domain_z_limit=100))
    assert result.status.iloc[0] == "invalid_input"
    assert pd.isna(result.prediction_g_l.iloc[0])
    assert result.status.iloc[1:].eq("ok").all()


def test_duplicates_and_time_order_are_not_guessed(model):
    _, bundle = model
    frame = example_inputs(bundle)
    frame.loc[1, ["sample_id", "batch_id"]] = frame.loc[0, ["sample_id", "batch_id"]].values
    frame.loc[2, "sensor_timestamp"] = pd.Timestamp(frame.loc[2, "measurement_timestamp"]) + pd.Timedelta(seconds=1)
    result = predict_frame(bundle, frame)
    assert result.status.iloc[:3].eq("invalid_input").all()
    assert result.prediction_g_l.iloc[:3].isna().all()


def test_release_copies_original_pipeline_and_evidence_without_refitting(model, tmp_path, monkeypatch):
    reference, bundle = model
    import src.model_registry as registry
    monkeypatch.setattr(registry, "train_study", lambda *a, **k: pytest.fail("No release training"))
    folder = create_release(reference, tmp_path)
    assert (folder / "pipeline.joblib").read_bytes() == Path(reference.pipeline_path).read_bytes()
    assert (folder / "model_metadata.json").read_bytes() == Path(reference.metadata_path).read_bytes()
    assert verify_release(folder, trusted=True)["parity_rows"] == 5
    stamp = (folder / "release_manifest.json").stat().st_mtime_ns
    assert create_release(reference, tmp_path) == folder
    assert (folder / "release_manifest.json").stat().st_mtime_ns == stamp
    with ZipFile(BytesIO(release_zip(folder))) as archive:
        assert "pipeline.joblib" in archive.namelist()
        assert "operation.lock" not in archive.namelist()
        assert not any("holdout_claim" in name for name in archive.namelist())
    report = evaluation_report(bundle)
    assert report["independent_test_metrics"] is None
    assert "CV only" in report["validation_status"]


@pytest.mark.parametrize("damage", ["pipeline", "metadata", "parity", "traversal", "version", "version_numpy", "version_xgboost"])
def test_release_corruption_rejected_before_deserialization(model, tmp_path, monkeypatch, damage):
    folder = create_release(model[0], tmp_path)
    import src.deployment as deployment
    monkeypatch.setattr(deployment.joblib, "load", lambda *a, **k: pytest.fail("Must check integrity before deserialization"))
    manifest_path = folder / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if damage in ("pipeline", "metadata", "parity"):
        name = {"pipeline": "pipeline.joblib", "metadata": "model_metadata.json", "parity": "parity_predictions.json"}[damage]
        (folder / name).write_bytes(b"corrupt")
    elif damage == "traversal":
        manifest["hashes"]["../escape.joblib"] = "00"
        manifest_path.write_text(json.dumps(manifest))
    else:
        key = {"version": "scikit_learn", "version_numpy": "numpy", "version_xgboost": "xgboost"}[damage]
        manifest["fitted_versions"][key] = "0.0"
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_release(folder, trusted=True)
    with pytest.raises(ValueError):
        release_zip(folder)


def test_untrusted_release_and_changed_source_rejected(model, tmp_path):
    with pytest.raises(ValueError, match="trusted"):
        verify_release(tmp_path)
    with pytest.raises(ValueError, match="changed"):
        create_release(replace(model[0], stamp=(0, 0, 0)), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_changed_parity_values_fail_even_when_file_hash_is_updated(model, tmp_path):
    folder = create_release(model[0], tmp_path)
    parity = folder / "parity_predictions.json"
    values = json.loads(parity.read_text())
    values["raw_model_prediction_g_l"][0] += 1
    parity.write_text(json.dumps(values))
    manifest_path = folder / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["hashes"][parity.name] = sha256(parity.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="predictions differ"):
        verify_release(folder, trusted=True)


def test_domain_failures_are_gaps(model):
    _, bundle = model
    frame = example_inputs(bundle)
    frame.loc[0, next(column for column in frame if column.startswith("raman_"))] = 1e8
    result = predict_frame(bundle, frame)
    assert result.status.iloc[0] == "out_of_domain"
    assert pd.isna(result.prediction_g_l.iloc[0])


@pytest.mark.parametrize("source", ["phase4", "phase7"])
def test_finalized_evidence_discovery_and_release_do_not_change_frozen_study(tmp_path, source):
    from src.data_generation import generate_dataset
    from src.data_integration import integrate_sources, sources_from_dataset
    from src.deployment import discover_deployment_models
    from src.modeling import ModelCase, TrainingConfig, train_study
    from src.modeling_data import fixed_batch_partition, modeling_dataset
    from src.modeling_storage import evaluate_once, save_study

    config = SimulationConfig(n_batches=6, duration_h=72, n_wavenumbers=71, seed=913)
    data = modeling_dataset(integrate_sources(sources_from_dataset(generate_dataset(config))))
    study = train_study(data, fixed_batch_partition(data.metadata.batch_id), [ModelCase("raw", "pls")], TrainingConfig(max_pls_components=2))
    root = tmp_path / "studies" if source == "phase4" else tmp_path / "reliability" / "glucose"
    save_study(study, root)
    evaluate_once(study, data, root)
    before = {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}
    found, issues = discover_deployment_models(tmp_path / "studies", tmp_path / "models", tmp_path / "reliability")
    assert len(found) == 1 and not issues
    report = evaluation_report(load_model(found[0]))
    assert report["independent_test_metrics"]
    assert report["partition"]["training_batches"] == list(study.partition.training_batches)
    assert verify_release(create_release(found[0], tmp_path / "releases"), trusted=True)["parity"] == "passed"
    after = {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}
    assert before == after
