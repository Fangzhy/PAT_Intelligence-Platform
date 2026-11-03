"""Trusted local artifacts, explicit calibration isolation, and no render-time fit."""

from dataclasses import replace
import json

import numpy as np
import pytest

from src.data_generation import SimulationConfig, generate_dataset
from src.model_registry import discover_models, load_model, prepare_demo_models
from src.replay import replay_batch


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    root = tmp_path_factory.mktemp("model-registry")
    config = SimulationConfig(n_batches=3, duration_h=72, n_wavenumbers=71, seed=412)
    references = prepare_demo_models(config, root)
    return root, config, references


def test_missing_models_are_graceful_and_discovery_does_not_create_artifacts(tmp_path):
    references, issues = discover_models(tmp_path / "phase4", tmp_path / "monitoring")
    assert references == [] and issues == []
    assert list(tmp_path.iterdir()) == []


def test_demo_models_are_saved_for_both_targets_and_use_separate_training_seed(registry):
    root, config, references = registry
    assert {reference.target for reference in references} == {"Glucose", "Lactate"}
    batch = replay_batch(generate_dataset(config), "B001")
    for reference in references:
        bundle = load_model(reference)
        assert bundle.metadata["calibration_config"]["seed"] != config.seed
        assert bundle.metadata["partition"]["training_batches"] == ["B001", "B002", "B003", "B004", "B005", "B006"]
        assert "holdout_metrics" not in bundle.metadata
        assert "CV only" in bundle.metadata["validation_status"]
        assert np.isfinite(bundle.pipeline.predict(batch.inputs.iloc[[10]])).all()
    assert not list(root.glob("*/holdout_claim.json"))


def test_valid_demo_artifacts_are_reused_without_training(registry, monkeypatch):
    root, config, references = registry
    import src.model_registry as registry_module

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Existing monitoring models must not be retrained")

    monkeypatch.setattr(registry_module, "train_study", forbidden_fit)
    again = prepare_demo_models(config, root)
    assert {reference.version for reference in again} == {reference.version for reference in references}
    found, issues = discover_models(root / "no-phase4", root)
    assert len(found) == 2 and not issues


def test_corrupt_metadata_and_missing_pipeline_are_reported(tmp_path, registry):
    root, _, references = registry
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "model_metadata.json").write_text('{"target":"Potency","target_unit":"g/L","model_version":"bad"}')
    found, issues = discover_models(tmp_path / "phase4", tmp_path)
    assert found == []
    assert len(issues) == 1
    reference = references[0]
    altered = replace(reference, version="incorrect-version")
    with pytest.raises(ValueError, match="changed"):
        load_model(altered)


def test_finalized_phase_four_pipeline_is_discovered_without_refitting(tmp_path):
    from src.data_integration import integrate_sources, sources_from_dataset
    from src.modeling import ModelCase, TrainingConfig, train_study
    from src.modeling_data import fixed_batch_partition, modeling_dataset
    from src.modeling_storage import evaluate_once, save_study

    config = SimulationConfig(n_batches=6, duration_h=72, n_wavenumbers=71, seed=913)
    data = modeling_dataset(integrate_sources(sources_from_dataset(generate_dataset(config))))
    study = train_study(data, fixed_batch_partition(data.metadata.batch_id), [ModelCase("raw", "pls")], TrainingConfig(max_pls_components=2))
    save_study(study, tmp_path / "phase4")
    assert discover_models(tmp_path / "phase4", tmp_path / "empty")[0] == []
    evaluate_once(study, data, tmp_path / "phase4")
    references, issues = discover_models(tmp_path / "phase4", tmp_path / "empty")
    assert len(references) == 1 and not issues
    bundle = load_model(references[0])
    assert bundle.metadata["artifact_source"] == "Phase 4 final model"
    assert bundle.metadata["study_id"] == study.study_id
    np.testing.assert_allclose(bundle.pipeline.predict(data.X), study.results[study.winner_id].estimator.predict(data.X))
