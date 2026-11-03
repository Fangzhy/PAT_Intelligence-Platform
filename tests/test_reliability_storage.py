"""Explicit preparation, isolated persistence, and read-only evidence loading."""

import pytest

from src.data_generation import SimulationConfig
from src.modeling_storage import holdout_is_sealed, save_study
from src.reliability_storage import (
    load_reliability_evidence, prepare_reliability_demo, reliability_demo_config,
)


CONFIG = SimulationConfig(n_batches=6, duration_h=72, n_wavenumbers=32, seed=12)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    root = tmp_path_factory.mktemp("reliability-storage")
    return root, prepare_reliability_demo(CONFIG, "Glucose", root)


def test_demo_calibration_is_separate_seed_and_keeps_independent_batches(saved):
    root, (cohort, study, evaluation, bundle) = saved
    config = reliability_demo_config(CONFIG)
    assert config.seed == CONFIG.seed + 2_000_007 and config.n_batches == 8
    assert study.partition.holdout_batches == ("B007", "B008")
    assert set(evaluation["predictions"].batch_id) == {"B007", "B008"}
    assert holdout_is_sealed(root / "glucose", cohort.cohort_id)
    assert bundle.target == "Glucose"


def test_preparation_reuses_completed_study_without_training_or_retesting(saved, monkeypatch):
    import src.reliability_storage as storage
    root, (_, study, evaluation, _) = saved
    def fail(*args, **kwargs):
        raise AssertionError("Completed evidence was trained or tested again")
    monkeypatch.setattr(storage, "train_study", fail)
    monkeypatch.setattr(storage, "evaluate_once", fail)
    _, repeated, result, _ = prepare_reliability_demo(CONFIG, "Glucose", root)
    assert repeated.study_id == study.study_id
    assert result["study_id"] == evaluation["study_id"]


def test_loading_unfinished_study_never_claims_or_evaluates_its_holdout(saved, tmp_path, monkeypatch):
    import src.reliability_storage as storage
    _, (cohort, study, _, _) = saved
    save_study(study, tmp_path)
    def fail(*args, **kwargs):
        raise AssertionError("Reading evidence attempted holdout evaluation")
    monkeypatch.setattr(storage, "evaluate_once", fail)
    loaded, evaluation, bundle = load_reliability_evidence(cohort, tmp_path)
    assert loaded.study_id == study.study_id and evaluation is None and bundle is None
    assert not holdout_is_sealed(tmp_path, cohort.cohort_id)


def test_missing_evidence_loading_creates_no_files(saved, tmp_path):
    _, (cohort, _, _, _) = saved
    assert load_reliability_evidence(cohort, tmp_path) == (None, None, None)
    assert not list(tmp_path.iterdir())


def test_demo_targets_use_separate_storage_even_when_cohort_ids_are_shared(saved):
    root, (glucose, _, _, _) = saved
    lactate, _, _, bundle = prepare_reliability_demo(CONFIG, "Lactate", root)
    assert lactate.cohort_id == glucose.cohort_id
    assert bundle.target == "Lactate"
    assert holdout_is_sealed(root / "lactate", lactate.cohort_id)
    assert load_reliability_evidence(glucose, root / "glucose")[2].target == "Glucose"


def test_sealed_cohort_with_missing_study_cannot_trigger_retraining(saved, tmp_path, monkeypatch):
    import src.reliability_storage as storage
    _, (cohort, _, _, _) = saved
    folder = tmp_path / "glucose" / cohort.cohort_id
    folder.mkdir(parents=True)
    (folder / "holdout_claim.json").write_text("{}", encoding="utf-8")
    def fail(*args, **kwargs):
        raise AssertionError("A sealed cohort attempted retraining")
    monkeypatch.setattr(storage, "train_study", fail)
    with pytest.raises(ValueError, match="sealed demo cohort"):
        prepare_reliability_demo(CONFIG, "Glucose", tmp_path)
