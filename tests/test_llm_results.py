"""Aggregate allowlist, frozen-test preservation, nulls, and review causality."""

from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pandas as pd
import pytest

from src.data_generation import SimulationConfig, generate_dataset
from src.data_integration import integrate_sources, sources_from_dataset
from src.llm_results import modeling_snapshot, reliability_snapshot, template_text
from src.modeling import ModelCase, TrainingConfig, train_study
from src.modeling_data import fixed_batch_partition, modeling_dataset
from src.modeling_storage import evaluate_once, save_study
from src.monitoring import MonitoringSettings
from src.reliability import (ReliabilityReview, ReliabilitySettings, available_reference_residuals,
    build_reliability_reference, distribution_shift, input_distribution_shift, residual_metrics)
from src.reliability_storage import load_reliability_evidence
from src.replay import replay_batch


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("ai-frozen-evidence")
    dataset = generate_dataset(SimulationConfig(n_batches=8, duration_h=72, n_wavenumbers=71))
    cohort = modeling_dataset(integrate_sources(sources_from_dataset(dataset)))
    study = train_study(cohort, fixed_batch_partition(cohort.metadata.batch_id), [ModelCase("raw", "pls")], TrainingConfig(max_pls_components=2))
    save_study(study, root)
    evaluate_once(study, cohort, root)
    study, evaluation, bundle = load_reliability_evidence(cohort, root)
    return root, dataset, cohort, study, evaluation, bundle


def test_snapshot_only_contains_aggregates_and_does_not_disclose_sealed_test(evidence):
    _, _, _, study, _, _ = evidence
    copy = replace(study, study_id="PRIVATE_STUDY", cohort_id="PRIVATE_COHORT",
                   training_metadata=study.training_metadata.assign(batch_id="PRIVATE_BATCH"))
    snapshot = modeling_snapshot(copy, None, source_kind="prepared_sources")
    assert snapshot.requires_consent
    assert "missing_test" in snapshot.payload["limitations"]
    assert all(row["scope"] != "independent_test" for row in snapshot.payload["facts"])
    for forbidden in ("PRIVATE", "sample_id", "batch_id", "reference_g_l", "measurement_timestamp", "raman_"):
        assert forbidden not in snapshot.serialized
    assert len(snapshot.payload["facts"]) < 30


def test_explanation_reads_frozen_evidence_without_modifying_any_saved_file(evidence, monkeypatch):
    root, _, _, study, evaluation, _ = evidence
    import src.modeling_storage as storage
    monkeypatch.setattr(storage, "evaluate_once", lambda *a, **kw: pytest.fail("Must not evaluate"))
    before = {str(path): sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}
    snapshot = modeling_snapshot(study, evaluation)
    assert "frozen_test" in snapshot.payload["limitations"]
    test_rmse = next(row for row in snapshot.payload["facts"] if row["label"].endswith("RMSEP"))
    expected = evaluation["metrics"].set_index("case_id").loc[study.winner_id, "RMSEP"]
    assert test_rmse["value"] == expected
    assert before == {str(path): sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="frozen study"):
        modeling_snapshot(study, {**evaluation, "study_id": "other"})


def test_undefined_metric_is_json_null_and_template_does_not_invent_it(evidence):
    _, _, _, study, _, _ = evidence
    result = study.results[study.winner_id]
    adjusted = replace(result, cv_metrics={**result.cv_metrics, "R2": np.nan})
    adjusted_study = replace(study, results={study.winner_id: adjusted})
    snapshot = modeling_snapshot(adjusted_study, None)
    assert "NaN" not in snapshot.serialized
    assert any(row["value"] is None for row in snapshot.payload["facts"])
    assert "Unavailable / not assessed" in template_text(snapshot)


def review_snapshot(evidence, elapsed):
    _, data, cohort, study, evaluation, bundle = evidence
    batch = replay_batch(data, "B001")
    cutoff = batch.metadata.measurement_timestamp.iloc[0] + pd.Timedelta(hours=elapsed)
    rules = ReliabilitySettings()
    history = ReliabilityReview("test").history(batch, cutoff, bundle, MonitoringSettings())
    residuals = available_reference_residuals(history, batch.assays, study.target, cutoff, rules)
    reference = build_reliability_reference(study, cohort)
    inputs = batch.inputs.loc[history.sample_id]
    input_shift = input_distribution_shift(reference.input_features, inputs, rules)
    prediction_shift = distribution_shift(pd.DataFrame({"prediction_g_l": reference.predictions_g_l}),
        pd.DataFrame({"prediction_g_l": history.loc[history.prediction_status.eq("ok"), "prediction_g_l"]}), rules)
    arguments = (study, evaluation, history, residuals, residual_metrics(residuals), input_shift, prediction_shift, rules)
    return reliability_snapshot(*arguments, context={"cutoff_utc": cutoff.isoformat()}), arguments, cutoff


def test_reliability_snapshot_uses_only_available_assays_and_revealed_predictions(evidence):
    early, _, _ = review_snapshot(evidence, 0)
    later, _, _ = review_snapshot(evidence, 78)
    fact = lambda snapshot, name: next(row["value"] for row in snapshot.payload["facts"] if row["label"] == name)
    assert fact(early, "Revealed observations") == 1
    assert fact(early, "Available assays") == 0
    assert fact(early, "Available assessed pairs — RMSE") is None
    assert fact(later, "Available assays") > 0
    assert early.identity != later.identity
    assert not any("sample_id" in key for key in early.payload)


def test_future_observations_or_assay_results_are_rejected_before_network(evidence):
    _, arguments, cutoff = review_snapshot(evidence, 78)
    _, _, early = review_snapshot(evidence, 0)
    with pytest.raises(ValueError, match="future"):
        reliability_snapshot(*arguments, context={"cutoff_utc": early.isoformat()})
    with pytest.raises(ValueError, match="timezone-aware"):
        reliability_snapshot(*arguments, context={"cutoff_utc": "2026-01-01"})
    first = reliability_snapshot(*arguments, context={"cutoff_utc": cutoff.isoformat(), "rules": "original"})
    second = reliability_snapshot(*arguments, context={"cutoff_utc": cutoff.isoformat(), "rules": "changed"})
    assert first.payload["facts"] == second.payload["facts"] and first.identity != second.identity
