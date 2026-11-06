"""Causal received-only reliability, exact baselines and read-only projections."""

from copy import deepcopy
from dataclasses import replace
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.acquisition import InMemoryTestAdapter, test_source_records as source_records
from src.acquired_reliability import (acquired_reliability_review, acquired_reliability_report_zip,
    acquired_target_history, received_assay_residuals)
from src.acquired_reliability_storage import prepare_acquired_reference, load_acquired_reference, reference_path
from src.data_generation import SimulationConfig, generate_dataset
from src.data_integration import integrate_sources, sources_from_dataset
from src.demo_data import generate_monitoring_dataset
from src.llm_results import acquired_reliability_snapshot, template_text
from src.model_registry import prepare_demo_models, load_model
from src.modeling_data import modeling_dataset
from src.monitoring import MonitoringSettings
from src.reliability import ReliabilitySettings
from src.stream_monitoring import AcquiredStreamSession, StreamSettings


CONFIG = SimulationConfig(n_batches=1, duration_h=24, n_wavenumbers=32)
NOW = pd.Timestamp("2026-11-05T15:00:00Z")
STREAM = StreamSettings(enforce_freshness=False, silence_seconds=10000)
RULES = ReliabilitySettings()


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("acquired-reliability")
    bundles = {ref.target: load_model(ref) for ref in prepare_demo_models(CONFIG, root / "models")}
    bundle = bundles["Glucose"]
    config = SimulationConfig(**bundle.metadata["calibration_config"])
    cohort = modeling_dataset(integrate_sources(sources_from_dataset(generate_dataset(config))), "Glucose")
    baseline = prepare_acquired_reference(bundle, cohort, root / "references")
    return root, bundles, cohort, baseline


def consume(prepared, *, assays=True):
    events = source_records(sources_from_dataset(generate_monitoring_dataset(CONFIG)))
    if not assays:
        events = [event for event in events if event["source"] != "assays"]
    session = AcquiredStreamSession("shared-acquired-reliability")
    adapter = InMemoryTestAdapter(events)
    session.poll_once(adapter, prepared[1], MonitoringSettings(), STREAM, limit=100, now=NOW)
    return session


def test_projection_reuses_predictions_and_baseline_without_fit_predict_or_poll(prepared, monkeypatch):
    session = consume(prepared)
    before = session.history().copy(deep=True)
    cursor = session.acquisition.cursor
    def forbidden(*args, **kwargs):
        pytest.fail("Reliability must only read the recorded stream")
    monkeypatch.setattr(Pipeline, "predict", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(session, "poll_once", forbidden)
    review = acquired_reliability_review(session, prepared[1]["Glucose"], RULES, STREAM, prepared[3], now=NOW, batch_id="MON-B001")
    pd.testing.assert_frame_equal(session.history(), before)
    assert session.acquisition.cursor == cursor
    assert review["history"].prediction_g_l.equals(before.glucose_prediction_g_l)
    assert review["metrics"].assessed_pairs.iloc[0] > 0
    assert review["input_shift"].status.eq("assessed").all()
    assert not review["history"].current_eligible.any()  # Historical freshness disabled.
    assert review["residuals"].pair_available_at_utc.eq(NOW).all()


def test_empty_collection_and_missing_baseline_are_explicit(prepared):
    empty = AcquiredStreamSession("empty")
    result = acquired_reliability_review(empty, prepared[1]["Glucose"], RULES, STREAM, now=NOW)
    assert result["history"].empty and result["residuals"].empty
    assert result["metrics"].assessed_pairs.iloc[0] == 0 and np.isnan(result["metrics"].RMSE.iloc[0])
    session = consume(prepared, assays=False)
    result = acquired_reliability_review(session, prepared[1]["Glucose"], RULES, STREAM, now=NOW)
    assert result["input_shift"].empty and not result["context"]["baseline_available"]
    assert result["residuals"].empty and result["metrics"].assessed_pairs.iloc[0] == 0


def test_assay_receipt_and_prediction_assessment_control_availability(prepared):
    session = consume(prepared)
    bundle = prepared[1]["Glucose"]
    history = acquired_target_history(session, bundle, STREAM, now=NOW)
    before = received_assay_residuals(session, history, bundle.target, RULES, now=NOW - pd.Timedelta(seconds=1))
    assert before.empty  # Old lab timestamps do not reveal unreceived assays.
    # A late prediction must not create bias retrospectively at lab result time.
    history["assessed_at_utc"] = NOW + pd.Timedelta(seconds=10)
    assert received_assay_residuals(session, history, bundle.target, RULES, now=NOW).empty
    after = received_assay_residuals(session, history, bundle.target, RULES, now=NOW + pd.Timedelta(seconds=10))
    assert after.pair_available_at_utc.eq(NOW + pd.Timedelta(seconds=10)).all()
    assert (after.result_timestamp < after.pair_available_at_utc).all()


def test_future_lab_result_withheld_and_unmatched_or_invalid_reference_not_normal(prepared):
    session = consume(prepared)
    history = acquired_target_history(session, prepared[1]["Glucose"], STREAM, now=NOW)
    assays = [delivery for delivery in session.acquisition.audit if delivery["source"] == "assays"]
    original = assays[0]["raw_event_json"]
    first = json.loads(assays[0]["raw_event_json"])
    first["data"]["result_timestamp"] = (NOW + pd.Timedelta(hours=1)).isoformat()
    assays[0]["raw_event_json"] = json.dumps(first)
    rows = received_assay_residuals(session, history, "Glucose", RULES, now=NOW)
    assert first["event_id"] not in set(rows.assay_event_id)
    assays[0]["raw_event_json"] = original
    last = json.loads(assays[-1]["raw_event_json"])
    last["data"]["glucose_reference_g_l"] = -1
    assays[-1]["raw_event_json"] = json.dumps(last)
    rows = received_assay_residuals(session, history.iloc[:0], "Glucose", RULES, now=NOW)
    assert rows.empty  # No observations in selected window.
    subset = history.copy()
    subset.loc[subset.sample_id.eq(first["data"]["sample_id"]), "sample_id"] = "unmatched-spectrum"
    rows = received_assay_residuals(session, subset, "Glucose", RULES, now=NOW)
    invalid = rows.loc[rows.assay_event_id.eq(last["event_id"])]
    assert len(invalid) == 1 and invalid.pair_status.eq("invalid_reference").all() and invalid.residual_alarm.isna().all()
    unmatched = rows.loc[rows.assay_event_id.eq(first["event_id"])]
    assert len(unmatched) == 1 and unmatched.pair_status.eq("prediction_unavailable").all()


def test_conflicting_assay_is_retained_as_unassessed_with_original_ledger(prepared):
    session = consume(prepared)
    event = next(event for event, (source, row) in session.acquisition.accepted.items() if source == "assays")
    session.acquisition.accepted.pop(event)
    session.acquisition.conflicted_ids.add(event)
    result = acquired_reliability_review(session, prepared[1]["Glucose"], RULES, STREAM, now=NOW)
    pair = result["residuals"].set_index("assay_event_id").loc[event]
    assert pair.pair_status == "invalid_reference" and pd.isna(pair.residual_g_l)
    assert "conflicted" in pair.reference_quality_flags
    assert result["metrics"].unassessed_pairs.iloc[0] >= 1


def test_later_source_failure_and_staleness_preserve_recorded_errors(prepared):
    session = consume(prepared)
    bundle = prepared[1]["Glucose"]
    first = acquired_reliability_review(session, bundle, RULES, STREAM, now=NOW)
    session.acquisition.last_error = "API disconnected"
    rules = StreamSettings(max_age_seconds=5, silence_seconds=10000)
    later = acquired_reliability_review(session, bundle, RULES, rules, now=NOW + pd.Timedelta(seconds=6))
    assert not later["history"].current_eligible.any()
    assert later["history"].freshness_flags.str.contains("stale").all()
    assert later["history"].source_health_flags.str.contains("API disconnected").all()
    np.testing.assert_allclose(first["history"].prediction_g_l, later["history"].prediction_g_l, equal_nan=True)
    np.testing.assert_allclose(first["residuals"].residual_g_l, later["residuals"].residual_g_l, equal_nan=True)


def test_fresh_current_status_future_observations_and_duplicate_predictions(prepared):
    session = consume(prepared)
    bundle = prepared[1]["Glucose"]
    for row in session.records.values():
        row["measurement_timestamp"] = NOW
        row["sensor_timestamp"] = NOW
    rules = StreamSettings(max_age_seconds=5, silence_seconds=10000)
    fresh = acquired_target_history(session, bundle, rules, now=NOW)
    assert fresh.current_eligible.any() and fresh.freshness_status.eq("fresh").all()
    first = next(iter(session.records.values()))
    first["measurement_timestamp"] = NOW + pd.Timedelta(seconds=1)
    future = acquired_target_history(session, bundle, rules, now=NOW)
    row = future.loc[future.spectrum_event_id.eq(first["spectrum_event_id"])].iloc[0]
    assert row.prediction_status == "not_yet_available" and not row.current_eligible
    second = list(session.records.values())[1]
    second["sample_id"] = first["sample_id"]
    duplicates = acquired_target_history(session, bundle, rules, now=NOW + pd.Timedelta(seconds=1))
    assert duplicates.loc[duplicates.sample_id.eq(first["sample_id"]), "prediction_status"].eq("ambiguous_prediction").all()


def test_rolling_bias_uses_receipt_order_instead_of_old_laboratory_order(prepared):
    session = consume(prepared)
    bundle = prepared[1]["Glucose"]
    history = acquired_target_history(session, bundle, STREAM, now=NOW)
    deliveries = [row for row in session.acquisition.audit if row["source"] == "assays"]
    # Both laboratory results are old, but the earlier result arrives last.
    for index, delivery in enumerate(deliveries):
        delivery["received_at_utc"] = (NOW + pd.Timedelta(seconds=len(deliveries) - index)).isoformat()
    pairs = received_assay_residuals(session, history, "Glucose", replace(RULES, rolling_pairs=2), now=NOW + pd.Timedelta(seconds=10))
    assert pairs.pair_available_at_utc.is_monotonic_increasing
    assessed = pairs.loc[pairs.pair_status.eq("assessed")]
    np.testing.assert_allclose(assessed.rolling_bias_g_l, assessed.residual_g_l.rolling(2, min_periods=2).mean(), equal_nan=True)


def test_finalized_model_baseline_preserves_frozen_test_artifacts(prepared, tmp_path):
    from src.modeling import ModelCase, TrainingConfig, train_study
    from src.modeling_storage import save_study, evaluate_once
    from src.model_registry import _reference
    cohort = prepared[2]
    from src.modeling_data import fixed_batch_partition
    study = train_study(cohort, fixed_batch_partition(cohort.metadata.batch_id), [ModelCase("raw", "pls")], config=TrainingConfig(max_pls_components=2))
    folder = save_study(study, tmp_path / "studies")
    evaluate_once(study, cohort, tmp_path / "studies")
    bundle = load_model(_reference(folder, "winner_pipeline.joblib", "winner_metadata.json", "Finalized development model", tmp_path / "studies"))
    before = {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}
    baseline = prepare_acquired_reference(bundle, cohort, tmp_path / "baseline")
    result = acquired_reliability_review(consume(prepared), bundle, RULES, STREAM, baseline, now=NOW)
    assert bundle.metadata["holdout_metrics"] and result["metrics"].assessed_pairs.iloc[0] == 0  # Different active model, no borrowed predictions.
    assert before == {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}


def test_batch_window_insufficient_data_and_model_mismatch(prepared):
    session = consume(prepared)
    result = acquired_reliability_review(session, prepared[1]["Glucose"], RULES, STREAM, prepared[3], now=NOW, window_hours=1)
    assert len(result["history"]) == 2
    assert result["input_shift"].status.eq("insufficient_data").all()
    bad = deepcopy(prepared[1]["Glucose"])
    bad.metadata["model_version"] += "-other"
    result = acquired_reliability_review(session, bad, RULES, STREAM, now=NOW)
    assert result["history"].prediction_status.eq("model_identity_mismatch").all()
    assert result["metrics"].assessed_pairs.iloc[0] == 0
    with pytest.raises(ValueError, match="different model"):
        acquired_reliability_review(session, bad, RULES, STREAM, prepared[3], now=NOW)


def test_explicit_baseline_is_training_only_and_reused_without_predict(prepared, tmp_path, monkeypatch):
    bundle, cohort = prepared[1]["Glucose"], prepared[2]
    expected = sum(cohort.metadata.batch_id.isin(bundle.metadata["partition"]["training_batches"]))
    calls = []
    original = bundle.pipeline.predict
    def predict(inputs):
        calls.append(len(inputs))
        assert set(cohort.metadata.loc[inputs.index].batch_id).isdisjoint(bundle.metadata["partition"]["holdout_batches"])
        return original(inputs)
    monkeypatch.setattr(bundle.pipeline, "predict", predict)
    reference = prepare_acquired_reference(bundle, cohort, tmp_path)
    assert calls == [expected]
    assert len(reference["training_row_ids"]) == expected
    assert prepare_acquired_reference(bundle, cohort, tmp_path) == reference
    assert calls == [expected]
    assert load_acquired_reference(bundle, tmp_path) == reference
    wrong = replace(cohort, cohort_id="a" * 64)
    with pytest.raises(ValueError, match="does not match"):
        prepare_acquired_reference(bundle, wrong, tmp_path)
    path = reference_path(bundle, tmp_path)
    changed = json.loads(path.read_text())
    changed["predictions_g_l"][0] += 1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="integrity"):
        load_acquired_reference(bundle, tmp_path)


def test_export_separates_operational_and_frozen_evidence_and_llm_has_only_aggregates(prepared):
    session = consume(prepared)
    bundle = prepared[1]["Glucose"]
    result = acquired_reliability_review(session, bundle, RULES, STREAM, prepared[3], now=NOW)
    with ZipFile(BytesIO(acquired_reliability_report_zip(result, session, bundle, prepared[3]))) as archive:
        assert "frozen_validation/recorded_evidence.json" in archive.namelist()
        assert "operational_review/residuals.csv" in archive.namelist()
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["cursor"] == session.acquisition.cursor and manifest["baseline_available"]
        evidence = json.loads(archive.read("frozen_validation/recorded_evidence.json"))
        assert evidence["holdout_metrics"] is None  # Demo never receives an independent-test claim.
    snapshot = acquired_reliability_snapshot(result, bundle)
    assert snapshot.requires_consent
    assert "MON-B001" not in snapshot.serialized and bundle.version not in snapshot.serialized
    assert "raw_event_json" not in snapshot.serialized and "raman_" not in snapshot.serialized
    assert template_text(snapshot)
