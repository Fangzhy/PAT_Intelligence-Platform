"""Frozen historical PCA over received sensor data, quality/freshness and audits."""

from copy import deepcopy
from dataclasses import replace
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.acquisition import InMemoryTestAdapter, test_source_records as source_records
from src.data_generation import SimulationConfig, generate_dataset
from src.data_integration import sources_from_dataset
from src.demo_data import generate_monitoring_dataset
from src.monitoring import MonitoringSettings
from src.multivariate_monitoring import fit_process_monitor, score_process_monitor, monitoring_model_manifest
from src.multivariate_storage import save_pca_reference, load_pca_reference, list_pca_references
from src.multivariate_stream import score_acquired_sensors, acquired_sensor_contributions, acquired_pca_report_zip
from src.stream_monitoring import AcquiredStreamSession, StreamSettings
from src.test_stream_source import PacedTestAdapter


CONFIG = SimulationConfig(n_batches=6, duration_h=72, n_wavenumbers=32, seed=42)
NOW = pd.Timestamp("2026-11-05T15:00:00Z")
RULES = StreamSettings(enforce_freshness=False, silence_seconds=1000)
MONITORING = MonitoringSettings()


@pytest.fixture(scope="module")
def model():
    data = generate_dataset(CONFIG)
    return fit_process_monitor(data.sensors, data.batches)


@pytest.fixture(scope="module")
def monitoring_data():
    return generate_monitoring_dataset(replace(CONFIG, n_batches=1))


def first_events(data):
    records = source_records(sources_from_dataset(data))
    return [deepcopy(next(event for event in records if event["source"] == name)) for name in ("batches", "sensors", "raman")]


def consume(records, *, rules=RULES, now=NOW):
    records = deepcopy(records)
    for sequence, event in enumerate(records):
        event["sequence"] = sequence
    adapter = InMemoryTestAdapter(records)
    session = AcquiredStreamSession("shared-pca")
    while not session.acquisition.complete:
        assert session.poll_once(adapter, {}, MONITORING, rules, limit=100, now=now)
    return session


def test_acquired_projection_matches_historical_math_without_fitting_or_cursor_changes(model, monitoring_data, monkeypatch):
    session = consume(source_records(sources_from_dataset(monitoring_data)))
    expected = score_process_monitor(model, monitoring_data.sensors)
    before = monitoring_model_manifest(model)
    cursor = session.acquisition.cursor
    ledger = session.acquisition.ledger().copy(deep=True)
    monkeypatch.setattr(StandardScaler, "fit", lambda *args, **kwargs: pytest.fail("No scaler fitting"))
    monkeypatch.setattr(PCA, "fit", lambda *args, **kwargs: pytest.fail("No PCA fitting"))
    monkeypatch.setattr(session, "poll_once", lambda *args, **kwargs: pytest.fail("Scoring must not poll"))
    actual = score_acquired_sensors(model, session, RULES, MONITORING, now=NOW)
    assert actual.status.eq("scored").all()
    assert actual.role.eq("Acquired monitoring").all()
    for column in ("PC1", "PC2", "t2", "spe", "anomaly_ratio"):
        np.testing.assert_allclose(actual[column], expected[column])
    event = actual.sensor_event_id.iloc[0]
    contributions = acquired_sensor_contributions(model, actual, event)
    assert contributions.spe_contribution.sum() == pytest.approx(actual.spe.iloc[0])
    assert monitoring_model_manifest(model) == before
    assert session.acquisition.cursor == cursor
    pd.testing.assert_frame_equal(session.acquisition.ledger(), ledger)


def test_later_staleness_or_read_failure_withholds_current_cards_but_retains_scores(model, monitoring_data):
    fresh = PacedTestAdapter(first_events(monitoring_data), started_at=NOW, clock=lambda: 0.0).records
    rules = StreamSettings(max_age_seconds=5, silence_seconds=1000)
    session = consume(fresh, rules=rules)
    first = score_acquired_sensors(model, session, rules, MONITORING, now=NOW)
    assert first.current_eligible.all() and first.freshness_status.eq("fresh").all()
    later = score_acquired_sensors(model, session, rules, MONITORING, now=NOW + pd.Timedelta(seconds=6))
    assert not later.current_eligible.any()
    assert later.freshness_status.eq("stale_or_future").all()
    np.testing.assert_array_equal(first.t2, later.t2)
    session.acquisition.last_error = "API disconnected"
    failed = score_acquired_sensors(model, session, rules, MONITORING, now=NOW)
    assert not failed.current_eligible.any()
    assert failed.source_health_flags.str.contains("API disconnected").all()
    np.testing.assert_array_equal(first.spe, failed.spe)


@pytest.mark.parametrize("offset", [-6, 6])
def test_stale_or_future_on_arrival_is_unscorable(model, monitoring_data, offset):
    fresh = PacedTestAdapter(first_events(monitoring_data), started_at=NOW, clock=lambda: 0.0).records
    for event in fresh:
        if event["source"] in ("sensors", "raman"):
            event["data"]["measurement_timestamp"] = (NOW + pd.Timedelta(seconds=offset)).isoformat()
    rules = StreamSettings(max_age_seconds=5)
    session = consume(fresh, rules=rules)
    scored = score_acquired_sensors(model, session, rules, MONITORING, now=NOW)
    assert scored.status.ne("scored").all()
    assert scored.t2.isna().all() and scored.spe.isna().all()
    assert scored.alarm.isna().all() and not scored.current_eligible.any()
    assert scored.arrival_freshness_flags.str.len().gt(0).all()


@pytest.mark.parametrize("column,value", [("ph", None), ("dissolved_oxygen_pct", 145)])
def test_quality_failures_are_gaps_with_unknown_alarm_states(model, monitoring_data, column, value):
    records = first_events(monitoring_data)
    records[1]["data"][column] = value
    session = consume(records)
    scored = score_acquired_sensors(model, session, RULES, MONITORING, now=NOW)
    assert scored.status.eq("invalid_input").all()
    assert scored.quality_flags.str.len().gt(0).all()
    assert scored.t2.isna().all() and scored.alarm.isna().all()
    with pytest.raises(ValueError, match="valid recorded"):
        acquired_sensor_contributions(model, scored, scored.sensor_event_id.iloc[0])


def test_later_conflicts_withdraw_pca_statistics_while_retaining_original_sensor_audit(model, monitoring_data):
    records = first_events(monitoring_data)
    conflict = deepcopy(records[1])
    conflict["data"]["ph"] += 0.1
    session = consume([*records, conflict])
    scored = score_acquired_sensors(model, session, RULES, MONITORING, now=NOW)
    assert len(scored) == 1
    assert scored.status.eq("invalidated_input").all()
    assert scored.ph.iloc[0] == records[1]["data"]["ph"]
    assert scored.spe.isna().all() and scored.alarm.isna().all()


def test_duplicate_delivery_does_not_duplicate_pca_rows_and_missing_metadata_is_flagged(model, monitoring_data):
    records = first_events(monitoring_data)
    session = consume([*records, deepcopy(records[1])])
    assert len(score_acquired_sensors(model, session, RULES, MONITORING, now=NOW)) == 1
    missing = consume(records[1:])
    scored = score_acquired_sensors(model, missing, RULES, MONITORING, now=NOW)
    assert scored.status.ne("scored").all()
    assert scored.quality_flags.str.contains("batch metadata").all()


def test_saved_pca_reference_round_trip_and_integrity(model, tmp_path, monitoring_data, monkeypatch):
    identity = save_pca_reference(model, CONFIG, tmp_path)
    assert identity in list_pca_references(tmp_path)
    monkeypatch.setattr(PCA, "fit", lambda *args, **kwargs: pytest.fail("Loading must not refit"))
    loaded, metadata = load_pca_reference(tmp_path, identity)
    assert metadata["historical_config"]["seed"] == CONFIG.seed
    assert monitoring_model_manifest(loaded) == monitoring_model_manifest(model)
    expected = score_process_monitor(model, monitoring_data.sensors)
    actual = score_process_monitor(loaded, monitoring_data.sensors)
    pd.testing.assert_frame_equal(actual, expected)
    path = tmp_path / identity / "reference.joblib"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        load_pca_reference(tmp_path, identity)
    with pytest.raises(ValueError, match="identity"):
        load_pca_reference(tmp_path, "../outside")


def test_received_only_export_separates_historical_calibration(model, monitoring_data):
    session = consume(first_events(monitoring_data))
    scored = score_acquired_sensors(model, session, RULES, MONITORING, now=NOW)
    with ZipFile(BytesIO(acquired_pca_report_zip(model, session, scored, RULES, MONITORING, now=NOW))) as archive:
        current = pd.read_csv(BytesIO(archive.read("acquired/observations.csv")))
        assert len(current) == 1 and current.sample_id.iloc[0] == "MON-B001-S00000"
        assert "historical_reference/reference_audit.csv" in archive.namelist()
        contributions = pd.read_csv(BytesIO(archive.read("acquired/spe_contributions.csv")))
        assert contributions.spe_contribution.sum() == pytest.approx(current.spe.iloc[0])
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["cursor"] == session.acquisition.cursor
        assert manifest["pca_reference"]["model_id"] == model.model_id


def test_empty_source_and_historical_freshness_are_explicit(model, monitoring_data):
    empty = AcquiredStreamSession("empty")
    result = score_acquired_sensors(model, empty, RULES, MONITORING, now=NOW)
    assert result.empty and "current_eligible" in result and "pca_model_id" in result
    assert empty.acquisition.cursor == 0
    session = consume(first_events(monitoring_data))
    historical = score_acquired_sensors(model, session, RULES, MONITORING, now=NOW)
    assert historical.status.eq("scored").all()
    assert historical.freshness_status.eq("not_enforced").all()
    assert not historical.current_eligible.any()


def test_shared_sensor_quality_rules_and_ambiguous_identities_remain_flagged(model, monitoring_data):
    records = first_events(monitoring_data)
    records[1]["data"]["ph"] = 7.4
    from src.data_integration import SENSOR_RANGES
    quality = MonitoringSettings(input_sensor_ranges={**SENSOR_RANGES, "ph": (6.7, 7.3)})
    session = consume(records)
    scored = score_acquired_sensors(model, session, RULES, quality, now=NOW)
    assert scored.quality_flags.str.contains("ph").all() and scored.t2.isna().all()
    duplicate = deepcopy(records[1])
    duplicate["event_id"] += "-different-delivery-id"
    ambiguous = consume([*records, duplicate])
    scored = score_acquired_sensors(model, ambiguous, RULES, MONITORING, now=NOW)
    assert len(scored) == 2 and scored.t2.isna().all()
    assert scored.quality_flags.str.contains("Ambiguous").all()


def test_pca_does_not_call_or_modify_the_predictive_pipeline(model, monitoring_data, tmp_path, monkeypatch):
    from sklearn.pipeline import Pipeline
    from src.model_registry import prepare_demo_models, load_model
    models = {reference.target: load_model(reference) for reference in prepare_demo_models(CONFIG, tmp_path)}
    session = AcquiredStreamSession("both-monitoring-models")
    adapter = InMemoryTestAdapter(first_events(monitoring_data))
    assert session.poll_once(adapter, models, MONITORING, RULES, now=NOW)
    before = session.history().copy(deep=True)
    means = {target: bundle.pipeline.named_steps["scale"].mean_.copy() for target, bundle in models.items()}
    monkeypatch.setattr(Pipeline, "predict", lambda *args, **kwargs: pytest.fail("PCA must not call the predictive model"))
    scored = score_acquired_sensors(model, session, RULES, MONITORING, now=NOW)
    assert scored.status.eq("scored").all()
    pd.testing.assert_frame_equal(session.history(), before)
    for target, bundle in models.items():
        np.testing.assert_array_equal(bundle.pipeline.named_steps["scale"].mean_, means[target])
