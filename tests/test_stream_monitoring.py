"""Online alignment, freshness, once-only inference, and conflict invalidation."""

from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from src.acquisition import AcquisitionError, AcquisitionPage, InMemoryTestAdapter, test_source_records as source_records, validate_page
from src.data_generation import SimulationConfig, generate_dataset
from src.data_integration import SourceTables, sources_from_dataset
from src.model_registry import load_model, prepare_demo_models
from src.monitoring import MonitoringSettings
from src.stream_monitoring import AcquiredStreamSession, StreamSettings, acquired_stream_report_zip, freshness_errors
from src.test_stream_source import PacedTestAdapter


CONFIG = SimulationConfig(n_batches=2, duration_h=24, n_wavenumbers=32)
NOW = pd.Timestamp("2026-11-03T12:00:00Z")
HISTORICAL = StreamSettings(enforce_freshness=False)
MONITORING = MonitoringSettings(domain_z_limit=100)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    refs = prepare_demo_models(CONFIG, tmp_path_factory.mktemp("stream-models"))
    return {reference.target: load_model(reference) for reference in refs}


@pytest.fixture(scope="module")
def records():
    sources = sources_from_dataset(generate_dataset(CONFIG))
    scoped = SourceTables(**{name: frame[frame.batch_id.eq("B001")].copy() for name, frame in sources.as_dict().items()})
    return source_records(scoped)


def small_records(records):
    return [json.loads(json.dumps(next(record for record in records if record["source"] == name))) for name in ("batches", "sensors", "raman")]


def adapter_for(records):
    return InMemoryTestAdapter([dict(record, sequence=index) for index, record in enumerate(records)])


def consume(records, models, rules=HISTORICAL, monitoring=MONITORING):
    adapter = adapter_for(records)
    session = AcquiredStreamSession("test")
    while not session.acquisition.complete:
        assert session.poll_once(adapter, models, monitoring, rules, limit=100, now=NOW), session.acquisition.last_error
    return session


def test_stream_prediction_equals_saved_pipeline_and_never_refits(records, models, monkeypatch):
    record_set = small_records(records)
    bundle = models["Glucose"]
    row = record_set[2]["data"]
    inputs = pd.DataFrame([{key: value for key, value in row.items() if key.startswith("raman_")}])
    for column in ("temperature_c", "ph", "dissolved_oxygen_pct", "feed_rate_l_h", "agitation_rpm"):
        inputs[column] = record_set[1]["data"][column]
    expected = float(np.asarray(bundle.pipeline.predict(inputs)).reshape(-1)[0])
    calls = []
    original = bundle.pipeline.predict
    def predict(values):
        assert "glucose_reference_g_l" not in values and "glucose_g_l" not in values
        calls.append(1)
        return original(values)
    def fail(*args, **kwargs):
        raise AssertionError("Streaming attempted to fit")
    monkeypatch.setattr(bundle.pipeline, "predict", predict)
    monkeypatch.setattr(bundle.pipeline, "fit", fail)
    session = consume(record_set, models)
    result = session.history().iloc[0]
    assert result.glucose_status == "ok" and result.glucose_prediction_g_l == pytest.approx(expected)
    assert result.glucose_model_version == bundle.version
    session.process(models, MONITORING, HISTORICAL, now=NOW + pd.Timedelta(seconds=1))
    assert calls == [1]
    assert session.history().glucose_prediction_time_utc.iloc[0] == result.glucose_prediction_time_utc


def test_pending_spectrum_waits_for_delayed_sensor_then_is_inferred_once(records, models):
    batch, sensor, spectrum = small_records(records)
    adapter = adapter_for([batch, spectrum, sensor])
    session = AcquiredStreamSession("pending")
    rules = StreamSettings(wait_seconds=5, enforce_freshness=False)
    session.poll_once(adapter, models, MONITORING, rules, limit=2, now=NOW)
    assert len(session.pending) == 1 and not session.records
    assert session.history().glucose_status.iloc[0] == "waiting_for_inputs"
    session.poll_once(adapter, models, MONITORING, rules, limit=1, now=NOW + pd.Timedelta(seconds=1))
    assert not session.pending and session.history().glucose_status.iloc[0] == "ok"


@pytest.mark.parametrize("fault", ["future_sensor", "other_batch", "missing_sensor", "missing_metadata"])
def test_alignment_never_uses_future_or_cross_batch_inputs(records, models, fault):
    batch, sensor, spectrum = small_records(records)
    selected = [batch, sensor, spectrum]
    if fault == "future_sensor":
        sensor["data"]["measurement_timestamp"] = (pd.Timestamp(spectrum["data"]["measurement_timestamp"]) + pd.Timedelta(minutes=1)).isoformat()
    elif fault == "other_batch":
        sensor["data"]["batch_id"] = "B002"
    elif fault == "missing_sensor":
        selected = [batch, spectrum]
    else:
        selected = [sensor, spectrum]
    row = consume(selected, models).history().iloc[0]
    assert row.glucose_status == "invalid_input" and np.isnan(row.glucose_prediction_g_l)
    assert row.alignment_status == "rejected"


def test_pending_inputs_expire_during_idle_source_without_advancing_cursor(records, models):
    batch, _, spectrum = small_records(records)
    first = adapter_for([batch, spectrum]).poll(0, 2)
    class IdleSource:
        def poll(self, cursor, limit):
            if cursor == 0:
                return AcquisitionPage(first.source_id, first.records, 2, True)
            return AcquisitionPage(first.source_id, [], 2, True, idle=True)
    session = AcquiredStreamSession("timeout")
    rules = StreamSettings(wait_seconds=2, enforce_freshness=False)
    session.poll_once(IdleSource(), models, MONITORING, rules, limit=5, now=NOW)
    assert len(session.pending) == 1
    session.poll_once(IdleSource(), models, MONITORING, rules, limit=5, now=NOW + pd.Timedelta(seconds=2))
    assert not session.pending and session.history().glucose_status.iloc[0] == "invalid_input"
    assert session.acquisition.cursor == 2 and not session.acquisition.complete


def test_sensor_age_rejects_stale_backward_match(records, models):
    batch, sensor, spectrum = small_records(records)
    spectrum["data"]["measurement_timestamp"] = (pd.Timestamp(sensor["data"]["measurement_timestamp"]) + pd.Timedelta(hours=1)).isoformat()
    row = consume([batch, sensor, spectrum], models).history().iloc[0]
    assert row.sensor_age_minutes == 60 and row.glucose_status == "stale_input"
    assert np.isnan(row.glucose_prediction_g_l)


def test_sensor_arriving_after_finalization_does_not_rewrite_a_gap(records, models):
    batch, sensor, spectrum = small_records(records)
    adapter = adapter_for([batch, spectrum, sensor])
    session = AcquiredStreamSession("late")
    rules = StreamSettings(wait_seconds=0, enforce_freshness=False)
    session.poll_once(adapter, models, MONITORING, rules, limit=2, now=NOW)
    assert session.history().glucose_status.iloc[0] == "invalid_input"
    session.poll_once(adapter, models, MONITORING, rules, limit=1, now=NOW + pd.Timedelta(seconds=1))
    assert session.history().glucose_status.iloc[0] == "invalid_input"
    assert np.isnan(session.history().glucose_prediction_g_l.iloc[0])


@pytest.mark.parametrize("late_source", ["sensor", "metadata"])
def test_late_input_cannot_satisfy_deadline_when_no_poll_ran_at_expiry(records, models, late_source):
    batch, sensor, spectrum = small_records(records)
    ordered = [batch, spectrum, sensor] if late_source == "sensor" else [sensor, spectrum, batch]
    adapter = adapter_for(ordered)
    session = AcquiredStreamSession("late-receipt")
    rules = StreamSettings(wait_seconds=2, enforce_freshness=False)
    session.poll_once(adapter, models, MONITORING, rules, limit=2, now=NOW)
    assert len(session.pending) == 1
    session.poll_once(adapter, models, MONITORING, rules, limit=1, now=NOW + pd.Timedelta(seconds=3))
    row = session.history().iloc[0]
    assert row.glucose_status == "invalid_input" and np.isnan(row.glucose_prediction_g_l)
    assert "after the input wait deadline" in row.quality_flags
    assert row.input_deadline_utc == NOW + pd.Timedelta(seconds=2)


@pytest.mark.parametrize("fault", ["missing_ph", "impossible_do", "missing_channel", "axis_mismatch", "domain"])
def test_bad_measurements_and_model_domain_failures_are_gaps(records, models, fault):
    batch, sensor, spectrum = small_records(records)
    expected = "invalid_input"
    if fault == "missing_ph":
        sensor["data"]["ph"] = None
    elif fault == "impossible_do":
        sensor["data"]["dissolved_oxygen_pct"] = 145
    elif fault == "missing_channel":
        spectrum["data"]["raman_400"] = None
    elif fault == "axis_mismatch":
        spectrum["data"]["raman_1801"] = spectrum["data"].pop("raman_1800")
        expected = "axis_mismatch"
    else:
        for column in spectrum["data"]:
            if column.startswith("raman_"):
                spectrum["data"][column] += 100
        expected = "out_of_domain"
    row = consume([batch, sensor, spectrum], models, monitoring=MonitoringSettings()).history().iloc[0]
    assert row.glucose_status == expected and np.isnan(row.glucose_prediction_g_l)


def test_wall_clock_stale_and_future_inputs_are_suppressed(records, models):
    selected = small_records(records)
    old = consume(selected, models, rules=StreamSettings()).history().iloc[0]
    assert old.glucose_status == "stale_input"
    for record in selected:
        for column in list(record["data"]):
            if column.endswith("timestamp"):
                record["data"][column] = (NOW + pd.Timedelta(seconds=3)).isoformat()
    future = consume(selected, models, rules=StreamSettings()).history().iloc[0]
    assert future.glucose_status == "future_input"
    for record in selected:
        for column in list(record["data"]):
            if column.endswith("timestamp"):
                record["data"][column] = NOW.isoformat()
    current = consume(selected, models, rules=StreamSettings()).history().iloc[0]
    assert current.glucose_status == "ok"


@pytest.mark.parametrize("offset, expected", [(-300, False), (-300.01, True), (2, False), (2.01, True)])
def test_freshness_boundary_is_inclusive(offset, expected):
    assert bool(freshness_errors(NOW + pd.Timedelta(seconds=offset), NOW, StreamSettings())) == expected


@pytest.mark.parametrize("conflict", [True, False])
def test_later_input_conflicts_withdraw_estimates_and_preserve_original_audit(records, models, conflict):
    batch, sensor, spectrum = small_records(records)
    duplicate = json.loads(json.dumps(sensor))
    if not conflict:
        duplicate["event_id"] += ":different-event"
    duplicate["data"]["ph"] += 0.1
    adapter = adapter_for([batch, sensor, spectrum, duplicate])
    session = AcquiredStreamSession("conflict")
    session.poll_once(adapter, models, MONITORING, HISTORICAL, limit=3, now=NOW)
    original = session.history().glucose_prediction_g_l.iloc[0]
    assert np.isfinite(original)
    session.poll_once(adapter, models, MONITORING, HISTORICAL, limit=1, now=NOW + pd.Timedelta(seconds=1))
    row = session.history().iloc[0]
    assert row.glucose_status == "invalidated_input" and np.isnan(row.glucose_prediction_g_l)
    assert row.glucose_original_prediction_g_l == original and row.glucose_original_status == "ok"
    assert len(session.invalidations) == 2 and row.invalidation_reason


def test_identical_redelivery_does_not_repeat_inference(records, models):
    batch, sensor, spectrum = small_records(records)
    session = consume([batch, sensor, spectrum, spectrum], models)
    assert len(session.records) == 1 and session.acquisition.cursor == 4
    assert session.acquisition.ledger().status.eq("duplicate").sum() == 1


def test_assays_are_not_inputs_and_future_results_remain_unavailable(records, models):
    baseline = consume(records, models)
    altered = json.loads(json.dumps(records))
    for record in altered:
        if record["source"] == "assays":
            record["data"]["glucose_reference_g_l"] = 99999
            record["data"]["result_timestamp"] = (NOW + pd.Timedelta(hours=1)).isoformat()
    reviewed = consume(altered, models)
    np.testing.assert_allclose(reviewed.history().glucose_prediction_g_l, baseline.history().glucose_prediction_g_l)
    assert reviewed.available_assays(now=NOW).empty
    assert len(reviewed.available_assays(now=NOW + pd.Timedelta(hours=1))) == 2


def test_sensor_only_stream_still_provides_direct_measurements(records):
    selected = [record for record in records if record["source"] in ("batches", "sensors")]
    session = consume(selected, {"Glucose": None, "Lactate": None})
    assert session.history().empty and len(session.sensor_history(HISTORICAL, now=NOW)) == 25
    missing_model = consume(small_records(records), {"Glucose": None, "Lactate": None}).history().iloc[0]
    assert missing_model.glucose_status == "model_unavailable" and np.isnan(missing_model.glucose_prediction_g_l)


def test_disconnect_keeps_history_and_health_exposes_data_silence(records, models):
    adapter = adapter_for(records)
    session = AcquiredStreamSession("health")
    session.poll_once(adapter, models, MONITORING, HISTORICAL, limit=3, now=NOW)
    original = session.history().copy()
    class Failing:
        def poll(self, cursor, limit):
            raise AcquisitionError("Disconnected")
    assert not session.poll_once(Failing(), models, MONITORING, HISTORICAL, now=NOW + pd.Timedelta(seconds=1))
    pd.testing.assert_frame_equal(session.history(), original)
    assert session.health(HISTORICAL, now=NOW)[0] == "Read failed — paused"
    session.acquisition.last_error = ""
    assert session.health(HISTORICAL, now=NOW + pd.Timedelta(seconds=11))[1] == ["No recent new source records"]


def test_paced_producer_exposes_only_due_records_and_explicit_idle_pages(records):
    clock = [0.0]
    adapter = PacedTestAdapter(records, started_at=NOW, clock=lambda: clock[0])
    first = adapter.poll(0, 100)
    assert len(first.records) == 3
    idle = adapter.poll(first.next_cursor, 100)
    assert idle.idle and idle.has_more and idle.next_cursor == 3
    validate_page(idle, 3, 100)
    assert all(pd.Timestamp(record["data"].get("measurement_timestamp", record["data"].get("start_timestamp"))) == NOW for record in first.records)
    clock[0] = 1
    second = adapter.poll(3, 100)
    assert len(second.records) == 2 and not second.idle
    assert pd.Timestamp(second.records[0]["data"]["measurement_timestamp"]) == NOW + pd.Timedelta(seconds=1)
    clock[0] = 1000
    assert not adapter.poll(5, 100).has_more


def test_quarantined_nonstring_event_identity_does_not_crash_processing(records, models):
    selected = small_records(records)
    selected[1]["event_id"] = ["bad", "id"]
    row = consume(selected, models).history().iloc[0]
    assert row.glucose_status == "invalid_input"


def test_report_contains_only_received_history_and_versioned_revisions(records, models):
    session = AcquiredStreamSession("report")
    session.poll_once(adapter_for(records), models, MONITORING, HISTORICAL, limit=3, now=NOW)
    with ZipFile(BytesIO(acquired_stream_report_zip(session, models, MONITORING, HISTORICAL, {"source": "unit"}, now=NOW))) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["cursor"] == 3 and manifest["models"]["Glucose"]["version"] == models["Glucose"].version
        predictions = pd.read_csv(BytesIO(archive.read("predictions.csv")))
        assert len(predictions) == 1
        assert "glucose_original_prediction_g_l" in predictions
        assays = pd.read_csv(BytesIO(archive.read("available_assays.csv")))
        assert assays.empty
        assert len(pd.read_csv(BytesIO(archive.read("delivery_audit.csv")))) == 3


@pytest.mark.parametrize("kwargs", [{"wait_seconds": -1}, {"max_age_seconds": 0}, {"silence_seconds": np.nan}, {"enforce_freshness": "yes"}])
def test_invalid_stream_rules_are_rejected(kwargs):
    with pytest.raises(ValueError):
        StreamSettings(**kwargs)
