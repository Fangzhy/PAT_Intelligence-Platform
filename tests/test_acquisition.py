"""Adapter contract, cursors, identity, quarantine, source handoff, and export."""

from dataclasses import replace
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from src.acquisition import (
    AcquisitionError, AcquisitionPage, AcquisitionSession, InMemoryTestAdapter,
    LocalHTTPAdapter, acquisition_report_zip, normalize_record,
    test_source_records as source_records, validate_page,
)
from src.data_generation import SimulationConfig, generate_dataset
from src.data_integration import integrate_sources, sources_from_dataset


@pytest.fixture(scope="module")
def sources():
    return sources_from_dataset(generate_dataset(SimulationConfig(n_batches=1, duration_h=24, n_wavenumbers=32)))


@pytest.fixture
def adapter(sources):
    return InMemoryTestAdapter(source_records(sources))


def collect(adapter, size=10):
    session = AcquisitionSession("unit")
    while not session.complete:
        assert session.poll_once(adapter, size, received_at="2026-11-03T12:00:00Z"), session.last_error
    return session


def sensor_event(sources):
    return next(record for record in source_records(sources) if record["source"] == "sensors")


def test_adapter_is_deterministic_returns_copies_and_preserves_original_sources(sources, adapter):
    originals = {name: frame.copy(deep=True) for name, frame in sources.as_dict().items()}
    first = adapter.poll(0, 5)
    assert first == InMemoryTestAdapter(source_records(sources)).poll(0, 5)
    assert first.next_cursor == 5 and first.has_more
    first.records[1]["data"]["ph"] = 100
    assert adapter.poll(0, 5).records[1]["data"]["ph"] != 100
    for name, frame in sources.as_dict().items():
        pd.testing.assert_frame_equal(frame, originals[name])


def test_full_acquisition_roundtrips_through_existing_integration(sources, adapter):
    session = collect(adapter)
    snapshot = session.snapshot()
    assert session.cursor == len(adapter.records) and not session.running
    assert len(session.accepted) == 53  # Batch + 25 sensors + 25 spectra + two assays.
    assert session.ledger().status.eq("accepted").all()
    result = integrate_sources(snapshot)
    expected = integrate_sources(sources)
    assert len(result.integrated) == len(expected.integrated) == 25
    assert result.integrated.sensor_matched.all()
    assert len(result.reference_pairs) == 2
    np.testing.assert_allclose(result.integrated.filter(regex="^raman_"), expected.integrated.filter(regex="^raman_"))
    assert not {"glucose_g_l", "lactate_g_l", "viable_cell_density_million_ml"} & set(snapshot.sensors)


def test_assays_are_delivered_at_result_time_and_receipt_time_is_separate(sources, adapter):
    records = adapter.records
    times = []
    for record in records:
        source, data = record["source"], record["data"]
        time = data["start_timestamp" if source == "batches" else "result_timestamp" if source == "assays" else "measurement_timestamp"]
        times.append(pd.Timestamp(time))
    assert times == sorted(times)
    first_assay = next(index for index, record in enumerate(records) if record["source"] == "assays")
    session = AcquisitionSession("causality")
    assert session.poll_once(adapter, first_assay, received_at="2026-11-03T12:00:00Z")
    assert session.snapshot().assays.empty
    assert session.poll_once(adapter, 1, received_at="2026-11-03T12:00:01Z")
    assert len(session.snapshot().assays) == 1
    assert pd.Timestamp(session.last_success_utc).year == 2026
    assert pd.Timestamp(session.snapshot().assays.collection_timestamp.iloc[0]).year == 2025


def test_duplicate_delivery_is_idempotent_but_retained_in_audit(sources):
    event = sensor_event(sources)
    records = [dict(event, sequence=0), dict(event, sequence=1)]
    session = collect(InMemoryTestAdapter(records))
    assert len(session.snapshot().sensors) == 1
    assert session.ledger().status.tolist() == ["accepted", "duplicate"]
    assert session.cursor == 2


@pytest.mark.parametrize("malformed", [False, True])
def test_conflicting_event_ids_quarantine_both_payloads_and_cannot_reappear(sources, malformed):
    event = sensor_event(sources)
    changed = json.loads(json.dumps(event))
    if malformed:
        del changed["data"]["ph"]
    else:
        changed["data"]["ph"] += 0.1
    records = [dict(event, sequence=0), dict(changed, sequence=1), dict(event, sequence=2)]
    session = collect(InMemoryTestAdapter(records))
    assert session.snapshot().sensors.empty
    assert session.ledger().status.eq("quarantined").all()
    assert session.cursor == 3


def test_distinct_event_ids_with_duplicate_samples_are_exposed_to_phase2(sources):
    records = source_records(sources)
    original = next(record for record in records if record["source"] == "sensors")
    changed = json.loads(json.dumps(original))
    changed["event_id"] += ":another-observation"
    changed["data"]["ph"] += 0.1
    records.append(dict(changed, sequence=len(records)))
    result = integrate_sources(collect(InMemoryTestAdapter(records)).snapshot())
    assert "duplicate_identity" in result.issues.issue.tolist()
    assert len(result.processed.sensors[result.processed.sensors._eligible]) == 24


def test_known_faults_are_quarantined_or_retained_as_quality_flags(sources):
    session = collect(InMemoryTestAdapter(source_records(sources, faults=True)))
    ledger = session.ledger()
    assert ledger.status.eq("duplicate").sum() == 1
    assert ledger.status.eq("quarantined").sum() == 2
    assert ledger["flags"].str.contains("ph").any()
    assert session.snapshot().sensors.ph.isna().sum() == 1


def test_numeric_missing_and_implausible_values_are_not_imputed(sources):
    event = sensor_event(sources)
    event["data"]["ph"] = "invalid"
    event["data"]["dissolved_oxygen_pct"] = 145
    _, row, flags = normalize_record(event)
    assert row["ph"] is None and row["dissolved_oxygen_pct"] == 145
    assert any("ph" in flag for flag in flags) and any("plausibility" in flag for flag in flags)


@pytest.mark.parametrize("fault", ["missing_key", "naive_time", "bad_id", "unknown_column", "unknown_source"])
def test_malformed_structural_records_are_quarantined(sources, fault):
    event = sensor_event(sources)
    if fault == "missing_key":
        del event["data"]["ph"]
    elif fault == "naive_time":
        event["data"]["measurement_timestamp"] = "2025-01-01T00:00:00"
    elif fault == "bad_id":
        event["data"]["sample_id"] = ""
    elif fault == "unknown_column":
        event["data"]["glucose_g_l"] = 8
    else:
        event["source"] = "equipment_command"
    event["sequence"] = 0
    session = collect(InMemoryTestAdapter([event]))
    assert len(session.accepted) == 0 and session.ledger().status.iloc[0] == "quarantined"


def test_raman_channels_are_canonicalized_and_schema_change_is_quarantined(sources):
    event = next(record for record in source_records(sources) if record["source"] == "raman")
    original = json.loads(json.dumps(event))
    event["data"] = dict(reversed(list(event["data"].items())))
    _, row, _ = normalize_record(event)
    coordinates = [float(key.removeprefix("raman_")) for key in row if key.startswith("raman_")]
    assert coordinates == sorted(coordinates)
    original["event_id"] += ":different-axis"
    original["data"]["raman_1801"] = original["data"].pop("raman_1800")
    session = collect(InMemoryTestAdapter([dict(event, sequence=0), dict(original, sequence=1)]))
    assert len(session.snapshot().raman) == 1
    assert "schema changed" in session.ledger()["flags"].iloc[-1]


def test_read_failure_and_source_change_leave_cursor_and_data_intact(adapter):
    session = AcquisitionSession("read")
    assert session.poll_once(adapter, 5)
    previous = session.ledger().copy()
    class Failing:
        def poll(self, cursor, limit):
            raise AcquisitionError("Source disconnected")
    session.running = True
    assert not session.poll_once(Failing(), 5)
    assert session.cursor == 5 and not session.running and session.last_error
    pd.testing.assert_frame_equal(session.ledger(), previous)
    class Changed:
        def poll(self, cursor, limit):
            return replace(adapter.poll(cursor, limit), source_id="different-source")
    assert not session.poll_once(Changed(), 5)
    assert "snapshot changed" in session.last_error and session.cursor == 5
    assert session.poll_once(adapter, 5)
    assert session.cursor == 10 and session.last_error == ""


@pytest.mark.parametrize("fault", ["cursor", "sequence", "version_type", "empty_continuation", "oversized_page"])
def test_protocol_failures_are_atomic_without_partial_page_commits(adapter, fault):
    page = adapter.poll(0, 5)
    if fault == "cursor":
        page = replace(page, next_cursor=100)
    elif fault == "sequence":
        page.records[-1]["sequence"] = 0
    elif fault == "version_type":
        page.records[0]["sequence"] = 0.0
    elif fault == "empty_continuation":
        page = AcquisitionPage(page.source_id, [], 0, True)
    else:
        page = adapter.poll(0, 10)
    class Invalid:
        def poll(self, cursor, limit):
            return page
    session = AcquisitionSession("protocol")
    assert not session.poll_once(Invalid(), 5)
    assert not session.audit and session.cursor == 0 and not session.accepted


def test_buffer_limit_stops_without_consuming_unrecorded_events(adapter):
    session = AcquisitionSession("capacity", max_deliveries=3)
    assert session.poll_once(adapter, 10) and session.cursor == 3
    assert not session.poll_once(adapter, 10)
    assert session.cursor == 3 and "buffer is full" in session.last_error


def test_snapshot_is_detached_and_export_contains_only_polled_records(adapter):
    session = AcquisitionSession("export")
    assert session.poll_once(adapter, 5)
    snapshot = session.snapshot()
    snapshot.sensors.loc[0, "ph"] = 100
    assert session.snapshot().sensors.ph.iloc[0] != 100
    with ZipFile(BytesIO(acquisition_report_zip(session, {"mode": "unit"}))) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["cursor"] == 5 and not manifest["complete"]
        ledger = pd.read_csv(BytesIO(archive.read("delivery_audit.csv")))
        assert len(ledger) == 5
        assays = pd.read_csv(BytesIO(archive.read("sources/assays.csv")))
        assert assays.empty


@pytest.mark.parametrize("url", ["https://127.0.0.1:8765", "http://example.com:8765", "file:///test", "http://127.0.0.1:0", "http://user:password@127.0.0.1:8765", "http://127.0.0.1:8765/other", "http://127.0.0.1:8765?query=1"])
def test_http_adapter_is_limited_to_explicit_loopback_test_endpoints(url):
    with pytest.raises(AcquisitionError, match="loopback"):
        LocalHTTPAdapter(url)
