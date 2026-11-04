"""Read-only test acquisition contract, adapters, and auditable collection.

Transport cursors are independent of source measurement/result timestamps.
These adapters consume source records; they never control equipment or fit models.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
import json
from numbers import Integral
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from src.data_integration import REQUIRED_COLUMNS, SENSOR_RANGES, SourceTables, _utc_timestamp


SCHEMA_VERSION = 1
MAX_PAGE_SIZE = 100
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
AUDIT_COLUMNS = ["sequence", "event_id", "source", "received_at_utc", "status", "flags", "raw_event_json"]
OPTIONAL_COLUMNS = {
    "sensors": {"elapsed_time_h"}, "raman": set(),
    "assays": {"collection_elapsed_time_h"},
    "batches": {"scenario", "description", "disturbance_start_h"},
}


class AcquisitionError(ValueError):
    """A transport/protocol problem which must not advance the consumer cursor."""


def _integer(value, minimum, maximum, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise AcquisitionError(f"{name} must be an integer from {minimum} to {maximum}.")


def _json_value(value):
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat() if pd.notna(value) else None
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _json_text(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class AcquisitionPage:
    source_id: str
    records: list[dict]
    next_cursor: int
    has_more: bool
    idle: bool = False

    def as_dict(self):
        return {"schema_version": SCHEMA_VERSION, "source_id": self.source_id, "records": self.records,
                "next_cursor": self.next_cursor, "has_more": self.has_more, "idle": self.idle}


class AcquisitionAdapter(Protocol):
    def poll(self, cursor: int, limit: int) -> AcquisitionPage:
        """Read a bounded page after a delivery cursor; never acknowledge by time."""
        ...


def validate_page(page, cursor, limit):
    _integer(cursor, 0, 10**12, "Cursor")
    _integer(limit, 1, MAX_PAGE_SIZE, "Page size")
    if not isinstance(page, AcquisitionPage) or not isinstance(page.source_id, str) or not 1 <= len(page.source_id) <= 200:
        raise AcquisitionError("A page needs a nonempty source snapshot identity.")
    if not isinstance(page.records, list) or len(page.records) > limit or not isinstance(page.has_more, bool):
        raise AcquisitionError("Invalid record page size or continuation flag.")
    _integer(page.next_cursor, 0, 10**12, "Next cursor")
    if not isinstance(page.idle, bool) or (page.idle and (page.records or not page.has_more)):
        raise AcquisitionError("Idle pages must be empty and keep the source open.")
    if page.next_cursor != cursor + len(page.records) or (not page.records and page.has_more and not page.idle):
        raise AcquisitionError("The page cursor does not advance by its delivery count.")
    for sequence, record in enumerate(page.records, start=cursor):
        if (not isinstance(record, dict) or not isinstance(record.get("sequence"), Integral)
                or isinstance(record.get("sequence"), bool) or record.get("sequence") != sequence):
            raise AcquisitionError("Record sequences must be consecutive from the requested cursor.")


def test_source_records(sources, *, faults=False):
    """Serialize separate source tables in availability order, excluding truth.

    Laboratory records are delivered at result time, not collection time. Source
    timestamps are historical simulation times; receipt timestamps are separate.
    """
    timed = []
    for priority, (source, frame) in enumerate(sources.as_dict().items()):
        time_column = "start_timestamp" if source == "batches" else "result_timestamp" if source == "assays" else "measurement_timestamp"
        for index, row in enumerate(frame.to_dict(orient="records")):
            data = {key: _json_value(value) for key, value in row.items()}
            event_id = f"{source}:{data.get('batch_id')}:{data.get('sample_id', index)}"
            event = {"event_id": event_id, "source": source, "data": data}
            time = _utc_timestamp(data.get(time_column))
            if pd.isna(time):
                raise AcquisitionError("A test source requires valid source timestamps before fault injection.")
            timed.append((time, -1 if source == "batches" else priority, index, event))
    timed.sort(key=lambda item: item[:3])
    records = [item[3] for item in timed]
    if faults:
        sensor_positions = [index for index, event in enumerate(records) if event["source"] == "sensors"]
        if sensor_positions:
            # Copy by JSON roundtrip, avoiding any mutation of supplied tables.
            records = json.loads(_json_text(records))
            position = sensor_positions[0]
            records.insert(position + 1, json.loads(_json_text(records[position])))
            bad = json.loads(_json_text(records[position]))
            bad["event_id"] += ":bad-time"
            bad["data"]["measurement_timestamp"] = "invalid-time"
            records.insert(position + 2, bad)
            missing = json.loads(_json_text(records[position]))
            missing["event_id"] += ":missing-key"
            del missing["data"]["ph"]
            records.insert(position + 3, missing)
        for event in records:
            if event["source"] == "sensors" and str(event["data"].get("sample_id", "")).endswith("00003"):
                event["data"]["ph"] = None
    return [{"sequence": sequence, **event} for sequence, event in enumerate(records)]


class InMemoryTestAdapter:
    """Independent polling adapter against a frozen JSON-like fixture source."""

    def __init__(self, records):
        self.records = json.loads(_json_text(records))
        self.source_id = "test-" + sha256(_json_text(self.records).encode()).hexdigest()[:24]

    def poll(self, cursor, limit):
        _integer(cursor, 0, len(self.records), "Cursor")
        _integer(limit, 1, MAX_PAGE_SIZE, "Page size")
        records = json.loads(_json_text(self.records[cursor:cursor + limit]))
        return AcquisitionPage(self.source_id, records, cursor + len(records), cursor + len(records) < len(self.records))


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise AcquisitionError("The local test adapter does not follow redirects.")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AcquisitionError("Duplicate JSON object keys are not supported.")
        result[key] = value
    return result


def _invalid_constant(value):
    raise AcquisitionError(f"Nonstandard JSON number: {value}.")


class LocalHTTPAdapter:
    """GET-only JSON polling from an explicitly configured loopback test server."""

    def __init__(self, base_url="http://127.0.0.1:8765", *, timeout_seconds=2.0):
        try:
            url = urlsplit(base_url)
            if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1")
                    or url.username is not None or url.password is not None or url.path not in ("", "/")
                    or url.query or url.fragment or url.port is None or not 1 <= url.port <= 65535):
                raise AcquisitionError("Use a local HTTP test source URL such as http://127.0.0.1:8765.")
        except (ValueError, AttributeError) as error:
            raise AcquisitionError("Use a valid loopback URL with an explicit port.") from error
        if not np.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 5:
            raise AcquisitionError("HTTP timeout must be greater than zero and at most five seconds.")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)

    def poll(self, cursor, limit):
        _integer(cursor, 0, 10**12, "Cursor")
        _integer(limit, 1, MAX_PAGE_SIZE, "Page size")
        request = Request(f"{self.base_url}/v1/records?{urlencode({'after': cursor, 'limit': limit})}", headers={"Accept": "application/json"}, method="GET")
        try:
            with build_opener(ProxyHandler({}), _NoRedirects()).open(request, timeout=self.timeout_seconds) as response:
                if response.headers.get_content_type() != "application/json":
                    raise AcquisitionError("The source did not return application/json.")
                payload = response.read(MAX_RESPONSE_BYTES + 1)
                if len(payload) > MAX_RESPONSE_BYTES:
                    raise AcquisitionError("The test-source response exceeds the 2 MiB page limit.")
            data = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
            if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != SCHEMA_VERSION:
                raise AcquisitionError("Unsupported acquisition schema version.")
            page = AcquisitionPage(data["source_id"], data["records"], data["next_cursor"], data["has_more"], data.get("idle", False))
            validate_page(page, cursor, limit)
            return page
        except (HTTPError, URLError, OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
            raise AcquisitionError(f"Local test source read failed: {error}") from error


def normalize_record(record):
    """Check schema/identity/time; retain numeric quality failures with flags."""
    source, data, event_id = record.get("source"), record.get("data"), record.get("event_id")
    if not isinstance(event_id, str) or not 1 <= len(event_id) <= 200:
        raise ValueError("Missing or invalid event_id")
    if not isinstance(source, str) or source not in REQUIRED_COLUMNS or not isinstance(data, dict):
        raise ValueError("Unknown source table or invalid record data")
    required = set(REQUIRED_COLUMNS[source])
    if not required.issubset(data):
        raise ValueError("Missing required columns: " + ", ".join(sorted(required - set(data))))
    spectral = [column for column in data if column.startswith("raman_")] if source == "raman" else []
    allowed = required | OPTIONAL_COLUMNS[source] | set(spectral)
    if set(data) - allowed:
        raise ValueError("Unsupported columns: " + ", ".join(sorted(set(data) - allowed)))
    row = dict(data)
    for key in ("batch_id",) if source == "batches" else ("batch_id", "sample_id"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"Missing or invalid {key}")
        row[key] = row[key].strip()
    timestamps = [key for key in required if key.endswith("timestamp")]
    for column in timestamps:
        value = _utc_timestamp(row[column])
        if pd.isna(value):
            raise ValueError(f"Invalid or timezone-naive {column}")
        row[column] = value.isoformat()
    if source == "assays" and row["result_timestamp"] < row["collection_timestamp"]:
        raise ValueError("Laboratory result precedes sample collection")
    if source == "raman":
        try:
            axis = np.array([float(column.removeprefix("raman_")) for column in spectral])
        except ValueError as error:
            raise ValueError("Invalid Raman channel headers") from error
        if len(axis) < 2 or not np.isfinite(axis).all() or len(set(axis)) != len(axis):
            raise ValueError("Raman records need at least two distinct finite channel coordinates")
        spectral = [column for _, column in sorted(zip(axis, spectral))]
        row = {key: value for key, value in row.items() if key not in spectral} | {key: row[key] for key in spectral}
    numeric = list(SENSOR_RANGES) if source == "sensors" else ["glucose_reference_g_l", "lactate_reference_g_l"] if source == "assays" else spectral
    flags = []
    for column in numeric:
        try:
            value = float(row[column]) if not isinstance(row[column], bool) else np.nan
        except (ValueError, TypeError):
            value = np.nan
        row[column] = value if np.isfinite(value) else None
        if row[column] is None:
            flags.append(f"Missing/invalid numeric {column}")
        elif column in SENSOR_RANGES and not SENSOR_RANGES[column][0] <= value <= SENSOR_RANGES[column][1]:
            flags.append(f"Sensor plausibility failure: {column}")
        elif source == "assays" and value < 0:
            flags.append(f"Negative reference: {column}")
    return source, row, flags


@dataclass
class AcquisitionSession:
    context_id: str
    max_deliveries: int = 10_000
    cursor: int = 0
    source_id: str | None = None
    running: bool = False
    complete: bool = False
    last_error: str = ""
    last_success_utc: str | None = None
    audit: list[dict] = field(default_factory=list)
    accepted: dict = field(default_factory=dict)
    event_hashes: dict = field(default_factory=dict)
    conflicted_ids: set = field(default_factory=set)
    spectral_columns: tuple | None = None

    def poll_once(self, adapter: AcquisitionAdapter, limit=10, *, received_at=None):
        """Commit validated pages; failed reads leave cursor and collected data intact."""
        if self.complete:
            self.running = False
            return True
        try:
            _integer(limit, 1, MAX_PAGE_SIZE, "Page size")
            remaining = self.max_deliveries - len(self.audit)
            if remaining <= 0:
                raise AcquisitionError("Collection buffer is full. Export or hand off the snapshot, then reset to collect again.")
            page = adapter.poll(self.cursor, min(limit, remaining))
            validate_page(page, self.cursor, min(limit, remaining))
            try:
                _json_text(page.as_dict())
            except (TypeError, ValueError) as error:
                raise AcquisitionError("The adapter returned a page outside the JSON record contract.") from error
            if self.source_id is not None and page.source_id != self.source_id:
                raise AcquisitionError("The source snapshot changed. Reset acquisition before reading the new source.")
            timestamp = pd.Timestamp.now(tz="UTC") if received_at is None else _utc_timestamp(received_at)
            if pd.isna(timestamp):
                raise AcquisitionError("Receipt time must include an explicit timezone.")
        except (AcquisitionError, OSError) as error:
            self.last_error = str(error)
            self.running = False
            return False
        self.source_id = page.source_id
        for record in page.records:
            flags, status = [], "accepted"
            event_id = record.get("event_id")
            try:
                if isinstance(event_id, str) and 1 <= len(event_id) <= 200:
                    fingerprint = sha256(_json_text({"source": record.get("source"), "data": record.get("data")}).encode()).hexdigest()
                    if event_id in self.conflicted_ids:
                        raise ValueError("Event ID has unresolved conflicting payloads")
                    if event_id in self.event_hashes and self.event_hashes[event_id] != fingerprint:
                        self.accepted.pop(event_id, None)
                        self.conflicted_ids.add(event_id)
                        for previous in self.audit:
                            if previous["event_id"] == event_id:
                                previous.update(status="quarantined", flags="Event ID has conflicting payloads")
                        raise ValueError("Event ID has conflicting payloads")
                    self.event_hashes[event_id] = fingerprint
                source, row, flags = normalize_record(record)
                if event_id in self.accepted:
                    status = "duplicate"
                else:
                    columns = tuple(key for key in row if key.startswith("raman_")) if source == "raman" else None
                    if columns is not None and self.spectral_columns is not None and columns != self.spectral_columns:
                        raise ValueError("Raman channel schema changed within one acquisition context")
                    self.accepted[event_id] = (source, row)
                    if columns is not None:
                        self.spectral_columns = columns
            except ValueError as error:
                status, flags = "quarantined", [str(error)]
            self.audit.append({"sequence": record["sequence"], "event_id": event_id, "source": record.get("source"),
                "received_at_utc": timestamp.isoformat(), "status": status, "flags": "; ".join(flags), "raw_event_json": _json_text(record)})
        self.cursor = page.next_cursor
        self.complete = not page.has_more
        self.running = self.running and not self.complete
        self.last_success_utc, self.last_error = timestamp.isoformat(), ""
        return True

    def snapshot(self):
        frames = {}
        for source, required in REQUIRED_COLUMNS.items():
            rows = [dict(row) for table, row in self.accepted.values() if table == source]
            columns = list(required) + list(self.spectral_columns or ()) if source == "raman" else list(required)
            frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
            if source == "raman" and self.spectral_columns:
                frame = frame.reindex(columns=list(required) + list(self.spectral_columns))
            frames[source] = frame
        return SourceTables(**frames)

    def ledger(self):
        return pd.DataFrame(self.audit, columns=AUDIT_COLUMNS)


def acquisition_report_zip(session, context):
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for source, frame in session.snapshot().as_dict().items():
            archive.writestr(f"sources/{source}.csv", frame.to_csv(index=False))
        archive.writestr("delivery_audit.csv", session.ledger().to_csv(index=False))
        archive.writestr("manifest.json", _json_text({"schema_version": SCHEMA_VERSION, "source_id": session.source_id,
            "cursor": session.cursor, "complete": session.complete, "last_success_utc": session.last_success_utc,
            "last_error": session.last_error, "context": context,
            "scope": "Only polled records; source time and receipt time are distinct. Test acquisition, no equipment control.",
            "policy": "Duplicate event IDs are idempotent; conflicting payloads and malformed identities/times are quarantined. Numeric quality failures remain flagged, without imputation."}))
    return output.getvalue()
