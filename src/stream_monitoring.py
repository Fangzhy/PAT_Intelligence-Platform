"""Online acquisition, causal input assembly, and frozen-pipeline inference.

Only already received records participate. Predictions are attempted once per
spectrum; later identity conflicts invalidate publication while retaining the
original estimate and an explicit revision audit. No fitting or imputation.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from src.acquisition import AcquisitionSession
from src.data_integration import _utc_timestamp
from src.modeling_data import SENSOR_COLUMNS, TARGETS
from src.modeling_storage import _json_text
from src.monitoring import PredictionResult, alarm_state, input_quality_errors, predict_observation
from src.prediction_reliability import reliability_record
from src.demo_data import source_provenance
from src.input_policy import shared_input_policy, read_input_policy
from src.monitoring_pipeline import integrate_received_sources, transformed_observation


@dataclass(frozen=True)
class StreamSettings:
    wait_seconds: float = 2.0
    enforce_freshness: bool = True
    max_age_seconds: float = 300.0
    future_tolerance_seconds: float = 2.0
    silence_seconds: float = 10.0

    def __post_init__(self):
        for name in ("wait_seconds", "max_age_seconds", "future_tolerance_seconds", "silence_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0 or (name in ("max_age_seconds", "silence_seconds") and value == 0):
                raise ValueError(f"{name} must be finite and {'positive' if name in ('max_age_seconds', 'silence_seconds') else 'nonnegative'}.")
        if not isinstance(self.enforce_freshness, bool):
            raise ValueError("enforce_freshness must be boolean.")


def _now(value=None):
    timestamp = pd.Timestamp.now(tz="UTC") if value is None else _utc_timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Stream assessment needs a timezone-aware UTC clock.")
    return timestamp


def freshness_errors(timestamp, now, settings, label="Measurement"):
    if not settings.enforce_freshness:
        return []
    value = _utc_timestamp(timestamp)
    if pd.isna(value):
        return [f"{label} timestamp is unavailable"]
    age = (_now(now) - value).total_seconds()
    if age > settings.max_age_seconds:
        return [f"{label} is stale relative to the current UTC clock"]
    if age < -settings.future_tolerance_seconds:
        return [f"{label} is in the future beyond clock tolerance"]
    return []


def _tables_and_ambiguity(acquisition):
    tables = defaultdict(list)
    identities, times = defaultdict(list), defaultdict(list)
    for event_id, (source, row) in acquisition.accepted.items():
        tables[source].append((event_id, row))
        key = (source, row["batch_id"]) if source == "batches" else (source, row["batch_id"], row["sample_id"])
        identities[key].append(event_id)
        if source != "batches":
            column = "collection_timestamp" if source == "assays" else "measurement_timestamp"
            times[(source, row["batch_id"], row[column])].append(event_id)
    ambiguous = {event_id for group in [*identities.values(), *times.values()] if len(group) > 1 for event_id in group}
    return tables, ambiguous


def _suppressed(target, bundle, status, flags):
    return PredictionResult(target, np.nan, status, tuple(flags), bundle.version if bundle else "Unavailable",
        bundle.metadata.get("validation_status", "Not independently validated") if bundle else "No model loaded",
        np.nan, pd.NaT, 0.0)


def _result_fields(result, settings):
    prefix = result.target.lower()
    return {
        **reliability_record(result, f"{prefix}_"),
        f"{prefix}_prediction_g_l": result.prediction_g_l,
        f"{prefix}_original_prediction_g_l": result.prediction_g_l,
        f"{prefix}_status": result.status, f"{prefix}_original_status": result.status,
        f"{prefix}_flags": "; ".join(result.flags), f"{prefix}_model_version": result.model_version,
        f"{prefix}_validation_status": result.validation_status,
        f"{prefix}_prediction_time_utc": result.prediction_time_utc,
        f"{prefix}_max_abs_z": result.max_abs_z,
        f"{prefix}_alarm": alarm_state(result.prediction_g_l, settings.prediction_limits[result.target]) if result.status == "ok" else "Not assessed",
    }


@dataclass
class AcquiredStreamSession:
    context_id: str
    acquisition: AcquisitionSession = field(init=False)
    records: dict = field(default_factory=dict)
    pending: dict = field(default_factory=dict)
    pending_views: dict = field(default_factory=dict)
    invalidations: list = field(default_factory=list)
    first_poll_utc: pd.Timestamp | None = None
    last_new_data_utc: pd.Timestamp | None = None

    def __post_init__(self):
        self.acquisition = AcquisitionSession(self.context_id)

    def pause(self):
        self.acquisition.running = False

    def poll_once(self, adapter, models, monitoring_settings, stream_settings, *, limit=10, now=None):
        requested_at = _now(now)
        if self.first_poll_utc is None:
            self.first_poll_utc = requested_at
        before = set(self.acquisition.accepted)
        success = self.acquisition.poll_once(adapter, limit, received_at=now)
        # With a real clock, measure receipt/assessment after a possibly slow
        # HTTP read. Explicit clocks remain deterministic in numerical tests.
        now = _now(now)
        if set(self.acquisition.accepted) - before:
            self.last_new_data_utc = now
        # Even idle/failed reads can expire already pending inputs. New values
        # are never inferred from records which were not previously received.
        self.process(models, monitoring_settings, stream_settings, now=now)
        return success

    def process(self, models, monitoring_settings, stream_settings, *, now=None):
        now = _now(now)
        if any(bundle is not None and bundle.metadata.get("input_policy") for bundle in models.values()):
            policy, _ = shared_input_policy(models)
            config, ranges = read_input_policy(policy)
            monitoring_settings = replace(monitoring_settings,
                max_sensor_age_minutes=config.sensor_tolerance_minutes, input_sensor_ranges=ranges)
        tables, ambiguous = _tables_and_ambiguity(self.acquisition)
        receipts = {}
        for delivery in self.acquisition.audit:
            if isinstance(delivery["event_id"], str):
                receipts.setdefault(delivery["event_id"], pd.Timestamp(delivery["received_at_utc"]))
        for event_id, row in tables["raman"]:
            if event_id not in self.records and event_id not in self.pending:
                self.pending[event_id] = {"row": dict(row), "received": receipts[event_id]}
        for event_id, record in self.records.items():
            if record["invalidation_reason"]:
                continue
            used = [record[key] for key in ("spectrum_event_id", "sensor_event_id", "batch_event_id") if record[key]]
            if any(key not in self.acquisition.accepted or key in ambiguous for key in used):
                reason = "Previously used input was removed or became ambiguous/conflicting"
                record.update(invalidation_reason=reason, invalidated_at_utc=now, quality_flags=reason)
                for target in TARGETS:
                    prefix = target.lower()
                    self.invalidations.append({"spectrum_event_id": event_id, "target": target,
                        "invalidated_at_utc": now, "model_version": record[f"{prefix}_model_version"],
                        "original_prediction_g_l": record[f"{prefix}_original_prediction_g_l"], "reason": reason})
                    record[f"{prefix}_prediction_g_l"] = np.nan
                    record[f"{prefix}_status"] = "invalidated_input"
                    record[f"{prefix}_flags"] = reason
                    record[f"{prefix}_alarm"] = "Not assessed"
        for event_id in list(self.pending):
            pending = self.pending[event_id]
            row = pending["row"]
            measured = pd.Timestamp(row["measurement_timestamp"])
            batches = [(key, data) for key, data in tables["batches"] if data["batch_id"] == row["batch_id"]]
            sensors = [(key, data) for key, data in tables["sensors"] if data["batch_id"] == row["batch_id"] and pd.Timestamp(data["measurement_timestamp"]) <= measured]
            sensor = max(sensors, key=lambda item: pd.Timestamp(item[1]["measurement_timestamp"])) if sensors else None
            batch = batches[0] if len(batches) == 1 else None
            age = (measured - pd.Timestamp(sensor[1]["measurement_timestamp"])).total_seconds() / 60 if sensor else np.nan
            reasons = []
            if event_id not in self.acquisition.accepted or event_id in ambiguous:
                reasons.append("Spectrum identity was removed or is ambiguous")
            if len(batches) > 1 or (batch and batch[0] in ambiguous):
                reasons.append("Batch metadata is ambiguous")
            if sensor and sensor[0] in ambiguous:
                reasons.append("Latest backward sensor identity/time is ambiguous")
            deadline = pending["received"] + pd.Timedelta(seconds=stream_settings.wait_seconds)
            if sensor and receipts[sensor[0]] > deadline:
                reasons.append("Sensor arrived after the input wait deadline")
            if batch and receipts[batch[0]] > deadline:
                reasons.append("Batch metadata arrived after the input wait deadline")
            waiting = []
            if not batch:
                waiting.append("Waiting for batch metadata")
            if not sensor:
                waiting.append("Waiting for a sensor at or before this spectrum")
            elif age > monitoring_settings.max_sensor_age_minutes:
                waiting.append("Waiting for a fresher backward sensor")
            expired = self.acquisition.complete or (now - pending["received"]).total_seconds() >= stream_settings.wait_seconds
            record = self._base_record(event_id, pending, sensor, batch, age, monitoring_settings)
            record.update(input_deadline_utc=deadline, sensor_received_at_utc=receipts[sensor[0]] if sensor else pd.NaT,
                          batch_received_at_utc=receipts[batch[0]] if batch else pd.NaT)
            if waiting and not reasons and not expired:
                record["quality_flags"] = "; ".join(waiting)
                record["alignment_status"] = "waiting_for_inputs"
                for target in TARGETS:
                    record.update(_result_fields(_suppressed(target, models.get(target), "waiting_for_inputs", waiting), monitoring_settings))
                self.pending_views[event_id] = record
                continue
            reasons.extend(waiting)
            if batch and measured < pd.Timestamp(batch[1]["start_timestamp"]):
                reasons.append("Spectrum precedes batch start")
            columns = [key for key in row if key.startswith("raman_")]
            inputs = pd.DataFrame([{f"raman_{float(key.removeprefix('raman_')):.12g}": row[key] for key in columns} | {key: sensor[1][key] if sensor else np.nan for key in SENSOR_COLUMNS}])
            sensor_time = pd.Timestamp(sensor[1]["measurement_timestamp"]) if sensor else pd.NaT
            quality = input_quality_errors(inputs, list(inputs.columns), measured, sensor_time, monitoring_settings.max_sensor_age_minutes, monitoring_settings.input_sensor_ranges)
            reasons.extend(quality)
            freshness = freshness_errors(measured, now, stream_settings, "Spectrum")
            if sensor:
                freshness.extend(freshness_errors(sensor_time, now, stream_settings, "Sensor"))
            reasons.extend(freshness)
            record["quality_flags"] = "; ".join(dict.fromkeys(reasons))
            record["assessed_at_utc"] = now
            record["alignment_status"] = "rejected" if reasons else "aligned"
            axis = np.array([float(key.removeprefix("raman_")) for key in columns])
            status = "future_input" if any("future" in reason for reason in freshness) else "stale_input" if freshness or (sensor and age > monitoring_settings.max_sensor_age_minutes) else "invalid_input"
            for target in TARGETS:
                result = _suppressed(target, models.get(target), status, reasons) if reasons else predict_observation(models.get(target), target, inputs, axis, measured, sensor_time, monitoring_settings)
                record.update(_result_fields(result, monitoring_settings))
            self.records[event_id] = record
            self.pending.pop(event_id)
            self.pending_views.pop(event_id, None)

    def _base_record(self, event_id, pending, sensor, batch, age, settings):
        row = pending["row"]
        measured = pd.Timestamp(row["measurement_timestamp"])
        record = {"sample_id": row["sample_id"], "batch_id": row["batch_id"], "measurement_timestamp": measured,
            "received_at_utc": pending["received"], "elapsed_time_h": (measured - pd.Timestamp(batch[1]["start_timestamp"])).total_seconds() / 3600 if batch else np.nan,
            "spectrum_event_id": event_id, "sensor_event_id": sensor[0] if sensor else "", "batch_event_id": batch[0] if batch else "",
            "sensor_timestamp": pd.Timestamp(sensor[1]["measurement_timestamp"]) if sensor else pd.NaT,
            "sensor_age_minutes": age, "alignment_status": "", "quality_flags": "",
            "invalidation_reason": "", "invalidated_at_utc": pd.NaT}
        record["assessed_at_utc"] = pd.NaT
        for column in SENSOR_COLUMNS:
            record[column] = sensor[1][column] if sensor else np.nan
            record[f"{column}_alarm"] = alarm_state(record[column], settings.sensor_limits[column])
        return record

    def history(self):
        rows = list(self.records.values()) + list(self.pending_views.values())
        return pd.DataFrame(rows).sort_values(["measurement_timestamp", "spectrum_event_id"]).reset_index(drop=True) if rows else pd.DataFrame()

    def sensor_history(self, stream_settings, *, now=None, monitoring_settings=None):
        now = _now(now)
        tables, ambiguous = _tables_and_ambiguity(self.acquisition)
        rows = []
        for event_id, data in tables["sensors"]:
            record = dict(data)
            flags = freshness_errors(record["measurement_timestamp"], now, stream_settings, "Sensor")
            if event_id in ambiguous:
                flags.append("Ambiguous sensor identity/time")
            frame = pd.DataFrame([data])
            flags.extend(input_quality_errors(frame, SENSOR_COLUMNS, data["measurement_timestamp"], data["measurement_timestamp"], 0,
                monitoring_settings.input_sensor_ranges if monitoring_settings is not None else None))
            record.update(sensor_event_id=event_id, sensor_quality_flags="; ".join(flags))
            record["measurement_timestamp"] = pd.Timestamp(record["measurement_timestamp"])
            rows.append(record)
        return pd.DataFrame(rows).sort_values("measurement_timestamp").reset_index(drop=True) if rows else pd.DataFrame(columns=["sample_id", "batch_id", "measurement_timestamp", *SENSOR_COLUMNS, "sensor_event_id", "sensor_quality_flags"])

    def available_assays(self, *, now=None):
        now = _now(now)
        snapshot = self.acquisition.snapshot()
        assays = snapshot.assays.copy()
        assays["collection_timestamp"] = pd.to_datetime(assays.collection_timestamp, utc=True)
        assays["result_timestamp"] = pd.to_datetime(assays.result_timestamp, utc=True)
        assays = assays[assays.result_timestamp <= now].copy()
        starts = {row["batch_id"]: pd.Timestamp(row["start_timestamp"]) for _, (table, row) in self.acquisition.accepted.items() if table == "batches"}
        assays["collection_elapsed_time_h"] = [(collected - starts[batch]).total_seconds() / 3600 if batch in starts else np.nan for batch, collected in zip(assays.batch_id, assays.collection_timestamp)]
        return assays

    def health(self, stream_settings, *, now=None):
        now = _now(now)
        if self.acquisition.last_error:
            return "Read failed — paused", [self.acquisition.last_error]
        last = self.last_new_data_utc if self.last_new_data_utc is not None else self.first_poll_utc
        warnings = ["No recent new source records"] if last is not None and (now - last).total_seconds() > stream_settings.silence_seconds else []
        status = "Complete" if self.acquisition.complete else "Receiving" if self.acquisition.running else "Paused"
        return status, warnings


def acquired_stream_report_zip(session, models, monitoring_settings, stream_settings, context, *, now=None):
    now = _now(now)
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        sources = session.acquisition.snapshot()
        for name, frame in sources.as_dict().items():
            archive.writestr(f"received/{name}.csv", frame.to_csv(index=False))
        try:
            policy = context.get("input_policy") or shared_input_policy(models)[0]
            archive.writestr("processing/input_policy.json", _json_text(policy))
            if not sources.raman.empty:
                integrated = integrate_received_sources(sources, policy)
                for name in ("integrated", "issues", "missing_summary", "outlier_summary"):
                    archive.writestr(f"processing/{name}.csv", getattr(integrated, name).to_csv(index=False))
                for name, frame in integrated.processed.as_dict().items():
                    archive.writestr(f"processing/{name}.csv", frame.to_csv(index=False))
            history = session.history()
            if not history.empty:
                aligned = history[history.alignment_status.eq("aligned") & history.invalidation_reason.eq("")]
                if not aligned.empty:
                    event = aligned.iloc[-1].spectrum_event_id
                    for target, bundle in models.items():
                        if bundle is not None:
                            try:
                                frame = transformed_observation(session, bundle, event)
                                frame["spectrum_event_id"] = event
                                archive.writestr(f"processing/latest_{target.lower()}_features.csv", frame.to_csv(index=False))
                            except (ValueError, TypeError, KeyError) as error:
                                archive.writestr(f"processing/{target.lower()}_features_unavailable.txt", str(error))
        except (ValueError, TypeError, KeyError) as error:
            archive.writestr("processing/unavailable.txt", str(error))
        for name, frame in (("predictions", session.history()), ("sensors", session.sensor_history(stream_settings, now=now, monitoring_settings=monitoring_settings)),
            ("available_assays", session.available_assays(now=now)), ("delivery_audit", session.acquisition.ledger()),
            ("prediction_invalidations", pd.DataFrame(session.invalidations))):
            archive.writestr(f"{name}.csv", frame.to_csv(index=False))
        archive.writestr("manifest.json", _json_text({"context": context, "exported_at_utc": now.isoformat(),
            "data_provenance": source_provenance(session.acquisition.snapshot().batches),
            "source_id": session.acquisition.source_id, "cursor": session.acquisition.cursor,
            "complete": session.acquisition.complete, "last_error": session.acquisition.last_error,
            "first_poll_utc": str(session.first_poll_utc), "last_new_data_utc": str(session.last_new_data_utc),
            "monitoring_rules": asdict(monitoring_settings), "stream_rules": asdict(stream_settings),
            "models": {target: {"version": bundle.version, "artifact_identity": bundle.metadata.get("artifact_identity"),
                "artifact_stamp": bundle.metadata.get("artifact_stamp")} if bundle else None for target, bundle in models.items()},
            "alignment": "Same batch, latest already received sensor at or before spectrum; bounded wait; no interpolation or refit.",
            "history": "Published estimates may be invalidated by later conflicts; original estimates/status and revision audit are retained.",
            "scope": "Received deliveries only. Assays are display/reliability references, never inference inputs. Test sources; no equipment control."}))
    return output.getvalue()
