"""Read-only PCA projection of the shared received sensor collection.

No adapter reads, cursor changes, fitting, or prediction-model calls occur here.
Arrival freshness determines scoring eligibility; current freshness is separate,
so an old valid historical score is retained without publishing it as current.
"""

from dataclasses import asdict
from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from src.acquisition import normalize_record
from src.demo_data import source_provenance
from src.modeling_storage import _json_text
from src.monitoring import input_quality_errors
from src.multivariate_monitoring import (KEY_COLUMNS, SENSOR_COLUMNS, monitoring_model_manifest,
    process_observations, score_process_monitor, spe_sensor_contributions)
from src.stream_monitoring import _now, _tables_and_ambiguity, freshness_errors


def score_acquired_sensors(model, session, stream_settings, monitoring_settings, *, now=None):
    now = _now(now)
    acquisition = session.acquisition
    tables, ambiguous = _tables_and_ambiguity(acquisition)
    receipts, sensors = {}, {}
    # Include removed/conflicted originals in the PCA audit as invalidated gaps.
    for delivery in acquisition.audit:
        event_id = delivery["event_id"]
        receipts.setdefault(event_id, pd.Timestamp(delivery["received_at_utc"]))
        if delivery["source"] != "sensors" or event_id in sensors:
            continue
        try:
            source, row, _ = normalize_record(json.loads(delivery["raw_event_json"]))
            if source == "sensors":
                sensors[event_id] = row
        except (ValueError, TypeError, KeyError):
            continue  # Malformed deliveries remain in the original delivery ledger.
    rows, extra_flags, arrival_flags, current_flags = [], [], [], []
    for event_id, sensor in sensors.items():
        measured, received = pd.Timestamp(sensor["measurement_timestamp"]), receipts[event_id]
        batches = [(key, data) for key, data in tables["batches"] if data["batch_id"] == sensor["batch_id"]]
        batch = batches[0] if len(batches) == 1 else None
        flags = []
        if event_id not in acquisition.accepted or event_id in acquisition.conflicted_ids:
            flags.append("Sensor delivery was quarantined, removed or conflicted")
        if event_id in ambiguous:
            flags.append("Ambiguous sensor sample identity or measurement time")
        elapsed = np.nan
        if batch is None or batch[0] in ambiguous:
            flags.append("Missing or ambiguous received batch metadata")
        else:
            elapsed = (measured - pd.Timestamp(batch[1]["start_timestamp"])).total_seconds() / 3600
            if receipts[batch[0]] > received + pd.Timedelta(seconds=stream_settings.wait_seconds):
                flags.append("Batch metadata arrived after the input wait deadline")
        arrival = freshness_errors(measured, received, stream_settings, "Sensor on arrival")
        current = freshness_errors(measured, now, stream_settings, "Sensor now")
        flags.extend(input_quality_errors(pd.DataFrame([sensor]), SENSOR_COLUMNS, measured, measured, 0,
            monitoring_settings.input_sensor_ranges))
        rows.append({"sensor_event_id": event_id, "sample_id": sensor["sample_id"], "batch_id": sensor["batch_id"],
            "elapsed_time_h": elapsed, "measurement_timestamp": measured, "received_at_utc": received,
            **{column: sensor[column] for column in SENSOR_COLUMNS}})
        extra_flags.append(flags)
        arrival_flags.append(arrival)
        current_flags.append(current)
    raw = pd.DataFrame(rows, columns=["sensor_event_id", *KEY_COLUMNS, "received_at_utc", *SENSOR_COLUMNS])
    working = raw.copy()
    # The shared protocol scopes sample identity by batch. Only the validation
    # key is composed; the original IDs are restored in every published table.
    working["sample_id"] = working.batch_id.astype(str) + "::" + working.sample_id.astype(str)
    screened = process_observations(working)
    reasons = ["; ".join(dict.fromkeys(([screened.iloc[i].quality_flags] if screened.iloc[i].quality_flags else [])
               + extra_flags[i] + arrival_flags[i])) for i in range(len(raw))]
    valid = pd.Series([not reason for reason in reasons], index=raw.index, dtype=bool)
    scored = score_process_monitor(model, working.loc[valid])
    scored.index = raw.index[valid]
    scored = scored.reindex(raw.index)
    for column in raw:
        scored[column] = raw[column]
    scored["role"] = "Acquired monitoring"
    scored["status"] = ["scored" if not reasons[i] else "invalidated_input" if raw.iloc[i].sensor_event_id not in acquisition.accepted
                        else "stale_or_future_on_arrival" if arrival_flags[i] else "invalid_input" for i in range(len(raw))]
    scored["quality_flags"] = pd.Series(reasons, index=raw.index, dtype=str)
    scored["arrival_freshness_flags"] = pd.Series(["; ".join(flags) for flags in arrival_flags], index=raw.index, dtype=str)
    scored["freshness_flags"] = pd.Series(["; ".join(flags) for flags in current_flags], index=raw.index, dtype=str)
    scored["freshness_status"] = ["not_enforced" if not stream_settings.enforce_freshness else "stale_or_future" if flags else "fresh" for flags in current_flags]
    _, health = session.health(stream_settings, now=now)
    health = list(dict.fromkeys(list(health) + ([acquisition.last_error] if acquisition.last_error else [])))
    scored["source_health_flags"] = "; ".join(health)
    scored["current_eligible"] = valid & scored.freshness_flags.eq("") & (not health) & stream_settings.enforce_freshness
    scored["pca_model_id"] = model.model_id
    return scored


def acquired_sensor_contributions(model, scored, event_id):
    row = scored.loc[scored.sensor_event_id.eq(event_id)]
    if len(row) != 1 or row.iloc[0].status != "scored":
        raise ValueError("Choose an observation with a valid recorded PCA score.")
    return spe_sensor_contributions(model, row)


def acquired_pca_report_zip(model, session, scored, rules, monitoring, *, now=None):
    now = _now(now)
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        manifest = {"pca_reference": monitoring_model_manifest(model), "source_id": session.acquisition.source_id,
            "cursor": session.acquisition.cursor, "acquisition_context": session.context_id, "as_of_utc": now.isoformat(),
            "stream_rules": asdict(rules), "quality_rules": asdict(monitoring),
            "data_provenance": source_provenance(session.acquisition.snapshot().batches),
            "scope": "Received sensor deliveries only; historical reference audit is separate. No new calibration or independent validation.",
            "freshness": "Invalid or stale-on-arrival rows are unscorable. Later staleness/read failure withholds current eligibility while retaining valid historical scores."}
        archive.writestr("manifest.json", _json_text(manifest))
        archive.writestr("acquired/observations.csv", scored.to_csv(index=False))
        archive.writestr("acquired/delivery_audit.csv", session.acquisition.ledger().to_csv(index=False))
        archive.writestr("historical_reference/reference_audit.csv", model.reference_audit.to_csv(index=False))
        valid = scored.loc[scored.status.eq("scored")]
        if not valid.empty:
            records = []
            for _, row in valid.iterrows():
                contribution = acquired_sensor_contributions(model, scored, row.sensor_event_id)
                contribution["sensor_event_id"] = row.sensor_event_id
                records.append(contribution)
            archive.writestr("acquired/spe_contributions.csv", pd.concat(records, ignore_index=True).to_csv(index=False))
    return output.getvalue()
