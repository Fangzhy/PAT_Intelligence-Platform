"""Read-only reliability of shared received predictions and delayed assays."""

from dataclasses import asdict
from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from src.acquisition import normalize_record
from src.acquired_reliability_storage import artifact_context
from src.modeling_data import SENSOR_COLUMNS
from src.modeling_storage import _json_text
from src.reliability import (HISTORY_COLUMNS, available_reference_residuals, distribution_shift,
    input_distribution_shift, residual_metrics)
from src.stream_monitoring import _now, _tables_and_ambiguity, freshness_errors


def acquired_target_history(session, bundle, stream_rules, *, now=None, batch_id=None, window_hours=24.0):
    """Reuse recorded inference only, preserving gaps and current health flags."""
    now = _now(now)
    if not np.isfinite(window_hours) or window_hours <= 0:
        raise ValueError("Review window must be finite and positive.")
    history = session.history().copy()
    prefix = bundle.target.lower() + "_"
    if history.empty:
        return pd.DataFrame(columns=[*HISTORY_COLUMNS, "received_at_utc", "assessed_at_utc", "spectrum_event_id",
                                     "freshness_flags", "source_health_flags", "current_eligible"])
    for column in HISTORY_COLUMNS:
        if prefix + column in history:
            history[column] = history[prefix + column]
    history["prediction_status"] = history[prefix + "status"]
    history["prediction_flags"] = history[prefix + "flags"]
    # Session contexts pin the selected artifacts. Also reject any mismatched
    # record rather than borrowing evidence from a different model version.
    mismatch = history.model_version.ne(bundle.version)
    history.loc[mismatch, "prediction_status"] = "model_identity_mismatch"
    history.loc[mismatch, "prediction_g_l"] = np.nan
    history = history.loc[pd.to_datetime(history.received_at_utc, utc=True) <= now].copy()
    if batch_id is not None:
        history = history.loc[history.batch_id.eq(batch_id)].copy()
    if history.empty:
        history["freshness_flags"], history["source_health_flags"], history["current_eligible"] = "", "", False
        return history
    anchor = min(now, history.measurement_timestamp.max())
    lower = anchor - pd.Timedelta(hours=window_hours)
    history = history.loc[history.measurement_timestamp >= lower].copy()
    # Defensive duplicate handling: preserve every row in the audit but prevent
    # duplicate sample identities from entering metrics or drift distributions.
    duplicate = history.duplicated(["batch_id", "sample_id"], keep=False)
    history.loc[duplicate, "prediction_status"] = "ambiguous_prediction"
    history.loc[duplicate, "prediction_g_l"] = np.nan
    not_yet_available = (history.measurement_timestamp > now) | (history.assessed_at_utc.notna() & (history.assessed_at_utc > now))
    history.loc[not_yet_available, "prediction_status"] = "not_yet_available"
    history.loc[not_yet_available, "prediction_g_l"] = np.nan
    history["freshness_flags"] = ["; ".join(freshness_errors(row.measurement_timestamp, now, stream_rules, "Spectrum")
        + freshness_errors(row.sensor_timestamp, now, stream_rules, "Sensor")) for row in history.itertuples()]
    _, health = session.health(stream_rules, now=now)
    history["source_health_flags"] = "; ".join(health)
    history["freshness_status"] = ["not_enforced" if not stream_rules.enforce_freshness else "stale_or_future" if flags else "fresh" for flags in history.freshness_flags]
    history["current_eligible"] = (history.prediction_status.eq("ok") & history.freshness_flags.eq("")
                                   & (not health) & stream_rules.enforce_freshness)
    return history.reset_index(drop=True)


def received_assay_residuals(session, history, target, rules, *, now=None):
    """Pair at max(lab result, receipt, prediction assessment), never backdate."""
    now = _now(now)
    _, ambiguous = _tables_and_ambiguity(session.acquisition)
    originals, receipts = {}, {}
    for delivery in session.acquisition.audit:
        event = delivery["event_id"]
        receipts.setdefault(event, pd.Timestamp(delivery["received_at_utc"]))
        if delivery["source"] != "assays" or event in originals:
            continue
        try:
            source, row, quality = normalize_record(json.loads(delivery["raw_event_json"]))
        except (ValueError, TypeError, KeyError):
            continue  # Raw malformed deliveries remain in the shared ledger.
        if source != "assays":
            continue
        flags = list(quality)
        if event not in session.acquisition.accepted or event in ambiguous:
            flags.append("Assay was quarantined, conflicted, removed or ambiguous")
        originals[event] = {**row, "assay_event_id": event, "assay_received_at_utc": receipts[event],
                            "reference_quality_flags": "; ".join(flags)}
    columns = ["sample_id", "batch_id", "collection_timestamp", "result_timestamp",
               "glucose_reference_g_l", "lactate_reference_g_l", "assay_event_id",
               "assay_received_at_utc", "reference_quality_flags"]
    assays = pd.DataFrame(originals.values(), columns=columns)
    for column in ("collection_timestamp", "result_timestamp", "assay_received_at_utc"):
        assays[column] = pd.to_datetime(assays[column], utc=True, format="mixed")
    # Restrict laboratory evidence to observations in the selected review window.
    if not history.empty:
        assays = assays.loc[assays.batch_id.isin(history.batch_id) & assays.collection_timestamp.between(
            history.measurement_timestamp.min(), history.measurement_timestamp.max())].copy()
    else:
        assays = assays.iloc[:0].copy()
    assays["laboratory_result_timestamp"] = assays.result_timestamp
    assays["result_timestamp"] = assays[["result_timestamp", "assay_received_at_utc"]].max(axis=1)
    # Missing predictions remain explicit unmatched references. Duplicate
    # prediction identities are already marked ambiguous in the displayed audit.
    pairing = history.drop_duplicates(["batch_id", "sample_id"]).copy()
    assessed = {(row.batch_id, row.sample_id): row.assessed_at_utc for row in pairing.itertuples()}
    assays["result_timestamp"] = [max(time, assessed.get((batch, sample), pd.NaT))
        if pd.notna(assessed.get((batch, sample), pd.NaT)) else time
        for time, batch, sample in zip(assays.result_timestamp, assays.batch_id, assays.sample_id)]
    reference_column = "glucose_reference_g_l" if target == "Glucose" else "lactate_reference_g_l"
    assays.loc[assays.reference_quality_flags.ne(""), reference_column] = np.nan
    residuals = available_reference_residuals(pairing, assays, target, now, rules)
    residuals["pair_available_at_utc"] = residuals.result_timestamp
    residuals["result_timestamp"] = residuals.laboratory_result_timestamp
    return residuals


def acquired_input_rows(session, history, bundle):
    """Original aligned raw values for descriptive drift; no transforms/predict."""
    rows = []
    axis = [f"raman_{value:.12g}" for value in bundle.metadata["spectral_axis_cm1"]]
    for record in history.itertuples():
        if record.alignment_status != "aligned" or record.invalidation_reason or record.prediction_status in ("ambiguous_prediction", "model_identity_mismatch", "not_yet_available"):
            continue
        spectrum = session.acquisition.accepted.get(record.spectrum_event_id)
        sensor = session.acquisition.accepted.get(record.sensor_event_id)
        if spectrum is None or sensor is None:
            continue
        raw = {key: value for key, value in spectrum[1].items() if key.startswith("raman_")}
        canonical = {f"raman_{float(key.removeprefix('raman_')):.12g}": value for key, value in raw.items()}
        compatible = set(canonical) == set(axis)
        rows.append({**{key: sensor[1][key] for key in SENSOR_COLUMNS},
                     **{key: canonical.get(key, np.nan) if compatible else np.nan for key in axis}})
    return pd.DataFrame(rows, columns=[*SENSOR_COLUMNS, *axis])


def acquired_reliability_review(session, bundle, rules, stream_rules, reference=None, *, now=None, batch_id=None, window_hours=24.0):
    now = _now(now)
    history = acquired_target_history(session, bundle, stream_rules, now=now, batch_id=batch_id, window_hours=window_hours)
    residuals = received_assay_residuals(session, history, bundle.target, rules, now=now)
    inputs = acquired_input_rows(session, history, bundle)
    if reference is not None and reference["artifact"] != artifact_context(bundle):
        raise ValueError("Training baseline belongs to a different model artifact.")
    if reference is None:
        input_shift = pd.DataFrame(columns=["feature", "status", "shift_flag"])
        prediction_shift = input_shift.copy()
    else:
        input_shift = input_distribution_shift(pd.DataFrame(reference["input_features"]), inputs, rules)
        prediction_shift = distribution_shift(pd.DataFrame({"prediction_g_l": reference["predictions_g_l"]}),
            history.loc[history.prediction_status.eq("ok"), ["prediction_g_l"]], rules)
    return {"history": history, "residuals": residuals, "metrics": residual_metrics(residuals),
        "input_shift": input_shift, "prediction_shift": prediction_shift,
        "context": {"artifact": artifact_context(bundle), "cursor": session.acquisition.cursor,
            "source_id": session.acquisition.source_id, "session_context": session.context_id,
            "as_of_utc": now.isoformat(), "batch_id": batch_id, "window_hours": window_hours,
            "observed_from_utc": history.measurement_timestamp.min().isoformat() if not history.empty else None,
            "observed_through_utc": history.measurement_timestamp.max().isoformat() if not history.empty else None,
            "input_drift_rows": len(inputs), "baseline_available": reference is not None,
            "reference_sha256": reference["sha256"] if reference else None,
            "rules": asdict(rules), "stream_rules": asdict(stream_rules)}}


def acquired_reliability_report_zip(review, session, bundle, reference):
    """Received-only operational tables, separate from saved validation metadata."""
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", _json_text(review["context"]))
        frozen = {key: bundle.metadata.get(key) for key in ("study_id", "cohort_id", "target", "validation_status", "cv_metrics", "holdout_metrics", "partition", "prediction_reliability")}
        archive.writestr("frozen_validation/recorded_evidence.json", _json_text(frozen))
        if reference:
            archive.writestr("training_reference/reference.json", _json_text(reference))
        for name in ("history", "residuals", "metrics", "input_shift", "prediction_shift"):
            archive.writestr(f"operational_review/{name}.csv", review[name].to_csv(index=False))
        ledger = session.acquisition.ledger()
        if not ledger.empty:
            ledger = ledger.loc[pd.to_datetime(ledger.received_at_utc, utc=True) <= pd.Timestamp(review["context"]["as_of_utc"])]
        archive.writestr("received/delivery_audit.csv", ledger.to_csv(index=False))
        archive.writestr("README.txt", "Operational assessed-pair errors are separate from frozen independent validation. No training, refitting, repeated inference or holdout evaluation. Pair availability is the latest laboratory result, receipt and prediction assessment time. Drift alone does not establish degradation. Empirical demonstration rules; no manufacturing validation.")
    return output.getvalue()
