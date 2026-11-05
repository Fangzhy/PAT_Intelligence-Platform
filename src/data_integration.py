"""Phase 2: auditable source preparation, batch-aware alignment, and quality flags.

Raw inputs are copied, never cleaned in place. Ambiguous identities/times and
invalid keys/times are excluded from matching, with an issue ledger. Numeric
missingness, plausibility failures, and exploratory outliers are retained.
"""

import csv
from dataclasses import asdict, dataclass
from io import BytesIO, StringIO
import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from src.data_generation import SyntheticDataset
from src.demo_data import PROVENANCE_COLUMNS, source_provenance


SENSOR_RANGES = {
    "temperature_c": (34.0, 40.0),
    "ph": (6.0, 8.0),
    "dissolved_oxygen_pct": (0.0, 100.0),
    "feed_rate_l_h": (0.0, 0.01),
    "agitation_rpm": (0.0, 600.0),
}
REFERENCE_COLUMNS = ["glucose_reference_g_l", "lactate_reference_g_l"]
REQUIRED_COLUMNS = {
    "sensors": ["sample_id", "batch_id", "measurement_timestamp", *SENSOR_RANGES],
    "raman": ["sample_id", "batch_id", "measurement_timestamp"],
    "assays": ["sample_id", "batch_id", "collection_timestamp", "result_timestamp", *REFERENCE_COLUMNS],
    "batches": ["batch_id", "start_timestamp"],
}
ISSUE_COLUMNS = ["source", "row_number", "batch_id", "sample_id", "column", "issue", "severity", "raw_value"]


@dataclass(frozen=True)
class IntegrationConfig:
    sensor_tolerance_minutes: float = 30.0
    assay_tolerance_minutes: float = 30.0
    sensor_direction: str = "backward"
    outlier_threshold: float = 3.5

    def __post_init__(self):
        for name in ("sensor_tolerance_minutes", "assay_tolerance_minutes", "outlier_threshold"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name == "outlier_threshold" and value == 0):
                raise ValueError(f"{name} must be finite and {'positive' if name == 'outlier_threshold' else 'nonnegative'}.")
        if self.sensor_direction not in ("backward", "nearest"):
            raise ValueError("sensor_direction must be backward or nearest.")


@dataclass
class SourceTables:
    sensors: pd.DataFrame
    raman: pd.DataFrame
    assays: pd.DataFrame
    batches: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {name: getattr(self, name) for name in REQUIRED_COLUMNS}


@dataclass
class IntegrationResult:
    raw: SourceTables
    processed: SourceTables
    integrated: pd.DataFrame
    reference_pairs: pd.DataFrame
    issues: pd.DataFrame
    missing_summary: pd.DataFrame
    outlier_summary: pd.DataFrame
    source_summary: pd.DataFrame
    wavenumbers_cm1: np.ndarray
    config: IntegrationConfig
    sensor_ranges: dict
    selected_batches: list[str] | None


def sources_from_dataset(dataset: SyntheticDataset) -> SourceTables:
    """Create four import-compatible sources without including simulated truth.

    Raman CSV headers encode their physical axis, e.g. raman_400, raman_405.
    They are not the generator's ordinal channel names.
    """
    raman = dataset.spectral_metadata[["sample_id", "batch_id", "measurement_timestamp"]].copy()
    intensities = dataset.spectra.loc[raman["sample_id"]].reset_index(drop=True).copy()
    intensities.columns = [f"raman_{value:.12g}" for value in dataset.wavenumbers_cm1]
    return SourceTables(
        sensors=dataset.sensors.copy(deep=True),
        raman=pd.concat([raman.reset_index(drop=True), intensities], axis=1),
        assays=dataset.offline_assays.copy(deep=True),
        batches=dataset.batches.copy(deep=True),
    )


def read_source_csv(content: bytes) -> pd.DataFrame:
    """Read UTF-8 CSV without hiding blanks or converting raw cell contents."""
    try:
        text = content.decode("utf-8-sig")
        reader = csv.reader(StringIO(text), strict=True)
        header = next(reader)
        if not header or any(not column.strip() for column in header) or len(set(header)) != len(header):
            raise ValueError("CSV column names must be nonempty and unique.")
        for row_number, row in enumerate(reader, start=1):
            if row and len(row) != len(header):
                raise ValueError(f"CSV data row {row_number} has {len(row)} fields; expected {len(header)}.")
        return pd.read_csv(StringIO(text), dtype=str, keep_default_na=False)
    except (UnicodeError, StopIteration, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError("Use a nonempty UTF-8, comma-separated CSV with a header row.") from error


def _utc_timestamp(value):
    """Require explicit timezone information instead of guessing a local zone."""
    try:
        if pd.isna(value) or not str(value).strip() or isinstance(value, (int, float)):
            return pd.NaT
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return pd.NaT
        return timestamp.tz_convert("UTC")
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def _blank(values: pd.Series) -> pd.Series:
    return values.isna() | values.astype("string").str.strip().eq("").fillna(True)


def modified_z_scores(values: pd.Series, minimum_observations: int = 8) -> tuple[pd.Series, str]:
    """Exploratory batch-local scores; abstain when MAD is zero or data is sparse."""
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    scores = pd.Series(np.nan, index=values.index, dtype=float)
    valid = numeric.dropna()
    if len(valid) < minimum_observations:
        return scores, "insufficient observations"
    median = valid.median()
    mad = (valid - median).abs().median()
    if mad == 0:
        return scores, "zero MAD; not evaluated"
    scores.loc[valid.index] = 0.6745 * (valid - median) / mad
    return scores, "evaluated"


def align_by_batch(
    left: pd.DataFrame, right: pd.DataFrame, *, left_time: str, right_time: str,
    tolerance_minutes: float, direction: str,
) -> pd.DataFrame:
    """Left-preserving time match within batch, restoring the original left order.

    Inputs must have valid UTC timestamps and unambiguous right-hand times.
    Matching picks an observed row; it does not synthesize or interpolate values.
    """
    if not np.isfinite(tolerance_minutes) or tolerance_minutes < 0:
        raise ValueError("Time tolerance must be finite and nonnegative.")
    if direction not in ("backward", "nearest"):
        raise ValueError("Direction must be backward or nearest.")
    if left[left_time].isna().any() or right[right_time].isna().any():
        raise ValueError("Remove invalid timestamps from matching inputs first.")
    if right.duplicated(["batch_id", right_time]).any():
        raise ValueError("Matching requires unique right-hand timestamps within each batch.")
    ordered = left.copy()
    ordered["_alignment_order"] = np.arange(len(ordered))
    merged = pd.merge_asof(
        ordered.sort_values(left_time, kind="stable"),
        right.sort_values(right_time, kind="stable"),
        by="batch_id", left_on=left_time, right_on=right_time,
        tolerance=pd.Timedelta(minutes=tolerance_minutes), direction=direction,
    )
    return merged.sort_values("_alignment_order").drop(columns="_alignment_order").reset_index(drop=True)


def integrate_sources(
    sources: SourceTables, config: IntegrationConfig = IntegrationConfig(), *,
    sensor_ranges: dict[str, tuple[float, float]] | None = None,
    batch_ids: list[str] | None = None,
) -> IntegrationResult:
    """Validate, flag, and integrate a source snapshot, retaining raw inputs.

    A spectral row anchors the sensor join. Each assay independently seeks a
    spectrum at collection time; sparse references are never carried forward
    over an entire batch. Invalid/ambiguous key rows are quarantined, not guessed.
    """
    ranges = dict(SENSOR_RANGES if sensor_ranges is None else sensor_ranges)
    if set(ranges) != set(SENSOR_RANGES):
        raise ValueError("Supply a lower and upper plausibility bound for each sensor.")
    for column, bounds in ranges.items():
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(f"Invalid plausibility bounds for {column}.")

    raw = SourceTables(**{name: frame.copy(deep=True) for name, frame in sources.as_dict().items()})
    for name, frame in raw.as_dict().items():
        missing_columns = set(REQUIRED_COLUMNS[name]) - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{name}: missing required columns: {', '.join(sorted(missing_columns))}.")
        if not frame.columns.is_unique:
            raise ValueError(f"{name}: column names must be unique.")
    spectral_columns = [column for column in raw.raman if column.startswith("raman_")]
    try:
        axis = np.array([float(column.removeprefix("raman_")) for column in spectral_columns])
    except ValueError as error:
        raise ValueError("Raman channel headers must encode wavenumbers, e.g. raman_400, raman_405.") from error
    if len(axis) < 2 or not np.isfinite(axis).all() or (np.diff(axis) <= 0).any():
        raise ValueError("Supply at least two Raman channels with strictly increasing numeric wavenumbers.")

    issue_records, missing_records, outlier_records = [], [], []
    frames = {}

    def flag(source, frame, mask, column, issue, severity="warning", raw_values=None):
        for index in frame.index[mask]:
            issue_records.append({
                "source": source, "row_number": int(frame.at[index, "_source_row"]),
                "batch_id": frame.at[index, "batch_id"],
                "sample_id": frame.at[index, "sample_id"] if "sample_id" in frame else "",
                "column": column, "issue": issue, "severity": severity,
                "raw_value": str(raw_values.at[index]) if raw_values is not None else "",
            })

    for name, original in raw.as_dict().items():
        active = original.copy()
        active["_source_row"] = np.arange(1, len(active) + 1)
        if batch_ids is not None:
            active = active[active["batch_id"].astype("string").str.strip().isin(batch_ids) | _blank(active["batch_id"])]
        optional = [column for column in ("scenario", "description", *PROVENANCE_COLUMNS) if name == "batches" and column in active]
        columns = REQUIRED_COLUMNS[name] + (spectral_columns if name == "raman" else []) + optional
        frame = active[columns + ["_source_row"]].reset_index(drop=True).copy()
        frame["_eligible"] = True
        timestamps = [column for column in columns if column.endswith("timestamp")]
        numeric_columns = list(SENSOR_RANGES) if name == "sensors" else spectral_columns if name == "raman" else REFERENCE_COLUMNS if name == "assays" else []
        for column in columns:
            if column in optional:
                continue
            values = frame[column].copy()
            missing_mask = _blank(values)
            invalid_mask = pd.Series(False, index=frame.index)
            if column in timestamps:
                frame[column] = pd.to_datetime(values.map(_utc_timestamp), utc=True)
                invalid_mask = ~missing_mask & frame[column].isna()
                frame.loc[frame[column].isna(), "_eligible"] = False
            elif column in numeric_columns:
                frame[column] = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
                invalid_mask = ~missing_mask & frame[column].isna()
            else:
                frame[column] = values.astype("string").str.strip().fillna("").astype(str)
                frame.loc[missing_mask, "_eligible"] = False
            flag(name, frame, missing_mask, column, "missing_value", raw_values=values)
            flag(name, frame, invalid_mask, column, "invalid_timestamp" if column in timestamps else "invalid_numeric", raw_values=values)
            missing_records.append({
                "source": name, "column": column, "rows": len(frame),
                "missing_raw": int(missing_mask.sum()), "invalid": int(invalid_mask.sum()),
                "missing_after_parse": int((missing_mask | invalid_mask).sum()),
                "missing_percent": 100 * (missing_mask | invalid_mask).mean() if len(frame) else 0.0,
            })
        identity = ["batch_id"] if name == "batches" else ["batch_id", "sample_id"]
        valid_identity = frame[identity].ne("").all(axis=1)
        duplicates = frame.duplicated(identity, keep=False) & valid_identity
        flag(name, frame, duplicates, ",".join(identity), "duplicate_identity", "error")
        frame.loc[duplicates, "_eligible"] = False
        if name != "batches":
            time_column = "collection_timestamp" if name == "assays" else "measurement_timestamp"
            duplicates = frame.duplicated(["batch_id", time_column], keep=False) & frame[time_column].notna() & frame["batch_id"].ne("")
            flag(name, frame, duplicates, time_column, "duplicate_timestamp", "error")
            frame.loc[duplicates, "_eligible"] = False
            for _, group in frame.groupby("batch_id", sort=False):
                out_of_order = group[time_column].diff() < pd.Timedelta(0)
                flag(name, frame, frame.index.isin(group.index[out_of_order]), time_column, "out_of_order_timestamp")
        # Consolidate numeric blocks after parsing the many Raman columns.
        frames[name] = frame.copy()

    batches = frames["batches"]
    starts = batches[batches["_eligible"]].set_index("batch_id")["start_timestamp"]
    for name in ("sensors", "raman", "assays"):
        frame = frames[name]
        batch_start = pd.to_datetime(frame["batch_id"].map(starts), utc=True)
        unknown = batch_start.isna() & frame["batch_id"].ne("")
        flag(name, frame, unknown, "batch_id", "unknown_or_invalid_batch", "error")
        time_column = "collection_timestamp" if name == "assays" else "measurement_timestamp"
        before_start = frame[time_column] < batch_start
        flag(name, frame, before_start, time_column, "before_batch_start", "error")
        frame.loc[unknown | before_start, "_eligible"] = False

    sensors = frames["sensors"]
    sensors["range_flag"] = False
    sensors["outlier_flag"] = False
    sensors["complete"] = sensors[list(SENSOR_RANGES)].notna().all(axis=1)
    for column, (lower, upper) in ranges.items():
        outside = sensors[column].notna() & ~sensors[column].between(lower, upper)
        flag("sensors", sensors, outside, column, "outside_sensor_range", raw_values=sensors[column])
        sensors.loc[outside, "range_flag"] = True
        for batch_id, group in sensors[sensors["_eligible"]].groupby("batch_id", sort=False):
            if column == "feed_rate_l_h":
                # Feed is deliberately stepped; a single batch-wide distribution
                # would label normal schedule phases as statistical outliers.
                scores = pd.Series(np.nan, index=group.index)
                status = "scheduled feed; range checks only"
            else:
                scores, status = modified_z_scores(group[column])
            outliers = scores.abs() > config.outlier_threshold
            flag("sensors", sensors, sensors.index.isin(group.index[outliers]), column, "potential_outlier")
            sensors.loc[group.index[outliers], "outlier_flag"] = True
            outlier_records.append({"batch_id": batch_id, "column": column, "observations": int(group[column].notna().sum()), "status": status, "potential_outliers": int(outliers.sum())})

    raman, assays = frames["raman"], frames["assays"]
    raman["spectrum_complete"] = raman[spectral_columns].notna().all(axis=1)
    assays["reference_complete"] = assays[REFERENCE_COLUMNS].notna().all(axis=1) & assays[REFERENCE_COLUMNS].ge(0).all(axis=1)
    for column in REFERENCE_COLUMNS:
        flag("assays", assays, assays[column] < 0, column, "negative_reference", raw_values=assays[column])
    impossible_delay = assays["result_timestamp"] < assays["collection_timestamp"]
    flag("assays", assays, impossible_delay, "result_timestamp", "result_before_collection", "error")
    assays.loc[impossible_delay, "_eligible"] = False

    spectral_input = raman[raman["_eligible"]].drop(columns="_eligible").rename(columns={"sample_id": "spectrum_sample_id", "measurement_timestamp": "spectrum_timestamp", "_source_row": "raman_source_row"})
    sensor_input = sensors[sensors["_eligible"]].drop(columns="_eligible").rename(columns={column: f"sensor_{column.lstrip('_')}" for column in sensors if column not in ("batch_id", "_eligible")})
    integrated = align_by_batch(
        spectral_input, sensor_input, left_time="spectrum_timestamp", right_time="sensor_measurement_timestamp",
        tolerance_minutes=config.sensor_tolerance_minutes, direction=config.sensor_direction,
    )
    integrated["sensor_matched"] = integrated["sensor_sample_id"].notna()
    integrated["sensor_time_delta_minutes"] = (integrated["sensor_measurement_timestamp"] - integrated["spectrum_timestamp"]).dt.total_seconds() / 60
    batch_input = batches[batches["_eligible"]].drop(columns=["_eligible", "_source_row"]).rename(columns={column: f"batch_{column}" for column in batches if column not in ("batch_id", "_eligible", "_source_row")})
    integrated = integrated.merge(batch_input, on="batch_id", how="left", validate="many_to_one")
    integrated["elapsed_time_h"] = (integrated["spectrum_timestamp"] - integrated["batch_start_timestamp"]).dt.total_seconds() / 3600
    assay_input = assays[assays["_eligible"]].drop(columns="_eligible").rename(columns={"sample_id": "assay_sample_id", "_source_row": "assay_source_row"})
    pairs = align_by_batch(
        assay_input, integrated, left_time="collection_timestamp", right_time="spectrum_timestamp",
        tolerance_minutes=config.assay_tolerance_minutes, direction="nearest",
    )
    pairs["spectrum_matched"] = pairs["spectrum_sample_id"].notna()
    pairs["collection_time_delta_minutes"] = (pairs["spectrum_timestamp"] - pairs["collection_timestamp"]).dt.total_seconds() / 60
    pairs["result_delay_h"] = (pairs["result_timestamp"] - pairs["collection_timestamp"]).dt.total_seconds() / 3600
    pairs["spectrum_reused"] = pairs["spectrum_matched"] & pairs.duplicated(["batch_id", "spectrum_sample_id"], keep=False)
    # Record join failures against original source row positions.
    for name, frame, mask, issue in (
        ("raman", integrated, ~integrated["sensor_matched"], "unmatched_sensor"),
        ("assays", pairs, ~pairs["spectrum_matched"], "unmatched_spectrum"),
        ("assays", pairs, pairs["spectrum_reused"], "spectrum_reused_by_assays"),
    ):
        source_row = "raman_source_row" if name == "raman" else "assay_source_row"
        sample_column = "spectrum_sample_id" if name == "raman" else "assay_sample_id"
        for _, row in frame[mask].iterrows():
            issue_records.append({"source": name, "row_number": int(row[source_row]), "batch_id": row["batch_id"], "sample_id": row[sample_column], "column": "timestamp alignment", "issue": issue, "severity": "warning", "raw_value": ""})

    return IntegrationResult(
        raw=raw, processed=SourceTables(**frames), integrated=integrated, reference_pairs=pairs,
        issues=pd.DataFrame(issue_records, columns=ISSUE_COLUMNS),
        missing_summary=pd.DataFrame(missing_records),
        outlier_summary=pd.DataFrame(outlier_records, columns=["batch_id", "column", "observations", "status", "potential_outliers"]),
        source_summary=pd.DataFrame([{"source": name, "input_rows": len(frame), "eligible_rows": int(frame["_eligible"].sum()), "excluded_from_matching": int((~frame["_eligible"]).sum())} for name, frame in frames.items()]),
        wavenumbers_cm1=axis, config=config, sensor_ranges=ranges,
        selected_batches=list(batch_ids) if batch_ids is not None else None,
    )


def available_references(reference_pairs: pd.DataFrame, as_of) -> pd.DataFrame:
    """Return reference values known by a timezone-aware cutoff, without features."""
    cutoff = _utc_timestamp(as_of)
    if pd.isna(cutoff):
        raise ValueError("The review cutoff must include an explicit timezone.")
    columns = ["assay_sample_id", "batch_id", "collection_timestamp", "result_timestamp", *REFERENCE_COLUMNS, "reference_complete"]
    return reference_pairs.loc[reference_pairs["result_timestamp"] <= cutoff, columns].copy()


def demonstrate_quality_issues(sources: SourceTables) -> SourceTables:
    """Make a deterministic, explicitly labeled faulty copy for learning."""
    copied = SourceTables(**{name: frame.copy(deep=True) for name, frame in sources.as_dict().items()})
    if len(copied.sensors) < 12 or len(copied.raman) < 5:
        raise ValueError("The quality example requires at least 12 sensor and 5 Raman rows.")
    copied.sensors.loc[copied.sensors.index[3], "ph"] = np.nan
    copied.sensors.loc[copied.sensors.index[7], "dissolved_oxygen_pct"] = 145.0
    copied.sensors["measurement_timestamp"] = copied.sensors["measurement_timestamp"].astype(object)
    copied.sensors.loc[copied.sensors.index[5], "measurement_timestamp"] = "invalid-time"
    copied.sensors = pd.concat([copied.sensors, copied.sensors.iloc[[10]]], ignore_index=True)
    channel = next(column for column in copied.raman if column.startswith("raman_"))
    copied.raman.loc[copied.raman.index[4], channel] = np.nan
    copied.raman["measurement_timestamp"] += pd.Timedelta(minutes=10)
    return copied


def source_csv_zip(sources: SourceTables) -> bytes:
    """Download four source CSVs; this is also a round-trip import example."""
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, frame in sources.as_dict().items():
            archive.writestr(f"{name}.csv", frame.to_csv(index=False))
    return output.getvalue()


def report_zip(result: IntegrationResult) -> bytes:
    """Export raw sources, processed tables, quality reports, and configuration."""
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, frame in result.raw.as_dict().items():
            archive.writestr(f"raw/{name}.csv", frame.to_csv(index=False))
        for name, frame in result.processed.as_dict().items():
            archive.writestr(f"processed/{name}.csv", frame.to_csv(index=False))
        for name in ("integrated", "reference_pairs", "issues", "missing_summary", "outlier_summary", "source_summary"):
            archive.writestr(f"reports/{name}.csv", getattr(result, name).to_csv(index=False))
        archive.writestr("manifest.json", json.dumps({
            "phase": 2, "config": asdict(result.config), "sensor_ranges": result.sensor_ranges,
            "data_provenance": source_provenance(result.processed.batches),
            "selected_batches": result.selected_batches, "timezone": "UTC",
            "raw_scope": "All original input rows; processed tables and reports use the selected batches, plus rows with missing batch IDs.",
            "duplicate_policy": "Exclude all ambiguous IDs/timestamps from matching; retain in raw/processed tables with flags.",
            "numeric_policy": "Invalid numeric cells become missing in processed copies; no interpolation, imputation, or outlier removal.",
            "assay_policy": "Retrospective collection-time pairing; result-time availability must be checked separately.",
            "spectral_axis_cm1": result.wavenumbers_cm1.tolist(),
            "limitations": "Educational synthetic/public-data workflow, not manufacturing validation or confirmed CPP limits.",
        }, indent=2))
    return output.getvalue()
