"""Known-answer checks for source integrity, time alignment, and quality reports."""

from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from src.data_generation import SimulationConfig, generate_dataset
from src.data_integration import (
    IntegrationConfig, SENSOR_RANGES, SourceTables, align_by_batch,
    available_references, demonstrate_quality_issues, integrate_sources,
    modified_z_scores, read_source_csv, report_zip, source_csv_zip,
    sources_from_dataset,
)


@pytest.fixture
def sources():
    return sources_from_dataset(generate_dataset(SimulationConfig(n_batches=2, duration_h=24, n_wavenumbers=32)))


def test_default_integration_keeps_observations_and_sparse_references(sources):
    result = integrate_sources(sources)
    assert len(result.integrated) == 50
    assert result.integrated["sensor_matched"].all()
    assert result.integrated["sensor_time_delta_minutes"].eq(0).all()
    assert len(result.reference_pairs) == 4
    assert result.reference_pairs["spectrum_matched"].all()
    assert result.reference_pairs["collection_time_delta_minutes"].eq(0).all()
    assert not any(column in result.integrated for column in ("glucose_g_l", "lactate_g_l", "glucose_reference_g_l"))
    assert result.source_summary["excluded_from_matching"].sum() == 0


def test_input_tables_and_returned_raw_snapshot_are_independent(sources):
    before = {name: frame.copy(deep=True) for name, frame in sources.as_dict().items()}
    result = integrate_sources(sources)
    for name, frame in before.items():
        pd.testing.assert_frame_equal(sources.as_dict()[name], frame)
        pd.testing.assert_frame_equal(result.raw.as_dict()[name], frame)
    sources.sensors.loc[0, "ph"] = 999
    assert result.raw.sensors.loc[0, "ph"] != 999


def test_matching_never_crosses_batches_and_includes_tolerance_boundary():
    time = pd.Timestamp("2025-01-01T00:00:00Z")
    left = pd.DataFrame({"batch_id": ["B1", "B2", "B1"], "spectrum_time": [time + pd.Timedelta(minutes=10), time, time + pd.Timedelta(minutes=11)]})
    right = pd.DataFrame({"batch_id": ["B1"], "sensor_time": [time], "value": [37.0]})
    matched = align_by_batch(left, right, left_time="spectrum_time", right_time="sensor_time", tolerance_minutes=10, direction="backward")
    assert matched["value"].iloc[0] == 37
    assert matched["value"].iloc[1:].isna().all()
    assert matched["batch_id"].tolist() == left["batch_id"].tolist()


def test_backward_does_not_choose_future_sensor_while_nearest_can():
    time = pd.Timestamp("2025-01-01T00:00:00Z")
    left = pd.DataFrame({"batch_id": ["B1"], "spectrum_time": [time]})
    right = pd.DataFrame({"batch_id": ["B1"], "sensor_time": [time + pd.Timedelta(minutes=1)], "value": [37.0]})
    arguments = dict(left_time="spectrum_time", right_time="sensor_time", tolerance_minutes=2)
    assert align_by_batch(left, right, direction="backward", **arguments)["value"].isna().all()
    assert align_by_batch(left, right, direction="nearest", **arguments)["value"].iloc[0] == 37


def test_equidistant_nearest_match_chooses_earlier_observation():
    time = pd.Timestamp("2025-01-01T00:00:00Z")
    left = pd.DataFrame({"batch_id": ["B1"], "spectrum_time": [time]})
    right = pd.DataFrame({"batch_id": ["B1", "B1"], "sensor_time": [time - pd.Timedelta(minutes=1), time + pd.Timedelta(minutes=1)], "value": [36, 38]})
    matched = align_by_batch(left, right, left_time="spectrum_time", right_time="sensor_time", tolerance_minutes=2, direction="nearest")
    assert matched["value"].iloc[0] == 36


def test_delayed_raman_clock_has_signed_delta_and_respects_tolerance(sources):
    sources.raman["measurement_timestamp"] += pd.Timedelta(minutes=10)
    broad = integrate_sources(sources, IntegrationConfig(sensor_tolerance_minutes=10))
    narrow = integrate_sources(sources, IntegrationConfig(sensor_tolerance_minutes=9))
    assert broad.integrated["sensor_time_delta_minutes"].eq(-10).all()
    assert broad.reference_pairs["collection_time_delta_minutes"].eq(10).all()
    assert not narrow.integrated["sensor_matched"].any()
    assert (narrow.issues["issue"] == "unmatched_sensor").sum() == 50


def test_missing_invalid_and_infinite_numeric_values_remain_missing(sources):
    sources.sensors.loc[0, "ph"] = np.nan
    sources.sensors["temperature_c"] = sources.sensors["temperature_c"].astype(object)
    sources.sensors.loc[1, "temperature_c"] = "fault"
    sources.sensors.loc[2, "agitation_rpm"] = np.inf
    channel = next(column for column in sources.raman if column.startswith("raman_"))
    sources.raman.loc[0, channel] = np.nan
    result = integrate_sources(sources)
    assert np.isnan(result.integrated.loc[0, "sensor_ph"])
    assert np.isnan(result.integrated.loc[1, "sensor_temperature_c"])
    assert np.isnan(result.integrated.loc[2, "sensor_agitation_rpm"])
    assert not result.integrated.loc[0, "spectrum_complete"]
    assert not result.integrated.loc[0, "sensor_complete"]
    assert result.integrated["sensor_matched"].all()
    assert {"invalid_numeric", "missing_value"}.issubset(result.issues["issue"])
    report = result.missing_summary.set_index(["source", "column"])
    assert report.loc[("sensors", "ph"), "missing_raw"] == 1
    assert report.loc[("sensors", "temperature_c"), "invalid"] == 1


def test_sensor_range_bounds_are_inclusive_and_values_are_retained(sources):
    sources.sensors.loc[:3, "temperature_c"] = [34, 40, 33.99, 40.01]
    result = integrate_sources(sources)
    flags = result.issues.query("issue == 'outside_sensor_range' and column == 'temperature_c'")
    assert flags["row_number"].tolist() == [3, 4]
    assert result.integrated.loc[2, "sensor_temperature_c"] == 33.99
    assert result.integrated.loc[3, "sensor_temperature_c"] == 40.01
    assert result.integrated["sensor_matched"].all()


def test_modified_z_scores_match_known_values_and_abstain_for_zero_mad():
    values = pd.Series([36.8, 36.9, 37, 37, 37, 37.1, 37.2, 42])
    scores, status = modified_z_scores(values)
    assert status == "evaluated"
    assert scores.iloc[-1] == pytest.approx(33.725)
    assert (scores.abs() > 3.5).sum() == 1
    scores, status = modified_z_scores(pd.Series([37.0] * 8 + [42.0]))
    assert scores.isna().all()
    assert status == "zero MAD; not evaluated"
    assert modified_z_scores(pd.Series([1, 2]))[1] == "insufficient observations"


def test_outlier_scores_are_batch_local_and_feed_schedule_is_not_scored(sources):
    sources.sensors["temperature_c"] = [36.8, 36.9, 37, 37, 37.1, 37.2] * 4 + [42] + [50.0] * 25
    result = integrate_sources(sources)
    flags = result.issues.query("issue == 'potential_outlier' and column == 'temperature_c'")
    assert flags["batch_id"].unique().tolist() == ["B001"]
    assert not result.issues.query("issue == 'potential_outlier' and column == 'feed_rate_l_h'").shape[0]
    assert result.outlier_summary.query("column == 'feed_rate_l_h'")["status"].eq("scheduled feed; range checks only").all()


def test_all_duplicate_sensor_candidates_are_excluded_without_dropping_raw_rows(sources):
    sources.sensors = pd.concat([sources.sensors, sources.sensors.iloc[[3]]], ignore_index=True)
    result = integrate_sources(sources)
    assert len(result.raw.sensors) == 51
    assert len(result.processed.sensors) == 51
    assert not result.processed.sensors.loc[[3, 50], "_eligible"].any()
    assert not result.integrated.loc[3, "sensor_matched"]
    assert result.issues.query("issue == 'duplicate_identity'")["row_number"].tolist() == [4, 51]


def test_duplicate_timestamps_are_ambiguous_even_with_distinct_ids(sources):
    sources.sensors.loc[1, "measurement_timestamp"] = sources.sensors.loc[0, "measurement_timestamp"]
    result = integrate_sources(sources)
    assert not result.processed.sensors.loc[[0, 1], "_eligible"].any()
    assert (result.issues["issue"] == "duplicate_timestamp").sum() == 2


def test_unordered_rows_are_flagged_and_sorted_for_matching(sources):
    sources.sensors = sources.sensors.iloc[::-1].reset_index(drop=True)
    result = integrate_sources(sources)
    assert result.integrated["sensor_matched"].all()
    assert result.integrated["sensor_time_delta_minutes"].eq(0).all()
    assert "out_of_order_timestamp" in result.issues["issue"].values


@pytest.mark.parametrize("value", ["bad-time", "2025-01-06T08:00:00", ""])
def test_invalid_or_timezone_naive_timestamp_is_excluded_and_reported(sources, value):
    sources.sensors["measurement_timestamp"] = sources.sensors["measurement_timestamp"].astype(object)
    sources.sensors.loc[0, "measurement_timestamp"] = value
    result = integrate_sources(sources)
    assert not result.integrated.loc[0, "sensor_matched"]
    assert not result.processed.sensors.loc[0, "_eligible"]
    assert result.raw.sensors.loc[0, "measurement_timestamp"] == value


def test_timezone_offsets_are_normalized_to_utc(sources):
    sources.sensors["measurement_timestamp"] = sources.sensors["measurement_timestamp"].dt.tz_convert("America/New_York").astype(str)
    result = integrate_sources(sources)
    assert result.integrated["sensor_time_delta_minutes"].eq(0).all()
    assert str(result.processed.sensors["measurement_timestamp"].dt.tz) == "UTC"


def test_unknown_batch_before_start_and_impossible_assay_result_are_reported(sources):
    sources.sensors.loc[0, "batch_id"] = "UNKNOWN"
    sources.sensors.loc[1, "measurement_timestamp"] = sources.batches.loc[0, "start_timestamp"] - pd.Timedelta(hours=1)
    sources.assays.loc[0, "result_timestamp"] = sources.assays.loc[0, "collection_timestamp"] - pd.Timedelta(minutes=1)
    result = integrate_sources(sources)
    assert {"unknown_or_invalid_batch", "before_batch_start", "result_before_collection"}.issubset(result.issues["issue"])
    assert len(result.reference_pairs) == 3


def test_duplicate_batch_metadata_does_not_produce_cartesian_matches(sources):
    sources.batches = pd.concat([sources.batches, sources.batches.iloc[[0]]], ignore_index=True)
    result = integrate_sources(sources)
    assert result.integrated["batch_id"].unique().tolist() == ["B002"]
    assert result.issues.query("source == 'batches' and issue == 'duplicate_identity'").shape[0] == 2


def test_batch_filter_preserves_full_raw_snapshot_and_original_row_numbers(sources):
    sources.sensors.loc[25, "ph"] = np.nan
    result = integrate_sources(sources, batch_ids=["B002"])
    assert len(result.raw.sensors) == 50
    assert result.integrated["batch_id"].unique().tolist() == ["B002"]
    assert result.reference_pairs["batch_id"].unique().tolist() == ["B002"]
    assert result.issues.query("issue == 'missing_value' and column == 'ph'")["row_number"].tolist() == [26]


def test_availability_uses_result_time_at_the_exact_boundary(sources):
    collection = sources.assays.loc[0, "collection_timestamp"]
    sources.assays.loc[0, "result_timestamp"] = collection + pd.Timedelta(hours=4)
    result = integrate_sources(sources)
    batch_pairs = result.reference_pairs.query("batch_id == 'B001'")
    assert available_references(batch_pairs, collection + pd.Timedelta(hours=3)).empty
    known = available_references(batch_pairs, collection + pd.Timedelta(hours=4))
    assert known["assay_sample_id"].tolist() == [sources.assays.loc[0, "sample_id"]]
    assert "sensor_temperature_c" not in known
    assert result.reference_pairs.loc[0, "spectrum_timestamp"] == collection
    with pytest.raises(ValueError, match="timezone"):
        available_references(batch_pairs, "2025-01-06T12:00:00")


def test_assay_reuse_of_a_spectrum_is_visible_and_not_averaged(sources):
    extra = sources.assays.iloc[[0]].copy()
    extra["sample_id"] = "REPLICATE"
    extra["collection_timestamp"] += pd.Timedelta(minutes=1)
    sources.assays = pd.concat([sources.assays, extra], ignore_index=True)
    result = integrate_sources(sources)
    reused = result.reference_pairs[result.reference_pairs["spectrum_reused"]]
    assert len(reused) == 2
    assert len(result.reference_pairs) == 5
    assert (result.issues["issue"] == "spectrum_reused_by_assays").sum() == 2


def test_no_target_or_arbitrary_extra_sensor_column_enters_integration(sources):
    sources.sensors["glucose_g_l"] = 999.0
    sources.sensors["unapproved_feature"] = 123
    result = integrate_sources(sources)
    assert "glucose_g_l" in result.raw.sensors
    assert not any("glucose_g_l" in column or "unapproved_feature" in column for column in result.integrated)


def test_quality_example_does_not_mutate_original_sources(sources):
    faulty = demonstrate_quality_issues(sources)
    assert len(sources.sensors) == 50
    assert sources.sensors["ph"].notna().all()
    result = integrate_sources(faulty)
    assert {"missing_value", "invalid_timestamp", "duplicate_identity", "outside_sensor_range", "unmatched_sensor"}.issubset(result.issues["issue"])
    assert not result.integrated["spectrum_complete"].all()


def test_csv_export_import_round_trip_reproduces_alignment(sources):
    with ZipFile(BytesIO(source_csv_zip(sources))) as archive:
        uploaded = SourceTables(**{name: read_source_csv(archive.read(f"{name}.csv")) for name in sources.as_dict()})
    original, imported = integrate_sources(sources), integrate_sources(uploaded)
    assert imported.integrated["sensor_matched"].all()
    np.testing.assert_allclose(original.integrated["sensor_temperature_c"], imported.integrated["sensor_temperature_c"])
    assert original.reference_pairs["assay_sample_id"].tolist() == imported.reference_pairs["assay_sample_id"].tolist()
    np.testing.assert_allclose(original.wavenumbers_cm1, imported.wavenumbers_cm1)


def test_csv_reader_preserves_blanks_and_rejects_ambiguous_headers():
    frame = read_source_csv(b"sample_id,value\n001,\n002,NA\n")
    assert frame["sample_id"].tolist() == ["001", "002"]
    assert frame["value"].tolist() == ["", "NA"]
    for content in (b"", b"x,x\n1,2\n", b"x,\n1,2\n", b"\xff"):
        with pytest.raises(ValueError):
            read_source_csv(content)


@pytest.mark.parametrize("content", [b"x,y\n1\n", b"x,y\n1,2,3\n", b'x,y\n1,"unclosed\n'])
def test_csv_reader_rejects_ragged_or_malformed_records(content):
    with pytest.raises(ValueError):
        read_source_csv(content)


def test_report_contains_raw_processed_and_reproducible_configuration(sources):
    result = integrate_sources(sources, batch_ids=["B001"])
    with ZipFile(BytesIO(report_zip(result))) as archive:
        assert {"raw/sensors.csv", "processed/raman.csv", "reports/reference_pairs.csv", "reports/issues.csv", "manifest.json"}.issubset(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["config"]["sensor_direction"] == "backward"
        assert manifest["selected_batches"] == ["B001"]
        assert len(read_source_csv(archive.read("raw/sensors.csv"))) == 50


@pytest.mark.parametrize("source", ["sensors", "raman", "assays", "batches"])
def test_missing_required_columns_have_actionable_error(sources, source):
    setattr(sources, source, sources.as_dict()[source].drop(columns="batch_id"))
    with pytest.raises(ValueError, match=f"{source}: missing required columns: batch_id"):
        integrate_sources(sources)


@pytest.mark.parametrize("changes", [
    {"sensor_tolerance_minutes": -1}, {"assay_tolerance_minutes": np.inf},
    {"sensor_direction": "forward"}, {"outlier_threshold": 0},
])
def test_invalid_alignment_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        IntegrationConfig(**changes)


def test_invalid_ranges_and_raman_axis_are_rejected(sources):
    with pytest.raises(ValueError, match="bounds"):
        integrate_sources(sources, sensor_ranges={**SENSOR_RANGES, "ph": (8, 6)})
    columns = [column for column in sources.raman if column.startswith("raman_")]
    sources.raman = sources.raman[["sample_id", "batch_id", "measurement_timestamp", *columns[::-1]]]
    with pytest.raises(ValueError, match="increasing"):
        integrate_sources(sources)


@pytest.mark.parametrize("source", ["sensors", "raman", "assays", "batches"])
def test_header_only_sources_do_not_crash(sources, source):
    setattr(sources, source, sources.as_dict()[source].iloc[:0].copy())
    result = integrate_sources(sources)
    if source in ("raman", "batches"):
        assert result.integrated.empty
    elif source == "sensors":
        assert not result.integrated["sensor_matched"].any()
    elif source == "assays":
        assert result.reference_pairs.empty
