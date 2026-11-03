"""Deterministic timing, observation identity, causal references, and replay gaps."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.data_generation import SimulationConfig, generate_dataset
from src.monitoring import MonitoringSettings
from src.replay import MonitoringSession, ReplayClock, known_assays, monitoring_report_zip, replay_batch, replay_fault_example


@pytest.fixture
def batch():
    return replay_batch(generate_dataset(SimulationConfig(n_batches=2, duration_h=24, n_wavenumbers=32)), "B001")


def test_clock_start_pause_resume_speed_and_completion_are_deterministic():
    clock = ReplayClock(rate=2)
    assert clock.cursor == -1
    clock.start(100, 10)
    assert clock.cursor == 0
    clock.tick(100.75, 10)
    assert clock.cursor == 1
    assert clock.credit == pytest.approx(0.5)
    clock.pause()
    clock.tick(999, 10)
    assert clock.cursor == 1
    clock.start(1000, 10)
    clock.tick(1000.25, 10)
    assert clock.cursor == 2
    clock.set_rate(4, 1000.25, 10)
    clock.tick(1001, 10)
    assert clock.cursor == 5
    clock.tick(1005, 10)
    assert clock.cursor == 9
    assert not clock.running
    clock.reset()
    assert clock.cursor == -1
    assert clock.credit == 0


def test_stepping_is_paused_only_and_clock_does_not_double_count_backwards_time():
    clock = ReplayClock()
    assert clock.step(5) == 0
    clock.start(10, 5)
    with pytest.raises(ValueError, match="Pause"):
        clock.step(5)
    clock.tick(9, 5)
    clock.tick(10, 5)
    assert clock.cursor == 0
    clock.tick(11, 5)
    assert clock.cursor == 1


@pytest.mark.parametrize("rate", [0, -1, np.nan, np.inf])
def test_invalid_replay_rate_is_rejected(rate):
    with pytest.raises(ValueError):
        ReplayClock(rate=rate).start(0, 5)


def test_single_point_replay_completes_and_invalid_time_is_rejected():
    clock = ReplayClock()
    clock.start(0, 1)
    assert clock.cursor == 0 and not clock.running
    with pytest.raises(ValueError):
        clock.start(np.nan, 5)


def test_manual_step_consumes_partial_progress_before_resuming():
    clock = ReplayClock()
    clock.start(0, 5)
    clock.tick(0.9, 5)
    clock.pause()
    clock.step(5)
    assert clock.cursor == 1 and clock.credit == 0
    clock.start(2, 5)
    clock.start(2.05, 5)  # Repeated starts cannot re-anchor a running clock.
    clock.tick(2.1, 5)
    assert clock.cursor == 1
    clock.tick(3, 5)
    assert clock.cursor == 2


def test_batch_adapter_preserves_ids_and_never_reads_simulated_truth(batch):
    dataset = generate_dataset(SimulationConfig(n_batches=2, duration_h=24, n_wavenumbers=32))
    altered = deepcopy(dataset)
    altered.truth[["glucose_g_l", "lactate_g_l"]] = 999999
    other = replay_batch(altered, "B001")
    pd.testing.assert_frame_equal(batch.inputs, other.inputs)
    assert not {"glucose_g_l", "lactate_g_l", "glucose_reference_g_l", "batch_id", "elapsed_time_h"} & set(batch.inputs.columns)
    assert batch.inputs.index.tolist() == batch.metadata.sample_id.tolist()
    assert set(batch.metadata.batch_id) == {"B001"}
    assert batch.inputs.shape == (25, 37)


def test_assays_are_revealed_only_at_result_time_and_keep_collection_time(batch):
    first = batch.assays.iloc[0]
    assert known_assays(batch, first.collection_timestamp).empty
    assert known_assays(batch, first.result_timestamp - pd.Timedelta(nanoseconds=1)).empty
    known = known_assays(batch, first.result_timestamp)
    assert known.sample_id.tolist() == [first.sample_id]
    assert known.collection_timestamp.iloc[0] == first.collection_timestamp
    with pytest.raises(ValueError, match="timezone"):
        known_assays(batch, "2025-01-06")


def test_history_never_reveals_future_rows_or_fills_missing_predictions(batch):
    session = MonitoringSession("test")
    settings = MonitoringSettings()
    assert session.history(batch, -1, {}, settings).empty
    prefix = session.history(batch, 3, {}, settings)
    assert len(prefix) == 4
    assert prefix.glucose_prediction_g_l.isna().all()
    assert prefix.glucose_status.eq("model_unavailable").all()
    stamp = prefix.glucose_prediction_time_utc.copy()
    repeated = session.history(batch, 3, {}, settings)
    pd.testing.assert_series_equal(stamp, repeated.glucose_prediction_time_utc)
    assert len(session.records) == 4
    session.history(batch, 12, {}, settings, mode="Historical review")
    assert len(session.history(batch, 2, {}, settings)) == 3
    session.reset()
    assert not session.records


def test_fault_example_is_a_copy_and_exposes_missing_and_implausible_values(batch):
    faulty = replay_fault_example(batch)
    assert np.isnan(faulty.inputs.iloc[3].ph)
    assert faulty.inputs.iloc[7].dissolved_oxygen_pct == 145
    assert faulty.inputs.iloc[10].isna().any()
    assert not batch.inputs.isna().any().any()
    first_spectral = batch.inputs.columns[0]
    assert faulty.inputs.iloc[12][first_spectral] == pytest.approx(batch.inputs.iloc[12][first_spectral] + 2)


def test_monitoring_report_contains_only_the_displayed_prefix_and_available_assays(batch):
    from io import BytesIO
    from zipfile import ZipFile
    import json
    session = MonitoringSession("export")
    settings = MonitoringSettings()
    session.history(batch, 20, {}, settings)
    history = session.history(batch, 1, {}, settings)
    cutoff = batch.metadata.iloc[1].measurement_timestamp
    known = known_assays(batch, cutoff)
    with ZipFile(BytesIO(monitoring_report_zip(history, known, settings, {"cutoff_utc": str(cutoff)}))) as archive:
        exported = pd.read_csv(BytesIO(archive.read("observations.csv")))
        assert len(exported) == 2
        assert "glucose_g_l" not in exported
        assert len(pd.read_csv(BytesIO(archive.read("available_assays.csv")))) == len(known)
        assert json.loads(archive.read("replay_manifest.json"))["phase"] == 5
