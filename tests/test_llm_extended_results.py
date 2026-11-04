"""Exploration/monitoring snapshots preserve scope, numbers, gaps and privacy."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from src.data_generation import SimulationConfig, generate_dataset
from src.llm_client import request_payload, LLMConfig
from src.llm_results import spectral_snapshot, process_snapshot, multivariate_snapshot, demo_snapshot
from src.monitoring import MonitoringSettings
from src.multivariate_monitoring import fit_process_monitor, score_process_monitor, spe_sensor_contributions


def value(snapshot, label):
    return next(fact["value"] for fact in snapshot.payload["facts"] if fact["label"] == label)


def spectral(**overrides):
    settings = {"prominence": 0.05, "window_length": 5, "smoothness": 1e5, "asymmetry": 0.01,
                "fft_window": "hann", "fft_detrend": "none", "band": (0, 0.01),
                "wavelet_level": None, "wavelet_threshold": None}
    arguments = dict(selected_count=20, reference_count=10, peak_count=2, settings=settings,
                     pca_variance=np.array([0.6, 0.3]), reconstruction_rmse=np.array([0.1, 0.3]),
                     context={"sample": "PRIVATE_SAMPLE", "references": ["PRIVATE_BATCH"]})
    arguments.update(overrides)
    axis = np.linspace(400, 540, 8)
    signal = np.array([1., -1.] * 4)
    return spectral_snapshot(axis, signal, np.zeros(8), np.zeros(8),
        {"selected_bins": 3, "fraction_of_analyzed_mean_square": 0.25}, **arguments)


def test_spectral_summary_preserves_axes_variance_and_residual_definition():
    snapshot = spectral()
    assert snapshot.payload["target"] == "Raman spectra"
    assert value(snapshot, "Raman channel spacing") == 20
    assert value(snapshot, "Cumulative retained reference variance") == pytest.approx(0.9)
    assert value(snapshot, "Mean reconstruction RMSE across all selected spectra") == pytest.approx(0.2)
    assert value(snapshot, "Raw minus SG-smoothed residual SD") == pytest.approx(np.sqrt(8 / 7))
    assert value(snapshot, "Retained fraction of analyzed mean square") == 0.25
    assert value(snapshot, "Wavelet soft threshold") is None
    assert "spectral_pca" in snapshot.payload["limitations"]
    assert "spectral_fft" in snapshot.payload["limitations"]
    assert "PRIVATE" not in snapshot.serialized and "sample_id" not in snapshot.serialized
    assert all(not isinstance(fact["value"], list) for fact in snapshot.payload["facts"])


def test_spectral_missing_pca_is_unknown_and_prepared_sources_require_consent():
    snapshot = spectral(pca_variance=None, reconstruction_rmse=None, source_kind="prepared_sources")
    assert value(snapshot, "Exploratory PCA reference variance") is None
    assert snapshot.requires_consent and "NaN" not in snapshot.serialized
    assert spectral(context={"sample": "changed"}).identity != spectral().identity


def process_frames():
    now = pd.Timestamp("2026-11-04T00:00:00Z")
    row = {"measurement_timestamp": now, "sample_id": "PRIVATE_SAMPLE", "batch_id": "PRIVATE_BATCH",
        "temperature_c": 37., "ph": 7., "dissolved_oxygen_pct": 50., "feed_rate_l_h": 0.004,
        "agitation_rpm": 300., "quality_flags": "", "glucose_status": "ok", "lactate_status": "out_of_domain",
        "glucose_prediction_g_l": 3.2, "lactate_prediction_g_l": 99.0}
    return pd.DataFrame([row]), pd.DataFrame(columns=["result_timestamp"]), now


def process(**kwargs):
    history, assays, now = process_frames()
    return process_snapshot(history, history, assays, MonitoringSettings(), cutoff=now, context="PRIVATE_CONTEXT", **kwargs)


def test_process_snapshot_never_fills_failed_predictions_or_forwards_private_rows():
    snapshot = process()
    assert value(snapshot, "Glucose displayed estimate at capture") == 3.2
    assert value(snapshot, "Lactate displayed estimate at capture") is None
    assert value(snapshot, "Lactate unavailable historical estimates") == 1
    assert "PRIVATE" not in snapshot.serialized and "sample_id" not in snapshot.serialized
    assert "monitoring" in snapshot.payload["limitations"]
    assert not any(fact["value"] == 99.0 for fact in snapshot.payload["facts"])


def test_stale_current_cards_are_withheld_without_rewriting_historical_coverage():
    snapshot = process(current_eligible=False, health_flag_count=1)
    assert value(snapshot, "Glucose displayed estimate at capture") is None
    assert value(snapshot, "Glucose eligible estimate at capture") is False
    assert value(snapshot, "Glucose accepted historical estimates") == 1
    assert value(snapshot, "Source health/freshness review flags") == 1


@pytest.mark.parametrize("damage", ["observation", "sensor", "assay", "naive_clock"])
def test_process_summary_rejects_future_evidence_and_invalid_clock(damage):
    history, assays, now = process_frames()
    sensors = history.copy()
    if damage == "observation": history["measurement_timestamp"] += pd.Timedelta(seconds=1)
    elif damage == "sensor": sensors["measurement_timestamp"] += pd.Timedelta(seconds=1)
    elif damage == "assay": assays = pd.DataFrame({"result_timestamp": [now + pd.Timedelta(seconds=1)]})
    else: now = now.tz_localize(None)
    with pytest.raises(ValueError):
        process_snapshot(history, sensors, assays, MonitoringSettings(), cutoff=now, context="test")


def test_stream_future_tolerance_does_not_relax_laboratory_availability():
    history, assays, now = process_frames()
    history["measurement_timestamp"] += pd.Timedelta(seconds=1)
    snapshot = process_snapshot(history, history, assays, MonitoringSettings(), cutoff=now, context="test", future_tolerance_seconds=2)
    assert snapshot is not None
    assays = pd.DataFrame({"result_timestamp": [now + pd.Timedelta(seconds=1)]})
    with pytest.raises(ValueError, match="future"):
        process_snapshot(history, history, assays, MonitoringSettings(), cutoff=now, context="test", future_tolerance_seconds=2)


def test_no_received_observations_means_no_manufactured_snapshot():
    _, assays, now = process_frames()
    assert process_snapshot(pd.DataFrame(), pd.DataFrame(), assays, MonitoringSettings(), cutoff=now, context="test") is None


@pytest.fixture(scope="module")
def mspc():
    data = generate_dataset(SimulationConfig(n_batches=6, duration_h=24, n_wavenumbers=32))
    model = fit_process_monitor(data.sensors, data.batches)
    scored = score_process_monitor(model, data.sensors)
    observation = scored.loc[scored.status.eq("scored")].iloc[[0]]
    contributions = spe_sensor_contributions(model, observation)
    return model, scored, observation, contributions


def test_multivariate_summary_uses_selected_charts_and_separate_inspection_without_refitting(mspc, monkeypatch):
    model, scored, observation, contributions = mspc
    monkeypatch.setattr(model.pca, "fit", lambda *a, **k: pytest.fail("Summary refitted PCA"))
    view = scored.iloc[:5]
    snapshot = multivariate_snapshot(model, view, observation, contributions, context={"sample": "PRIVATE_SAMPLE"})
    assert value(snapshot, "Selected chart observations") == 5
    assert value(snapshot, "Empirical per-chart calibration quantile") == model.quantile
    assert value(snapshot, "Separately inspected observation SPE") == observation.spe.iloc[0]
    component_sum = sum(fact["value"] for fact in snapshot.payload["facts"] if fact["label"].endswith("SPE contribution"))
    assert component_sum == pytest.approx(observation.spe.iloc[0])
    assert "empirical" in snapshot.payload["limitations"]
    assert "scenario" not in snapshot.serialized and "PRIVATE" not in snapshot.serialized


def test_unscorable_multivariate_observation_does_not_become_normal_or_get_statistics(mspc):
    model, _, observation, contributions = mspc
    invalid = observation.copy()
    invalid["status"] = "invalid"
    invalid["alarm"] = True
    snapshot = multivariate_snapshot(model, invalid, invalid, contributions, context="invalid")
    assert value(snapshot, "Selected chart unscorable observations") == 1
    assert value(snapshot, "Selected chart Either-chart exceedances") == 0
    assert value(snapshot, "Separately inspected observation SPE") is None
    assert not any(fact["label"].endswith("SPE contribution") for fact in snapshot.payload["facts"])


@pytest.mark.parametrize("page,expected", [("Spectral Analytics", "not temporal Hz"),
    ("Process Monitoring", "captured historical"), ("Multivariate Monitoring", "joint false-alarm")])
def test_new_page_prompts_have_specific_scientific_scope(page, expected):
    snapshot = demo_snapshot(page)
    payload = request_payload("vendor/model:free", {}, snapshot, "Learning explanation")
    assert expected in payload["messages"][0]["content"]
    assert json.loads(payload["messages"][1]["content"])["page"] == page
