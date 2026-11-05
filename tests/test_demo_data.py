"""Independent demo populations, provenance, import compatibility and no leakage."""

from dataclasses import asdict, replace
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from src.acquisition import AcquisitionSession, InMemoryTestAdapter, acquisition_report_zip, test_source_records as source_records
from src.data_generation import SimulationConfig, generate_dataset
from src.data_integration import integrate_sources, report_zip, source_csv_zip, sources_from_dataset
from src.demo_data import (MONITORING_SEED_OFFSET, PROVENANCE_COLUMNS, demo_provenance,
    generate_development_dataset, generate_monitoring_dataset, monitoring_demo_config, source_provenance)
from src.modeling import ModelCase, TrainingConfig, train_study
from src.modeling_data import fixed_batch_partition, modeling_dataset
from src.modeling_storage import evaluate_once, load_study, save_study, study_manifest
from src.stream_monitoring_ui import _make_adapter, fixture_records


CONFIG = SimulationConfig(n_batches=6, duration_h=72, n_wavenumbers=71, seed=42)


def test_monitoring_uses_different_seed_measurements_and_disjoint_ids():
    development = generate_development_dataset(CONFIG)
    monitoring = generate_monitoring_dataset(CONFIG)
    assert monitoring_demo_config(CONFIG).seed == 3_000_059
    assert monitoring_demo_config(CONFIG).seed == CONFIG.seed + MONITORING_SEED_OFFSET
    for name in ("sensors", "truth", "spectral_metadata", "offline_assays", "batches"):
        old, new = getattr(development, name), getattr(monitoring, name)
        assert set(old.batch_id).isdisjoint(new.batch_id)
        assert new.batch_id.str.startswith("MON-").all()
        if "sample_id" in new:
            assert set(old.sample_id).isdisjoint(new.sample_id)
    assert set(monitoring.spectra.index) == set(monitoring.spectral_metadata.sample_id)
    assert set(monitoring.offline_assays.sample_id).issubset(monitoring.spectra.index)
    assert not np.allclose(development.spectra, monitoring.spectra)
    assert not np.allclose(development.sensors.ph, monitoring.sensors.ph)
    assert not np.allclose(development.offline_assays.glucose_reference_g_l, monitoring.offline_assays.glucose_reference_g_l)
    assert CONFIG.seed == 42
    np.testing.assert_array_equal(development.wavenumbers_cm1, monitoring.wavenumbers_cm1)


def test_development_measurements_and_existing_cohort_identity_remain_unchanged():
    original = generate_dataset(CONFIG)
    development = generate_development_dataset(CONFIG)
    for name in ("sensors", "truth", "spectra", "spectral_metadata", "offline_assays"):
        pd.testing.assert_frame_equal(getattr(original, name), getattr(development, name))
    pd.testing.assert_frame_equal(original.batches, development.batches[original.batches.columns])
    old = modeling_dataset(integrate_sources(sources_from_dataset(original)))
    new = modeling_dataset(integrate_sources(sources_from_dataset(development)))
    assert old.cohort_id == new.cohort_id
    pd.testing.assert_frame_equal(old.X, new.X)
    assert new.data_provenance["sources"][0]["data_role"] == "historical_development"
    assert not any(key in new.X for key in PROVENANCE_COLUMNS)


def test_both_populations_reproduce_and_distinct_seeds_reset_source_identity():
    one, two = generate_monitoring_dataset(CONFIG), generate_monitoring_dataset(CONFIG)
    pd.testing.assert_frame_equal(one.spectra, two.spectra)
    assert demo_provenance(CONFIG, monitoring=True) == demo_provenance(CONFIG, monitoring=True)
    assert demo_provenance(CONFIG, monitoring=True)["data_source_id"] != demo_provenance(CONFIG)["data_source_id"]
    changed = demo_provenance(replace(CONFIG, seed=43), monitoring=True)
    assert changed["data_source_id"] != demo_provenance(CONFIG, monitoring=True)["data_source_id"]
    config = replace(CONFIG, n_batches=8)
    expanded = generate_monitoring_dataset(config)
    pd.testing.assert_frame_equal(one.spectra, expanded.spectra.loc[one.spectra.index])


def test_received_provenance_survives_acquisition_integration_and_exports():
    config = replace(CONFIG, n_batches=1, duration_h=24, n_wavenumbers=32)
    sources = sources_from_dataset(generate_monitoring_dataset(config))
    session = AcquisitionSession("provenance")
    assert session.poll_once(InMemoryTestAdapter(source_records(sources)), 100)
    received = session.snapshot()
    assert session.ledger().status.eq("accepted").all()
    provenance = source_provenance(received.batches)
    assert provenance["sources"][0]["generation_seed"] == config.seed + MONITORING_SEED_OFFSET
    assert json.loads(provenance["sources"][0]["generation_config_json"])["seed"] == config.seed + MONITORING_SEED_OFFSET
    integrated = integrate_sources(received)
    assert source_provenance(integrated.processed.batches) == provenance
    for blob in (acquisition_report_zip(session, {}), report_zip(integrated)):
        with ZipFile(BytesIO(blob)) as archive:
            assert json.loads(archive.read("manifest.json"))["data_provenance"] == provenance
    with ZipFile(BytesIO(source_csv_zip(received))) as archive:
        batches = pd.read_csv(BytesIO(archive.read("batches.csv")))
        assert batches.data_role.eq("monitoring_demo").all()
        assert set(PROVENANCE_COLUMNS).issubset(batches)


def test_unrecorded_external_source_does_not_acquire_invented_seed():
    assert source_provenance(pd.DataFrame({"batch_id": ["external"]})) == {"status": "not_recorded", "sources": []}
    partial = source_provenance(pd.DataFrame({"data_role": ["external"], "generation_seed": [np.nan]}))
    assert partial["sources"][0]["generation_seed"] is None


def test_stream_and_acquisition_populations_match_and_stream_time_rebase_preserves_measurements():
    config = replace(CONFIG, n_batches=1, duration_h=24, n_wavenumbers=32)
    expected = source_records(sources_from_dataset(generate_monitoring_dataset(config)))
    assert fixture_records(config, "MON-B001", False) == expected
    context = {"source": "Fresh in-app test stream", "simulation": asdict(config), "batch": "MON-B001", "faults": False, "pace": 1.0}
    adapter = _make_adapter(context)
    raw = next(row["data"] for row in expected if row["source"] == "raman")
    rebased = next(row["data"] for row in adapter.records if row["source"] == "raman")
    assert raw["measurement_timestamp"] != rebased["measurement_timestamp"]
    assert {key: value for key, value in raw.items() if key.startswith("raman_")} == {key: value for key, value in rebased.items() if key.startswith("raman_")}
    assert next(row["data"] for row in adapter.records if row["source"] == "batches")["data_role"] == "monitoring_demo"


def test_new_monitoring_data_does_not_modify_sealed_development_study(tmp_path, monkeypatch):
    config = replace(CONFIG, n_batches=8)
    cohort = modeling_dataset(integrate_sources(sources_from_dataset(generate_development_dataset(config))))
    study = train_study(cohort, fixed_batch_partition(cohort.metadata.batch_id), [ModelCase("raw", "pls")], TrainingConfig(max_pls_components=2))
    save_study(study, tmp_path)
    evaluate_once(study, cohort, tmp_path)
    before = {path.name: path.read_bytes() for path in (tmp_path / cohort.cohort_id).iterdir()}
    monkeypatch.setattr("src.modeling.train_study", lambda *args, **kwargs: pytest.fail("Monitoring must never train"))
    monitoring = generate_monitoring_dataset(config)
    assert set(study.partition.training_batches + study.partition.holdout_batches).isdisjoint(monitoring.batches.batch_id)
    assert study_manifest(study)["data_provenance"]["sources"][0]["generation_seed"] == CONFIG.seed
    assert load_study(tmp_path, cohort.cohort_id).study_id == study.study_id
    assert before == {path.name: path.read_bytes() for path in (tmp_path / cohort.cohort_id).iterdir()}
