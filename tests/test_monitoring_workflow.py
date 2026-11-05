"""Saved input rules, reusable preprocessing, received-only reports and API contract."""

from copy import deepcopy
from dataclasses import replace
from io import BytesIO
import json
from threading import Thread
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from src.acquisition import AcquisitionError, InMemoryTestAdapter, MonitoringAPIAdapter, test_source_records as source_records
from src.data_generation import SimulationConfig, generate_dataset
from src.data_integration import IntegrationConfig, SENSOR_RANGES, integrate_sources, sources_from_dataset
from src.demo_data import generate_monitoring_dataset
from src.input_policy import make_input_policy, shared_input_policy
from src.model_registry import ModelBundle
from src.modeling import ModelCase, TrainingConfig, train_study
from src.modeling_data import fixed_batch_partition, modeling_dataset
from src.modeling_storage import evaluate_once, save_study
from src.monitoring import MonitoringSettings, predict_observation
from src.monitoring_pipeline import integrate_received_sources, transformed_observation
from src.stream_monitoring import AcquiredStreamSession, StreamSettings, acquired_stream_report_zip
from src.test_acquisition_source import make_test_server


@pytest.fixture(scope="module")
def saved_model(tmp_path_factory):
    config = SimulationConfig(n_batches=8, duration_h=72, n_wavenumbers=71, seed=715)
    ranges = dict(SENSOR_RANGES, ph=(6.7, 7.3))
    integration = integrate_sources(sources_from_dataset(generate_dataset(config)), IntegrationConfig(sensor_tolerance_minutes=5), sensor_ranges=ranges)
    cohort = modeling_dataset(integration)
    study = train_study(cohort, fixed_batch_partition(cohort.metadata.batch_id), [ModelCase("preprocessed", "pls")], TrainingConfig(max_pls_components=2))
    root = tmp_path_factory.mktemp("workflow-policy")
    save_study(study, root)
    evaluate_once(study, cohort, root)
    from src.model_registry import discover_models, load_model
    reference = discover_models(root, root / "none")[0][0]
    return config, load_model(reference), root


def test_input_policy_is_saved_with_exact_integration_and_quality_settings(saved_model):
    _, bundle, _ = saved_model
    policy, recorded = shared_input_policy({"Glucose": bundle})
    assert recorded
    assert policy["integration"]["sensor_tolerance_minutes"] == 5
    assert policy["sensor_ranges"]["ph"] == [6.7, 7.3]
    assert bundle.pipeline.input_policy_ == bundle.metadata["input_policy"]


def test_inference_cannot_loosen_saved_tolerance_or_ranges(saved_model):
    config, bundle, _ = saved_model
    from src.replay import replay_batch
    batch = replay_batch(generate_dataset(config), "B001")
    row, metadata = batch.inputs.iloc[[20]].copy(), batch.metadata.iloc[20]
    result = predict_observation(bundle, "Glucose", row, batch.axis, metadata.measurement_timestamp,
        metadata.measurement_timestamp - pd.Timedelta(minutes=6), MonitoringSettings(max_sensor_age_minutes=240, domain_z_limit=100))
    assert result.status == "invalid_input" and any("too old" in flag for flag in result.flags)
    row["ph"] = 7.4  # Within generic demo plausibility, outside this model's saved rule.
    result = predict_observation(bundle, "Glucose", row, batch.axis, metadata.measurement_timestamp,
        metadata.sensor_timestamp, MonitoringSettings(domain_z_limit=100))
    assert result.status == "invalid_input" and any("ph" in flag for flag in result.flags)


def test_conflicting_or_retrospective_saved_policies_do_not_silently_change(saved_model):
    _, bundle, _ = saved_model
    other = deepcopy(bundle)
    other.metadata["input_policy"]["integration"]["sensor_tolerance_minutes"] = 10
    with pytest.raises(ValueError, match="matching"):
        shared_input_policy({"Glucose": bundle, "Lactate": other})
    other.metadata["input_policy"]["integration"]["sensor_direction"] = "nearest"
    with pytest.raises(ValueError, match="nearest"):
        shared_input_policy({"Glucose": other})


def test_engine_applies_saved_quality_rules_without_relying_on_ui_configuration(saved_model):
    config, bundle, _ = saved_model
    records = source_records(sources_from_dataset(generate_monitoring_dataset(replace(config, n_batches=1))))
    first = [deepcopy(next(event for event in records if event["source"] == name)) for name in ("batches", "sensors", "raman")]
    first[1]["data"]["ph"] = 7.4
    for sequence, event in enumerate(first):
        event["sequence"] = sequence
    session = AcquiredStreamSession("saved-policy-in-engine")
    assert session.poll_once(InMemoryTestAdapter(first), {"Glucose": bundle}, MonitoringSettings(max_sensor_age_minutes=240), StreamSettings(enforce_freshness=False))
    row = session.history().iloc[0]
    assert row.glucose_status == "invalid_input"
    assert "ph" in row.quality_flags
    assert np.isnan(row.glucose_prediction_g_l)


def test_received_integration_preprocessing_and_export_match_saved_artifacts(saved_model, monkeypatch):
    config, bundle, root = saved_model
    original_artifacts = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    sources = sources_from_dataset(generate_monitoring_dataset(replace(config, n_batches=1)))
    adapter = InMemoryTestAdapter(source_records(sources))
    session = AcquiredStreamSession("single-shared-collection")
    policy = bundle.metadata["input_policy"]
    monitoring = MonitoringSettings(max_sensor_age_minutes=5, input_sensor_ranges=policy["sensor_ranges"], domain_z_limit=100)
    rules = StreamSettings(enforce_freshness=False)
    monkeypatch.setattr(type(bundle.pipeline), "fit", lambda *args, **kwargs: pytest.fail("No model fitting while monitoring"))
    while not session.acquisition.complete:
        assert session.poll_once(adapter, {"Glucose": bundle}, monitoring, rules, limit=100)
    actual = integrate_received_sources(session.acquisition.snapshot(), policy)
    expected = integrate_sources(sources, IntegrationConfig(**policy["integration"]), sensor_ranges=policy["sensor_ranges"])
    pd.testing.assert_frame_equal(actual.integrated, expected.integrated)
    history = session.history()
    event = history.iloc[0].spectrum_event_id
    features = transformed_observation(session, bundle, event)
    from src.replay import replay_batch
    inputs = replay_batch(generate_monitoring_dataset(replace(config, n_batches=1)), "MON-B001").inputs.iloc[[0]]
    raw = bundle.pipeline.named_steps["features"].transform(inputs)
    scaled = bundle.pipeline.named_steps["scale"].transform(raw)
    np.testing.assert_allclose(features.preprocessed_value, np.asarray(raw).ravel())
    np.testing.assert_allclose(features.standardized_value, np.asarray(scaled).ravel())
    with ZipFile(BytesIO(acquired_stream_report_zip(session, {"Glucose": bundle}, monitoring, rules, {"input_policy": policy}))) as archive:
        assert json.loads(archive.read("processing/input_policy.json")) == policy
        assert "processing/integrated.csv" in archive.namelist()
        assert "processing/latest_glucose_features.csv" in archive.namelist()
        assert "received/raman.csv" in archive.namelist()
    assert original_artifacts == {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_monitoring_api_reads_contract_and_keeps_token_out_of_records():
    config = SimulationConfig(n_batches=1, duration_h=24, n_wavenumbers=32)
    producer = InMemoryTestAdapter(source_records(sources_from_dataset(generate_monitoring_dataset(config))))
    server = make_test_server(producer)
    worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        adapter = MonitoringAPIAdapter(f"http://127.0.0.1:{server.server_port}", bearer_token="test-secret-value")
        page = adapter.poll(0, 5)
        assert page.next_cursor == 5
        assert page.records[0]["data"]["batch_id"] == "MON-B001"
        assert "test-secret-value" not in json.dumps(page.as_dict())
        assert adapter.headers["Authorization"] == "Bearer test-secret-value"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_validated_model_acquisition_to_process_monitoring_shares_all_processing_evidence(saved_model, tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from sklearn.pipeline import Pipeline
    config, bundle, root = saved_model
    before = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    monkeypatch.setattr(Pipeline, "fit", lambda *args, **kwargs: pytest.fail("Acquisition must not train"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=90)
    app.session_state["simulation_config"] = replace(config, n_batches=6)
    app.session_state["modeling_state_root"] = str(root)
    app.session_state["monitoring_models_root"] = str(tmp_path / "unused-demo-models")
    app.run()
    app.switch_page("pages/9_Test_Acquisition.py").run()
    app.selectbox(key="stream_source").select("Historical in-app test source").run()
    assert app.number_input(key="stream_sensor_age").disabled
    assert app.number_input(key="stream_sensor_age").value == 5
    assert not any(button.key == "prepare_stream_models" for button in app.button)
    app.button(key="acq_poll").click().run()
    assert not app.exception
    session = app.session_state["stream_monitoring_session"]
    cursor = session.acquisition.cursor
    history = session.history().copy()
    monkeypatch.setattr(Pipeline, "predict", lambda *args, **kwargs: pytest.fail("Navigation must not infer an observation again"))
    app.switch_page("pages/5_Process_Monitoring.py").run()
    assert not app.exception
    assert app.session_state["stream_monitoring_session"] is session
    assert session.acquisition.cursor == cursor
    pd.testing.assert_frame_equal(session.history(), history)
    assert any("sensor_matched" in frame.value for frame in app.dataframe)
    assert any("standardized_value" in frame.value for frame in app.dataframe)
    assert any("glucose_status" in frame.value for frame in app.dataframe)
    assert before == {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("url", ["http://remote.example", "https://user:secret@example.com", "file:///tmp/source", "https://example.com?token=secret"])
def test_monitoring_api_rejects_insecure_or_embedded_credentials(url):
    with pytest.raises(AcquisitionError):
        MonitoringAPIAdapter(url)
