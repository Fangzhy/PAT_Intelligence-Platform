"""Fresh, paced source through actual HTTP and saved online inference end to end."""

from threading import Thread

import numpy as np
import pandas as pd

from src.acquisition import LocalHTTPAdapter, test_source_records as source_records
from src.data_generation import SimulationConfig, generate_dataset
from src.data_integration import sources_from_dataset
from src.model_registry import load_model, prepare_demo_models
from src.monitoring import MonitoringSettings
from src.stream_monitoring import AcquiredStreamSession, StreamSettings
from src.test_acquisition_source import make_test_server
from src.test_stream_source import PacedTestAdapter


def test_actual_http_fresh_source_idle_recovery_and_saved_predictions(tmp_path):
    config = SimulationConfig(n_batches=1, duration_h=24, n_wavenumbers=32)
    models = {ref.target: load_model(ref) for ref in prepare_demo_models(config, tmp_path)}
    records = source_records(sources_from_dataset(generate_dataset(config)))
    clock = [0.0]
    start = pd.Timestamp.now(tz="UTC")
    source = PacedTestAdapter(records, started_at=start, clock=lambda: clock[0])
    server = make_test_server(source)
    worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        adapter = LocalHTTPAdapter(f"http://127.0.0.1:{server.server_port}")
        session = AcquiredStreamSession("http-end-to-end")
        monitoring, rules = MonitoringSettings(domain_z_limit=100), StreamSettings()
        assert session.poll_once(adapter, models, monitoring, rules, limit=100, now=start)
        assert session.acquisition.cursor == 3
        assert session.history().glucose_status.iloc[0] == "ok"
        assert np.isfinite(session.history().glucose_prediction_g_l.iloc[0])
        assert session.available_assays(now=start).empty
        assert session.poll_once(adapter, models, monitoring, rules, limit=100, now=start + pd.Timedelta(milliseconds=100))
        assert session.acquisition.cursor == 3 and not session.acquisition.complete
        clock[0] = 6
        assert session.poll_once(adapter, models, monitoring, rules, limit=100, now=start + pd.Timedelta(seconds=6))
        assert len(session.available_assays(now=start + pd.Timedelta(seconds=6))) == 1
        clock[0] = 60
        assert session.poll_once(adapter, models, monitoring, rules, limit=100, now=start + pd.Timedelta(seconds=60))
        assert session.acquisition.complete and len(session.history()) == 25
        assert session.history().glucose_status.eq("ok").all()
        assert session.history().glucose_model_version.eq(models["Glucose"].version).all()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
