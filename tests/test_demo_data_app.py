"""Provenance and the separate monitoring population across the visible workflow."""

from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from src.app_data import load_dataset, load_monitoring_dataset
from src.data_generation import SimulationConfig
from src.demo_data import MONITORING_SEED_OFFSET, source_provenance


ROOT = Path(__file__).resolve().parents[1]
CONFIG = SimulationConfig(n_batches=6, duration_h=72, n_wavenumbers=71)


def test_acquisition_handoff_keeps_monitoring_provenance_and_development_config(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Collection must not train or evaluate a test")
    monkeypatch.setattr("src.modeling.train_study", forbidden)
    monkeypatch.setattr("src.modeling_storage.evaluate_holdout", forbidden)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    app.session_state["simulation_config"] = CONFIG
    app.session_state["modeling_state_root"] = str(tmp_path / "modeling")
    app.session_state["monitoring_models_root"] = str(tmp_path / "models")
    app.run()
    assert not app.exception
    assert any("historical development" in caption.value for caption in app.caption)
    app.switch_page("pages/9_Test_Acquisition.py").run()
    assert app.selectbox(key="stream_fixture_batch").value == "MON-B001"
    assert any("Generation seed: 3000059" in caption.value for caption in app.caption)
    app.button(key="acq_poll").click().run()
    assert not app.exception
    snapshot = app.session_state["acquisition_session"].snapshot()
    assert source_provenance(snapshot.batches)["sources"][0]["generation_seed"] == CONFIG.seed + MONITORING_SEED_OFFSET
    expected = load_monitoring_dataset(CONFIG).spectra.iloc[0].to_numpy()
    columns = [key for key in snapshot.raman if key.startswith("raman_")]
    np.testing.assert_allclose(snapshot.raman.iloc[0][columns].to_numpy(dtype=float), expected)
    assert not np.allclose(expected, load_dataset(CONFIG).spectra.iloc[0])
    cursor = app.session_state["acquisition_session"].cursor
    app.switch_page("pages/5_Process_Monitoring.py").run()
    assert not app.exception
    assert app.session_state["stream_monitoring_session"].acquisition.cursor == cursor
    app.switch_page("pages/2_Data_Integration.py").run()
    assert not app.exception
    assert source_provenance(app.session_state["integration_result"].processed.batches)["sources"][0]["data_role"] == "historical_development"
    assert app.session_state["simulation_config"] == CONFIG
    assert not list(tmp_path.iterdir())


def test_replay_uses_independent_demo_and_export_context_carries_provenance(tmp_path):
    app = AppTest.from_file(str(ROOT / "pages/5_Process_Monitoring.py"), default_timeout=90)
    app.session_state["simulation_config"] = CONFIG
    app.session_state["modeling_state_root"] = str(tmp_path / "modeling")
    app.session_state["monitoring_models_root"] = str(tmp_path / "models")
    app.run()
    app.button(key="replay_step").click().run()
    assert not app.exception
    history = next(item.value for item in app.dataframe if "glucose_status" in item.value)
    assert history.batch_id.tolist() == ["MON-B001"]
    assert history.sample_id.tolist() == ["MON-B001-S00000"]
    assert any("Generation seed: 3000059" in caption.value for caption in app.caption)
    assert app.session_state["simulation_config"].seed == 42
