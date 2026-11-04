"""Manual network actions, private caches, consent, and actual page integration."""

from pathlib import Path
import json

import pytest
import requests
from streamlit.testing.v1 import AppTest

from src.data_generation import SimulationConfig
from src.llm_client import LLMConfig, local_result
from src.reliability_storage import prepare_reliability_demo


ROOT = Path(__file__).resolve().parents[1]
CONFIG = SimulationConfig(n_batches=6, duration_h=72, n_wavenumbers=71)
HARNESS = '''
import json
import streamlit as st
from src.llm_results import demo_snapshot, ResultSnapshot
from src.llm_ui import render_explanation
snapshot = demo_snapshot()
data = snapshot.payload
if st.checkbox("Change current results", key="change"):
    data["facts"][0]["value"] = 4.2
    data["snapshot_id"] = "f" * 64
if st.session_state.get("prepared_source"):
    data["source_kind"] = "prepared_sources"
render_explanation(ResultSnapshot(json.dumps(data)), key="test")
'''


@pytest.fixture
def fake_ai(monkeypatch):
    import src.llm_ui as ui
    monkeypatch.setattr(requests, "request", lambda *a, **kw: pytest.fail("Unexpected real network request"))
    monkeypatch.setattr(ui, "_configuration", lambda: LLMConfig("test-secret", "vendor/primary:free", "vendor/backup:free"))
    calls = []
    class Service:
        def generate(self, snapshot, config, **kwargs):
            calls.append(snapshot)
            result = local_result(snapshot)
            result.update(source="backup", model="vendor/backup:free", reason="success", reported_cost=0)
            result["sections"]["interpretation"] = "Review these recorded metrics in their stated evidence scope."
            return result
    monkeypatch.setattr(ui, "ExplanationService", lambda runtime: Service())
    return calls


def test_manual_request_cache_template_switch_and_session_privacy(fake_ai):
    app = AppTest.from_string(HARNESS).run()
    assert not app.exception and fake_ai == []
    app.button(key="ai_test_explain").click().run()
    assert not app.exception and len(fake_ai) == 1
    assert any("Backup LLM" in caption.value for caption in app.caption)
    app.run()
    app.button(key="ai_test_explain").click().run()
    assert len(fake_ai) == 1
    app.button(key="ai_test_template").click().run()
    assert len(fake_ai) == 1 and any("Template summary" in caption.value for caption in app.caption)
    app.button(key="ai_test_explain").click().run()
    assert len(fake_ai) == 2
    another = AppTest.from_string(HARNESS).run()
    assert not another.exception and len(another.get("download_button")) == 0
    assert "test-secret" not in json.dumps(app.session_state["ai_test_cache"])


def test_changed_snapshot_or_style_never_presents_previous_text_as_current(fake_ai):
    app = AppTest.from_string(HARNESS).run()
    app.button(key="ai_test_explain").click().run()
    app.checkbox(key="change").check().run()
    assert not app.exception and len(fake_ai) == 1
    assert len(app.get("download_button")) == 0
    assert any("settings changed" in caption.value for caption in app.caption)
    app.button(key="ai_test_explain").click().run()
    app.radio(key="ai_test_style").set_value("Technical explanation").run()
    assert len(app.get("download_button")) == 0 and len(fake_ai) == 2


def test_imported_summary_requires_explicit_consent_but_local_template_does_not(fake_ai):
    app = AppTest.from_string(HARNESS)
    app.session_state["prepared_source"] = True
    app.run()
    assert app.button(key="ai_test_explain").disabled
    app.button(key="ai_test_template").click().run()
    assert not app.exception and fake_ai == []
    app.checkbox(key="ai_test_consent").check().run()
    app.button(key="ai_test_explain").click().run()
    assert not app.exception and len(fake_ai) == 1
    app.checkbox(key="change").check().run()
    assert not app.checkbox(key="ai_test_consent").value
    assert app.button(key="ai_test_explain").disabled


def test_disabled_ai_still_produces_template_with_no_network(monkeypatch):
    import src.llm_ui as ui
    monkeypatch.setattr(ui, "_configuration", lambda: LLMConfig("secret", enabled=False))
    monkeypatch.setattr(requests, "request", lambda *a, **kw: pytest.fail("Disabled AI connected"))
    app = AppTest.from_string(HARNESS).run()
    app.button(key="ai_test_explain").click().run()
    assert not app.exception
    assert any("Template summary" in caption.value for caption in app.caption)


def test_modeling_explanation_does_not_train_again_or_open_holdout(tmp_path, fake_ai, monkeypatch):
    app = AppTest.from_file(str(ROOT / "pages/4_Predictive_Modeling.py"), default_timeout=90)
    app.session_state["modeling_state_root"] = str(tmp_path)
    app.session_state["simulation_config"] = CONFIG
    app.run()
    app.multiselect(key="model_experiments").set_value(["PLS · Raw spectra"])
    app.slider(key="model_components").set_value(2)
    app.button(key="train_models").click().run()
    before = [metric.value for metric in app.metric]
    import src.modeling as modeling
    import src.modeling_storage as storage
    monkeypatch.setattr(modeling, "train_study", lambda *a, **kw: pytest.fail("Explanation refit a model"))
    monkeypatch.setattr(storage, "evaluate_once", lambda *a, **kw: pytest.fail("Explanation opened the holdout"))
    app.button(key="ai_modeling_explain").click().run()
    assert not app.exception and len(fake_ai) == 1
    assert [metric.value for metric in app.metric] == before
    assert "missing_test" in fake_ai[0].payload["limitations"]
    assert not list(tmp_path.glob("*/holdout_claim.json"))


def test_reliability_clock_change_hides_stale_explanation_and_keeps_assays_causal(tmp_path, fake_ai):
    root = tmp_path / "reliability"
    prepare_reliability_demo(CONFIG, "Glucose", root)
    app = AppTest.from_file(str(ROOT / "pages/7_Model_Reliability.py"), default_timeout=90)
    app.session_state["simulation_config"] = CONFIG
    app.session_state["reliability_state_root"] = str(root)
    app.run()
    app.button(key="reliability_latest").click().run()
    app.button(key="ai_reliability_explain").click().run()
    assert not app.exception and len(fake_ai) == 1
    app.slider(key="reliability_clock").set_value(0.0).run()
    assert not app.exception and len(fake_ai) == 1
    assert not any(button.key == "ai_reliability_download" for button in app.get("download_button"))
    app.button(key="ai_reliability_explain").click().run()
    snapshot = fake_ai[-1]
    available = next(fact for fact in snapshot.payload["facts"] if fact["label"] == "Available assays")
    assert available["value"] == 0
