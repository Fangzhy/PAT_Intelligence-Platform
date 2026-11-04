"""Upgrade cached imports in isolated interpreters without hiding normal errors."""

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]


def isolated(code):
    # Reloading exception/dataclass definitions in the pytest interpreter would
    # itself invalidate references held by unrelated test modules. A subprocess
    # accurately models a separate long-running Streamlit server instead.
    result = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", textwrap.dedent(code)],
                            cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]


def test_no_cached_llm_modules_means_no_imports_or_reload():
    isolated('''
        import sys
        from src.llm_bootstrap import refresh_legacy_llm_modules
        assert not refresh_legacy_llm_modules()
        assert not any(name in sys.modules for name in ("src.llm_results", "src.llm_client", "src.llm_ui"))
    ''')


def test_current_modules_are_not_reloaded_and_model_state_is_preserved(monkeypatch):
    import src.llm_bootstrap as bootstrap
    import src.llm_results
    import src.llm_client
    import src.llm_ui
    import src.modeling
    scientific_module = sys.modules["src.modeling"]
    result_class = src.llm_results.ResultSnapshot
    monkeypatch.setattr(bootstrap.importlib, "reload", lambda *a: pytest.fail("Current modules must not reload"))
    assert not bootstrap.refresh_legacy_llm_modules()
    assert src.llm_results.ResultSnapshot is result_class
    assert sys.modules["src.modeling"] is scientific_module


@pytest.mark.parametrize("standalone", [False, True])
def test_legacy_cached_exports_recover_before_spectral_page_import(standalone):
    isolated(f'''
        from pathlib import Path
        import requests
        from streamlit.testing.v1 import AppTest
        import src.llm_results as results
        import src.llm_client as client
        import src.llm_ui as ui
        import src.modeling as modeling
        from src.data_generation import SimulationConfig

        def forbidden(*args, **kwargs):
            raise AssertionError("Import recovery attempted network access or model training")
        requests.request = forbidden
        modeling.train_study = forbidden
        original_scientific_module = modeling
        # Simulate the previously working two-page explanation module version.
        for name in ("spectral_snapshot", "process_snapshot", "multivariate_snapshot"):
            delattr(results, name)
        delattr(ui, "render_captured_explanation")
        results.PROMPT_VERSION = client.PROMPT_VERSION = ui.PROMPT_VERSION = "legacy"

        entry = "pages/3_Spectral_Analytics.py" if {standalone!r} else "app.py"
        app = AppTest.from_file(str(Path(entry).resolve()), default_timeout=60)
        app.session_state["simulation_config"] = SimulationConfig(n_batches=6, duration_h=24, n_wavenumbers=32)
        app.run()
        if not {standalone!r}:
            app.switch_page("pages/3_Spectral_Analytics.py").run()
        assert not app.exception, str(app.exception)
        assert callable(results.spectral_snapshot)
        assert callable(results.process_snapshot)
        assert callable(results.multivariate_snapshot)
        assert callable(ui.render_captured_explanation)
        assert results.PROMPT_VERSION == client.PROMPT_VERSION == ui.PROMPT_VERSION
        assert results.PROMPT_VERSION != "legacy"
        assert modeling is original_scientific_module
        assert modeling.train_study is forbidden
        app.run()
        assert not app.exception
    ''')
