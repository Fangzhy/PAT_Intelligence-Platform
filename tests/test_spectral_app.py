"""Phase 3 controls, plot units, reference scope, and cross-phase regressions."""

import json
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from src.data_generation import SimulationConfig


ROOT = Path(__file__).resolve().parents[1]
PAGE = "pages/3_Spectral_Analytics.py"


def page():
    return AppTest.from_file(str(ROOT / PAGE), default_timeout=60).run()


def test_spectral_laboratory_loads_with_reference_and_projected_batches():
    app = page()
    assert not app.exception
    assert app.metric[0].value == "870"
    assert app.metric[1].value == "281"
    assert app.metric[2].value == "435"
    assert len(app.tabs) == 6
    variance = next(frame.value for frame in app.dataframe if "explained_variance_ratio" in frame.value)
    assert np.isfinite(variance["explained_variance_ratio"]).all()
    assert 0 < variance["cumulative_explained_variance"].iloc[-1] <= 1.00000001


def test_comparison_panels_preserve_frequency_coefficient_and_derivative_axes():
    app = page()
    app.multiselect(key="transform_comparison").set_value([
        "FFT magnitude", "Wavelet decomposition", "Savitzky–Golay first derivative", "1-D Laplacian",
    ]).run()
    assert not app.exception
    figures = [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]
    labels = [figure["layout"].get("xaxis", {}).get("title", {}).get("text", "") for figure in figures]
    ylabels = [figure["layout"].get("yaxis", {}).get("title", {}).get("text", "") for figure in figures]
    assert "Spatial frequency (cycles per cm⁻¹)" in labels
    assert "Concatenated coefficient index (A then details)" in labels
    assert "dI/dν (a.u. per cm⁻¹)" in ylabels
    assert "d²I/dν² (a.u. per (cm⁻¹)²)" in ylabels


def test_region_window_and_wavelet_controls_remain_valid_when_region_shrinks():
    app = page()
    app.selectbox(key="spectral_sg_window").select(31).run()
    app.slider(key="spectral_region").set_value((600.0, 1600.0)).run()
    assert not app.exception
    assert app.metric[1].value == "201"
    app.slider(key="spectral_region").set_value((400.0, 430.0)).run()
    assert not app.exception
    assert app.metric[1].value == "7"
    assert app.selectbox(key="spectral_sg_window").value <= 7
    app.selectbox(key="spectral_wavelet").select("db4").run()
    assert not app.exception
    assert any("too short" in warning.value for warning in app.warning)


def test_no_reference_batches_abstains_for_msc_and_pca_but_keeps_raw_analysis():
    app = page()
    app.multiselect(key="spectral_references").set_value([]).run()
    app.multiselect(key="transform_comparison").set_value(["Raw", "MSC"]).run()
    assert not app.exception
    assert app.metric[2].value == "0"
    assert any("reference batch for MSC" in warning.value for warning in app.warning)
    assert any("at least three complete spectra" in warning.value for warning in app.warning)
    app.multiselect(key="spectral_batches").set_value([]).run()
    assert not app.exception
    assert any("Select at least one batch" in warning.value for warning in app.warning)


def test_phase_two_quality_snapshot_is_available_and_incomplete_spectra_are_reported():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app.switch_page("pages/2_Data_Integration.py").run()
    app.checkbox(key="quality_demo").check().run()
    assert app.metric[1].value == "868"
    app.switch_page(PAGE).run()
    app.radio(key="spectral_source").set_value("Latest Phase 2 prepared Raman").run()
    assert not app.exception
    assert app.metric[0].value == "869"
    excluded = next(frame.value for frame in app.dataframe if "reason" in frame.value)
    assert excluded["sample_id"].tolist() == ["B001-S00004"]
    app.switch_page("pages/2_Data_Integration.py").run()
    assert not app.exception
    # Returning to Phase 2 may reset that page's demonstration widget; the
    # current result must remain internally consistent in either mode.
    expected = "868" if app.checkbox(key="quality_demo").value else "870"
    assert app.metric[1].value == expected
    app.switch_page("pages/1_Overview.py").run()
    assert not app.exception
    assert app.metric[0].value == "6"


def test_missing_phase_two_snapshot_has_actionable_guidance():
    app = page()
    app.radio(key="spectral_source").set_value("Latest Phase 2 prepared Raman").run()
    assert not app.exception
    assert any("Run Data Integration first" in warning.value for warning in app.warning)


def test_pca_form_runs_composite_preprocessing_on_reference_batches():
    app = AppTest.from_file(str(ROOT / PAGE), default_timeout=60)
    app.session_state["simulation_config"] = SimulationConfig(n_batches=3, duration_h=72, n_wavenumbers=71)
    app.run()
    app.multiselect(key="spectral_references").set_value(["B001", "B002"]).run()
    app.selectbox(key="pca_components").select(2)
    app.selectbox(key="pca_scatter").select("msc")
    app.checkbox(key="pca_baseline").check()
    app.checkbox(key="pca_autoscale").check()
    app.selectbox(key="pca_derivative").select(1)
    app.button(key="fit_pca").click().run()
    assert not app.exception
    assert app.metric[2].value == "146"
    variance = next(frame.value for frame in app.dataframe if "explained_variance_ratio" in frame.value)
    assert len(variance) == 2
    app.checkbox(key="pca_smoothing").uncheck()
    app.button(key="fit_pca").click().run()
    assert not app.exception
    assert any("require a Savitzky" in warning.value for warning in app.warning)
