"""Reference-only fitting, analytical PCA reconstruction, and source adapters."""

import numpy as np
import pytest

from src.chemometrics import fit_spectral_pca, project_spectral_pca
from src.data_generation import SimulationConfig, generate_dataset
from src.data_integration import demonstrate_quality_issues, integrate_sources, sources_from_dataset
from src.preprocessing import SpectralPreprocessor
from src.spectral_data import raman_from_integration, raman_from_simulation


def test_rank_two_pca_explains_known_variation_and_reconstructs_exactly():
    axis = np.arange(8)
    rng = np.random.default_rng(12)
    basis = np.array([[1, 2, 3, 4, 3, 2, 1, 0], [0, 1, 0, -1, 0, 1, 0, -1]])
    spectra = rng.normal(size=(20, 2)) @ basis + 4
    pipeline = fit_spectral_pca(spectra[:15], axis, n_components=2)
    scores, residual = project_spectral_pca(pipeline, spectra)
    assert scores.shape == (20, 2)
    np.testing.assert_allclose(residual, 0, atol=1e-13)
    assert pipeline.named_steps["pca"].explained_variance_ratio_.sum() == pytest.approx(1)


def test_msc_scaler_and_pca_are_not_refitted_when_projecting_extreme_rows():
    axis = np.linspace(400, 1800, 41)
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(20, 41)) + 5 + np.sin(axis / 100)
    preprocessing = SpectralPreprocessor(axis, scatter="msc", window_length=5)
    pipeline = fit_spectral_pca(reference, axis, n_components=3, autoscale=True, preprocessor=preprocessing)
    msc = pipeline.named_steps["spectra"].msc_
    before = [msc.reference_.copy(), pipeline.named_steps["scale"].mean_.copy(), pipeline.named_steps["scale"].scale_.copy(), pipeline.named_steps["pca"].components_.copy()]
    project_spectral_pca(pipeline, reference[:2] * 1e4 + 100)
    after = [msc.reference_, pipeline.named_steps["scale"].mean_, pipeline.named_steps["scale"].scale_, pipeline.named_steps["pca"].components_]
    for first, second in zip(before, after):
        np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(msc.reference_, reference.mean(axis=0))
    assert not hasattr(preprocessing, "msc_")  # The caller's template is cloned.


def test_mean_centering_uses_only_reference_statistics():
    reference = np.array([[1, 2, 3, 4], [2, 5, 3, 7], [4, 1, 5, 6], [3, 4, 2, 9]], dtype=float)
    pipeline = fit_spectral_pca(reference, np.arange(4), n_components=2, autoscale=True)
    np.testing.assert_allclose(pipeline.named_steps["scale"].mean_, reference.mean(axis=0))
    np.testing.assert_allclose(pipeline.named_steps["scale"].scale_, reference.std(axis=0, ddof=0))
    together, _ = project_spectral_pca(pipeline, reference[:2])
    separately = np.vstack([project_spectral_pca(pipeline, row)[0] for row in reference[:2]])
    np.testing.assert_allclose(together, separately, atol=1e-12)


@pytest.mark.parametrize("components", [0, 4, 1.5])
def test_invalid_pca_component_count_is_rejected(components):
    with pytest.raises(ValueError):
        fit_spectral_pca(np.arange(20).reshape(4, 5), np.arange(5), n_components=components)


def test_pca_rejects_constant_or_missing_reference_data():
    with pytest.raises(ValueError, match="variation"):
        fit_spectral_pca(np.ones((4, 5)), np.arange(5), n_components=2)
    with pytest.raises(ValueError, match="finite"):
        fit_spectral_pca(np.full((4, 5), np.nan), np.arange(5), n_components=2)


def test_simulation_adapter_preserves_spectral_identity_without_truth_features():
    dataset = generate_dataset(SimulationConfig(n_batches=2, duration_h=12, n_wavenumbers=32))
    collection = raman_from_simulation(dataset)
    assert collection.intensities.shape == (26, 32)
    assert collection.metadata["sample_id"].tolist() == dataset.spectral_metadata["sample_id"].tolist()
    assert "glucose_g_l" not in collection.metadata
    assert collection.excluded.empty


def test_phase_two_adapter_reports_incomplete_rows_and_preserves_accepted_identity():
    dataset = generate_dataset(SimulationConfig(n_batches=2, duration_h=12, n_wavenumbers=32))
    result = integrate_sources(demonstrate_quality_issues(sources_from_dataset(dataset)))
    collection = raman_from_integration(result)
    assert collection.intensities.shape == (25, 32)
    assert collection.excluded["sample_id"].tolist() == ["B001-S00004"]
    assert np.isfinite(collection.intensities).all()
    assert collection.metadata["sample_id"].iloc[0] == "B001-S00000"
    assert collection.metadata["elapsed_time_h"].iloc[0] == pytest.approx(1 / 6)
