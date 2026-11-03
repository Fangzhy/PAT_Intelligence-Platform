"""Analytical derivatives, known Fourier signals, and wavelet reconstruction."""

import numpy as np
import pywt
import pytest
from scipy import fft

from src.spectral_transformations import (
    discrete_fourier_transform, finite_difference_derivative, fourier_analysis,
    frequency_band_features, laplacian_1d, reconstruct_fourier,
    reconstruct_wavelet, wavelet_decompose, wavelet_denoise, wavelet_features,
)


def test_direct_dft_matches_known_four_point_signal_and_fft():
    signal = np.array([1, 0, -1, 0])
    np.testing.assert_allclose(discrete_fourier_transform(signal), [0, 2, 0, 2], atol=1e-12)
    np.testing.assert_allclose(discrete_fourier_transform(signal), fft.fft(signal), atol=1e-12)


@pytest.mark.parametrize("spacing", [0.2, 2.0, 5.0])
def test_derivatives_and_laplacian_match_quadratic_at_edges_and_interior(spacing):
    axis = np.arange(21) * spacing
    signal = axis**2 + 3 * axis + 2
    np.testing.assert_allclose(finite_difference_derivative(signal, axis, 1), 2 * axis + 3, atol=1e-10)
    np.testing.assert_allclose(finite_difference_derivative(signal, axis, 2), 2, atol=1e-10)
    np.testing.assert_allclose(laplacian_1d(signal, axis), 2, atol=1e-10)


def test_known_sinusoid_amplitude_frequency_and_mean_square():
    size, spacing = 64, 0.25
    signal = 3 + 2 * np.sin(2 * np.pi * 8 * np.arange(size) / size)
    result = fourier_analysis(signal, np.arange(size) * spacing)
    assert result.frequencies[8] == pytest.approx(8 / (size * spacing))
    assert result.amplitude[0] == pytest.approx(3)
    assert result.amplitude[8] == pytest.approx(2)
    assert result.power[8] == pytest.approx(2)
    assert result.power.sum() == pytest.approx(11)
    assert result.nyquist == pytest.approx(2)


def test_even_length_nyquist_bin_is_not_doubled():
    signal = (-1.0) ** np.arange(64)
    result = fourier_analysis(signal, np.arange(64))
    assert result.amplitude[-1] == pytest.approx(1)
    assert result.power[-1] == pytest.approx(1)


@pytest.mark.parametrize("size", [63, 64])
@pytest.mark.parametrize("window", ["none", "hann"])
@pytest.mark.parametrize("detrend", ["none", "constant", "linear"])
def test_fft_round_trip_and_parseval_for_odd_even_tapered_and_detrended_signals(size, window, detrend):
    signal = np.random.default_rng(5).normal(size=size) + np.linspace(0, 4, size)
    result = fourier_analysis(signal, np.arange(size) * 2, window=window, detrend=detrend)
    np.testing.assert_allclose(reconstruct_fourier(result), result.analyzed_signal, atol=2e-14)
    assert result.power.sum() == pytest.approx(np.mean(result.analyzed_signal**2))


def test_frequency_filter_removes_known_high_frequency_component():
    sample = np.arange(128)
    slow = np.sin(2 * np.pi * 4 * sample / 128)
    fast = 0.5 * np.sin(2 * np.pi * 30 * sample / 128)
    result = fourier_analysis(slow + fast, sample)
    filtered = reconstruct_fourier(result, upper=0.1)
    np.testing.assert_allclose(filtered, slow, atol=1e-13)
    features = frequency_band_features(result, upper=0.1)
    assert features["band_mean_square"] == pytest.approx(0.5)
    assert features["fraction_of_analyzed_mean_square"] == pytest.approx(0.8)


def test_fft_magnitude_loses_phase_but_complex_reconstruction_preserves_it():
    sample = np.arange(64)
    first, second = np.sin(2 * np.pi * 4 * sample / 64), np.cos(2 * np.pi * 4 * sample / 64)
    a, b = fourier_analysis(first, sample), fourier_analysis(second, sample)
    np.testing.assert_allclose(a.amplitude, b.amplitude, atol=1e-13)
    assert not np.allclose(a.coefficients, b.coefficients)
    np.testing.assert_allclose(reconstruct_fourier(b), second, atol=1e-13)


def test_hann_reconstruction_does_not_claim_to_restore_untapered_input():
    signal = np.ones(32)
    result = fourier_analysis(signal, np.arange(32), window="hann")
    np.testing.assert_allclose(reconstruct_fourier(result), np.hanning(32), atol=1e-14)
    assert not np.allclose(reconstruct_fourier(result), signal)
    assert frequency_band_features(fourier_analysis(np.zeros(32), np.arange(32)))["fraction_of_analyzed_mean_square"] == 0


@pytest.mark.parametrize("wavelet", ["db2", "db4", "haar"])
@pytest.mark.parametrize("size", [127, 128])
def test_wavelet_round_trip_and_zero_threshold_preserve_original_length(wavelet, size):
    signal = np.random.default_rng(8).normal(size=size)
    decomposition = wavelet_decompose(signal, wavelet=wavelet, level=3)
    np.testing.assert_allclose(reconstruct_wavelet(decomposition), signal, atol=1e-12)
    denoised, threshold = wavelet_denoise(decomposition, threshold_scale=0)
    assert threshold == 0
    np.testing.assert_allclose(denoised, signal, atol=1e-12)
    assert decomposition.labels == ["A3", "D3", "D2", "D1"]


def test_wavelet_denoising_improves_known_smooth_signal_with_gaussian_noise():
    axis = np.linspace(0, 2 * np.pi, 512)
    truth = np.sin(axis) + 0.2 * np.sin(3 * axis)
    noisy = truth + np.random.default_rng(10).normal(0, 0.2, len(axis))
    result = wavelet_decompose(noisy, wavelet="db4", level=4)
    denoised, threshold = wavelet_denoise(result)
    assert threshold > 0
    assert np.mean((denoised - truth) ** 2) < np.mean((noisy - truth) ** 2)
    features = wavelet_features(result)
    assert features["band"].tolist() == ["A4", "D4", "D3", "D2", "D1"]
    assert (features["squared_norm"] >= 0).all()


def test_constant_wavelet_signal_stays_finite():
    result = wavelet_decompose(np.zeros(64), wavelet="haar", level=3)
    denoised, _ = wavelet_denoise(result)
    np.testing.assert_array_equal(denoised, np.zeros(64))


@pytest.mark.parametrize("bounds", [(-1, 0.1), (0.2, 0.1), (0, 1), (np.nan, 0.1)])
def test_invalid_frequency_bands_are_rejected(bounds):
    result = fourier_analysis(np.arange(32), np.arange(32))
    with pytest.raises(ValueError):
        reconstruct_fourier(result, *bounds)


def test_nonuniform_fft_and_invalid_wavelet_levels_are_rejected():
    with pytest.raises(ValueError, match="uniform"):
        fourier_analysis([1, 2, 3, 4], [1, 2, 4, 5])
    for level in (0, 99, 1.5):
        with pytest.raises(ValueError, match="level"):
            wavelet_decompose(np.arange(32), level=level)
    with pytest.raises(ValueError, match="wavelet"):
        wavelet_decompose(np.arange(32), wavelet="not-a-wavelet")
    assert pywt.dwt_max_level(32, pywt.Wavelet("db2").dec_len) >= 1
