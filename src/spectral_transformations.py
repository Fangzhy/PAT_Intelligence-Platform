"""Derivatives, Fourier analysis, and wavelets with explicit numerical conventions."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
import pywt
from scipy import fft
from scipy.signal import detrend as scipy_detrend

from src.preprocessing import spectral_array, validate_axis


def finite_difference_derivative(values, axis, order=1):
    """First gradient or second derivative, including second-order one-sided edges."""
    matrix, vector = spectral_array(values, minimum_channels=4)
    axis = validate_axis(axis, matrix.shape[1], uniform=True)
    if order == 1:
        result = np.gradient(matrix, axis, axis=1, edge_order=2)
    elif order == 2:
        spacing_squared = (axis[1] - axis[0]) ** 2
        result = np.empty_like(matrix)
        result[:, 1:-1] = (matrix[:, 2:] - 2 * matrix[:, 1:-1] + matrix[:, :-2]) / spacing_squared
        result[:, 0] = (2 * matrix[:, 0] - 5 * matrix[:, 1] + 4 * matrix[:, 2] - matrix[:, 3]) / spacing_squared
        result[:, -1] = (2 * matrix[:, -1] - 5 * matrix[:, -2] + 4 * matrix[:, -3] - matrix[:, -4]) / spacing_squared
    else:
        raise ValueError("Finite-difference derivative order must be 1 or 2.")
    return result[0] if vector else result


def laplacian_1d(values, axis):
    """The one-dimensional Laplacian is the second spatial derivative, not a transform."""
    return finite_difference_derivative(values, axis, order=2)


def discrete_fourier_transform(values):
    """Direct O(N^2) DFT for small numerical teaching examples, not full matrices."""
    matrix, vector = spectral_array(values, minimum_channels=2)
    if not vector or matrix.shape[1] > 2048:
        raise ValueError("Direct DFT expects one short vector (at most 2048 values); use FFT for larger inputs.")
    signal = matrix[0]
    indices = np.arange(len(signal))
    return np.exp(-2j * np.pi * indices[:, None] * indices / len(signal)) @ signal


@dataclass
class FourierResult:
    frequencies: np.ndarray
    coefficients: np.ndarray
    amplitude: np.ndarray
    power: np.ndarray
    prepared_signal: np.ndarray
    analyzed_signal: np.ndarray
    nyquist: float
    window: str
    detrend: str


def fourier_analysis(values, axis, *, window="none", detrend="none") -> FourierResult:
    """One-sided FFT amplitude and mean-square contribution per frequency bin.

    Amplitude accounts for window coherent gain. Power is |FFT|^2/N^2 with
    interior bins doubled, so its sum equals mean(analyzed_signal^2). This is
    per-bin mean square, not a spectral density. Frequencies are cycles per
    unit of Raman shift (cm^-1), not temporal Hz or molecular frequencies.
    """
    matrix, vector = spectral_array(values)
    if not vector:
        raise ValueError("Fourier exploration expects one spectrum.")
    axis = validate_axis(axis, matrix.shape[1], uniform=True)
    signal = matrix[0]
    if detrend == "constant":
        prepared = signal - signal.mean()
    elif detrend == "linear":
        prepared = scipy_detrend(signal, type="linear")
    elif detrend == "none":
        prepared = signal.copy()
    else:
        raise ValueError("Detrending must be none, constant, or linear.")
    if window not in ("none", "hann"):
        raise ValueError("Window must be none or hann.")
    taper = np.ones(len(signal)) if window == "none" else np.hanning(len(signal))
    analyzed = prepared * taper
    coefficients = fft.rfft(analyzed)
    weights = np.full(len(coefficients), 2.0)
    weights[0] = 1
    if len(signal) % 2 == 0:
        weights[-1] = 1
    spacing = axis[1] - axis[0]
    return FourierResult(
        frequencies=fft.rfftfreq(len(signal), d=spacing), coefficients=coefficients,
        amplitude=weights * np.abs(coefficients) / taper.sum(),
        power=weights * np.abs(coefficients) ** 2 / len(signal) ** 2,
        prepared_signal=prepared, analyzed_signal=analyzed, nyquist=1 / (2 * spacing),
        window=window, detrend=detrend,
    )


def _frequency_mask(result, lower, upper):
    upper = result.nyquist if upper is None else upper
    if not np.isfinite([lower, upper]).all() or not 0 <= lower <= upper <= result.nyquist:
        raise ValueError("Frequency band must satisfy 0 <= lower <= upper <= Nyquist.")
    return (result.frequencies >= lower) & (result.frequencies <= upper)


def reconstruct_fourier(result: FourierResult, lower=0.0, upper=None):
    """Keep complex coefficients in a band and reconstruct the analyzed signal.

    With no masking this reconstructs the detrended/windowed signal exactly.
    It does not undo a Hann taper or restore a removed trend. A hard frequency
    cutoff can cause ringing, and FFT treats endpoints as periodically joined.
    """
    mask = _frequency_mask(result, lower, upper)
    return fft.irfft(np.where(mask, result.coefficients, 0), n=len(result.analyzed_signal))


def frequency_band_features(result: FourierResult, lower=0.0, upper=None):
    mask = _frequency_mask(result, lower, upper)
    band_power = float(result.power[mask].sum())
    total = float(result.power.sum())
    return {"selected_bins": int(mask.sum()), "band_mean_square": band_power, "fraction_of_analyzed_mean_square": band_power / total if total > 0 else 0.0}


@dataclass
class WaveletResult:
    coefficients: list[np.ndarray]
    original_length: int
    wavelet: str
    level: int
    mode: str

    @property
    def labels(self):
        return [f"A{self.level}"] + [f"D{level}" for level in range(self.level, 0, -1)]


def wavelet_decompose(values, *, wavelet="db2", level=3):
    matrix, vector = spectral_array(values)
    if not vector:
        raise ValueError("Wavelet exploration expects one spectrum.")
    try:
        basis = pywt.Wavelet(wavelet)
    except (ValueError, TypeError) as error:
        raise ValueError("Choose a supported discrete wavelet.") from error
    maximum = pywt.dwt_max_level(matrix.shape[1], basis.dec_len)
    if isinstance(level, bool) or not isinstance(level, Integral) or not 1 <= level <= maximum:
        raise ValueError(f"Wavelet level must be an integer between 1 and {maximum} for this signal and wavelet.")
    coefficients = pywt.wavedec(matrix[0], basis, mode="symmetric", level=level)
    return WaveletResult(coefficients, matrix.shape[1], wavelet, level, "symmetric")


def reconstruct_wavelet(result: WaveletResult):
    """Trim symmetric-padding reconstruction to the original (possibly odd) length."""
    return pywt.waverec(result.coefficients, result.wavelet, mode=result.mode)[:result.original_length]


def wavelet_denoise(result: WaveletResult, *, threshold_scale=1.0):
    """Universal soft threshold from finest-detail MAD, leaving approximation intact."""
    if not np.isfinite(threshold_scale) or threshold_scale < 0:
        raise ValueError("Wavelet threshold scale must be finite and nonnegative.")
    finest = result.coefficients[-1]
    sigma = np.median(np.abs(finest - np.median(finest))) / 0.6744897501960817
    threshold = float(threshold_scale * sigma * np.sqrt(2 * np.log(result.original_length)))
    coefficients = [result.coefficients[0].copy()]
    coefficients.extend(pywt.threshold(detail, threshold, mode="soft") if threshold > 0 else detail.copy() for detail in result.coefficients[1:])
    denoised = pywt.waverec(coefficients, result.wavelet, mode=result.mode)[:result.original_length]
    return denoised, threshold


def wavelet_features(result: WaveletResult):
    """Coefficient summaries; symmetric padding means these are not conserved energies."""
    return pd.DataFrame([
        {"band": label, "coefficients": len(values), "squared_norm": float(values @ values), "mean": float(values.mean()), "std": float(values.std(ddof=0))}
        for label, values in zip(result.labels, result.coefficients)
    ])
