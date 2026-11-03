"""Spectral preprocessing with explicit shape, axis, and fit/transform contracts."""

from numbers import Integral

import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def spectral_array(values, minimum_channels=3):
    """Return a private 2-D float copy and whether the input was a single vector."""
    array = np.array(values, dtype=float, copy=True)
    vector = array.ndim == 1
    if vector:
        array = array[None, :]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < minimum_channels:
        raise ValueError(f"Provide a spectrum or matrix with at least {minimum_channels} channels and one row.")
    if not np.isfinite(array).all():
        raise ValueError("Spectra must be finite. Review missing/invalid rows in Phase 2; no automatic filling is performed.")
    return array, vector


def validate_axis(axis, n_channels=None, *, uniform=False):
    """Validate an increasing physical spectral axis; never resample it silently."""
    axis = np.array(axis, dtype=float, copy=True)
    if axis.ndim != 1 or len(axis) < 3 or not np.isfinite(axis).all() or (np.diff(axis) <= 0).any():
        raise ValueError("The spectral axis must contain at least three finite, strictly increasing values.")
    if n_channels is not None and len(axis) != n_channels:
        raise ValueError("Spectral axis length does not match the number of channels.")
    if uniform and not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=1e-7, atol=1e-10):
        raise ValueError("This method requires uniformly spaced spectral channels; no resampling is performed.")
    return axis


def select_region(values, axis, lower, upper):
    """Select a contiguous, inclusive physical region without changing raw inputs."""
    matrix, vector = spectral_array(values)
    axis = validate_axis(axis, matrix.shape[1])
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("Choose finite region bounds with lower < upper.")
    mask = (axis >= lower) & (axis <= upper)
    if mask.sum() < 3:
        raise ValueError("The selected region must contain at least three channels.")
    selected = matrix[:, mask]
    return (selected[0] if vector else selected), axis[mask]


def estimate_baseline(values, *, smoothness=1e5, asymmetry=0.01, iterations=15):
    """Asymmetric least-squares baseline with a discrete second-difference penalty.

    Minimizes sum(w * (y-z)^2) + lambda * ||D2 z||^2, iteratively assigning
    smaller weights above z. Assumes predominantly positive peaks on a smooth
    background. Lambda depends on channel spacing; this is not a calibrated
    fluorescence model. It operates separately on each spectrum.
    """
    matrix, vector = spectral_array(values)
    if not np.isfinite(smoothness) or smoothness <= 0 or not 0 < asymmetry < 0.5:
        raise ValueError("Baseline smoothness must be positive and asymmetry must lie between 0 and 0.5.")
    if isinstance(iterations, bool) or not isinstance(iterations, Integral) or iterations < 1:
        raise ValueError("Baseline iterations must be a positive integer.")
    size = matrix.shape[1]
    second_difference = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(size - 2, size), format="csc")
    penalty = smoothness * (second_difference.T @ second_difference)
    baselines = np.empty_like(matrix)
    for row_index, signal in enumerate(matrix):
        weights = np.ones(size)
        for _ in range(iterations):
            baseline = spsolve(sparse.diags(weights, format="csc") + penalty, weights * signal)
            updated = np.where(signal > baseline, asymmetry, 1 - asymmetry)
            if np.array_equal(weights, updated):
                break
            weights = updated
        baselines[row_index] = baseline
    return baselines[0] if vector else baselines


def snv(values):
    """Row-wise standard normal variate using the sample standard deviation."""
    matrix, vector = spectral_array(values)
    std = matrix.std(axis=1, ddof=1, keepdims=True)
    tolerance = np.finfo(float).eps * np.maximum(1, np.abs(matrix).max(axis=1, keepdims=True))
    if (std <= tolerance).any():
        raise ValueError("SNV is undefined for a constant or numerically constant spectrum.")
    transformed = (matrix - matrix.mean(axis=1, keepdims=True)) / std
    return transformed[0] if vector else transformed


def savitzky_golay(values, axis, *, window_length=11, polyorder=3, derivative=0):
    """Smooth or differentiate along channels using physical spacing and interp edges."""
    matrix, vector = spectral_array(values)
    axis = validate_axis(axis, matrix.shape[1], uniform=True)
    for name, value in (("window_length", window_length), ("polyorder", polyorder), ("derivative", derivative)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer.")
    if window_length < 3 or window_length % 2 == 0 or window_length > len(axis):
        raise ValueError("Savitzky–Golay window must be odd, at least 3, and no longer than the spectrum.")
    if not 0 <= derivative <= polyorder < window_length:
        raise ValueError("Require 0 <= derivative <= polynomial order < window length.")
    result = savgol_filter(matrix, window_length, polyorder, deriv=derivative, delta=axis[1] - axis[0], axis=1, mode="interp")
    return result[0] if vector else result


class MultiplicativeScatterCorrection(TransformerMixin, BaseEstimator):
    """Learn the reference mean spectrum on fit data only, then reuse it.

    Fits x_i = a_i + b_i * reference and returns (x_i - a_i) / b_i.
    Per-spectrum regression coefficients are computed at transform time; the
    reference is never recomputed from projected/inference rows.
    """

    def fit(self, X, y=None):
        matrix, vector = spectral_array(X)
        if vector:
            raise ValueError("Fit MSC using a 2-D reference matrix.")
        self.reference_ = matrix.mean(axis=0)
        self.n_features_in_ = matrix.shape[1]
        centered = self.reference_ - self.reference_.mean()
        self.reference_variation_ = float(centered @ centered)
        if self.reference_variation_ <= np.finfo(float).eps * max(1, float(self.reference_ @ self.reference_)):
            raise ValueError("MSC requires a nonconstant reference spectrum.")
        return self

    def transform(self, X):
        check_is_fitted(self, ["reference_", "reference_variation_"])
        matrix, vector = spectral_array(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("MSC channel count differs from its fitted reference.")
        reference_centered = self.reference_ - self.reference_.mean()
        means = matrix.mean(axis=1)
        slopes = (matrix - means[:, None]) @ reference_centered / self.reference_variation_
        if (np.abs(slopes) <= 1e-12).any():
            raise ValueError("MSC cannot correct a spectrum with zero or near-zero reference slope.")
        intercepts = means - slopes * self.reference_.mean()
        result = (matrix - intercepts[:, None]) / slopes[:, None]
        return result[0] if vector else result


class SpectralPreprocessor(TransformerMixin, BaseEstimator):
    """Region -> optional baseline -> scatter -> optional SG smoothing/derivative.

    Dataset-learned MSC references are learned only by fit. Row-wise methods
    have no cross-sample fitted state. Centering/scaling belongs to the next
    pipeline step, so its training-only statistics can be inspected separately.
    """

    def __init__(self, wavenumbers_cm1, region=None, baseline=False, baseline_smoothness=1e5,
                 baseline_asymmetry=0.01, scatter="none", window_length=None,
                 polyorder=3, derivative=0):
        self.wavenumbers_cm1 = wavenumbers_cm1
        self.region = region
        self.baseline = baseline
        self.baseline_smoothness = baseline_smoothness
        self.baseline_asymmetry = baseline_asymmetry
        self.scatter = scatter
        self.window_length = window_length
        self.polyorder = polyorder
        self.derivative = derivative

    def _before_scatter(self, matrix):
        selected = matrix[:, self.region_mask_].copy()
        if self.baseline:
            selected -= estimate_baseline(selected, smoothness=self.baseline_smoothness, asymmetry=self.baseline_asymmetry)
        return selected

    def fit(self, X, y=None):
        matrix, vector = spectral_array(X)
        if vector:
            raise ValueError("Fit preprocessing using a 2-D reference matrix.")
        axis = validate_axis(self.wavenumbers_cm1, matrix.shape[1])
        if self.scatter not in ("none", "snv", "msc"):
            raise ValueError("Scatter correction must be none, snv, or msc.")
        self.n_features_in_ = matrix.shape[1]
        self.region_mask_ = np.ones(len(axis), dtype=bool)
        if self.region is not None:
            select_region(matrix, axis, *self.region)
            self.region_mask_ = (axis >= self.region[0]) & (axis <= self.region[1])
        self.output_axis_ = axis[self.region_mask_]
        if self.window_length is None and self.derivative != 0:
            raise ValueError("Spectral derivatives require a Savitzky–Golay window.")
        self.msc_ = None
        if self.scatter == "msc":
            self.msc_ = MultiplicativeScatterCorrection().fit(self._before_scatter(matrix))
        # Validate every selected operation before accepting the fitted object.
        self.transform(matrix)
        return self

    def transform(self, X):
        check_is_fitted(self, ["output_axis_", "region_mask_"])
        matrix, vector = spectral_array(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("Preprocessing channel count differs from the fit data.")
        result = self._before_scatter(matrix)
        if self.scatter == "snv":
            result = snv(result)
        elif self.scatter == "msc":
            result = self.msc_.transform(result)
        if self.window_length is not None:
            result = savitzky_golay(result, self.output_axis_, window_length=self.window_length, polyorder=self.polyorder, derivative=self.derivative)
        return result[0] if vector else result
