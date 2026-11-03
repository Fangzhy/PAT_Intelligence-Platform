"""Fixed spectral recipes with learned MSC references fitted inside each fold."""

import numpy as np
import pandas as pd
import pywt
from scipy import fft
from scipy.signal import detrend
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.modeling_data import SENSOR_COLUMNS
from src.preprocessing import SpectralPreprocessor, estimate_baseline, validate_axis


FEATURE_SETS = {
    "raw": "Raw spectra",
    "preprocessed": "Baseline + MSC + SG spectra",
    "first_derivative": "SG first derivatives",
    "second_derivative": "SG second derivatives",
    "fft": "FFT amplitudes and band summaries",
    "wavelet": "Wavelet coefficients and summaries",
    "summary": "Fixed peak/area/ratio summaries",
    "sensors": "Five process sensors",
    "combined": "Preprocessed spectra + sensors",
}


class SpectralFeatures(TransformerMixin, BaseEstimator):
    """Transform the same named inputs identically at training and inference.

    Regions, windows, Fourier bands, and wavelet choices are fixed a priori in
    protocol v1. MSC reference spectra are learned from fit rows only. Extra
    input columns cannot silently become features.
    """

    def __init__(self, wavenumbers_cm1, feature_set="raw"):
        self.wavenumbers_cm1 = wavenumbers_cm1
        self.feature_set = feature_set

    def _input(self, X):
        if not isinstance(X, pd.DataFrame) or X.empty or not X.columns.is_unique:
            raise ValueError("Supply a nonempty DataFrame with unique input column names.")
        required = list(self.spectral_columns_) + (list(SENSOR_COLUMNS) if self.feature_set in ("sensors", "combined") else [])
        missing = set(required) - set(X.columns)
        if missing:
            raise ValueError(f"Missing model inputs: {', '.join(sorted(missing))}.")
        matrix = X[list(self.spectral_columns_)].to_numpy(dtype=float)
        if not np.isfinite(X[required].to_numpy(dtype=float)).all():
            raise ValueError("Model inputs must be finite; automatic imputation is not enabled.")
        return matrix

    def fit(self, X, y=None):
        if self.feature_set not in FEATURE_SETS:
            raise ValueError("Unknown feature recipe.")
        self.axis_ = validate_axis(self.wavenumbers_cm1, uniform=True)
        if len(self.axis_) < 11:
            raise ValueError("Feature recipes require at least eleven spectral channels.")
        self.spectral_columns_ = tuple(f"raman_{value:.12g}" for value in self.axis_)
        matrix = self._input(X)
        self.fit_row_ids_ = tuple(X.index.astype(str))
        self.n_features_in_ = len(X.columns)
        self.preprocessor_ = None
        if self.feature_set in ("preprocessed", "first_derivative", "second_derivative", "combined"):
            derivative = {"first_derivative": 1, "second_derivative": 2}.get(self.feature_set, 0)
            self.preprocessor_ = SpectralPreprocessor(self.axis_, baseline=True, scatter="msc", window_length=11, derivative=derivative).fit(matrix)
        _, names = self._extract(X, matrix)
        self.output_names_ = np.asarray(names, dtype=object)
        return self

    def _extract(self, X, matrix):
        if self.feature_set == "raw":
            return matrix.copy(), list(self.spectral_columns_)
        if self.feature_set == "sensors":
            return X[list(SENSOR_COLUMNS)].to_numpy(dtype=float), list(SENSOR_COLUMNS)
        if self.preprocessor_ is not None:
            values = self.preprocessor_.transform(matrix)
            names = [f"{self.feature_set}_{value:.12g}" for value in self.axis_]
            if self.feature_set == "combined":
                return np.column_stack([values, X[list(SENSOR_COLUMNS)].to_numpy(dtype=float)]), names + list(SENSOR_COLUMNS)
            return values, names
        if self.feature_set == "fft":
            size = len(self.axis_)
            frequencies = fft.rfftfreq(size, d=self.axis_[1] - self.axis_[0])
            nyquist = 1 / (2 * (self.axis_[1] - self.axis_[0]))
            selected = (frequencies > 0) & (frequencies <= 0.8 * nyquist)
            taper = np.hanning(size)
            coefficients = fft.rfft(detrend(matrix, axis=1, type="linear") * taper, axis=1)
            weights = np.full(len(frequencies), 2.0)
            weights[0] = 1
            if size % 2 == 0:
                weights[-1] = 1
            amplitude = np.abs(coefficients) * weights / taper.sum()
            power = np.abs(coefficients)**2 * weights / size**2
            total = power.sum(axis=1, keepdims=True)
            normalized = np.divide(power, total, out=np.zeros_like(power), where=total > 0)
            features = [amplitude[:, selected]]
            names = [f"fft_amplitude_{value:.9g}" for value in frequencies[selected]]
            edges = [0.0, 0.2, 0.5, 0.8, 1.0]
            for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
                mask = (frequencies >= lower * nyquist) & (frequencies < upper * nyquist if index < 3 else frequencies <= upper * nyquist)
                features.extend([power[:, mask].sum(axis=1)[:, None], normalized[:, mask].sum(axis=1)[:, None]])
                names.extend([f"fft_band_{index}_mean_square", f"fft_band_{index}_fraction"])
            entropy = -(normalized * np.log(np.maximum(normalized, np.finfo(float).tiny))).sum(axis=1)
            features.append(entropy[:, None])
            return np.column_stack(features), names + ["fft_power_entropy"]
        if self.feature_set == "wavelet":
            wavelet = pywt.Wavelet("db2")
            level = min(3, pywt.dwt_max_level(len(self.axis_), wavelet.dec_len))
            coefficients = pywt.wavedec(matrix, wavelet, mode="symmetric", level=level, axis=1)
            labels = [f"A{level}"] + [f"D{value}" for value in range(level, 0, -1)]
            names, features = [], []
            for label, band in zip(labels, coefficients):
                features.append(band)
                names.extend(f"wavelet_{label}_{index}" for index in range(band.shape[1]))
                features.extend([(band**2).sum(axis=1)[:, None], band.mean(axis=1)[:, None], band.std(axis=1)[:, None]])
                names.extend([f"wavelet_{label}_squared_norm", f"wavelet_{label}_mean", f"wavelet_{label}_std"])
            return np.column_stack(features), names
        # Summary regions are illustrative, fixed before observing response data.
        corrected = matrix - estimate_baseline(matrix)
        features, names, areas = [], [], []
        for center in (1003, 1120, 1340, 1450, 1655):
            region = np.abs(self.axis_ - center) <= 45
            if region.sum() < 2:
                raise ValueError("Summary features need two or more channels in each fixed band around 1003, 1120, 1340, 1450, and 1655 cm^-1.")
            area = np.trapezoid(np.maximum(corrected[:, region], 0), self.axis_[region], axis=1)
            areas.append(area)
            features.extend([corrected[:, region].max(axis=1)[:, None], area[:, None]])
            names.extend([f"peak_{center}", f"positive_area_{center}"])
        features.extend([(areas[1] / np.maximum(areas[3], 1e-12))[:, None], (areas[2] / np.maximum(areas[4], 1e-12))[:, None]])
        return np.column_stack(features), names + ["area_ratio_1120_1450", "area_ratio_1340_1655"]

    def transform(self, X):
        check_is_fitted(self, ["output_names_"])
        values, names = self._extract(X, self._input(X))
        if list(self.output_names_) != names or not np.isfinite(values).all():
            raise ValueError("Engineered features must be finite and preserve fitted feature order.")
        return pd.DataFrame(values, columns=names, index=X.index)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["output_names_"])
        return self.output_names_.copy()
