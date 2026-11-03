"""Reference-fitted exploratory PCA; no prediction or calibrated monitoring limits."""

from numbers import Integral

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import SpectralPreprocessor, spectral_array


def fit_spectral_pca(reference_spectra, axis, *, n_components=3, autoscale=False, preprocessor=None):
    """Fit every learned operation using only the explicitly supplied references.

    Returning a fitted sklearn Pipeline preserves one identical transform path
    for reference, projected, and later inference rows. Autoscaling is optional;
    mean centering is always fitted on reference rows.
    """
    matrix, vector = spectral_array(reference_spectra)
    if vector or len(matrix) < 2:
        raise ValueError("PCA requires at least two reference spectra in a matrix.")
    if isinstance(n_components, bool) or not isinstance(n_components, Integral) or n_components < 1:
        raise ValueError("PCA component count must be a positive integer.")
    preprocessing = clone(preprocessor) if preprocessor is not None else SpectralPreprocessor(axis)
    transformed = preprocessing.fit_transform(matrix)
    if n_components > min(len(matrix) - 1, transformed.shape[1]):
        raise ValueError("PCA components cannot exceed min(reference rows - 1, selected channels).")
    if not np.any(np.var(transformed, axis=0) > np.finfo(float).eps ** 2):
        raise ValueError("Reference spectra have no between-sample variation after preprocessing.")
    scaler = StandardScaler(with_mean=True, with_std=autoscale)
    scaled = scaler.fit_transform(transformed)
    pca = PCA(n_components=n_components, svd_solver="full").fit(scaled)
    return Pipeline([("spectra", preprocessing), ("scale", scaler), ("pca", pca)])


def project_spectral_pca(pipeline, spectra):
    """Project without refitting; return scores and reconstruction RMSE per row.

    RMSE is calculated in centered/scaled PCA input coordinates, not raw Raman
    units unless preprocessing/scaling make those units equivalent. These values
    are exploratory and have no fitted alarm threshold.
    """
    matrix, vector = spectral_array(spectra)
    if vector:
        matrix = matrix.copy()
    processed = pipeline.named_steps["spectra"].transform(matrix)
    scaled = pipeline.named_steps["scale"].transform(processed)
    pca = pipeline.named_steps["pca"]
    scores = pca.transform(scaled)
    reconstructed = pca.inverse_transform(scores)
    rmse = np.sqrt(np.mean((scaled - reconstructed) ** 2, axis=1))
    return scores, rmse
