"""Known-answer preprocessing and fit/transform integrity checks."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from src.preprocessing import (
    MultiplicativeScatterCorrection, SpectralPreprocessor, estimate_baseline,
    savitzky_golay, select_region, snv, validate_axis,
)


def test_snv_centers_and_normalizes_each_row_without_mutation():
    values = np.array([[1, 2, 5, 3], [10, 12, 18, 14]], dtype=float)
    original = values.copy()
    transformed = snv(values)
    np.testing.assert_allclose(transformed.mean(axis=1), 0, atol=1e-14)
    np.testing.assert_allclose(transformed.std(axis=1, ddof=1), 1)
    np.testing.assert_allclose(transformed[0], transformed[1])
    np.testing.assert_array_equal(values, original)


def test_msc_recovers_reference_from_additive_and_multiplicative_scatter():
    reference = np.array([1, 2, 4, 2, 1.5], dtype=float)
    model = MultiplicativeScatterCorrection().fit(np.vstack([reference, reference]))
    corrected = model.transform(np.vstack([3 + 2 * reference, -1 + 0.5 * reference]))
    np.testing.assert_allclose(corrected, np.vstack([reference, reference]), atol=1e-12)
    before = model.reference_.copy()
    model.transform([100 + 20 * reference])
    np.testing.assert_array_equal(model.reference_, before)


def test_msc_requires_fitted_nonconstant_reference_and_nonzero_slope():
    with pytest.raises(NotFittedError):
        MultiplicativeScatterCorrection().transform([1, 2, 3])
    with pytest.raises(ValueError, match="nonconstant"):
        MultiplicativeScatterCorrection().fit(np.ones((3, 6)))
    model = MultiplicativeScatterCorrection().fit([[1, 2, 4, 2]])
    with pytest.raises(ValueError, match="slope"):
        model.transform([[1, 1, 1, 1]])
    with pytest.raises(ValueError, match="channel"):
        model.transform([[1, 2, 3]])


def test_asymmetric_baseline_preserves_a_known_linear_background():
    axis = np.linspace(400, 1800, 101)
    background = 0.2 + 0.0003 * axis
    result = estimate_baseline(background)
    np.testing.assert_allclose(result, background, atol=2e-8)
    np.testing.assert_array_equal(background, 0.2 + 0.0003 * axis)


@pytest.mark.parametrize("derivative", [0, 1, 2])
def test_savitzky_golay_matches_analytical_polynomial_in_physical_units(derivative):
    axis = np.linspace(0, 3, 61)
    signal = axis**3 + 2 * axis**2 + 4 * axis + 1
    expected = [signal, 3 * axis**2 + 4 * axis + 4, 6 * axis + 4][derivative]
    actual = savitzky_golay(signal, axis, window_length=9, polyorder=3, derivative=derivative)
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_region_selection_is_inclusive_and_preserves_input():
    axis = np.arange(400, 451, 5)
    values = np.arange(22).reshape(2, 11)
    selected, selected_axis = select_region(values, axis, 410, 430)
    np.testing.assert_array_equal(selected_axis, [410, 415, 420, 425, 430])
    np.testing.assert_array_equal(selected, values[:, 2:7])
    selected[0, 0] = -1
    assert values[0, 2] == 2


def test_preprocessor_reuses_training_msc_reference_and_consistent_feature_order():
    axis = np.linspace(400, 1800, 31)
    reference = np.sin(np.arange(31) / 5) + 3
    training = np.vstack([reference, 1.2 * reference + 0.2, 0.8 * reference - 0.2])
    preprocessing = SpectralPreprocessor(axis, region=(600, 1600), scatter="msc", window_length=5).fit(training)
    reference_before = preprocessing.msc_.reference_.copy()
    projected = np.vstack([2 * reference + 4, 0.5 * reference - 1])
    together = preprocessing.transform(projected)
    separate = np.vstack([preprocessing.transform(row) for row in projected])
    np.testing.assert_allclose(together, separate, atol=1e-12)
    np.testing.assert_array_equal(preprocessing.msc_.reference_, reference_before)
    assert together.shape[1] == len(preprocessing.output_axis_)
    assert clone(preprocessing).get_params()["scatter"] == "msc"


@pytest.mark.parametrize("values", [np.ones(10), [1, np.nan, 2], [1, np.inf, 2], []])
def test_snv_rejects_undefined_or_missing_spectra(values):
    with pytest.raises(ValueError):
        snv(values)


@pytest.mark.parametrize("changes", [
    {"window_length": 4}, {"window_length": 51}, {"polyorder": 11},
    {"derivative": 4}, {"window_length": 5.5},
])
def test_invalid_sg_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        savitzky_golay(np.arange(21), np.arange(21), **changes)


@pytest.mark.parametrize("axis", [[1, 3, 2], [1, 1, 2], [1, np.nan, 3]])
def test_invalid_axes_are_rejected(axis):
    with pytest.raises(ValueError):
        validate_axis(axis)


def test_no_automatic_resampling_for_nonuniform_sg_input():
    with pytest.raises(ValueError, match="uniform"):
        savitzky_golay(np.arange(5), [1, 2, 3, 5, 7], window_length=3, polyorder=2)


@pytest.mark.parametrize("changes", [{"smoothness": 0}, {"asymmetry": 0.5}, {"iterations": 0}, {"asymmetry": np.nan}])
def test_invalid_baseline_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        estimate_baseline(np.arange(10), **changes)


def test_preprocessing_rejects_inconsistent_channels_and_undefined_derivatives():
    axis = np.arange(10)
    with pytest.raises(ValueError, match="window"):
        SpectralPreprocessor(axis, derivative=1).fit(np.ones((2, 10)))
    preprocessing = SpectralPreprocessor(axis).fit(np.ones((2, 10)))
    with pytest.raises(ValueError, match="channel"):
        preprocessing.transform(np.ones((2, 9)))
