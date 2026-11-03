"""Feature contracts, fixed spectral summaries, and training-only MSC state."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import FEATURE_SETS, SpectralFeatures
from src.modeling_data import SENSOR_COLUMNS
from src.preprocessing import estimate_baseline


@pytest.fixture
def feature_inputs():
    axis = np.linspace(400, 1800, 71)
    rng = np.random.default_rng(12)
    spectra = 1 + np.sin(axis / 100)[None, :] + rng.normal(0, 0.1, (10, len(axis)))
    frame = pd.DataFrame(spectra, columns=[f"raman_{value:.12g}" for value in axis], index=[f"row{index}" for index in range(10)])
    frame[list(SENSOR_COLUMNS)] = rng.normal(size=(10, len(SENSOR_COLUMNS)))
    return frame, axis


@pytest.mark.parametrize("recipe", list(FEATURE_SETS))
def test_feature_names_order_and_single_row_inference_match_batch_transform(feature_inputs, recipe):
    X, axis = feature_inputs
    original = X.copy(deep=True)
    extractor = SpectralFeatures(axis, recipe).fit(X.iloc[:7])
    together = extractor.transform(X.iloc[7:])
    separately = pd.concat([extractor.transform(X.iloc[[index]]) for index in range(7, 10)])
    np.testing.assert_allclose(together, separately, rtol=1e-9, atol=1e-9)
    assert list(together.columns) == list(extractor.get_feature_names_out())
    assert np.isfinite(together).all().all()
    pd.testing.assert_frame_equal(X, original)
    # Input column order and extra columns cannot alter declared feature order.
    reordered = X.iloc[7:, ::-1].assign(glucose_g_l=999, batch_id="do-not-model")
    np.testing.assert_allclose(extractor.transform(reordered), together)


def test_msc_reference_is_training_only_and_never_recomputed_on_projected_rows(feature_inputs):
    X, axis = feature_inputs
    extractor = SpectralFeatures(axis, "preprocessed").fit(X.iloc[:7])
    reference = extractor.preprocessor_.msc_.reference_.copy()
    training = X.iloc[:7][list(extractor.spectral_columns_)].to_numpy()
    np.testing.assert_allclose(reference, (training - estimate_baseline(training)).mean(axis=0))
    changed = X.iloc[7:].copy()
    changed[list(extractor.spectral_columns_)] *= 100
    extractor.transform(changed)
    np.testing.assert_array_equal(extractor.preprocessor_.msc_.reference_, reference)
    assert set(extractor.fit_row_ids_) == set(X.index[:7])


def test_summary_areas_scale_with_signal_amplitude_and_ratios_are_invariant(feature_inputs):
    X, axis = feature_inputs
    spectrum = 0.2 + sum(np.exp(-0.5 * ((axis - center) / 15)**2) for center in (1003, 1120, 1340, 1450, 1655))
    X.loc[X.index[0], X.columns.str.startswith("raman_")] = spectrum
    X.loc[X.index[1], X.columns.str.startswith("raman_")] = 2 * spectrum
    features = SpectralFeatures(axis, "summary").fit_transform(X.iloc[:2])
    np.testing.assert_allclose(features.filter(like="positive_area_").iloc[1], 2 * features.filter(like="positive_area_").iloc[0], rtol=1e-7)
    np.testing.assert_allclose(features.filter(like="area_ratio_").iloc[1], features.filter(like="area_ratio_").iloc[0], rtol=1e-7)


def test_fft_band_fractions_cover_total_power(feature_inputs):
    X, axis = feature_inputs
    features = SpectralFeatures(axis, "fft").fit_transform(X)
    fractions = features.filter(regex="fft_band_.*_fraction$")
    np.testing.assert_allclose(fractions.sum(axis=1), 1)
    assert features["fft_power_entropy"].ge(0).all()


def test_declared_sensors_only_and_combined_features_use_exactly_five_sensor_inputs(feature_inputs):
    X, axis = feature_inputs
    sensor = SpectralFeatures(axis, "sensors").fit_transform(X)
    assert sensor.columns.tolist() == list(SENSOR_COLUMNS)
    np.testing.assert_array_equal(sensor, X[list(SENSOR_COLUMNS)])
    combined = SpectralFeatures(axis, "combined").fit_transform(X)
    assert combined.shape[1] == len(axis) + 5
    assert combined.columns[-5:].tolist() == list(SENSOR_COLUMNS)


def test_missing_and_invalid_inference_inputs_are_rejected(feature_inputs):
    X, axis = feature_inputs
    extractor = SpectralFeatures(axis, "raw").fit(X)
    with pytest.raises(ValueError, match="Missing model inputs"):
        extractor.transform(X.drop(columns="raman_400"))
    broken = X.copy()
    broken.loc[broken.index[0], "raman_400"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        extractor.transform(broken)
    with pytest.raises(ValueError, match="Unknown"):
        SpectralFeatures(axis, "not-a-recipe").fit(X)
