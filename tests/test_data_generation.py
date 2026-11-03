"""Check scientific invariants and source identity, not just execution paths."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.data_generation import (
    CultureState, SimulationConfig, advance_culture, gaussian_peak,
    generate_dataset, simulate_raman_spectra,
)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(SimulationConfig(n_batches=4, duration_h=144, sample_interval_h=2, n_wavenumbers=141))


def test_feed_step_preserves_amounts_and_accounts_for_dilution():
    # No growth, death, uptake, or production: only feed and dilution remain.
    result = advance_culture(
        CultureState(2, 1, 4, 2), dt_h=1, feed_rate_l_h=0.1,
        growth_rate_h=0, death_rate_h=0, maintenance_rate=0, lactate_rate=0,
    )
    assert result.volume_l == pytest.approx(2.1)
    assert result.glucose_g_l * result.volume_l == pytest.approx(2 * 4 + 0.1 * 150)
    assert result.viable_cell_density_million_ml * result.volume_l == pytest.approx(2)
    assert result.lactate_g_l * result.volume_l == pytest.approx(4)


def test_growth_and_glucose_uptake_share_the_same_yield():
    initial = CultureState(2, 1, 4, 0)
    result = advance_culture(
        initial, dt_h=0.1, feed_rate_l_h=0, growth_rate_h=0.03,
        death_rate_h=0, maintenance_rate=0, lactate_rate=0, cell_yield=3,
    )
    biomass_increase = (result.viable_cell_density_million_ml - initial.viable_cell_density_million_ml) * initial.volume_l
    glucose_consumed = (initial.glucose_g_l - result.glucose_g_l) * initial.volume_l
    assert biomass_increase == pytest.approx(3 * glucose_consumed)


def test_glucose_depletion_limits_growth_without_negative_concentration():
    initial = CultureState(2, 1, 0.0001, 0)
    result = advance_culture(
        initial, dt_h=0.1, feed_rate_l_h=0, growth_rate_h=0.1,
        death_rate_h=0, maintenance_rate=0, cell_yield=3,
    )
    assert result.glucose_g_l == pytest.approx(0, abs=1e-12)
    assert result.viable_cell_density_million_ml - 1 == pytest.approx(3 * 0.0001)


@pytest.mark.parametrize("changes", [
    {"n_batches": 0}, {"n_batches": 1.5}, {"n_batches": True},
    {"duration_h": 0}, {"duration_h": np.nan}, {"duration_h": np.inf},
    {"sample_interval_h": -1}, {"sample_interval_h": 7},
    {"sample_interval_h": 200}, {"n_wavenumbers": 10}, {"seed": -1},
])
def test_invalid_config_is_rejected(changes):
    with pytest.raises(ValueError):
        SimulationConfig(**changes)


@pytest.mark.parametrize("changes", [
    {"dt_h": 0}, {"feed_rate_l_h": -1}, {"growth_rate_h": np.nan},
    {"cell_yield": 0}, {"death_rate_h": 20},
])
def test_invalid_balance_inputs_are_rejected(changes):
    arguments = {"dt_h": 0.1, "feed_rate_l_h": 0.002, "growth_rate_h": 0.03, **changes}
    with pytest.raises(ValueError):
        advance_culture(CultureState(2, 1, 4, 0.2), **arguments)


def test_same_config_reproduces_every_source_exactly():
    config = SimulationConfig(n_batches=2, duration_h=6, n_wavenumbers=32)
    first, second = generate_dataset(config), generate_dataset(config)
    for name in ("truth", "sensors", "spectra", "spectral_metadata", "offline_assays", "batches"):
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name))
    np.testing.assert_array_equal(first.wavenumbers_cm1, second.wavenumbers_cm1)


def test_seed_changes_the_generated_process():
    config = SimulationConfig(n_batches=1, duration_h=6, n_wavenumbers=32)
    assert not generate_dataset(config).truth.equals(generate_dataset(replace(config, seed=43)).truth)


def test_adding_batches_preserves_existing_batch_sources():
    config = SimulationConfig(n_batches=1, duration_h=6, n_wavenumbers=32)
    first, expanded = generate_dataset(config), generate_dataset(replace(config, n_batches=2))
    for name in ("truth", "sensors", "spectral_metadata", "offline_assays", "batches"):
        expanded_frame = getattr(expanded, name)
        preserved = expanded_frame[expanded_frame["batch_id"] == "B001"].reset_index(drop=True)
        pd.testing.assert_frame_equal(getattr(first, name), preserved)
    pd.testing.assert_frame_equal(first.spectra, expanded.spectra.loc[first.spectra.index])


def test_spectral_resolution_does_not_change_other_sources():
    config = SimulationConfig(n_batches=1, duration_h=6, n_wavenumbers=32)
    low, high = generate_dataset(config), generate_dataset(replace(config, n_wavenumbers=64))
    for name in ("truth", "sensors", "offline_assays"):
        pd.testing.assert_frame_equal(getattr(low, name), getattr(high, name))


def test_timestamp_grid_and_batch_identity_are_preserved(dataset):
    assert len(dataset.sensors) == 4 * 73
    assert dataset.sensors["sample_id"].is_unique
    assert str(dataset.sensors["measurement_timestamp"].dt.tz) == "UTC"
    for batch_id, frame in dataset.sensors.groupby("batch_id"):
        assert frame["elapsed_time_h"].iloc[[0, -1]].tolist() == [0, 144]
        np.testing.assert_allclose(frame["elapsed_time_h"].diff().dropna(), 2)
        assert frame["measurement_timestamp"].is_monotonic_increasing
        elapsed = (frame["measurement_timestamp"] - frame["measurement_timestamp"].iloc[0]).dt.total_seconds() / 3600
        np.testing.assert_allclose(elapsed, frame["elapsed_time_h"])
        assert frame["sample_id"].str.startswith(batch_id + "-").all()
    assert dataset.spectra.index.tolist() == dataset.spectral_metadata["sample_id"].tolist()
    assert dataset.sensors["sample_id"].tolist() == dataset.truth["sample_id"].tolist()


def test_process_and_spectral_values_are_finite_and_physically_admissible(dataset):
    for frame in (dataset.sensors, dataset.truth, dataset.spectra):
        assert np.isfinite(frame.select_dtypes(include="number").to_numpy()).all()
    for column in ("volume_l", "glucose_g_l", "lactate_g_l", "viable_cell_density_million_ml"):
        assert (dataset.truth[column] >= 0).all()
    assert dataset.sensors["dissolved_oxygen_pct"].between(0, 100).all()
    assert (dataset.sensors["feed_rate_l_h"] >= 0).all()
    for _, frame in dataset.truth.groupby("batch_id"):
        assert frame["volume_l"].is_monotonic_increasing
        assert frame["volume_l"].iloc[0] == 2
        assert frame["viable_cell_density_million_ml"].iloc[-1] > frame["viable_cell_density_million_ml"].iloc[0]


def test_sensor_observations_exclude_latent_concentrations(dataset):
    assert "glucose_g_l" not in dataset.sensors
    assert "lactate_g_l" not in dataset.sensors
    assert "viable_cell_density_million_ml" not in dataset.sensors
    assert not np.array_equal(dataset.sensors["temperature_c"], dataset.truth["temperature_c"])
    assert dataset.spectra.columns.str.startswith("raman_").all()


def test_uniform_raman_axis_and_measurement_dimensions(dataset):
    assert dataset.wavenumbers_cm1[[0, -1]].tolist() == [400, 1800]
    np.testing.assert_allclose(np.diff(dataset.wavenumbers_cm1), 10)
    assert dataset.spectra.shape == (len(dataset.spectral_metadata), 141)


def test_gaussian_peak_has_expected_center_and_width():
    np.testing.assert_allclose(gaussian_peak(np.array([990, 1000, 1010]), 1000, 10), [np.exp(-0.5), 1, np.exp(-0.5)])
    with pytest.raises(ValueError):
        gaussian_peak(np.array([1000]), 1000, 0)


def test_increasing_glucose_changes_its_spectral_signal(dataset):
    original_truth = dataset.truth[dataset.truth["batch_id"] == "B001"].copy()
    increased_truth = original_truth.copy()
    increased_truth["glucose_g_l"] += 1
    original = simulate_raman_spectra(original_truth, dataset.wavenumbers_cm1, np.random.default_rng(10))
    increased = simulate_raman_spectra(increased_truth, dataset.wavenumbers_cm1, np.random.default_rng(10))
    difference = (increased - original).to_numpy()
    peak_index = np.argmin(abs(dataset.wavenumbers_cm1 - 1120))
    far_index = np.argmin(abs(dataset.wavenumbers_cm1 - 600))
    assert (difference[:, peak_index] > 0.04).all()
    np.testing.assert_allclose(difference[:, far_index], 0, atol=1e-12)


@pytest.mark.parametrize("axis", [np.array([400, 401, 405]), np.array([405, 404]), np.array([400, np.nan])])
def test_invalid_raman_axis_is_rejected(dataset, axis):
    with pytest.raises(ValueError):
        simulate_raman_spectra(dataset.truth, axis, np.random.default_rng(0))


def test_reference_assays_use_collection_time_and_delayed_result_time(dataset):
    assays = dataset.offline_assays
    assert assays["sample_id"].is_unique
    assert set(assays["sample_id"]).issubset(dataset.truth["sample_id"])
    linked = dataset.truth.set_index("sample_id").loc[assays["sample_id"]]
    assert linked["measurement_timestamp"].tolist() == assays["collection_timestamp"].tolist()
    assert linked["batch_id"].tolist() == assays["batch_id"].tolist()
    delays = (assays["result_timestamp"] - assays["collection_timestamp"]).dt.total_seconds() / 3600
    assert delays.between(2, 6).all()
    assert (assays[["glucose_reference_g_l", "lactate_reference_g_l"]] >= 0).all().all()
    assert not np.array_equal(assays["glucose_reference_g_l"], linked["glucose_g_l"])


def test_known_disturbance_is_labeled_and_changes_process_signals(dataset):
    batch = dataset.truth[dataset.truth["batch_id"] == "B004"]
    early = batch[batch["elapsed_time_h"] < 90]
    late = batch[batch["elapsed_time_h"] > 120]
    assert (early["disturbance_fraction"] == 0).all()
    assert (late["disturbance_fraction"] == 1).all()
    assert late["temperature_c"].mean() > early["temperature_c"].mean() + 0.5
    assert late["dissolved_oxygen_pct"].mean() < early["dissolved_oxygen_pct"].mean() - 15
    assert dataset.batches.loc[3, "scenario"] == "Deliberately disturbed"


def test_observation_frequency_does_not_materially_change_underlying_process():
    config = SimulationConfig(n_batches=1, duration_h=30, sample_interval_h=1, n_wavenumbers=32)
    coarse, fine = generate_dataset(config), generate_dataset(replace(config, sample_interval_h=0.5))
    columns = ["volume_l", "glucose_g_l", "lactate_g_l", "viable_cell_density_million_ml"]
    # Both numerical integrations should agree at common observation times.
    np.testing.assert_allclose(coarse.truth[columns], fine.truth.iloc[::2][columns], atol=0.002, rtol=0.001)
