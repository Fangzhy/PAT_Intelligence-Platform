"""Reproducible, illustrative fed-batch cultures and Raman measurements.

All coefficients are teaching assumptions, not fitted manufacturing parameters.
Truth, sensors, spectra, and laboratory references remain separate data sources.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    """Controls shared by all batches; the observation grid includes both endpoints."""

    n_batches: int = 6
    duration_h: float = 144.0
    sample_interval_h: float = 1.0
    n_wavenumbers: int = 281
    seed: int = 42

    def __post_init__(self) -> None:
        for name, minimum in (("n_batches", 1), ("n_wavenumbers", 32), ("seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name in ("duration_h", "sample_interval_h"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        intervals = self.duration_h / self.sample_interval_h
        if intervals < 1 or not np.isclose(intervals, round(intervals), rtol=0, atol=1e-8):
            raise ValueError("duration_h must be an integer multiple of sample_interval_h.")


@dataclass(frozen=True)
class CultureState:
    """Concentrations are per culture volume; cell density is in 10^6 cells/mL."""

    volume_l: float
    viable_cell_density_million_ml: float
    glucose_g_l: float
    lactate_g_l: float


@dataclass
class SyntheticDataset:
    """Separate modalities connected by batch_id and stable sample_id keys.

    spectra contains only intensity columns, indexed by sample_id. Column j maps
    to wavenumbers_cm1[j]. spectral_metadata supplies time and batch information.
    """

    sensors: pd.DataFrame
    truth: pd.DataFrame
    spectra: pd.DataFrame
    spectral_metadata: pd.DataFrame
    wavenumbers_cm1: np.ndarray
    offline_assays: pd.DataFrame
    batches: pd.DataFrame


def advance_culture(
    state: CultureState,
    *,
    dt_h: float,
    feed_rate_l_h: float,
    growth_rate_h: float,
    death_rate_h: float = 0.002,
    feed_glucose_g_l: float = 150.0,
    cell_yield: float = 3.0,
    maintenance_rate: float = 0.002,
    lactate_rate: float = 0.002,
) -> CultureState:
    """Advance one explicit mass-balance step, including dilution by feed.

    Glucose uptake q_G = mu / Y_XG + maintenance. If the proposed uptake
    exceeds available glucose, scale uptake, growth, and lactate production by
    the same fraction. Death still occurs. No cells or lactate enter in the feed.
    Use small steps: the generator uses at most 0.1 h. There is no harvest flow.
    """
    values = (
        state.volume_l, state.viable_cell_density_million_ml,
        state.glucose_g_l, state.lactate_g_l, dt_h, feed_rate_l_h,
        growth_rate_h, death_rate_h, feed_glucose_g_l, cell_yield,
        maintenance_rate, lactate_rate,
    )
    if not np.isfinite(values).all() or min(values) < 0:
        raise ValueError("Culture states and rates must be finite and nonnegative.")
    if state.volume_l <= 0 or dt_h <= 0 or cell_yield <= 0:
        raise ValueError("Volume, time step, and cell yield must be positive.")
    if death_rate_h * dt_h > 1:
        raise ValueError("Time step is too large for the death rate.")

    volume = state.volume_l
    cells = state.viable_cell_density_million_ml * volume
    new_volume = volume + feed_rate_l_h * dt_h
    available_glucose = state.glucose_g_l * volume + feed_rate_l_h * feed_glucose_g_l * dt_h
    requested_uptake = (growth_rate_h / cell_yield + maintenance_rate) * cells * dt_h
    nutrient_fraction = min(1.0, available_glucose / requested_uptake) if requested_uptake > 0 else 1.0

    new_cells = cells * (1 + (growth_rate_h * nutrient_fraction - death_rate_h) * dt_h)
    new_glucose = max(0.0, available_glucose - requested_uptake * nutrient_fraction)
    new_lactate = state.lactate_g_l * volume + lactate_rate * nutrient_fraction * cells * dt_h
    return CultureState(
        volume_l=new_volume,
        viable_cell_density_million_ml=new_cells / new_volume,
        glucose_g_l=new_glucose / new_volume,
        lactate_g_l=new_lactate / new_volume,
    )


def gaussian_peak(wavenumbers_cm1: np.ndarray, center_cm1: float, width_cm1: float) -> np.ndarray:
    """Return a unit-height Gaussian; width is its standard deviation in cm^-1."""
    if not np.isfinite(width_cm1) or width_cm1 <= 0:
        raise ValueError("Peak width must be finite and positive.")
    return np.exp(-0.5 * ((np.asarray(wavenumbers_cm1) - center_cm1) / width_cm1) ** 2)


def _process_conditions(
    time_h: float, cell_density: float, duration_h: float,
    feed_factor: float, phase: float, abnormal: bool,
) -> dict[str, float]:
    """Illustrative controlled signals and an optional late process disturbance."""
    disturbance = float(np.clip((time_h / duration_h - 0.65) / 0.10, 0, 1)) if abnormal else 0.0
    feed = (0.002 if time_h >= 24 else 0.0) + (0.0015 if time_h >= 72 else 0.0)
    return {
        "temperature_c": 37.0 + 0.10 * np.sin(time_h / 12 + phase) + 0.8 * disturbance,
        "ph": 7.05 - 0.003 * cell_density + 0.025 * np.sin(time_h / 15 + phase) - 0.18 * disturbance,
        "dissolved_oxygen_pct": max(0.0, 58 - 0.65 * cell_density + 2 * np.sin(time_h / 18 + phase) - 20 * disturbance),
        "feed_rate_l_h": feed * feed_factor * (1 - 0.35 * disturbance),
        "agitation_rpm": 200 + 4 * cell_density + 5 * np.sin(time_h / 20 + phase) + 12 * disturbance,
        "disturbance_fraction": disturbance,
    }


def _simulate_process(
    config: SimulationConfig, batch_index: int, rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Integrate latent process states; observation frequency does not set solver steps."""
    batch_id = f"B{batch_index + 1:03d}"
    abnormal = (batch_index + 1) % 4 == 0
    start = pd.Timestamp("2025-01-06T08:00:00Z") + pd.Timedelta(days=14 * batch_index)
    times = np.linspace(0, config.duration_h, round(config.duration_h / config.sample_interval_h) + 1)
    mu_max = rng.uniform(0.050, 0.060)
    feed_factor = rng.uniform(0.90, 1.10)
    phase = rng.uniform(0, 2 * np.pi)
    state = CultureState(2.0, rng.uniform(0.4, 0.6), rng.uniform(4.0, 5.0), rng.uniform(0.1, 0.2))
    rows = []

    for sample_index, time_h in enumerate(times):
        conditions = _process_conditions(time_h, state.viable_cell_density_million_ml, config.duration_h, feed_factor, phase, abnormal)
        rows.append({
            "sample_id": f"{batch_id}-S{sample_index:05d}",
            "batch_id": batch_id,
            "elapsed_time_h": time_h,
            "measurement_timestamp": start + pd.to_timedelta(time_h, unit="h"),
            "volume_l": state.volume_l,
            "viable_cell_density_million_ml": state.viable_cell_density_million_ml,
            "glucose_g_l": state.glucose_g_l,
            "lactate_g_l": state.lactate_g_l,
            **conditions,
        })
        if sample_index == len(times) - 1:
            break
        # Split at feed switches as well as observation times. Feed is discontinuous
        # at 24 h and 72 h, so an integration step must not straddle those switches.
        stop_h = times[sample_index + 1]
        boundaries = [time_h] + [switch for switch in (24.0, 72.0) if time_h < switch < stop_h] + [stop_h]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            substeps = int(np.ceil((right - left) / 0.1))
            dt_h = (right - left) / substeps
            for step in range(substeps):
                current_time = left + step * dt_h
                conditions = _process_conditions(current_time, state.viable_cell_density_million_ml, config.duration_h, feed_factor, phase, abnormal)
                oxygen_factor = float(np.clip(conditions["dissolved_oxygen_pct"] / 40, 0, 1))
                capacity_factor = max(0.0, 1 - state.viable_cell_density_million_ml / 28)
                mu = mu_max * state.glucose_g_l / (0.5 + state.glucose_g_l) * capacity_factor * oxygen_factor
                state = advance_culture(
                    state, dt_h=dt_h, feed_rate_l_h=conditions["feed_rate_l_h"],
                    growth_rate_h=mu, lactate_rate=0.0015 + 0.025 * mu,
                )

    metadata = {
        "batch_id": batch_id,
        "start_timestamp": start,
        "scenario": "Deliberately disturbed" if abnormal else "Nominal simulation",
        "disturbance_start_h": 0.65 * config.duration_h if abnormal else np.nan,
        "description": "Late feed reduction, lower DO and pH, and elevated temperature; synthetic baseline shift." if abnormal else "Between-batch variation in inoculum, growth rate, and feed delivery.",
    }
    return pd.DataFrame(rows), metadata


def _observe_sensors(truth: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add sensor noise and a small batch-specific offset; preserve truth separately."""
    sensors = truth[["sample_id", "batch_id", "elapsed_time_h", "measurement_timestamp"]].copy()
    noise_levels = {
        "temperature_c": 0.04, "ph": 0.008, "dissolved_oxygen_pct": 0.6,
        "feed_rate_l_h": 0.00004, "agitation_rpm": 1.0,
    }
    for column, noise_std in noise_levels.items():
        offset = rng.normal(0, noise_std * 0.5)
        values = truth[column].to_numpy() + offset + rng.normal(0, noise_std, len(truth))
        if column == "dissolved_oxygen_pct":
            values = np.clip(values, 0, 100)
        elif column == "feed_rate_l_h":
            values = np.maximum(values, 0)
        sensors[column] = values
    return sensors


def simulate_raman_spectra(
    truth: pd.DataFrame, wavenumbers_cm1: np.ndarray, rng: np.random.Generator,
) -> pd.DataFrame:
    """Mix illustrative analyte peaks, cell background, fluorescence, and noise.

    Peak positions and sensitivities are arbitrary teaching choices. They are
    not verified Raman band assignments. Intensity is in arbitrary units.
    """
    axis = np.asarray(wavenumbers_cm1, dtype=float)
    if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
        raise ValueError("The Raman axis must be a finite one-dimensional array.")
    spacing = np.diff(axis)
    if np.any(spacing <= 0) or not np.allclose(spacing, spacing[0]):
        raise ValueError("The Raman axis must be increasing and uniformly spaced.")

    shift = rng.normal(0, 1.2)
    glucose_peaks = gaussian_peak(axis, 1120 + shift, 16) + 0.7 * gaussian_peak(axis, 1340 + shift, 22)
    lactate_peaks = gaussian_peak(axis, 1135 + shift, 18) + 0.8 * gaussian_peak(axis, 1450 + shift, 24)
    cell_peaks = gaussian_peak(axis, 1003 + shift, 14) + 0.8 * gaussian_peak(axis, 1655 + shift, 30)
    signal = (
        0.07 * truth["glucose_g_l"].to_numpy()[:, None] * glucose_peaks
        + 0.13 * truth["lactate_g_l"].to_numpy()[:, None] * lactate_peaks
        + 0.008 * truth["viable_cell_density_million_ml"].to_numpy()[:, None] * cell_peaks
    )
    normalized_axis = (axis - axis.mean()) / np.ptp(axis)
    progress = truth["elapsed_time_h"].to_numpy() / truth["elapsed_time_h"].iloc[-1]
    baseline = (
        0.3 + rng.uniform(-0.04, 0.04) + 0.15 * progress[:, None]
        + rng.uniform(0.05, 0.15) * normalized_axis
        + 0.12 * normalized_axis**2
        + 0.18 * gaussian_peak(axis, 1200, 350)
        + 0.22 * truth["disturbance_fraction"].to_numpy()[:, None]
    )
    spectra = rng.uniform(0.92, 1.08) * signal + baseline + rng.normal(0, 0.015, signal.shape)
    return pd.DataFrame(
        spectra, index=pd.Index(truth["sample_id"], name="sample_id"),
        columns=[f"raman_{index:04d}" for index in range(len(axis))],
    )


def _sample_offline_assays(truth: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Sample approximately daily, plus the final point; results arrive 2-6 h later."""
    targets = np.append(np.arange(0, truth["elapsed_time_h"].iloc[-1], 24), truth["elapsed_time_h"].iloc[-1])
    indices = np.unique(np.searchsorted(truth["elapsed_time_h"].to_numpy(), targets))
    sampled = truth.iloc[indices]
    assays = sampled[["sample_id", "batch_id", "elapsed_time_h", "measurement_timestamp"]].copy()
    assays = assays.rename(columns={"elapsed_time_h": "collection_elapsed_time_h", "measurement_timestamp": "collection_timestamp"})
    assays["result_timestamp"] = assays["collection_timestamp"] + pd.to_timedelta(rng.uniform(2, 6, len(assays)), unit="h")
    for analyte, std in (("glucose", 0.08), ("lactate", 0.04)):
        assays[f"{analyte}_reference_g_l"] = np.maximum(0, sampled[f"{analyte}_g_l"].to_numpy() + rng.normal(0, std, len(assays)))
    return assays.reset_index(drop=True)


def generate_dataset(config: SimulationConfig = SimulationConfig()) -> SyntheticDataset:
    """Generate independent batches with stable IDs and isolated random streams.

    Batch i uses SeedSequence([seed, i]); adding batches preserves earlier ones.
    Separate process/sensor/spectral/assay streams prevent a change to spectral
    resolution from changing the process or reference observations.
    """
    axis = np.linspace(400, 1800, config.n_wavenumbers)
    sensor_tables, truth_tables, spectral_tables, assay_tables, batch_records = [], [], [], [], []
    for batch_index in range(config.n_batches):
        streams = np.random.SeedSequence([config.seed, batch_index]).spawn(4)
        process_rng, sensor_rng, raman_rng, assay_rng = [np.random.default_rng(stream) for stream in streams]
        truth, metadata = _simulate_process(config, batch_index, process_rng)
        truth_tables.append(truth)
        sensor_tables.append(_observe_sensors(truth, sensor_rng))
        spectral_tables.append(simulate_raman_spectra(truth, axis, raman_rng))
        assay_tables.append(_sample_offline_assays(truth, assay_rng))
        batch_records.append(metadata)

    all_truth = pd.concat(truth_tables, ignore_index=True)
    return SyntheticDataset(
        sensors=pd.concat(sensor_tables, ignore_index=True),
        truth=all_truth,
        spectra=pd.concat(spectral_tables),
        spectral_metadata=all_truth[["sample_id", "batch_id", "elapsed_time_h", "measurement_timestamp"]].copy(),
        wavenumbers_cm1=axis,
        offline_assays=pd.concat(assay_tables, ignore_index=True),
        batches=pd.DataFrame(batch_records),
    )
