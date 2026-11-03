# Synthetic data assumptions and provenance

Every value is generated locally. No external dataset, manufacturing data,
equipment connection, or fitted product model is used. Numerical choices are
illustrative, not operating recommendations.

## Process and numerical method

| Quantity | Assumption |
| --- | --- |
| Initial volume | 2 L |
| Initial cell density | Uniform 0.4–0.6 million cells/mL per batch |
| Initial glucose | Uniform 4–5 g/L per batch |
| Initial lactate | Uniform 0.1–0.2 g/L per batch |
| Maximum specific growth rate | Uniform 0.050–0.060 /h per batch |
| Glucose half-saturation | 0.5 g/L |
| Cell-density capacity | 28 million cells/mL, before feed dilution |
| Death rate | 0.002 /h |
| Feed glucose | 150 g/L |
| Cell yield on glucose | 3 (million cells/mL)/(g/L) |
| Glucose maintenance | 0.002 (g/L)/(million cells/mL)/h |
| Lactate coefficient | 0.0015 + 0.025 × growth rate; units explained in the learning guide |
| Feed schedule | 0 before 24 h; 0.002 L/h from 24 h; 0.0035 L/h from 72 h |
| Batch feed multiplier | Uniform 0.90–1.10 |
| Growth DO factor | clip(DO / 40, 0, 1), DO in % air saturation |
| Integration | Explicit amount balances, steps <= 0.1 h, split at feed switches |
| Observations | Uniform grid including both endpoints |
| Default setup | 6 batches, 144 h, 1 h interval, 281 channels, seed 42 |

Feed has no cells or lactate. Harvest, evaporation, sampling-volume loss, and gas
mass balances are omitted. A nutrient limiter scales uptake, growth, and lactate
production together if available glucose is insufficient. Death is a simple loss
of viable cells; released material is not modeled.

Cell-density amount is represented as density × volume. Its constant conversion
factor to absolute cell count cancels when calculating density again. It must
not be interpreted as grams of biomass.

## Controlled signals and sensor noise

Noiseless signals have a random batch phase:

- Temperature: 37 + 0.10 sin(t/12 + phase), in °C.
- pH: 7.05 − 0.003 X + 0.025 sin(t/15 + phase).
- DO: max(0, 58 − 0.65 X + 2 sin(t/18 + phase)), in % air saturation.
- Agitation: 200 + 4 X + 5 sin(t/20 + phase), in rpm.
- Feed follows the schedule and multiplier above.

These are surrogate relationships, without mechanistic mixing, heat, acid-base,
or control-loop models. Only the DO factor directly modifies growth.

| Sensor | Independent noise standard deviation |
| --- | --- |
| Temperature | 0.04 °C |
| pH | 0.008 |
| DO | 0.6 percentage points |
| Feed | 0.00004 L/h |
| Agitation | 1 rpm |

Each sensor has a constant batch offset drawn from a zero-mean normal
distribution with half its listed noise standard deviation. Observed DO is
clipped to [0, 100] and feed to nonnegative values as part of the observation
model. This is generation behavior, not data cleaning. Zero true flow can have
small positive observed readings. Raw observations are preserved afterward.

## Deliberately disturbed batches

Every fourth batch (B004, B008, ...) is disturbed. Let d ramp from 0 to 1 between
65% and 75% of batch duration. The disturbance:

- adds 0.8d °C to temperature;
- subtracts 0.18d from pH;
- subtracts 20d percentage points from DO, with a floor of zero;
- multiplies feed by 1 − 0.35d;
- adds 12d rpm to agitation;
- adds 0.22d to the Raman baseline.

Feed and oxygen changes also affect process states. Scenario labels describe
known generation settings, not predicted labels or validated alarms.

## Raman measurement model

The default grid is 400–1800 cm^-1 in 5 cm^-1 steps. Gaussian widths below are
standard deviations. Positions are arbitrary examples, not measured molecular
band assignments.

| Contribution | Centers (cm^-1) | Widths (cm^-1) | Relative amplitudes | Multiplier |
| --- | --- | --- | --- | --- |
| Glucose | 1120, 1340 | 16, 22 | 1, 0.7 | 0.07 per g/L |
| Lactate | 1135, 1450 | 18, 24 | 1, 0.8 | 0.13 per g/L |
| Cell-associated | 1003, 1655 | 14, 30 | 1, 0.8 | 0.008 per million cells/mL |

One batch axis shift is sampled from N(0, 1.2^2) cm^-1. A gain uniform over
0.92–1.08 multiplies analyte and cell signals. Define u as the centered axis
divided by its full range. Background is:

```text
0.3 + batch_offset + 0.15 * elapsed_time / batch_duration
+ batch_slope * u + 0.12 * u**2
+ 0.18 * Gaussian(center=1200, width=350)
+ 0.22 * disturbance_fraction
```

Batch offset is uniform over [−0.04, 0.04], slope over [0.05, 0.15]. Independent
N(0, 0.015^2) noise is added to each intensity. All intensities are arbitrary
units; this is not a calibrated photon-counting or instrument model.

## Assays, keys, and time

Offline samples use the first available observation at or after each 24 h target,
plus the final point, with duplicate selected indices removed. Glucose assay
noise has standard deviation 0.08 g/L, lactate 0.04 g/L. Negative assay values are
clipped to zero. Results arrive uniformly 2–6 h after collection and may arrive
after the batch ends.

Timestamps are UTC. The first illustrative batch starts at 08:00 UTC on
2025-01-06; batch starts are spaced by 14 days. These are synthetic historical
timestamps, not actual records. Elapsed time is the cross-batch comparison axis.

`sample_id` links simultaneous synthetic observations; `batch_id` identifies an
independent batch. Tables stay separate. Phase 1 performs no import, real-world
alignment, interpolation, or quality remediation.

## Reproducibility and future model use

Use `generate_dataset(SimulationConfig(...))`. Each batch has a seed sequence
based on the master seed and zero-based batch index. Four child streams isolate
process, sensor, spectral, and assay randomness. Adding batches preserves
earlier batches; changing spectral resolution preserves process and assay data.
Changing observation spacing changes noise draws and can slightly change the
integration path.

Independent streams generate different realizations of the same assumed model;
they do not reproduce all manufacturing variability. Reproducibility is tested
in the pinned environment. Changing code or library versions requires checking
it again.

Truth, IDs, times, and scenario labels must not silently become concentration
features. Later phases must declare valid inputs, retain an independent batch
holdout, and fit learned preprocessing on training data only. No model is fitted
in Phase 1, so no train/test split or predictive performance is reported.
