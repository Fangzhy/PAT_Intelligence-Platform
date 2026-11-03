# Vaccine-PAT-Intelligence-Platform

A learning-oriented Python application for exploring a synthetic recombinant
protein fed-batch bioreactor and, in future phases, developing PAT analytics.
**Phases 1–3 are implemented.** The default measurements are synthetic. This is an
educational portfolio demonstration, not a manufacturing-validated system.

## Start locally in VS Code

Open a PowerShell terminal in this project folder. Python 3.11 and dependencies
are already installed in `.venv` for this checkout. Start the dashboard with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally <http://localhost:8501>.
Stop the server with **Ctrl+C**. Activation is optional when you use the explicit
interpreter path above.

For a fresh checkout, install [uv](https://docs.astral.sh/uv/getting-started/installation/)
and run:

```powershell
uv venv --python 3.11 .venv
uv pip sync --python .venv\Scripts\python.exe requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Alternatively, if Python 3.11 and the Windows Python launcher are installed:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

In VS Code, use **Python: Select Interpreter** and choose
`.venv\Scripts\python.exe`. NumPy and pandas handle data, Streamlit and Plotly
provide the interface, and pytest runs the checks. `requirements.txt` pins the
full dependency environment; `requirements.in` records direct dependency ranges.
Phase 3 adds SciPy, scikit-learn, and PyWavelets. If you already have an older
checkout environment, synchronize it with the updated `requirements.txt` before
starting the app.

## Explore the Overview dashboard

1. Start with seed **42**, **6 batches**, and **144 hours**. The initial page
   generates this dataset automatically.
2. Use **Batches to explore** to compare trajectories. Select a process variable
   to inspect temperature, pH, DO, feed flow, or agitation.
3. Compare **B001** with **B004**. B004 has a deliberately introduced late
   disturbance. Scenario labels describe the simulator, not detected anomalies.
4. Inspect the separate **simulation truth** chart for glucose, lactate, cell
   density, or volume. These values are not soft-sensor predictions.
5. Open **Raman snapshot** and move the observation index to inspect spectra.
6. Open **Data sources & assays** to compare sample collection and result times,
   inspect raw sensor observations, or download selected sensor rows.
7. Change simulation settings and click **Generate dataset**. Keeping the same
   seed and settings reproduces the same measurements.

## Learn step by step

Read [the Phase 1 learning guide](docs/phase_1_learning_guide.md) for equations,
a worked numerical example, code explanations, exercises, and three interview
questions with sample answers. The [simulation assumptions](docs/synthetic_data_assumptions.md)
document units, coefficients, noise, and scientific limits.

1. Understand feed, concentration, cell growth, and dilution.
2. Trace one mass-balance step in `advance_culture()`.
3. Understand NumPy arrays, pandas tables, seeds, and Gaussian peaks.
4. Explore the dashboard and explain the source of each displayed value.
5. Continue with Data Integration after understanding the measurement sources.

## Phase 2: integrate sources and inspect quality

Open **Data Integration** from the Streamlit sidebar. The Overview's most recent
simulation settings are shared across both pages; direct entry uses the defaults.

1. Keep **Synthetic dataset** selected. The default dataset aligns 870 spectra
   with 870 sensor rows and pairs 42 laboratory samples at collection time.
2. Enable **Add known data-quality issues**. Review missing cells, duplicate
   IDs/timestamps, an invalid timestamp, impossible DO, and unmatched rows.
   This creates a separate faulty copy and preserves the Overview dataset.
3. Compare a **5 minute** and **30 minute** sensor tolerance with the faulty copy.
   Its Raman timestamps are shifted by +10 minutes, making tolerance meaningful.
4. Move **Review elapsed time** from 0 h to 6 h. Initial laboratory results appear
   only after their result timestamps. Retrospective label pairs still describe
   the original collection time.
5. Under **Raw & processed sources**, download the source CSV ZIP. Extract it,
   select **Upload CSV sources**, and upload `sensors.csv`, `raman.csv`,
   `assays.csv`, and `batches.csv` to reproduce the workflow.
6. Download the integration report ZIP, which includes raw inputs, processed
   copies, aligned data, reference pairs, quality tables, and a rule manifest.

Read the [Phase 2 learning guide](docs/phase_2_learning_guide.md) for schemas,
equations, a numerical example, source eligibility rules, function explanations,
limitations, exercises, and three interview questions with sample answers.
Use only synthetic or appropriately licensed public data for imports.

## Phase 3: explore spectra and transformations

Open **Spectral Analytics** in the sidebar. The new laboratory uses the Overview
simulation or the latest successful Phase 2 prepared Raman snapshot. Incomplete
or ineligible Raman rows are reported and excluded without filling intensities.

1. Keep the defaults and compare raw overlays, mean spectra, peaks, estimated
   background, and raw-minus-smoothed residuals.
2. Compare baseline correction, SNV, MSC, finite-difference or Savitzky–Golay
   derivatives, FFT magnitude, and wavelet representations. Each panel keeps
   the correct axis and units.
3. In **Fourier laboratory**, vary the retained frequency band, detrending, and
   Hann window. Reconstruction preserves phase and reconstructs the analyzed
   signal; it does not undo tapering or removed trends.
4. In **Wavelet laboratory**, inspect approximation/detail coefficients and
   change the denoising threshold. A zero threshold gives a reconstruction check.
5. In **PCA exploration**, use B001–B003 as reference batches and project B004–B006.
   Change preprocessing/scaling, inspect scores, loadings, explained variance,
   and the largest reconstruction residuals. These are exploratory views, not
   glucose predictions or validated alarms.
6. Download selected-spectrum transformations, Fourier values, wavelet summaries,
   or PCA scores/loadings for inspection.

Read the [Phase 3 learning guide](docs/phase_3_learning_guide.md) for equations,
worked examples, fit/transform rules, limitations, three interview questions with
answers, and verification results. Dataset-learned MSC references, centering,
scaling, and PCA components are fitted on reference batches only.

## Project layout

```text
app.py                          Streamlit Overview and plotting
src/data_generation.py          Process balances and measurement simulation
src/app_data.py                 Cached synthetic dataset shared across pages
src/data_integration.py         CSV preparation, quality flags, and time matching
src/preprocessing.py            Baseline, SNV, MSC, SG, region selection
src/spectral_transformations.py Derivatives, FFT, inverse FFT, and wavelets
src/chemometrics.py             Reference-fitted PCA pipeline and projection
src/spectral_data.py            Raman adapters with explicit exclusions
pages/2_Data_Integration.py     Integration, quality, and availability dashboard
pages/3_Spectral_Analytics.py    Interactive spectral analysis laboratory
tests/test_data_generation.py   Scientific and data identity checks
tests/test_app.py               Streamlit startup and interaction checks
tests/test_data_integration.py  Time, provenance, missingness, and quality checks
tests/test_integration_app.py   Navigation and integration page interactions
tests/test_preprocessing.py     Known answers and training-only preprocessing
tests/test_spectral_transformations.py
tests/test_chemometrics.py      PCA, leakage, reconstruction, source identity
tests/test_spectral_app.py      Spectral controls and cross-phase navigation
docs/phase_1_learning_guide.md   Equations, code walkthrough, interview practice
docs/phase_2_learning_guide.md   Integration rules, exercises, interview practice
docs/phase_3_learning_guide.md   Spectral methods, regression checks, interview practice
docs/synthetic_data_assumptions.md
.streamlit/config.toml          Theme and disabled usage telemetry
requirements.in                Direct dependencies
requirements.txt               Pinned complete environment
```

The user-authored master plan and existing Git configuration are preserved.
Future modules and pages will be added when their phase is approved; there are
no empty placeholder implementations. Data is generated in memory and cached by
configuration. No external data source, credentials, or model artifact is needed.
Optional CSV uploads remain in the local Streamlit session. No dependency
changes were needed for Phase 2.

## Run checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Checks cover mass-balance invariants, glucose depletion, invalid inputs,
reproducibility, batch identity, UTC timestamps, uniform spectral spacing, assay
delays, spectral response to glucose, the designed disturbance, and Streamlit
startup and interactions. Phase 2 adds batch-boundary and timestamp-tolerance
tests, delayed-reference availability, duplicate quarantine, missing-value
preservation, CSV round trips, report exports, and cross-page navigation.
Phase 3 adds analytical derivative/DFT/FFT examples, normalization and filtering
checks, odd/even FFT and wavelet reconstruction, denoising on a known noisy
signal, fit/transform leakage checks, PCA reconstruction, plot units, and
cross-phase data handoff. Run the entire suite to check all completed phases.

No concentration prediction model, batch-aware predictive comparison, calibrated
multivariate monitoring limits, or deployment is implemented yet. **Development
stops after Phase 3 until you explicitly request Phase 4.**
