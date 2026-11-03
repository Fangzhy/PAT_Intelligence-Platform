# Vaccine-PAT-Intelligence-Platform

A learning-oriented Python application for exploring a synthetic recombinant
protein fed-batch bioreactor and, in future phases, developing PAT analytics.
**Phases 1–7 are implemented.** The default measurements are synthetic. This is an
educational portfolio demonstration, not a manufacturing-validated system.
An optional **Data Acquisition** page, labeled **Test mode**, provides separate polling
adapters against in-app and local HTTP test sources.

## Workflow navigation

The sidebar follows the data workflow, with Overview first and named sections:

| Section | Pages in order |
|---|---|
| Overview | Overview |
| Data | Data Acquisition → Data Integration & Quality |
| Analytics | Spectral Analytics → Predictive Modeling |
| Monitoring | Process Monitoring → Multivariate Monitoring → Model Reliability |

Data Integration's source selector includes an **Open Data Acquisition** shortcut.
Acquisition keeps its **Test mode** badge visible and sends a detached snapshot
back to integration. Source/adapter code remains separate from the page UI.

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
starting the app. Phase 4 adds pinned XGBoost while retaining all 47 earlier
dependency versions.

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

## Phase 4: compare soft-sensor models with batch-aware validation

Open **Predictive Modeling** in the sidebar. Model labels are sparse offline
glucose or lactate assays, not latent simulated truth. The default modeling
cohort expands the six Overview batches to eight while preserving their original
measurements: B001–B006 are training batches and B007–B008 are held out.

1. Keep the default glucose target and inspect the fixed batch assignments.
   At 144 h there are 56 labeled assay rows: 42 training and 14 held out.
2. Choose experiments and click **Train and compare (CV only)**. The planned
   13-experiment comparison includes all nine feature families and all four
   model types: PLS, Linear Regression, Random Forest, and XGBoost.
3. Inspect pooled RMSECV, latent-variable selection, out-of-fold predictions,
   residuals, feature stability, and prescribed noise/baseline sensitivity.
   Learned MSC references and scalers are fitted inside each training fold.
4. Finish training-only comparisons before clicking **Freeze comparison and
   evaluate held-out batches once**. The same final batches and rows evaluate
   every successful experiment; the CV-selected winner stays fixed.
5. Revisit the page to reload saved results. The local holdout record disables
   further tuning after disclosure. Choose a genuinely new cohort for further
   independent-validation claims.
6. Download the study report. It includes the fitted CV-selected pipeline;
   holdout scores appear only after final evaluation. Final pipeline/metadata
   and audit files are saved under ignored `.pat_state/modeling/`.

The page can also use the latest successful Phase 2 prepared sources. It reports
input exclusions and uses one common row cohort for every selected feature/model
comparison. Prior inspection of a dataset elsewhere cannot be undone by a batch
split or local evaluation gate.

Read the [Phase 4 learning guide](docs/phase_4_learning_guide.md) for the feature
recipes, PLS and validation mathematics, code walkthrough, numerical examples,
test outcomes, limitations, and three interview questions with sample answers.

## Phase 5: replay monitoring with saved model inference

Open **Process Monitoring** in the sidebar. The page replays the Overview's
synthetic sensor/Raman observations, using saved glucose and lactate pipelines.
It contains no physical equipment connection or control action.

1. Choose a batch and inspect the saved model selections. Default separate demo
   models have been prepared locally. **Prepare / reuse saved demo models** can
   create matching artifacts for another Overview configuration; it is an explicit
   training action on a different calibration seed, never part of ordinary replay.
2. Click **Start / resume**, **Pause**, or **Advance one observation**. Change
   observations per second while preserving replay progress. Reset clears the
   current replay history. Navigation to another page pauses replay.
3. Inspect direct sensor trends and separately labeled model estimates. Missing
   inputs, implausible sensors, mismatched spectral axes, domain failures, and
   invalid predictions create recorded gaps; simulated truth is never substituted.
4. Edit advisory limits or the illustrative feature-domain threshold. Changing
   the batch, mode, models, or rules starts a fresh paused context, keeping each
   history internally consistent. No limits are presented as validated CPP limits.
5. Enable **Inject known replay faults** and use **Historical review** to inspect
   indices 3, 7, 10, and 12. Moving backward hides future observations and assays.
6. Inspect delayed references: only assays whose result times have arrived are
   shown. Historical review extends after the last observation to reveal final
   laboratory reports without inventing new process observations.
7. Download the revealed monitoring report with versioned predictions, quality
   and alarm states, available references, and the replay/rule manifest.

Finalized Phase 4 winner pipelines are discovered automatically. Their test record
is not modified. The separate default demo models use calibration seed 1000045
for replay seed 42 and are explicitly labeled **grouped CV only**. Default
artifacts live under ignored `.pat_state/monitoring/models/`; missing models
still allow sensor-only replay. No dependency changes were needed for Phase 5.

Read the [Phase 5 learning guide](docs/phase_5_learning_guide.md) for replay and
inference mathematics, source/time separation, code explanations, limitations,
verification outcomes, and three interview questions with sample answers.

## Phase 6: multivariate statistical process monitoring

Open **Multivariate Monitoring** in the sidebar and click **Fit reference
monitoring model**. The page uses observed temperature, pH, dissolved oxygen,
feed flow, and agitation. No simulated concentrations or scenario labels enter
the PCA matrix. It provides:

- Hotelling's T², SPE/Q, and a maximum statistic-to-limit ratio over elapsed time.
- Nominal/disturbed batch comparison, exceedance fractions, invalid input counts,
  and designed disturbance annotations separate from detector flags.
- PCA scores, loadings, explained variance, and additive SPE residual contributions.
- Editable nominal reference batches, one to four components, and per-chart
  empirical quantiles; fitting occurs only on the explicit calibration action.
- A downloadable historical report containing all observations, batch summaries,
  reference row audit, loadings, fitted parameters, limits, and assumptions.

The default six-batch split fits scaling/PCA on **B001–B003**, estimates limits
from separate nominal **B005**, and evaluates untouched **B004 and B006**.
The split is by whole batch. Chart controls never refit the model. Calibration
survives page navigation within the session; changing the Overview simulation
requires a new fit. Navigating here also pauses Phase 5 replay.

Limits are empirical observation-level quantiles, not validated false-alarm
probabilities. This initial PCA pools time-varying batch trajectories; temporal
correlation, few independent batches, and repeated use of two charts matter.
With the default seed 42 and 144 h duration, B004 flags at 58/145 observations,
including 47/51 after its designed disturbance onset. Nominal B006 also flags
at 21/145 observations, and B004's first flag precedes the disturbance. These
results illustrate both detection and nuisance alarms; they do not establish a
validated detector or an accurate disturbance-onset estimate.

Missing/nonfinite or implausible sensor values remain unscorable with gaps and
unknown alarm states. The original data is preserved without interpolation.
This page performs historical comparison; the Phase 5 replay export still
contains only revealed observations. No dependencies were added for Phase 6.

Read the [Phase 6 learning guide](docs/phase_6_learning_guide.md) for the
mathematics, a worked example, code walkthrough, test outcomes, limitations,
and three interview questions with sample answers.

## Phase 7: model reliability and lifecycle validation

Open **Model Reliability** in the sidebar. Select glucose or lactate and either
prepare a separate frozen demo or read a matching saved Phase 4 study. The page
adds:

- Recorded training, grouped CV, and independent-batch RMSEP/MAE/R²/bias, with
  batch-level test results and the original CV-selected winner.
- Saved-pipeline historical review with a time cutoff, exact sample/time assay
  pairing after result availability, residual limits, and rolling bias alerts.
- Descriptive sensor, Raman level/baseline-proxy, and prediction-distribution
  shift screens against original training references.
- Missing-input, model-domain, spectral-axis, and invalid-prediction flags with
  unavailable estimates preserved as gaps.
- Controlled copies illustrating pH offset, Raman baseline shift, assay bias,
  and invalid/out-of-domain inputs; raw data and frozen results are preserved.
- A ZIP report separating previously frozen validation evidence from the
  revealed-only operational review, including source, model version, rules,
  timestamps, reference coverage, and quality flags.

For a first walkthrough, click **Prepare / reuse frozen reliability demo**, then
**Review through latest assay**. Move **Review elapsed time (h)** backward and
confirm future reference results disappear. Changing residual/drift limits
does not retrain or reopen evaluation. Source/batch/fault/domain changes create
a fresh review context. Navigating here pauses Phase 5 replay.

The demo uses a separate calibration seed (Overview seed + 2000007), at least
eight batches, raw-Raman PLS with training-only grouped CV, and a saved one-time
independent test on the final two batches. Target-specific local records live
under ignored `.pat_state/reliability/`; preparation is explicit and completed
records are reused. Existing Phase 4 studies are read without claiming an
unfinished holdout. Finalize those explicitly on Predictive Modeling first.

Shift is measured with empirical Wasserstein distance divided by training SD.
Its editable threshold is descriptive, not a calibrated probability. Early
batch windows, changing process phase, and prediction rejection can alter the
distribution. A shift flag does not prove sensor drift or model degradation;
assay bias can also produce large prediction-minus-reference errors.

Read the [Phase 7 learning guide](docs/phase_7_learning_guide.md) for equations,
code examples, default numerical results, limitations, tests, and three
interview questions with sample answers. No dependencies were added.

## Optional extension: separate test acquisition adapters

The original app had synthetic replay and CSV import, but no independent polling
adapter. **Data Acquisition** now provides that prototype while preserving the
simulator and existing monitoring pages.

1. Open **Data → Data Acquisition** in the sidebar. Keep **In-app test source** selected
   and click **Poll once**, or **Connect / start polling** for one poll per second.
2. Inspect collected sensors, spectra, delayed assays, batch metadata, source
   timestamps, receipt timestamps, and the delivery audit. Pause/reset as needed.
3. Enable **Include test delivery faults**, apply settings, and poll. Identical
   duplicate deliveries are suppressed; invalid timestamps/schema are quarantined;
   missing numeric observations stay missing with quality flags.
4. Click **Send snapshot to Data Integration**, then open Data Integration in
   the sidebar. **Acquired test snapshot** is selected automatically. The normal
   alignment/quality workflow runs on a detached snapshot; subsequent collection
   does not silently change it. Spectral Analytics can use this Phase 2 handoff.
5. Download the collected source CSVs, original delivery JSON audit, and cursor/
   source manifest. Only polled records are included.

To exercise the HTTP adapter against a separate local process, run another
terminal at the project root:

```powershell
.\.venv\Scripts\python.exe -m src.test_acquisition_source --port 8765
```

Select **Local HTTP test source**, use `http://127.0.0.1:8765`, click **Apply
source settings**, then poll. The default server serves a fixed 24 h synthetic
batch with 71 Raman channels. Add `--faults` for delivery/quality examples.
Stop it with **Ctrl+C**; failed reads pause the collector without advancing its
cursor or deleting collected data. Restarting the identical source can resume;
a changed source identity requires an explicit collection reset.

Both adapters implement `poll(cursor, limit)` and return the same bounded JSON
page contract. Collection handles source identity, duplicate/conflicting event
IDs, channel consistency, schema/time validation, and numeric quality flags.
The HTTP client is GET-only, limited to explicitly configured loopback test
URLs, and uses a read timeout and response-size limit. Navigation pauses polling.

This is historical test-record replay through an acquisition boundary, not a
live equipment, OPC UA, or MQTT connection. Measurement and laboratory result
times remain separate from current receipt time. The session buffer is bounded
at 10,000 deliveries; it is not a durable historian or exactly-once message
system. No new dependencies or equipment control actions were added. Phase 8
deployment remains a separate future phase.

Read the [acquisition adapter learning guide](docs/acquisition_adapter_learning_guide.md)
for the protocol, examples, extension points, limitations, tests, and three
interview questions with sample answers.

## Project layout

```text
app.py                          Grouped Streamlit page router and workflow order
pages/1_Overview.py              Synthetic bioreactor Overview and plotting
src/data_generation.py          Process balances and measurement simulation
src/app_data.py                 Cached synthetic dataset shared across pages
src/navigation.py               Replay pause behavior independent of cached data imports
src/data_integration.py         CSV preparation, quality flags, and time matching
src/preprocessing.py            Baseline, SNV, MSC, SG, region selection
src/spectral_transformations.py Derivatives, FFT, inverse FFT, and wavelets
src/chemometrics.py             Reference-fitted PCA pipeline and projection
src/spectral_data.py            Raman adapters with explicit exclusions
src/feature_engineering.py      Nine fixed, fit/transform-consistent feature families
src/modeling_data.py            Assay-labeled common cohort and batch partitions
src/modeling.py                 Grouped CV search, four models, and final scoring
src/validation.py               Regression metrics and fold audit records
src/modeling_storage.py         Saved studies, one-time holdout gate, model/report export
src/model_registry.py           Trusted saved model discovery and separate demo calibration
src/monitoring.py               Input/domain validation, saved inference, advisory alarms
src/replay.py                   Replay clock, revealed history, causal assay views, reports
src/multivariate_monitoring.py   Reference-fitted PCA, separate empirical limits, scores, audit/report
src/reliability.py              Frozen metrics, causal assay residuals, drift, and reports
src/reliability_storage.py      Read-only evidence loading and isolated explicit demo preparation
src/acquisition.py              Adapter contract, polling, deduplication/quarantine, snapshots, audit
src/test_acquisition_source.py  Standalone loopback JSON test server and CLI
pages/2_Data_Integration.py     Integration, quality, and availability dashboard
pages/3_Spectral_Analytics.py    Interactive spectral analysis laboratory
pages/4_Predictive_Modeling.py   Comparative soft-sensor modeling laboratory
pages/5_Process_Monitoring.py    Replay controls, real model estimates, and advisory alerts
pages/6_Multivariate_Monitoring.py PCA control charts, batch comparison, scores, and residuals
pages/7_Model_Reliability.py     Frozen evidence, review clock, residuals, drift, and provenance
pages/9_Test_Acquisition.py      Optional in-app/HTTP polling lab and explicit Phase 2 handoff
tests/test_data_generation.py   Scientific and data identity checks
tests/test_app.py               Streamlit startup and interaction checks
tests/test_navigation_app.py    Sidebar order, visible test mode, integration shortcut and state
tests/test_data_integration.py  Time, provenance, missingness, and quality checks
tests/test_integration_app.py   Navigation and integration page interactions
tests/test_preprocessing.py     Known answers and training-only preprocessing
tests/test_spectral_transformations.py
tests/test_chemometrics.py      PCA, leakage, reconstruction, source identity
tests/test_spectral_app.py      Spectral controls and cross-phase navigation
tests/test_feature_engineering.py
tests/test_modeling.py          Metrics, label provenance, CV isolation, and all models
tests/test_modeling_storage.py  Evaluation gate and fitted-artifact inference
tests/test_modeling_app.py      Training actions, persistence, sealing, and navigation
tests/test_model_registry.py    Model discovery, calibration isolation, artifact reuse
tests/test_monitoring.py        Alarm boundaries, input/domain checks, true inference
tests/test_replay.py            Clock transitions, source identity, causal histories
tests/test_monitoring_app.py    Controls, fault gaps, model preparation, navigation
tests/test_multivariate_monitoring.py T²/Q known answers, isolation, disturbance, quality, reports
tests/test_multivariate_app.py  Calibration actions, chart units, configuration, navigation
tests/test_reliability.py       Known residual/drift answers, causality, gaps, reference isolation
tests/test_reliability_storage.py Frozen reuse, holdout isolation, separate targets, sealed state
tests/test_reliability_app.py   Explicit preparation, time controls, saved studies, navigation
tests/test_acquisition.py       Cursor/identity/quality contracts, source round trips, snapshot export
tests/test_acquisition_http.py  Real loopback reads, timeouts, disconnects, invalid HTTP/JSON
tests/test_acquisition_app.py   Controls, detached handoff, page navigation, visible connection errors
docs/phase_1_learning_guide.md   Equations, code walkthrough, interview practice
docs/phase_2_learning_guide.md   Integration rules, exercises, interview practice
docs/phase_3_learning_guide.md   Spectral methods, regression checks, interview practice
docs/phase_4_learning_guide.md   Model comparison, interpretation, tests, interview practice
docs/phase_5_learning_guide.md   Replay, inference, reliability, tests, interview practice
docs/phase_6_learning_guide.md   PCA monitoring, empirical limits, false alarms, interview practice
docs/phase_7_learning_guide.md   Reliability evidence, drift, delayed residuals, interview practice
docs/acquisition_adapter_learning_guide.md Test-source protocol, usage, assumptions, interview practice
docs/synthetic_data_assumptions.md
.streamlit/config.toml          Theme and disabled usage telemetry
requirements.in                Direct dependencies
requirements.txt               Pinned complete environment
```

The user-authored master plan and existing Git configuration are preserved.
Future modules and pages will be added when their phase is approved; there are
no empty placeholder implementations. Data is generated in memory and cached by
configuration. No external data source or credentials are needed. Soft-sensor
monitoring loads local saved pipelines; its explicit demo preparation action
can supply them. Optional CSV uploads and acquired snapshots remain in the local
Streamlit session.
No dependency changes were needed for Phases 2, 5, 6, or 7.

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
Phase 4 adds feature consistency, sparse assay provenance, common cohorts,
whole-batch CV, unchanged fitting when test labels change, all four model types,
noise sensitivity, persistent split/evaluation guards, trusted model inference,
report contents, explicit training actions, and cross-phase regressions.
Phase 5 adds deterministic replay timing, start/pause/resume/step/reset, inclusive
alarm boundaries, invalid/negative prediction gaps, training-statistic domain
checks, causal assay visibility, saved-model reuse without fitting, model/source
identity, revealed-only exports, and navigation pause behavior.
Phase 6 adds analytical T²/Q calculations, reference-only scale/PCA fitting,
separate nominal limit calibration, complete batch partition guards, invalid
input gaps, reproducible reports, disturbance detection, correct chart units,
explicit fitting, and navigation/configuration regression checks.
Phase 7 adds frozen evidence loading, separate reproducible demo studies,
sample/result-time residual causality, rolling bias, known Wasserstein distances,
constant/insufficient-data screens, training-reference isolation, guarded saved
inference, report scope, source/axis checks, and earlier-phase navigation.
The optional acquisition extension adds real loopback HTTP transport checks,
atomic cursor progress, duplicate/conflict handling, malformed/changed-source
recovery, buffer bounds, receipt/source time separation, and explicit snapshot
handoff through existing integration without changing the simulator.

Live equipment connections, manufacturing validation, validated false-alarm
rates, and Phase 8 deployment remain future work. **Development stops after
Phase 7 until you explicitly request Phase 8.**
