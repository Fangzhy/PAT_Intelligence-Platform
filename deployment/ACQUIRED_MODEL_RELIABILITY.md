# Model reliability for shared incoming observations

The **Acquired data reliability** view reads the same session used by Data
Acquisition, Process Monitoring and acquired Multivariate Monitoring. It reuses
recorded glucose/lactate predictions. Opening the view, changing display rules,
exporting, or capturing an explanation never fits a transformer/model, repeats
incoming-data inference, or evaluates a holdout.

## First walkthrough

1. Restart Streamlit after updating the application.
2. Develop/select and finalize a model on Predictive Modeling, or explicitly
   prepare the separate CV-only teaching models on Process Monitoring.
3. Open Data Acquisition, select the saved model and a monitoring source, then
   poll or start collection. Historical in-app data is convenient for a quick
   walkthrough; current availability stays unavailable when freshness is disabled.
4. Open Model Reliability. Acquisition requests **Acquired data reliability**.
   Select an active target, received batch, and review lookback (default 24 h).
   With daily synthetic assays, use a 72 h lookback to inspect a rolling bias
   requiring three assessed pairs; shorter windows can legitimately be unassessed.
5. Review the active model's recorded historical CV/test evidence separately
   from its operational assay residuals, rolling bias and assessment coverage.
6. Expand **Original training reliability baseline**. Choose the development
   source that actually produced this model, then click **Prepare / save matching
   training baseline**. Separate stream demo models use **Model demo calibration**.
   Finalized synthetic models use **Historical synthetic development cohort**
   with the original Overview settings; uploaded models require the original
   successful Data Integration snapshot via **Latest prepared development sources**.
7. Inspect input/prediction shift and the quality/freshness audit. Export the
   report. Optionally capture the view, preview its aggregate facts, and request
   an LLM explanation or the local template summary.

The current source must implement the existing JSON acquisition contract.
Assay events are optional: a Raman-only instrument cannot establish operational
prediction accuracy. Supply laboratory references through the shared source's
`assays` records when they become available. A vendor-specific instrument/LIMS
connector or separate assay-upload interface is not introduced here.

## Historical evidence and exact artifact matching

Acquired mode inherits the model selection from the shared workspace, rather
than selecting a different reliability-demo model. The model's identity, version,
target, study/cohort and artifact stamp identify the training baseline. Record
version mismatches are unassessed. Source/model/input-rule changes retain the
existing shared behavior: a fresh paused collection context is created.

Frozen historical performance is read from the selected model's saved metadata.
CV-only teaching models remain labeled CV-only; operational errors never grant
them an independent-test claim. The historical review mode remains available
for its original saved-study workflow and pauses acquisition when selected.

## Explicit training baseline

Baseline preparation verifies the original development cohort hash, target,
input schema, training partition and row count against the selected model. It
uses **only original training rows** to record raw sensor/spectral indicator
distributions and saved-model training predictions. It performs no fitting,
selection, test disclosure or monitoring-data calibration. Missing or mismatched
development inputs leave drift unavailable; they never trigger a replacement demo.

An exact matching saved baseline is reused without predicting again. Baselines
are JSON artifacts in ignored `.pat_state/acquired_reliability/`, with a content
digest and model identity checks. They do not modify existing model/study files.
Public-demo baselines are isolated in each visitor's temporary state root.

## Causal assay assessment

- Match exact batch ID, sample ID and collection/measurement timestamp.
- Use only actually received assays. Laboratory result time alone does not
  make a result available to this application.
- A pair becomes available at the latest of laboratory result time, first
  receipt time, and the prediction's assessment time. Preserve all three times.
- Rolling bias follows pair availability order within the selected batch/window.
- Missing, invalid, duplicate/conflicted or unmatched references, unavailable
  predictions and time mismatches remain explicit unassessed rows with unknown
  alarm states. Malformed payloads remain in the original delivery ledger.
- RMSE, MAE, bias and R² use assessed pairs only, accompanied by reference and
  assessment counts. Without assessed pairs, prediction accuracy is not assessed.
  R² remains undefined with insufficient or constant references.
- No simulated truth, target interpolation or nearest-time assay substitution
  fills a missing reference. Assays never enter the predictive input pipeline.

## Windows, quality, freshness and drift

The lookback is anchored at the latest received measurement in the selected
batch, so historical fixtures do not vanish merely because wall-clock time is
later. Assay availability is still evaluated at the actual current UTC clock.
Later source silence/read failures or staleness withhold current eligibility
while preserving valid recorded predictions and operational residuals. Current
status additionally requires enabled wall-clock freshness. Invalidated inputs
retain their original audit and withheld predictions.

Input shift uses aligned, non-invalidated received raw sensor values and raw
spectral mean/edge indicators. Axes must match; mismatches leave spectral
indicators unknown. Prediction shift uses accepted recorded estimates. At least
five finite values per distribution are required. Wasserstein distance divided
by training SD is a descriptive effect-size screen; constant references use an
explicit changed/unchanged check. It is not a calibrated alarm probability.

Training data can cover a whole batch while the review covers only one process
phase. Phase differences, rejected observations and sparse assays affect these
comparisons. A distribution flag alone does not demonstrate model degradation,
a sensor fault or a physical root cause. PCA T²/SPE retain their own historical
reference on Multivariate Monitoring; the predictive pipelines retain their own
saved preprocessing. No combined reliability score is invented.

## Reports, explanations and verification

The ZIP separates `frozen_validation/`, `training_reference/`,
`operational_review/` and the original `received/delivery_audit.csv`. The manifest
records model identity, cursor, source/session identity, as-of time, selected
batch/window, rule settings, baseline identity and input coverage. The delivery
ledger can cover the whole received session; operational tables use the selected
batch/window. No unreceived source rows are exported.

Explanations use a manually captured aggregate snapshot and the existing
primary/backup/template chain. No raw spectra, assay rows, identifiers,
endpoints or credentials enter the outbound explanation. External sources
require the existing consent control. Timer reruns never request explanations.
Eligibility and assessment remain deterministic Python checks.

Tests cover exact training baseline matching, no fit/repeated inference,
delayed/future assays, conflicts, receipt-order bias, missing evidence, freshness,
exports, aggregate explanations, shared navigation/cursor and frozen artifact
preservation. No real device or external LLM call is required for these checks.
