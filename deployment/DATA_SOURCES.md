# Historical development and monitoring demo populations

The app generates distinct synthetic measurements for development and for
monitoring. The simulator equations, physical spectral axis and saved-model
preprocessing remain unchanged.

| Use | Seed at default settings | Batch/sample identities |
|---|---|---|
| Overview, spectral exploration, synthetic modeling and its held-out test | 42 | B001; B001-S00000 |
| Data Acquisition in-app fixture, Process Monitoring replay and streams, operational reliability review | 3000059 | MON-B001; MON-B001-S00000 |

The monitoring seed is the development seed plus **3,000,017**. This creates
separate deterministic random draws while keeping the walkthrough reproducible.
Changing the Overview configuration changes the associated monitoring-demo
configuration and source identity. Duration, observation interval, channel count
and batch count follow Overview; its seed and measurements are separate.

Historical development measurements and cohort fingerprints retain their
previous values. Existing sealed studies remain sealed. Monitoring observations
do not automatically become training inputs or cause retraining/test evaluation.
Separately prepared demo calibration models retain their existing independent
calibration seeds and CV-only evidence labels.

## Try it

1. On Overview, keep seed 42 and generate historical development data.
2. Open Data Acquisition. Its caption shows monitoring generation seed 3000059
   and the fixture selector offers MON-B001, MON-B002, and so on.
3. Poll records and expand **Received data provenance**. Open Process Monitoring
   to inspect this same collection, integration/quality results and predictions.
4. Open Process Monitoring. Replay, historical review and both in-app stream
   timing modes use this same separate monitoring population.
5. Choose a saved development model or explicitly prepare a separate demo
   calibration. Accepted observations use its saved preprocessing/model;
   rejected observations remain gaps with recorded reasons.
6. Model Reliability keeps frozen historical test results separate from the
   operational review of MON- observations. That review does not create a new
   independent-test record.

Multivariate Monitoring currently calibrates and compares historical Overview
sensor batches. Acquisition and Process Monitoring now share one collector;
the multivariate page remains a historical comparison.

## Provenance contract

Generated batch metadata and `batches.csv` include these optional fields:

| Field | Meaning |
|---|---|
| `data_role` | `historical_development` or `monitoring_demo` |
| `generation_seed` | Seed actually used to generate the measurements |
| `development_seed` | Development configuration that the demo accompanies |
| `data_source_id` | Deterministic identity from role, generation configuration and protocol |
| `generator_protocol` | Version of the population-separation policy |
| `generation_config_json` | Actual simulator configuration, including generation seed |

These fields are metadata and never enter the prediction feature matrix.
The configuration identity describes the generated population; the acquisition
transport also retains its separate source snapshot ID, event IDs and cursor.

The fields survive collection snapshots, Data Integration preparation and
CSV exports. Acquisition/integration/stream manifests expose the received
provenance; replay/reliability report contexts include the monitoring policy.
New modeling-study and model metadata retain the development source declaration.
Older saved studies remain readable with provenance marked unrecorded.

Uploaded/HTTP provenance is **source-declared**, not authenticated instrument
identity. Producers may omit the optional fields; absence stays unrecorded.
The app does not infer an external source's seed from local settings. A fresh
timestamp or new transport identity alone does not prove new measurements.

Monitoring snapshots are not offered as a development-page data source.
An explicitly uploaded copy may still contain previously monitored observations;
training makes those observations development data for that study. Their reuse
cannot be claimed as unseen monitoring evidence. Distinct default seeds do not
establish independence from arbitrary uploads.

## Standalone HTTP producer

```powershell
.\.venv\Scripts\python.exe -m src.test_acquisition_source --port 8765 --seed 42 --stream --duration 24 --channels 71
```

`--seed` specifies the development basis. The command prints and records actual
monitoring seed 3000059, MON- identities and the generation configuration.
Its measurements are generated before fresh timestamp rebasing; rebasing only
changes when records become available. Without `--stream`, the same population
uses historical timestamps for retrospective testing.

The producer is still a localhost test adapter. This change adds no physical
instrument connector and does not alter model applicability/range thresholds.

## Verification

Tests check that monitoring spectra, sensors and reference assays differ;
historical values and cohort fingerprints remain unchanged; IDs join correctly
across all modalities; sources reproduce; provenance survives preparation and
exports; stream rebasing preserves monitoring measurements; and collection never
trains or changes sealed historical artifacts.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
