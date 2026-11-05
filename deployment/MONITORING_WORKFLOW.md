# Monitoring acquisition and the saved processing pipeline

The two workflows have separate roles:

```text
Historical demo / uploaded development files
  -> Data Integration & Quality -> Predictive Modeling -> frozen saved pipeline

New synthetic monitoring population / compatible JSON API
  -> Data Acquisition -> saved integration and quality rules
  -> saved preprocessing and scaler -> saved prediction model
  -> Process Monitoring: input, processing, prediction and audit results
```

## Run the workflow

1. Prepare historical data on Data Integration & Quality. Keep **backward**
   matching for models intended for online use. Save the intended tolerance
   and sensor plausibility ranges by training from **Latest prepared sources**
   in Predictive Modeling. The direct synthetic modeling option uses default
   integration settings.
2. Train/select using grouped CV and complete the existing one-time independent
   test when ready. Newly created studies save the integration configuration
   and quality ranges with the model metadata and fitted pipeline. Test results
   do not select or recalibrate these rules.
3. Open **Monitoring -> Data Acquisition**. Select fresh or historical synthetic
   monitoring data, a local HTTP test source, or a compatible Raman HTTP API.
   The synthetic population has its separate monitoring seed and MON- IDs.
4. Select the saved glucose/lactate pipeline. Its recorded backward matching,
   sensor tolerance and plausibility ranges are reused. The saved tolerance is
   displayed as read-only. Different target models must have matching input
   policies for one shared run.
5. Click **Poll once** or **Connect / start polling**. Acquisition performs no
   model training. It preserves received source rows, original delivery payloads,
   source/receipt times and quality decisions.
6. Open **Process Monitoring**. The app selects acquired-stream mode and keeps
   the same collector, received records, cursor and existing inference attempts.
   Active polling can continue between these two views. Other analysis pages
   pause collection. There is no detached handoff to the development page.

## What Process Monitoring shows

- **Received monitoring inputs:** accepted Raman, sensor, assay or batch records.
  Rejected deliveries remain in the delivery audit with their original payloads.
- **Integration and quality with saved rules:** same development integration code,
  backward matches, time differences, completeness, sensor range/statistical
  flags and issue records, evaluated on the currently received tables.
- **Saved preprocessing and standardized inputs:** select a model and aligned
  observation to inspect its fitted feature transformer and scaler outputs.
- **Online predictions:** concentrations, model version, inference time,
  applicability/range checks, advisory alarms, withheld estimates and reasons.
- **Assays and audit:** received laboratory results appear only when available;
  they are not prediction inputs. Later identity conflicts withdraw affected
  estimates while retaining their original values and revision history.

Per-observation inference retains the original already-received same-batch
backward sensor match, deadline and freshness decision. The integration report
describes the current received snapshot; it does not revise past inference.
Batch-local modified-z warnings can change as more observations arrive and are
descriptive warnings, as in development, rather than automatic exclusion gates.
Missing inputs are not imputed; axes are not resampled. No model or learned
preprocessing reference is fitted from incoming measurements.

The report ZIP includes `received/` tables, `processing/input_policy.json`,
integration/quality reports, latest inspectable transformed features per model,
prediction history, delayed assays, provenance and delivery/invalidation audits.
An unavailable processing result is recorded explicitly.

## Saved rules and older models

New models record `input_policy` in metadata and `input_policy_` on the fitted
pipeline. Their policy includes integration settings, sensor plausibility ranges,
missing-value handling and exact-axis requirements. Loading verifies the pipeline
and metadata agree. A saved retrospective **nearest** policy is rejected for
online monitoring; develop a new study using backward matching.

Older frozen studies are not overwritten or retested. If their policy was never
recorded, the UI explicitly identifies the default demo rules as a fallback;
it does not claim those rules came from that model. Prepare a genuinely new
development cohort for a new frozen study when saved-rule evidence is required.
No-model collection still works, with concentrations unavailable. Separate
CV-only teaching models remain labeled as such; completing a synthetic test
does not establish suitability for a real process.

Source/model/rule changes start a fresh paused monitoring context. Changing a
page or a displayed source table alone does not reset the collection.

## Synthetic and HTTP sources

For the local test producer, use another terminal:

```powershell
.\.venv\Scripts\python.exe -m src.test_acquisition_source --port 8765 --seed 42 --stream --duration 24 --channels 71
```

Choose **Local HTTP test source**, URL `http://127.0.0.1:8765`, and match the saved
model's spectral axis. Without `--stream`, choose historical review/freshness
settings explicitly. No timestamp override establishes actual current freshness.

**Raman HTTP API** accepts an explicitly configured HTTP(S) base URL. Remote
sources require HTTPS. It calls `GET <base>/v1/records?after=<cursor>&limit=<n>`
and expects `application/json` using the existing acquisition schema:

| Page field | Contract |
|---|---|
| `schema_version` | Integer 1 |
| `source_id` | Stable identity of this source run |
| `records` | List of `{sequence, event_id, source, data}` |
| `source` | `batches`, `sensors`, `raman`, or `assays` |
| `next_cursor` | Requested cursor plus delivered record count |
| `has_more` | Boolean; keep true for an open stream |
| `idle` | Optional true for an empty page from an open stream |

Records need sample/batch identities, explicit-timezone timestamps and the
existing per-table schema in `src/data_integration.py::REQUIRED_COLUMNS`.
Raman headers encode physical shifts, e.g. `raman_400`; the complete axis must
match the selected saved pipeline. Process sensors and batch metadata are also
needed by the current inference contract. Offline assays can arrive later.

If required, set `PAT_MONITORING_API_TOKEN` in the server process environment
before starting Streamlit; it becomes a bearer Authorization header. Do not put
credentials into URLs. Tokens are not saved in source contexts, CSVs, manifests
or AI summaries. Redirects are disabled and page size/timeouts remain bounded.
Public demo mode offers only the in-app sources.

This is a generic JSON adapter. A vendor-native Raman API needs a bridge/mapping
for its endpoints, authentication and measurement format. No vendor-specific
compatibility is claimed without instrument documentation and a test endpoint.

## Checks

Tests verify shared cursor/history, navigation behavior, saved-rule enforcement,
exact preprocessing parity, received-only export scope, artifact preservation,
API schema/auth-header handling and separation from development data.
