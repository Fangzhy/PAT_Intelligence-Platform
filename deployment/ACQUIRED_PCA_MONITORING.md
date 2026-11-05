# Acquired sensor monitoring with a frozen historical PCA reference

Multivariate Monitoring offers two views. **Historical reference comparison**
fits and evaluates the existing historical sensor workflow. **Acquired sensor
monitoring** reads the shared monitoring collection and applies that fitted PCA
reference to newly received sensor observations.

## First walkthrough

1. On Overview, generate historical development data with six batches. For a
   quick walkthrough, use seed 42, 72 hours and a one-hour observation interval.
2. Open Multivariate Monitoring in **Historical reference comparison** mode.
3. Keep fitting batches B001–B003, limit-calibration batch B005, two components
   and the 0.99 empirical quantile. Click **Fit reference monitoring model**.
   No new acquisition observations enter the scaler, PCA or limit calibration.
4. Optionally click **Save fitted historical PCA reference**. This writes an
   application-created local artifact to `.pat_state/multivariate/`. After a
   restart, use **Load saved PCA reference** to restore it explicitly.
5. Open Data Acquisition, select the separate synthetic monitoring source or a
   compatible JSON API, then poll or start collection. For deterministic
   historical testing, select **Historical in-app test source**; freshness will
   be explicitly marked as not enforced. Fresh mode uses current UTC times.
6. Open Multivariate Monitoring. Acquisition requests **Acquired sensor
   monitoring** mode. The page reuses the same session, adapter and cursor.
   If no reference is ready, collection pauses while you fit/load one; received
   records remain intact. Resume the shared acquisition after reference setup.
7. Inspect T², SPE, statistic-to-limit ratio, quality/freshness flags and one
   observation's additive SPE contributions. Return to Process Monitoring to
   inspect glucose/lactate estimates from the same collection.

## Independent fitted preprocessing

The PCA matrix contains the five observed process sensors: temperature, pH,
dissolved oxygen, feed rate and agitation. Scaling statistics and components are
fitted only on the selected historical fitting batches. Empirical T²/SPE limits
come from separate historical nominal limit batches.

Incoming sensors use that scaler and PCA directly. They never update the means,
scales, loadings, eigenvalues, component count or limits. Contributions are squared
standardized reconstruction residuals and sum to SPE. Their magnitude identifies
what to inspect, not the physical cause of an excursion.

The glucose/lactate soft sensors retain their own saved feature transformers,
scalers and estimators. PCA scoring never calls or changes those pipelines.
It also works with sensor-only received data when no prediction model is selected;
valid received batch metadata is needed for sample/time context.

## Quality, freshness and source health

- Only already received sensor deliveries and batch metadata are used. Malformed
  deliveries stay in the shared original delivery ledger.
- Missing/nonfinite or implausible values, shared saved sensor-quality failures,
  ambiguous identities/times and missing/late batch metadata are unscorable.
- Identical repeat deliveries produce one PCA observation. Later conflicts
  invalidate affected original sensor rows, retain their raw audit, and leave
  NaN statistics and unknown alarm states. Invalid data is never filled in.
- Freshness on arrival compares measurement time with recorded receipt time,
  using the shared acquisition rules. Stale/future-on-arrival rows are gaps.
- Later staleness, source silence or a read failure withholds current cards.
  Historical valid statistics remain available with current freshness/health
  flags. With wall-clock freshness disabled, historical scores are available
  but current cards remain unavailable.

New stream observations have role **Acquired monitoring**, distinct from fitting,
limit calibration and historical test batches. Their scores do not constitute a
new independent validation. Empirical per-chart limits do not guarantee false
alarm probabilities under repeated, correlated batch observations.

## Shared collection and persistence

The acquired view can start, pause or poll the existing shared source. These
controls use the existing adapter, soft-sensor rules and collection buffer;
they do not establish another API connection or reset the cursor. Scoring,
chart selection, contributions and exports are read-only operations. Other
analysis views, including historical PCA comparison, pause collection.

Reference save/load is explicit. The artifact contains the fitted scaler, PCA,
limits, historical batch assignment/audit and source configuration. Loading
checks protocol, pinned scikit-learn version, artifact digest and metadata parity.
Only application-created local references are loaded; there is no model-upload
deserialization. Public demo storage remains visitor-scoped and temporary.

Changing Overview settings clears the active reference as before; an explicitly
saved reference can be loaded again. Loading does not recalibrate against the
new observations. Changes to a reference after examining evaluation results
make subsequent comparisons exploratory.

## Report and verification

The acquired PCA report separates `acquired/observations.csv`, received delivery
audit and sensor contributions from `historical_reference/reference_audit.csv`.
Its manifest records PCA parameters/limits, source ID, cursor, provenance,
quality/freshness rules and export time. It exports only received observations.

Tests cover numerical parity, no refitting, separate model preprocessing,
shared cursor/history across all three pages, missing/invalid/conflicted inputs,
arrival/current freshness, read failures, explicit save/load, integrity checks,
contribution sums and received-only report scope.
