# Seven-minute technical demonstration

Start locally with `streamlit run app.py` or use a verified published URL.
Default data are synthetic. In public mode use six batches, 72 hours, and
71 channels. Open pages once before an interview to warm appropriate caches.

| Time | Action | Explain |
|---|---|---|
| 0–1 min | Overview: compare B001 and disturbed B004 | Sensor observations, simulator truth, laboratory assays, and predictions have different meanings. The disturbance is intentionally simulated. |
| 1–2 min | Data Acquisition: poll once, send snapshot; inspect Data Integration & Quality | Transport acceptance is separate from scientific quality. Preserve source/receipt times, duplicates, gaps, units, and assay availability. |
| 2–3 min | Spectral Analytics: compare raw, SNV/derivative, Fourier and wavelet views | These transformations answer different questions. Learn any fitted preprocessing inside training folds; keep axes meaningful. |
| 3–4 min | Predictive Modeling: show grouped batch partition and saved evidence | Select features and latent variables using training-batch CV. Freeze the winner before opening the independent batch test once. Avoid running a large search during the presentation. |
| 4–5 min | Process Monitoring: keep Replay, explicitly prepare demo models, step a sample | Prediction uses a frozen pipeline and input quality/domain screens. Invalid readings give gaps. Demo calibration is CV-only. No automatic equipment action. |
| 5–6 min | Multivariate Monitoring and Model Reliability: explain baseline/domain/holdout views | PCA T²/Q charts describe multivariate departures; they do not identify root cause or prove a CQA. Explain why a shifted batch challenges extrapolation. |
| 6–7 min | Deployment & Reporting: prepare/reuse demo, select model, run example, create release ZIP | Show the model card, versioned CSV, retained evaluation, integrity and reload parity. Explain the difference between successful serialization and predictive validity. |

For an optional live-data demonstration, choose **Acquired stream** on Process
Monitoring, keep **Fresh in-app test stream**, explicitly prepare models, then
poll/start. Show sensor alignment, a pending spectrum, freshness, and the audit
history. Navigation pauses monitoring. Fresh stream timestamps are rebased
synthetic times; a fast producer is not a real bioreactor connection.

Failure demonstration: download the inference example, delete a required input
column, upload it, and run. The app rejects the schema. Then put a blank or
non-finite value in one numeric cell: that row becomes an invalid-input gap.
Do not replace a bad value with a label or clip a negative prediction.

If no saved model exists, the app remains usable and asks for explicit
preparation/finalization. Public temporary models may vanish after a reconnect.
Keep a downloaded report and trusted release for the discussion. Finish by
stating the next engineering step: prospective external validation, source and
unit contracts, durable provenance, and lifecycle controls before real use.
