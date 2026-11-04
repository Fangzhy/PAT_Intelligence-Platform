# Phase 8 interview practice

## 1. How would you move a Raman or NIR model from development to monitoring?

I would freeze the entire fitted transformation, scaling, and regression
pipeline, together with the axis, units, input schema, calibration batch IDs,
versions, and evaluation evidence. I would verify predictions after reloading
and then challenge the model on prospective representative batches. In this
app the deployment page packages that fitted pipeline, runs guarded CSV
inference, and compares reload predictions. Those are engineering checks;
synthetic data do not establish instrument or manufacturing validation.

Follow-up: Why not apply SNV or fit MSC again at runtime? Per-spectrum SNV can
be deterministic, but learned references, scaling statistics, selected features,
and latent-variable choices must retain their training meaning. I would deploy
the exact pipeline and never call `fit` on production observations.

## 2. Your model passes its checksum and parity test. Is it ready for process decisions?

No. The checksum detects accidental file changes and parity detects changed
serialization behavior. Neither establishes accuracy, source authenticity, or
acceptable process risk. I would require representative external validation,
assay traceability, input quality and freshness controls, domain monitoring,
documented intended use, and lifecycle change control. For a density-based
soft sensor I would make the same distinction between a reproducible
correlation and a validated concentration estimate over the operating range.

Follow-up: How do you treat out-of-range inputs? Preserve them in the audit,
flag the domain or quality failure, and withhold the estimate rather than
inventing a plausible value. An advisory alarm is separate from an automatic
control action.

## 3. How would you explain the deployment architecture and its next limitation?

The Streamlit interface calls pure data and analytics modules. Acquisition
adapters collect test events; quality and time alignment determine which inputs
are eligible; saved pipelines produce predictions without training during
polling. Public visitors receive separate temporary study folders. GitHub CI
defines Windows and Linux checks, and Cloud reads a pinned Python environment.
The next limitation is durability and operational validation: temporary files
are not a production registry or audit service, and simulated measurements do
not establish real process performance.

Follow-up: What would a maintained deployment need? Versioned trusted artifacts,
authenticated durable storage, quotas, logs and monitoring, rollback, controlled
dependency updates, and prospective review of drift and recalibration. Keep
retraining separate from the inference service and retain the approved baseline.

Practice in your own words: compare Raman/NIR chemometrics with density-based
soft sensors, explain grouped CV versus frozen test evidence, distinguish FFT
frequency content from derivatives and localized wavelets, then explain which
claims this particular synthetic demonstration can support.
