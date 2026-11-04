# Scientific scope and assumptions

The simulator represents a simplified recombinant-protein fed-batch process,
not a validated vaccine manufacturing process. Its mass balances, feed schedule,
growth/consumption relations, noise, Raman peaks, and late disturbance are
designed for teaching. Shared simulator structure can make prediction easier
than on real instruments or independent biological processes. Full coefficients
and units are documented in `docs/synthetic_data_assumptions.md` where available.

Raman intensities are arbitrary units on the declared axis. The app does not
simulate every instrument transfer effect, fluorescence mechanism, path length,
matrix interference, biological variant, scale-up effect, or assay uncertainty.
No automatic unit conversion or spectral interpolation is performed by the
deployment predictor. An identical column count is insufficient if the actual
axis or units differ; the operator must honor the exported input contract.

Glucose/lactate are illustrative concentration soft sensors. They are not
potency or purity assays. A correlation with process variables does not prove
a CPP/CQA relationship. Labels are offline laboratory references; result
availability and collection time must be handled separately. Simulator truth
is never a substitute for predictions in monitoring.

Grouped batch CV avoids splitting correlated observations from one batch across
training and validation. Hyperparameter selection can still make CV optimistic.
Only already-recorded frozen independent-batch metrics are exported. A new seed
within the same simulator is weaker evidence than independent process data.
Phase 5/8 demo calibration is separately seeded and labeled CV-only. Holdout
claims persist within the relevant stored cohort; deleting files or restarting
a temporary public session is not a regulated evaluation-control system.

Plausibility, timing, univariate feature-z screens, PCA diagnostics, and alarm
thresholds are advisory teaching tools. They do not guarantee measurement
accuracy, calibrated uncertainty, root cause, or process acceptability. Historical
CSV prediction checks relative sensor timing; live acquired-stream monitoring
also checks receipt and source freshness. No real instrument protocol, control
action, authenticated production acquisition, or GMP validation is claimed.

Serialization parity establishes that saved/reloaded fitted code predicts
within tolerance on recorded examples. It does not measure accuracy, regulatory
fitness, instrument transfer, or cross-platform equivalence until actually
tested on the target platform. Hashes establish integrity, not trust. Python
joblib artifacts must come from a trusted source in the matching environment.
