# Optional OpenRouter result explanations

The **Spectral Analytics**, **Predictive Modeling**, **Process Monitoring**,
**Multivariate Monitoring**, and **Model Reliability** pages contain an
**Explain these results · AI or local summary** panel. The existing calculations,
predictions, alarms, training workflow, frozen holdout, and original reports keep
their existing behavior. AI adds educational interpretation of an aggregate
snapshot. It never fits a model, calculates replacement metrics, opens a holdout,
fills a prediction gap, or controls equipment.

## Start locally

Synchronize the updated `requirements.txt`; `python-dotenv==1.2.4` is the only
new package. `requests` is now also declared directly at its existing pin.

```powershell
uv pip sync --python .venv\Scripts\python.exe requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Keep your real key in the ignored `.env`. `.env.example` contains empty/nonsecret
examples. Your existing setting names remain supported:

```dotenv
OPEN_ROUTER_API=your-existing-key
OPENROUTER_MODEL=dots-studio/dots-3-note-preview:free
OPENROUTER_MODEL_BACKUP=nvidia/nemotron-3.5-super-vl:free
OPENROUTER_ENABLED=true
OPENROUTER_PUBLIC_ENABLED=false
```

`OPENROUTER_API_KEY` is an alias for `OPEN_ROUTER_API`. Configuration precedence
is environment variables, then the `[llm]` Streamlit secret section, then the
project-root `.env`. Parsing `.env` does not modify global environment variables
or expand embedded environment references. The base API URL is fixed to the
official OpenRouter endpoint; no custom endpoint from an upload is accepted.

On Predictive Modeling, train/reuse a saved comparison, then open the explanation
panel. On Model Reliability, explicitly prepare/reuse a frozen demo or select a
saved study. Move the review clock to the observations you want to discuss.
Preview the outgoing summary, choose **Learning explanation** or **Technical
explanation**, and click **Explain these results**. **Use local template summary**
always works without a key or internet access. Download the separate explanation
JSON to retain the facts, explanation source, model, attempt reasons, prompt
version, snapshot fingerprint, and generation time.

## Spectral and monitoring views

On **Model Reliability → Acquired data reliability**, select an active monitoring
target and received batch/window. Click **Capture visible view for explanation**,
then explain that fixed historical snapshot. Aggregate facts cover recorded
model evidence, assessed operational pairs/errors, current eligibility counts,
baseline availability and distribution flags. No raw measurements, identities
or credentials are sent. Missing assays/baselines remain unassessed, and no
independent-test claim is added to CV-only models. Timer reruns never call an LLM;
the existing primary/backup/template fallback and external-data consent apply.

On **Spectral Analytics**, the panel explains the current region, selected
spectrum statistics, raw-minus-SG residual SD, Fourier band fraction, wavelet
threshold, and available exploratory PCA summary. It sends no raw curve,
scores, loadings, or source identifiers. Removing the reference scope leaves
PCA evidence unavailable. Explained reference variance is not prediction
accuracy; the reconstruction RMSE summary includes all selected spectra,
including reference spectra. Changing view/preprocessing/reference settings
invalidates the previous explanation. Prepared Raman sources require consent.

On **Multivariate Monitoring**, explicitly fit a reference monitoring model
first. The panel reads its existing empirical T-squared/SPE limits, retained
variance, selected chart coverage and exceedance counts. It also reads the
separately selected residual-inspection observation and its valid sensor
contributions. Chart selection and residual inspection are different scopes.
No additional fit occurs. Invalid inspection statistics stay unavailable;
unscorable chart rows are counted separately. Reference quantiles do not
guarantee a joint false-alarm probability or process acceptability.

On **Process Monitoring**, first reveal/review observations or poll an acquired
test stream. Click **Capture visible view for explanation**, then **Explain
these results**. The capture step itself sends nothing externally. It freezes
the visible summary, including input/eligibility flags, advisory bounds,
available-assay counts, and eligible displayed predictions. Stale or otherwise
withheld current cards remain unavailable in a new capture even when past
accepted predictions remain in the audit. Received sensor and spectrum cards
may concern different observations in an all-batch stream view; the summary
does not establish a causal relationship between them.

Replay, Historical review, fresh test streams, historical test streams, and
local HTTP test streams use the same manual capture behavior. HTTP-source
summaries require consent. Capture time and data cutoff are displayed and
included in the downloaded explanation, without entering the LLM prompt.
Forward progress can retain a clearly labeled historical capture; changed facts
produce a warning to capture again. Clock rewinds, reset/empty history,
source/model/rule/batch-view changes, and stream invalidation revisions clear
the capture. There are no automatic LLM calls on timer or polling reruns.
Pause replay/acquisition first if you want the live display to stay still;
explicit LLM requests can briefly occupy the Streamlit UI while awaiting a
response. This is not a real-time acquisition service guarantee.

## Fallback behavior

The application manages primary → backup → local template. It sends at most one
completion request per distinct configured model for each explicit uncached
action. OpenRouter may route that model among its own providers internally.

| Condition | Action |
|---|---|
| Primary returns a complete, acceptable structured response | Show Primary LLM |
| Model-specific rate limit, unavailability, timeout, invalid/incomplete explanation | Try the configured backup once |
| Backup also fails | Show the local template |
| Shared daily/minute quota, authentication/account restriction | Skip further models and show the template |
| Key missing, AI disabled, or imported summary lacks consent | No external request; template remains available |
| Configured model is paid, missing, or lacks structured-output support | Skip it; use the next permitted option |

Switching models cannot bypass the shared account quota. The client checks
`/key`, caches the advisory counter briefly, and refreshes it after a model rate
error. Known exhausted daily quota is held until the next UTC day. Provider
failures enter a bounded cooldown; subsequent clicks can temporarily skip that
model. Retry hints are respected. A failed template result can be retried through
the Explain button; successful explanations are reused for the identical context.

Only IDs ending in `:free` are permitted. Before inference, a cached model
catalog must confirm zero prompt/completion/request pricing and structured-output
support. Requests also set zero prompt/completion provider price ceilings and
enable no paid plugins, tools, or alternate paid model routes. The returned model
identity and reported cost are checked. These controls do not promise future
availability: free models, quotas, and provider policies may change.

Outbound requests have short connect/read timeouts, bounded response sizes, and
a 25-second scheduling budget checked before requests and between received
chunks. This is not a hard real-time guarantee; network/DNS/socket scheduling can
add overhead. No background explanation requests or automatic retry loop run.

Process-local usage limits allow at most three completion attempts per minute
and ten per UTC day per visitor, and thirty per UTC day per shared key in this
process. Failed completion attempts count. These protective counters and
cooldowns clear on process restart and are not a durable multi-instance budget
service. OpenRouter remains authoritative; other apps using the key can consume
the same quota, and its usage counter may lag.

## Data, grounding, and privacy

Python builds an allowlisted summary: target, known model/feature labels,
aggregate errors, counts, review thresholds, evidence scope, and limitations.
Comparisons include the CV winner, inspected case, and up to four leading CV
cases; the included count is stated. No raw spectra, sensor/assay rows, batch or
sample IDs, file paths, credentials, uploaded text, or simulator truth is sent.
Private source/rule/clock/study identity contributes only to the snapshot hash.
Only already-recorded independent-test evidence is included. Reliability
summaries reject observations or assay results after the review cutoff.

Prepared/integration-source summaries require a checkbox for the current
snapshot before transmission. Inspect the preview first. Summaries can still
contain sensitive aggregate results: consent is not anonymization or a guarantee
of provider confidentiality. Provider logging/training/retention policies vary.
Do not send confidential data without the appropriate authorization and policy.

Facts and units are rendered by Python, with display rounding and full numeric
values retained in the JSON. Generated prose references evidence IDs and is
required to omit numerical literals; the application rejects numerals, unknown
references, missing required limitations, malformed responses, URLs/code, and
common unsupported validation/control claims. Review checks and limitations are
selected from application-authored statements. The same checks apply to both
models. Prose is displayed as text, never executed or rendered as HTML.

These checks reduce errors; they do **not** prove every sentence is semantically
correct. Read the interpretation against the authoritative facts. The template
is explicitly labeled as locally generated rather than AI output. A historical
distribution shift does not prove sensor drift or model degradation, and
synthetic evidence does not establish manufacturing validation.

Successful outputs are cached only in the visitor's session (four entries per
panel). A changed source, saved study/selection, cutoff, scenario, rule, style,
configuration, or prompt version changes the cache identity. Previous text is
hidden until a matching summary is available. Shared runtime state contains
only bounded counters, availability metadata, and cooldowns—not prompt text,
responses, raw keys, or process datasets. Downloads omit private cache keys and
raw remote error messages.

## Community Cloud

The public demonstration defaults to template-only operation. To enable a
shared key, add this section to Streamlit's Secrets settings; never commit an
actual key:

```toml
[llm]
api_key = "your-existing-key"
model = "dots-studio/dots-3-note-preview:free"
model_backup = "nvidia/nemotron-3.5-super-vl:free"
enabled = true
public_enabled = true
```

Visitors cannot enable the operator's public-AI switch. When enabled, all
visitors share that key's quota. Keep public input synthetic unless separately
authorized, and expect a template when capacity is exhausted. For dependable
multi-instance service, introduce durable authenticated usage controls and
retention policies before wider deployment.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.deployment_check
.\.venv\Scripts\python.exe -m src.deployment_smoke
.\.venv\Scripts\python.exe -m src.llm_check
# Explicit, optional synthetic requests; consumes free-request quota:
.\.venv\Scripts\python.exe -m src.llm_check --live
# Optional checks of each new explanation scope, using synthetic facts only:
.\.venv\Scripts\python.exe -m src.llm_check --live --page spectral
.\.venv\Scripts\python.exe -m src.llm_check --live --page process
.\.venv\Scripts\python.exe -m src.llm_check --live --page multivariate
```

The CLI live check reads environment/`.env` configuration and tests each model
separately without fallback so a failed primary cannot be hidden by a successful
backup. The normal suite mocks network responses and requires no key. It covers
primary priority, free-only restrictions, duplicate-model suppression, outage
and response-validation fallback, account-wide limits, template behavior,
redaction, input/response bounds, consent, private caches, stale-result hiding,
future-data rejection, and unchanged frozen evidence. CI never runs live calls.

## Import errors after updating a running app

A long-running Streamlit process can keep an older imported explanation module
while rerunning a newly updated page. This can report a missing function even
when that function exists on disk and imports successfully in a fresh Python
process. The entrypoint and explanation pages now check for missing exports or
inconsistent explanation protocols and refresh already-loaded LLM modules in
dependency order. Ordinary reruns leave current modules unchanged. The guard
does not clear scientific/modeling modules, saved models, or session state and
does not make an API call.

Refresh the browser after pulling an update. If the old server has not picked
up the updated entrypoint/page code, stop it with Ctrl+C in its terminal and run
`.\.venv\Scripts\python.exe -m streamlit run app.py` again. Restarting clears
session-only state; local saved artifacts remain in `.pat_state/`.

Official references: [model fallback](https://openrouter.ai/docs/guides/routing/model-fallbacks),
[rate limits](https://openrouter.ai/docs/api_reference/limits),
[structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs),
[provider policies](https://openrouter.ai/docs/guides/privacy/provider-logging).

## Prediction reliability evidence

The explanation summaries can include the saved model's declared/decision
ranges, applicability-assessment status, decision-range all-row errors and
coverage, and captured applicability/range rejection counts. These are
deterministic aggregate facts. AI does not set gates, restore rejected
predictions, calibrate limits, or evaluate the independent test. Declared demo
ranges and empirical PLS screens do not establish manufacturing validation.
