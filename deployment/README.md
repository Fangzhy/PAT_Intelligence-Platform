# Deploy the synthetic portfolio app

The application entrypoint is `app.py`. Use Python **3.11**, `requirements.txt`,
and `packages.txt`. No equipment, database, credentials, or prebuilt model is
required to open the default synthetic Overview and replay screens. Models are
prepared only by explicit buttons. Missing models show guidance instead of
substitute predictions. The optional inference API is deferred.

Optional [AI result explanations](AI_EXPLANATIONS.md) use a separate OpenRouter
configuration. No key is needed for the main app or its offline template
summaries. Shared public AI is disabled unless the operator enables it.

## Verify the release locally

From the repository root in PowerShell:

```powershell
uv pip check --python .venv\Scripts\python.exe
.\.venv\Scripts\python.exe -m src.deployment_check
.\.venv\Scripts\python.exe -m pytest -q
$env:PAT_DEPLOYMENT_MODE = "public_demo"
.\.venv\Scripts\python.exe -m src.deployment_smoke
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Unset the variable with `Remove-Item Env:PAT_DEPLOYMENT_MODE` to return to local
mode. Alternatively copy `.streamlit/secrets.toml.example` to the ignored
`.streamlit/secrets.toml`. Public mode gives each browser session separate
temporary study/model/release folders, limits simulation size, and offers only
in-app acquisition sources. Default Process Monitoring remains **Replay**.
Public mode is a demonstration configuration, not authentication or a sandbox
for arbitrary code. Do not upload confidential process or patient data.

In a pip-created environment, `python -m pip check` is equivalent to the uv
check above. This checkout's uv-created environment does not include pip.

`requirements.in` documents direct ranges; `requirements.txt` pins all packages.
Linux uses the official `xgboost-cpu` distribution at the same XGBoost version;
other platforms use `xgboost`. Install one distribution, not both. The Linux
wheel-resolution check is not a substitute for executing the Linux CI job.
The GitHub workflow runs the full suite, preflight, and server health smoke on
Ubuntu and Windows. Review both jobs before publishing an update.

## Publish through Streamlit Community Cloud

1. Review `git diff` and `git status`. Commit the application, tests, workflow,
   deployment guides, requirements, and example config to your chosen release
   branch. Keep `.pat_state/`, `.venv/`, uploaded data, and secrets out of Git.
   The existing repository is
   [Fangzhy/Vaccine-PAT-Intelligence-Platform](https://github.com/Fangzhy/Vaccine-PAT-Intelligence-Platform).
   This checkout uses `dev`; Cloud must select the branch containing Phase 8.
   A local commit does not update GitHub: push the reviewed branch first.
2. Sign into [Streamlit Community Cloud](https://share.streamlit.io/) with the
   GitHub account that can access the repository, then select **Create app**.
3. Select that repository, your release branch, and **`app.py`** as the main file.
   In **Advanced settings**, explicitly select **Python 3.11**. The local
   `.python-version` alone does not configure Cloud's Python selection.
4. Paste the contents of `.streamlit/secrets.toml.example` into Secrets:

   ```toml
   [deployment]
   mode = "public_demo"
   ```

5. Deploy. Review build logs and the app preflight. Open the resulting URL in a
   signed-out/private browser to verify public access. Do not change an existing
   private app's visibility without checking its content and intended audience.
6. Follow [the demonstration guide](DEMO_GUIDE.md). Check all nine pages, missing
   model handling, explicit demo preparation, replay, CSV inference, report and
   model downloads. Verify that two sessions have separate model state.
7. Record the actual URL, deployed Git commit, Python/package versions, date,
   and smoke-test result below. Only then advertise the app as publicly live.

**Publication record:** No hosted URL is recorded by this implementation.
Local checks and a CI definition alone do not prove a successful Cloud build.

## Model release and inference

Open **Lifecycle → Deployment & Reporting**. Finalized Phase 4 and Phase 7
artifacts retain their recorded independent synthetic-batch evaluation. The
separate demo models have **grouped CV only**; exporting them does not create
independent-test evidence. Select a trusted model, inspect its card, and run the
synthetic CSV example. Input headers and units must exactly match its schema.
Extra/missing headers reject the file; invalid/ambiguous rows give gaps with
flags. Maximum 1,000 rows. This is historical inference: relative sensor age is
checked, not current wall-clock freshness. Acquired-stream monitoring performs
the separate live freshness checks.

**Create / verify release bundle** copies the fitted feature transformer,
scaler, estimator, original metadata, schema, pinned requirements, model card,
license, and available serialization parity examples into a content-addressed
release folder. The ZIP excludes training datasets and mutable studies. JSON
evaluation and versioned predictions can be downloaded separately. Original
metadata may contain training batch identifiers; review it before sharing.

For a bundle created by this app, extract to a folder retaining its
`pat-...` name and verify in the pinned environment:

```python
from src.deployment import verify_release
verify_release("path/to/pat-<release-id>", trusted=True)
```

Joblib is executable serialization. Never use this verification function on
an uploaded or unknown model. Checksums detect file corruption, not malicious
provenance. No model upload or ZIP import is exposed in the app. Keep this
source checkout with the bundle: the pipeline uses the app's Python classes;
the ZIP is not a standalone executable or an inference service. The saved
pipeline preserves fitted preprocessing; running `fit` on inference data would
break its meaning. Parity checks compare raw predictions and are distinct from
guarded operational output and accuracy evaluation.

## Maintenance and recovery

Cloud sessions and local hosted files can disappear on restart, sleep, or
redeploy. Download artifacts before leaving. Session isolation preserves each
visitor's evaluation gate during the session; it is not a durable regulated
audit trail. Session folders have no cross-process retention/cleanup guarantee.
A durable multiuser service would need authenticated storage, quotas, retention,
monitoring, and an approved model registry.

For dependency mismatch, restore the pinned environment and select Python 3.11;
do not bypass version guards. For a corrupt release, retain it for diagnosis
and select a known valid source artifact. For slow demos, use six batches,
72 hours, 71 spectral channels, and the small demo calibration button. Model
searches remain potentially expensive and explicit. To roll back, deploy a
previous tested commit together with its matching dependency file and models.
Release creation/verification also checks the saved NumPy, pandas, SciPy,
scikit-learn, and XGBoost versions before loading a release for parity.

Official references: [Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy),
[Cloud secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management),
[scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html),
[XGBoost installation](https://xgboost.readthedocs.io/en/stable/install.html).
