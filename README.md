# Vaccine-PAT-Intelligence-Platform

A learning-oriented Python application for exploring a synthetic recombinant
protein fed-batch bioreactor and, in future phases, developing PAT analytics.
**Phase 1 only is implemented.** Every measurement is synthetic. This is an
educational portfolio demonstration, not a manufacturing-validated system.

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
5. Review the tests and interview questions before approving Phase 2.

## Project layout

```text
app.py                          Streamlit Overview and plotting
src/data_generation.py          Process balances and measurement simulation
tests/test_data_generation.py   Scientific and data identity checks
tests/test_app.py               Streamlit startup and interaction checks
docs/phase_1_learning_guide.md   Equations, code walkthrough, interview practice
docs/synthetic_data_assumptions.md
.streamlit/config.toml          Theme and disabled usage telemetry
requirements.in                Direct dependencies
requirements.txt               Pinned complete environment
```

The user-authored master plan and existing Git configuration are preserved.
Future modules and pages will be added when their phase is approved; there are
no empty placeholder implementations. Data is generated in memory and cached by
configuration. No external data source, credentials, or model artifact is needed.

## Run checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Checks cover mass-balance invariants, glucose depletion, invalid inputs,
reproducibility, batch identity, UTC timestamps, uniform spectral spacing, assay
delays, spectral response to glucose, the designed disturbance, and Streamlit
startup and interactions.

Phase 1 does not perform data integration, spectral preprocessing, model
training, prediction, anomaly detection, or deployment. **Development stops here
until you explicitly request Phase 2.**
