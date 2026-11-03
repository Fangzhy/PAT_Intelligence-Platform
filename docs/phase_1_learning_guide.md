# Phase 1: build the process before building a model

The deliverable is an interactive Overview of independent synthetic bioreactor
batches. Understand where the measurements come from before trying to predict a
concentration from them. Only Phase 1 is implemented.

## 1. Understand the scientific problem

Living cells grow in a fed-batch vessel for recombinant protein production.
Fresh nutrient solution enters without a continuous harvest. Feed supplies
glucose and increases volume. Cells consume glucose and produce lactate.
Temperature, pH, dissolved oxygen, and mixing characterize the culture environment.

The simulator does not calculate recombinant protein concentration, purity, or
potency. These batches are a simplified cell culture example, not a vaccine
manufacturing recipe.

PAT uses timely measurements and process understanding to support product
quality. See the [FDA PAT framework](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/pat-framework-innovative-pharmaceutical-development-manufacturing-and-quality-assurance).
This application demonstrates scientific programming and data behavior; it
provides no evidence of manufacturing performance or product quality.

### Process variable, CPP, and CQA

- A **process variable** describes operation, such as temperature or feed flow.
- A **critical quality attribute (CQA)** is a product property that must meet
  suitable criteria to ensure desired quality. Product-specific examples can
  include defined purity or potency attributes.
- A **critical process parameter (CPP)** requires evidence that its variability
  affects a CQA. Monitoring a variable does not establish criticality.

These distinctions follow [ICH Q8(R2), glossary](https://database.ich.org/sites/default/files/Q8_R2_Guideline.pdf).
No CPP/CQA relationship is established by the synthetic generator.

## 2. Translate the process into equations

| Symbol | Python representation | Unit |
| --- | --- | --- |
| t | elapsed_time_h | h |
| V | volume_l | L |
| X | viable_cell_density_million_ml | 10^6 cells/mL |
| G | glucose_g_l | g/L |
| L | lactate_g_l | g/L |
| F | feed_rate_l_h | L/h |
| G_f | feed_glucose_g_l | g/L |
| mu, k_d | growth_rate_h, death_rate_h | 1/h |

We use a glucose-saturation relationship with a crowding term:

\[
\mu=\mu_{\max}\frac{G}{K_G+G}\max\left(0,1-\frac{X}{X_{\max}}\right)f_{O_2},
\qquad f_{O_2}=\operatorname{clip}(DO/40,0,1).
\]

The glucose factor is zero without glucose and 1/2 when G = K_G. At high glucose
it approaches one. The crowding term slows growth near a chosen cell capacity.
The DO factor is illustrative, not a calibrated oxygen-transfer or respiration
model. Temperature and pH are simulated signals; their effects on growth are not
modeled.

These relationships are appropriate for teaching because they are interpretable
and produce changing concentrations. A more elaborate model would require
measured parameters and additional process knowledge.

The continuous concentration balances are:

\[
\frac{dV}{dt}=F,\qquad
\frac{dX}{dt}=(\mu-k_d)X-\frac{F}{V}X,
\]

\[
\frac{dG}{dt}=\frac{F}{V}(G_f-G)-q_GX,\qquad
\frac{dL}{dt}=q_LX-\frac{F}{V}L.
\]

Uptake and production use:

\[
q_G=\mu/Y_{XG}+m_G,\qquad q_L=0.0015+0.025\mu.
\]

The coefficients in q_L carry units so q_L is in
(g/L)/(10^6 cells/mL)/h. Yield Y_XG is in (10^6 cells/mL)/(g/L),
and maintenance m_G has the same units as q_G.

### How Python approximates the equations

`advance_culture()` updates amounts using an explicit Euler step and divides by
the new culture volume:

\[
V_{new}=V+F\Delta t,
\]

\[
G_{available}=GV+FG_f\Delta t,\qquad U=q_GXV\Delta t,
\qquad r=\begin{cases}\min(1,G_{available}/U)&U>0\\1&U=0.\end{cases}
\]

\[
X_{new}=\frac{XV[1+(r\mu-k_d)\Delta t]}{V_{new}},\quad
G_{new}=\frac{G_{available}-rU}{V_{new}},\quad
L_{new}=\frac{LV+r q_LXV\Delta t}{V_{new}}.
\]

Scaling by r prevents consuming more glucose than is available. It also reduces
growth and lactate production consistently. This limiter does not model
maintenance priority or detailed starvation physiology. The generator takes
steps of at most 0.1 h and splits steps at feed changes. Observation spacing
controls retained points, not the maximum solver step.

### A small numerical example

Suppose V = 2 L, X = 1 million cells/mL, G = 4 g/L, and L = 0.2 g/L.
Let F = 0.002 L/h, G_f = 150 g/L, mu = 0.03/h, k_d = 0.002/h,
Y_XG = 3, m_G = 0.002, q_L = 0.002. Take a single 1 h step for arithmetic;
the generator normally takes smaller steps.

1. New volume: 2 + 0.002 = **2.002 L**.
2. Available glucose: 4 × 2 + 0.002 × 150 = **8.3 g**.
3. Uptake coefficient: 0.03 / 3 + 0.002 = **0.012**.
4. Glucose consumption: 0.012 × 1 × 2 × 1 = **0.024 g**; r = 1.
5. New glucose: (8.3 − 0.024) / 2.002 = **4.13387 g/L**.
6. New cell density: 2 × (1 + 0.03 − 0.002) / 2.002 = **1.02697 million cells/mL**.
7. New lactate: (0.2 × 2 + 0.002 × 1 × 2) / 2.002 = **0.20180 g/L**.

Glucose increased because feed addition exceeded consumption. All concentrations
also account for dilution. Reproduce this calculation in Python:

```python
from src.data_generation import CultureState, advance_culture

initial = CultureState(
    volume_l=2,
    viable_cell_density_million_ml=1,
    glucose_g_l=4,
    lactate_g_l=0.2,
)
next_state = advance_culture(
    initial, dt_h=1, feed_rate_l_h=0.002, growth_rate_h=0.03,
)
print(next_state)
```

## 3. Build synthetic Raman measurements

A spectrum is a row vector of intensities. The default axis has 281 points from
400 to 1800 cm^-1, spaced by 5 cm^-1.

\[
I(\nu,t)=a_{batch}\sum_j a_j c_j(t)
\exp\left[-\frac{(\nu-\nu_j-\delta_{batch})^2}{2\sigma_j^2}\right]
+b(\nu,t)+\epsilon(\nu,t).
\]

- Each concentration c_j multiplies a Gaussian spectral contribution.
- Peaks near 1120 and 1135 cm^-1 overlap deliberately.
- Batch gain and peak shift mimic instrument differences.
- Background b includes curvature, a broad feature, baseline drift, and an
  additional shift in disturbed batches.
- Epsilon is Gaussian measurement noise, with standard deviation 0.015.

A Gaussian with center 1120 and width 16 has value **1** at 1120 and
**exp(-1/2) ≈ 0.6065** at 1136. A one g/L glucose increase contributes 0.07
intensity units from that peak at its exact center, before other contributions.

Peak positions are arbitrary teaching choices, not experimental band
assignments. Concentration affects spectra through the intended simulated
measurement mechanism. It is not copied into the sensor table or spectral
feature columns. Future models must still exclude truth, identifiers, times,
and scenario labels unless their use is scientifically justified.

## 4. Read the code in this order

### `src/data_generation.py`

1. **`SimulationConfig`** is a dataclass: a named collection of settings. It
   rejects invalid grids, negative settings, and too few spectral channels.
   `frozen=True` prevents reassignment and allows use as a cache key.
2. **`CultureState`** names the four state variables rather than relying on
   unexplained array positions.
3. **`advance_culture()`** performs one mass-balance step. Keyword arguments make
   units and assumptions easier to check at each call site.
4. **`_simulate_process()`** runs the time loop. A leading underscore identifies
   an internal helper rather than the intended public API.
5. **`_observe_sensors()`** creates a separate noisy observation table while
   preserving truth.
6. **`gaussian_peak()`** creates one peak; **`simulate_raman_spectra()`** combines
   peaks and nuisance variation using NumPy arrays.
7. **`_sample_offline_assays()`** samples roughly daily and records separate
   collection and result timestamps, with reference measurement error.
8. **`generate_dataset()`** orchestrates these components and returns a
   `SyntheticDataset` containing the source tables and spectral axis.

### NumPy: arrays and broadcasting

```python
concentrations = truth["glucose_g_l"].to_numpy()  # (n_samples,)
peak = gaussian_peak(axis, 1120, 16)            # (n_channels,)
signal = concentrations[:, None] * peak        # (n_samples, n_channels)
```

`[:, None]` turns a vector into a column. NumPy broadcasts this column across
the peak vector, building one spectrum per time point without a Python loop over
each spectral channel.

`np.random.default_rng()` creates an explicit random-number generator.
`SeedSequence([seed, batch_index]).spawn(4)` separates process, sensor, spectral,
and assay randomness. Repeating a configuration reproduces every source; adding
batches leaves earlier batches unchanged. Changing the observation grid changes
noise draws and may cause small numerical integration differences.

### pandas: labels and source identity

`DataFrame` is a labeled table. `batch_id` identifies a culture; `sample_id`
identifies a measurement occasion. Timestamps are UTC and elapsed time is hours.
`spectra.loc[sample_id]` retrieves one spectrum. Its numerical matrix has no
concentration or metadata columns.

| Object | Rows / entries | Contents |
| --- | --- | --- |
| sensors | One per observation | Noisy direct sensor readings |
| truth | One per observation | Internal states and noiseless signals |
| spectra | One per spectrum, indexed by sample_id | Numeric Raman channels |
| spectral_metadata | One per spectrum | Batch, sample, and time |
| offline_assays | One per laboratory sample | References and two timestamps |
| batches | One per batch | Start time and designed scenario |
| wavenumbers_cm1 | One value per channel | Axis matching spectral columns |

At defaults, each batch has 144 / 1 + 1 = **145** observations. Six batches have
**870** spectra with **281** channels each and **42** offline glucose/lactate
sample pairs. In the executed default simulation, final cell density spans about
**21.6–25.5 million cells/mL** and final glucose **12.2–15.2 g/L**. These are
outputs of the chosen assumptions, not experimentally verified process ranges.

### `pages/1_Overview.py`: reruns and caching

Streamlit executes the script when you interact with a widget. A form groups
simulation settings until **Generate dataset** is clicked. `@st.cache_data`
stores up to four configurations so chart selection does not rerun the solver.
Plotly supplies hover details and chart navigation. Dashboard filters do not
overwrite source measurements.

The original Overview code now lives in `pages/1_Overview.py`. Start the app
with the same `streamlit run app.py` command: `app.py` registers grouped pages
and runs the selected screen. Dataset caching remains in `src/app_data.py`.

## 5. Try the dashboard and interpret the results

Run the README command, keep the defaults, and complete these checkpoints:

1. **Explain one trajectory.** Inspect glucose and feed flow. Explain how
   addition, consumption, growth, and dilution affect concentration.
2. **Compare batches.** Select B001 and B004. B004's disturbance begins at 65% of
   its duration and reaches full strength at 75%. Check feed, temperature, pH,
   and DO. Separate imposed changes from effects of the growth equations.
3. **Inspect measurement ambiguity.** Move through Raman snapshots. Explain why
   a higher baseline alone cannot imply higher glucose.
4. **Check provenance.** Identify a sensor observation, latent truth, and an
   assay. Explain why collection and result times must both be retained.
5. **Check reproducibility.** Generate twice with seed 42, then try 43. The first
   two runs match; the third changes the realization.

Scenario labels describe generation. No statistical detector has assessed them.
No dashboard line is a trained-model prediction.

## 6. Verify the computations

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The initial executed suite passed **38 tests**. It checks known-answer dilution
and yield examples, glucose depletion, invalid inputs, repeatability, batch
identity, source separation, UTC timestamps, assay delays, spectral spacing,
Gaussian values, intended glucose response, and the designed disturbance.
A coarse/fine observation comparison checks numerical agreement at common times.

Streamlit checks use its official
[AppTest interface](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
to execute the page, change selections, move the spectral slider, regenerate
with new settings, and handle an empty batch selection. These functional tests
do not establish scientific validation or replace visual inspection.

The local Streamlit server also started successfully, and its root page and
health endpoint both returned HTTP 200. Browser preview infrastructure was
unavailable during development, so visual layout inspection remains outstanding.

## 7. Limits to explain clearly

- Parameters, peaks, and noise are assumptions without cell-line or instrument
  calibration.
- Protein production, viability decline, oxygen-transfer physics, other
  nutrients, harvest, and most metabolic behavior are omitted. Lactate is
  produced and diluted; a metabolic consumption switch is not modeled.
- Noise is mainly independent Gaussian noise with simple batch offsets. Real
  data can have correlated noise, saturation, missingness, and faults.
- Data cleaning, timestamp joining, preprocessing, PCA, PLS, and predictions are
  later phases.
- Time points within a batch are correlated. Future validation must hold out
  independent batches, with the same holdout retained across model comparisons.
- Synthetic success cannot establish manufacturing validation, CQA prediction,
  process safety, or validated CPP limits.

See [synthetic_data_assumptions.md](synthetic_data_assumptions.md) for coefficients.

## 8. Three interview questions and sample answers

**1. How would you distinguish a process variable from a CPP and a CQA?**

“A process variable describes operation, such as feed flow or temperature. A CQA
is a product property that needs to meet appropriate quality criteria. Calling
temperature a CPP requires process-specific evidence that temperature variation
affects a CQA. Here I monitor temperature, but I cannot establish criticality
because I have no product quality labels or supporting studies.”

**2. Why keep truth, sensors, spectra, and reference assays separate?**

“They represent different information sources. The simulator knows the true
concentration, sensors have measurement noise, Raman is an indirect multivariate
measurement, and assays have error and reporting delay. Keeping sources separate
makes provenance clear and helps prevent target leakage. I would apply the same
discipline to an NIR or density-based soft sensor.”

**3. How would you assess whether a Raman glucose soft sensor generalizes?**

“I would align spectra and assays by collection time while respecting when
results become available. I would hold out independent batches and use
batch-aware cross-validation within training data. Preprocessing and model
selection would use training data only. I would compare models on the same
untouched holdout and inspect errors by batch. Synthetic results demonstrate
the workflow, not manufacturing validation.”

## 9. Phase boundary

Phase 1 supplies the environment, simulator, Overview, learning materials, and
tests. Phase 2 will add multisource alignment, duplicate and missing-data checks,
sensor range assessment, and data quality reporting. It will preserve raw data
and will not automatically interpolate missing values. **Do not start Phase 2
until the user explicitly requests it.**
