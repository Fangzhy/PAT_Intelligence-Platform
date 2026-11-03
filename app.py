"""Phase 1: explore synthetic bioreactor batches and learn their assumptions."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app_data import load_dataset
from src.navigation import pause_monitoring_on_navigation
from src.data_generation import SimulationConfig


SENSOR_VARIABLES = {
    "Temperature (°C)": "temperature_c",
    "pH (dimensionless)": "ph",
    "Dissolved oxygen (% air saturation)": "dissolved_oxygen_pct",
    "Feed flow (L/h)": "feed_rate_l_h",
    "Agitation (rpm)": "agitation_rpm",
}
TRUTH_VARIABLES = {
    "Glucose (g/L)": "glucose_g_l",
    "Lactate (g/L)": "lactate_g_l",
    "Viable cell density (10⁶ cells/mL)": "viable_cell_density_million_ml",
    "Culture volume (L)": "volume_l",
}
COLORS = ["#0F766E", "#2563EB", "#D97706", "#C2415C", "#7C3AED", "#4D7C0F", "#0369A1", "#A16207", "#BE185D", "#475569"]


def trend_chart(frame: pd.DataFrame, column: str, label: str, color_map: dict):
    """Use elapsed time to compare batch trajectories on the same process clock."""
    figure = px.line(
        frame, x="elapsed_time_h", y=column, color="batch_id",
        color_discrete_map=color_map,
        labels={"elapsed_time_h": "Elapsed process time (h)", column: label, "batch_id": "Batch"},
        hover_data={"measurement_timestamp": True},
    )
    figure.update_layout(height=360, margin=dict(l=10, r=10, t=15, b=10), legend=dict(orientation="h", y=1.12))
    return figure


def main() -> None:
    st.set_page_config(page_title="Vaccine PAT | Overview", page_icon="🧪", layout="wide")
    pause_monitoring_on_navigation()
    st.caption("VACCINE PAT INTELLIGENCE PLATFORM · OVERVIEW")
    st.title("Explore the process. Understand the measurements.")
    st.write("A synthetic recombinant protein fed-batch bioreactor, from process sensors to Raman spectra and laboratory references.")
    st.info("Educational synthetic data · No manufacturing validation · Overview concentrations are simulation truth or synthetic assays. Separately labeled trained-model estimates are available on Predictive Modeling and Process Monitoring.")

    saved_config = st.session_state.get("simulation_config", SimulationConfig())
    with st.sidebar:
        st.header("Simulation setup")
        with st.form("simulation_setup"):
            seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=saved_config.seed, step=1, key="seed")
            n_batches = st.slider("Number of batches", min_value=2, max_value=10, value=saved_config.n_batches, key="n_batches")
            duration = st.selectbox("Batch duration (h)", [72, 144, 240], index=[72, 144, 240].index(saved_config.duration_h), key="duration")
            interval = st.selectbox("Observation interval (h)", [0.5, 1.0, 2.0], index=[0.5, 1.0, 2.0].index(saved_config.sample_interval_h), key="interval")
            st.form_submit_button("Generate dataset", type="primary")
        st.caption("The same configuration reproduces the same dataset. Every fourth batch includes a known synthetic disturbance.")

    config = SimulationConfig(n_batches=n_batches, duration_h=duration, sample_interval_h=interval, seed=seed)
    st.session_state["simulation_config"] = config
    dataset = load_dataset(config)
    batch_ids = dataset.batches["batch_id"].tolist()
    color_map = dict(zip(batch_ids, COLORS))
    selected = st.sidebar.multiselect("Batches to explore", batch_ids, default=batch_ids, key="batch_filter")
    if not selected:
        st.warning("Select at least one batch in the sidebar to view the measurements.")
        return

    sensors = dataset.sensors[dataset.sensors["batch_id"].isin(selected)]
    truth = dataset.truth[dataset.truth["batch_id"].isin(selected)]
    assays = dataset.offline_assays[dataset.offline_assays["batch_id"].isin(selected)]
    batches = dataset.batches[dataset.batches["batch_id"].isin(selected)]
    metrics = st.columns(4)
    metrics[0].metric("Selected batches", len(selected))
    metrics[1].metric("Sensor time points", f"{len(sensors):,}")
    metrics[2].metric("Raman channels / spectrum", len(dataset.wavenumbers_cm1))
    metrics[3].metric("Offline sample pairs", len(assays))

    process_tab, raman_tab, sources_tab, learn_tab = st.tabs(["Process overview", "Raman snapshot", "Data sources & assays", "Learn Phase 1"])

    with process_tab:
        st.subheader("Direct sensor observations")
        sensor_label = st.selectbox("Process variable", list(SENSOR_VARIABLES), key="sensor_variable")
        st.plotly_chart(trend_chart(sensors, SENSOR_VARIABLES[sensor_label], sensor_label, color_map), width="stretch", key="sensor_trends")
        st.caption("These are simulated sensor observations with measurement noise and batch offsets. Feed readings may show small zero-flow noise.")

        st.subheader("Latent process states — simulation truth")
        truth_label = st.selectbox("Simulated state", list(TRUTH_VARIABLES), key="truth_variable")
        st.plotly_chart(trend_chart(truth, TRUTH_VARIABLES[truth_label], truth_label, color_map), width="stretch", key="truth_trends")
        st.caption("The simulator knows these values exactly. A real process would require reference assays or a validated measurement model to estimate them.")

        st.subheader("Batch summary")
        summary = sensors.groupby("batch_id", sort=False).agg(
            sensor_time_points=("sample_id", "size"),
            mean_temperature_c=("temperature_c", "mean"),
            mean_ph=("ph", "mean"),
            minimum_do_pct=("dissolved_oxygen_pct", "min"),
            mean_feed_l_h=("feed_rate_l_h", "mean"),
        ).reset_index().merge(batches[["batch_id", "scenario"]], on="batch_id", validate="one_to_one")
        st.dataframe(summary, hide_index=True, width="stretch")
        st.caption("Scenario labels come from simulation design. They are not anomaly detection results or quality decisions.")

    with raman_tab:
        st.subheader("A spectrum is a vector of measurements")
        st.write("Choose a batch and collection time to inspect one synthetic Raman spectrum. Peak overlap, baseline changes, and noise make concentration estimation a later modeling task.")
        raman_batch = st.selectbox("Spectrum batch", selected, key="raman_batch")
        sample_metadata = dataset.spectral_metadata[dataset.spectral_metadata["batch_id"] == raman_batch]
        sample_number = st.slider("Observation index", 0, len(sample_metadata) - 1, 0, key="raman_sample")
        sample = sample_metadata.iloc[sample_number]
        intensity = dataset.spectra.loc[sample["sample_id"]].to_numpy()
        figure = go.Figure(go.Scatter(x=dataset.wavenumbers_cm1, y=intensity, mode="lines", line=dict(color=color_map[raman_batch], width=2)))
        figure.update_layout(
            xaxis_title="Raman shift (cm⁻¹)", yaxis_title="Intensity (arbitrary units)",
            height=400, margin=dict(l=10, r=10, t=15, b=10),
        )
        st.plotly_chart(figure, width="stretch", key="raman_spectrum")
        st.caption(f"{sample['sample_id']} · {sample['elapsed_time_h']:g} h · {sample['measurement_timestamp']} · Uniform 400–1800 cm⁻¹ axis")
        st.warning("Peak locations and analyte sensitivities are illustrative. This is not an experimentally measured Raman spectrum or a source of verified molecular band assignments.")

    with sources_tab:
        st.subheader("Keep each source identifiable")
        st.dataframe(pd.DataFrame([
            {"Source": "Direct sensors", "Meaning": "Noisy temperature, pH, DO, feed, and agitation observations", "Time / key": "measurement_timestamp; sample_id; batch_id"},
            {"Source": "Latent simulation truth", "Meaning": "Internal concentrations, cell density, volume, and noiseless sensor states", "Time / key": "measurement_timestamp; sample_id; batch_id"},
            {"Source": "Raman spectra", "Meaning": "281 intensity channels by default; analyte signal + nuisance variation", "Time / key": "spectral_metadata links sample_id to each matrix row"},
            {"Source": "Offline reference assays", "Meaning": "Noisy glucose and lactate assays, approximately daily", "Time / key": "collection_timestamp differs from result_timestamp"},
            {"Source": "Model predictions", "Meaning": "Fitted soft-sensor estimates on Predictive Modeling", "Time / key": "Assay/sample ID; CV or held-out batch role"},
        ]), hide_index=True, width="stretch")
        st.subheader("Offline references: collection time versus result time")
        st.write("An assay describes the sample at collection time. Its result becomes available 2–6 simulated hours later. Open Data Integration in the sidebar to inspect time alignment, availability, and data quality.")
        st.dataframe(assays, hide_index=True, width="stretch")
        st.subheader("Batch design metadata")
        st.dataframe(batches, hide_index=True, width="stretch")
        with st.expander("Inspect raw sensor observations"):
            st.dataframe(sensors, hide_index=True, width="stretch")
        st.download_button("Download selected sensor observations (CSV)", sensors.to_csv(index=False).encode("utf-8"), file_name="synthetic_sensor_observations.csv", mime="text/csv")

    with learn_tab:
        st.subheader("What do the variables mean?")
        st.markdown("""
- **Temperature and pH:** characterize the culture environment.
- **Dissolved oxygen (DO):** oxygen availability, expressed here as percent air saturation.
- **Feed flow:** nutrient solution entering the vessel; it also changes culture volume.
- **Agitation:** mixing speed. This teaching model does not calculate oxygen transfer from impeller physics.
- **Glucose, lactate, and viable cell density:** nutrient, metabolite, and living cell concentration.

A **process variable** describes operation. A **CQA** is a product property relevant to desired quality, such as an appropriately defined purity or potency attribute. A **confirmed CPP** needs process-specific evidence that its variability affects a CQA. No CQA or CPP relationship is established here.
""")
        st.latex(r"\mu = \mu_{\max}\frac{G}{K_G+G}\left(1-\frac{X}{X_{\max}}\right)f_{O_2}")
        st.latex(r"\frac{dX}{dt}=(\mu-k_d)X-\frac{F}{V}X,\quad \frac{dG}{dt}=\frac{F}{V}(G_f-G)-q_GX,\quad \frac{dL}{dt}=q_LX-\frac{F}{V}L")
        st.write("Growth saturates with glucose and slows as cell density approaches a chosen capacity. Feed adds glucose and dilutes the culture. The solver updates amounts, then divides by the new volume, using steps of at most 0.1 h.")
        st.latex(r"I(\nu)=\sum_j a_j c_j\exp\left[-\frac{(\nu-\nu_j)^2}{2\sigma_j^2}\right]+b(\nu,t)+\epsilon")
        st.write("Raman intensity combines concentration-dependent Gaussian peaks, a changing background, batch instrument variation, and random noise. NumPy builds these arrays; pandas keeps their batch, sample, and time labels; Plotly draws interactive charts; Streamlit reruns the page when you interact.")
        st.subheader("Try three small experiments")
        st.markdown("""
1. Compare B001 with B004 near the last third of the batch. Inspect temperature, DO, and feed flow.
2. Inspect glucose alongside the feed schedule. Explain why a concentration depends on both nutrient addition and consumption.
3. Keep the seed fixed and regenerate; then change the seed. Separate reproducibility from between-batch variation.
""")
        st.caption("Full equations, a numerical example, implementation walkthrough, limitations, and interview questions are in docs/phase_1_learning_guide.md. Phase 2 is available as Data Integration in the sidebar.")
        st.markdown("Scientific context: [FDA PAT framework](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/pat-framework-innovative-pharmaceutical-development-manufacturing-and-quality-assurance) · [ICH Q8(R2)](https://database.ich.org/sites/default/files/Q8_R2_Guideline.pdf)")


if __name__ == "__main__":
    main()
