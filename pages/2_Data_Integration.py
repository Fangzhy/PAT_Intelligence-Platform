"""Phase 2: integrate identifiable sources and review data quality."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_data import load_dataset
from src.demo_data_ui import render_received_provenance
from src.navigation import pause_monitoring_on_navigation
from src.data_generation import SimulationConfig
from src.data_integration import (
    IntegrationConfig, REQUIRED_COLUMNS, SENSOR_RANGES, SourceTables,
    available_references, demonstrate_quality_issues, integrate_sources,
    read_source_csv, report_zip, source_csv_zip, sources_from_dataset,
)


@st.cache_data(show_spinner="Aligning sources and assessing data quality…", max_entries=4)
def run_integration(sources, config, ranges, selected):
    """Changing the review-time slider should not repeat parsing and alignment."""
    result = integrate_sources(sources, config, sensor_ranges=ranges, batch_ids=selected)
    return result, report_zip(result)


def main():
    st.set_page_config(page_title="PAT | Data Integration", page_icon="🔗", layout="wide")
    pause_monitoring_on_navigation()
    st.caption("PAT INTELLIGENCE PLATFORM")
    st.title("Align the sources. Inspect the quality.")
    st.write("Connect process sensors, Raman measurements, laboratory references, and batch metadata while preserving the original observations.")
    st.info("Educational workflow · No automatic interpolation or outlier removal · Range and statistical flags require review; they are not validated CPP limits or product quality decisions.")

    with st.sidebar:
        st.header("Integration inputs")
        if st.session_state.get("integration_source") == "Acquired test snapshot":
            st.session_state.pop("integration_source", None)
        sources_available = ["Synthetic dataset", "Upload CSV sources"]
        remembered_source = st.session_state.get("development_source_preference", sources_available[0])
        source_mode = st.radio("Data source", sources_available, index=sources_available.index(remembered_source) if remembered_source in sources_available else 0, key="integration_source")
        st.session_state["development_source_preference"] = source_mode
        st.caption("Historical development data for training and validation. Receive new monitoring inputs under Monitoring → Data Acquisition.")
    simulation_config = st.session_state.get("simulation_config", SimulationConfig())
    if source_mode == "Synthetic dataset":
        sources = sources_from_dataset(load_dataset(simulation_config))
        st.sidebar.caption(f"Overview configuration: seed {simulation_config.seed}, {simulation_config.n_batches} batches, {simulation_config.duration_h:g} h. Change it on the Overview page.")
        demo = st.sidebar.checkbox("Add known data-quality issues", key="quality_demo")
        if demo:
            sources = demonstrate_quality_issues(sources)
            st.warning("Faulty copy for learning: missing pH and one Raman value, DO = 145%, an invalid sensor timestamp, a duplicated sensor row, and Raman timestamps shifted by +10 minutes. The Overview dataset is unchanged.")
    else:
        st.caption("Use synthetic or appropriately licensed public data. Upload all four CSVs with the documented schema. Timestamps require an explicit timezone, such as 2025-01-06T08:00:00Z.")
        uploaded = {}
        for name in REQUIRED_COLUMNS:
            uploaded[name] = st.file_uploader(f"{name}.csv", type=["csv"], key=f"upload_{name}")
        if not all(file is not None for file in uploaded.values()):
            st.warning("Upload sensors.csv, raman.csv, assays.csv, and batches.csv to run integration.")
            templates = sources_from_dataset(load_dataset(simulation_config))
            st.download_button("Download example source CSVs (ZIP)", source_csv_zip(templates), file_name="pat_source_examples.zip", mime="application/zip")
            with st.expander("Required CSV columns", expanded=True):
                show_schema()
            return
        try:
            sources = SourceTables(**{name: read_source_csv(file.getvalue()) for name, file in uploaded.items()})
        except ValueError as error:
            st.error(f"Could not read the source files: {error}")
            return

    # Include source-only batch IDs so missing metadata can be inspected too.
    render_received_provenance(sources.batches)
    if "data_role" in sources.batches and sources.batches.data_role.eq("monitoring_demo").any():
        st.info("These are separate monitoring demo observations. Keep them separate from the historical development cohort when assessing saved-model behavior.")
    if any("batch_id" not in frame for frame in sources.as_dict().values()):
        st.error("Every source requires a batch_id column. See the CSV schema in docs/phase_2_learning_guide.md.")
        return
    batch_ids = sorted(set().union(*(set(frame["batch_id"].dropna().astype(str).str.strip()) for frame in sources.as_dict().values())) - {""})
    selected = st.sidebar.multiselect("Batches to integrate", batch_ids, default=batch_ids, key="integration_batches")
    if not selected:
        st.warning("Select at least one batch to integrate.")
        return

    with st.sidebar:
        st.subheader("Alignment rules")
        direction = st.selectbox("Sensor matching", ["backward", "nearest"], key="sensor_direction")
        st.caption("Backward uses the last observed sensor row at or before a spectrum. Nearest is retrospective and may choose a future row. Equal-distance nearest ties choose the earlier row.")
        sensor_tolerance = st.number_input("Sensor tolerance (minutes)", min_value=0.0, max_value=240.0, value=30.0, step=5.0, key="sensor_tolerance")
        assay_tolerance = st.number_input("Assay collection tolerance (minutes)", min_value=0.0, max_value=240.0, value=30.0, step=5.0, key="assay_tolerance")
        threshold = st.number_input("Potential-outlier |z*| threshold", min_value=1.0, max_value=10.0, value=3.5, step=0.5, key="outlier_threshold")

    with st.expander("Sensor plausibility ranges — editable teaching assumptions"):
        defaults = pd.DataFrame([{"sensor": column, "lower": lower, "upper": upper} for column, (lower, upper) in SENSOR_RANGES.items()])
        edited = st.data_editor(defaults, disabled=["sensor"], hide_index=True, width="stretch", key="sensor_ranges")
        st.caption("Bounds are inclusive. Range failures retain their values and receive flags. These are illustrative sensor plausibility bounds, not confirmed CPP operating limits.")
    ranges = {row.sensor: (row.lower, row.upper) for row in edited.itertuples()}
    try:
        result, report_archive = run_integration(
            sources, IntegrationConfig(sensor_tolerance, assay_tolerance, direction, threshold),
            ranges, selected,
        )
    except (ValueError, TypeError) as error:
        st.error(f"Integration needs corrected inputs: {error}")
        return

    st.session_state["integration_result"] = result
    integrated, pairs = result.integrated, result.reference_pairs
    metrics = st.columns(4)
    metrics[0].metric("Eligible spectra", len(integrated))
    metrics[1].metric("Spectra with sensor match", int(integrated["sensor_matched"].sum()))
    metrics[2].metric("Assays paired with spectrum", int(pairs["spectrum_matched"].sum()))
    flagged_rows = result.issues[["source", "row_number"]].drop_duplicates()
    metrics[3].metric("Source rows flagged for review", len(flagged_rows))
    st.caption("A match confirms time proximity, not measurement quality. Inspect completeness and range/outlier flags before using values. Missing batch-ID rows remain in the selected scope for review.")

    alignment_tab, quality_tab, raw_tab, learn_tab = st.tabs(["Aligned measurements", "Data quality report", "Raw & processed sources", "Information"])
    with alignment_tab:
        st.subheader("Raman-anchored sensor observations")
        view_columns = ["batch_id", "spectrum_sample_id", "spectrum_timestamp", "sensor_sample_id", "sensor_measurement_timestamp", "sensor_time_delta_minutes", "sensor_matched", "spectrum_complete", "sensor_complete", "sensor_range_flag", "sensor_outlier_flag"]
        st.dataframe(integrated[view_columns], hide_index=True, width="stretch")
        st.caption("Signed delta = sensor time − spectrum time. Backward matches have delta ≤ 0. Missing matches remain missing. Ambiguous or invalid key/time rows are excluded with source-row issues.")
        st.subheader("Offline assays paired at sample collection time")
        pair_columns = ["batch_id", "assay_sample_id", "collection_timestamp", "result_timestamp", "result_delay_h", "spectrum_sample_id", "collection_time_delta_minutes", "spectrum_matched", "spectrum_reused", "reference_complete", "glucose_reference_g_l", "lactate_reference_g_l"]
        st.dataframe(pairs[pair_columns], hide_index=True, width="stretch")
        st.caption("These are retrospective label pairs, not live-available concentrations. Each assay seeks one nearest spectrum in its batch; labels are not forward-filled. Reuse of a spectrum by multiple assays is explicitly flagged.")

        st.subheader("Which laboratory results were available then?")
        review_batches = result.processed.batches.loc[result.processed.batches["_eligible"], "batch_id"].tolist()
        if review_batches:
            review_batch = st.selectbox("Availability batch", review_batches, key="availability_batch")
            start = result.processed.batches.set_index("batch_id").loc[review_batch, "start_timestamp"]
            times = integrated.loc[integrated["batch_id"] == review_batch, "elapsed_time_h"]
            max_time = max(1.0, float(times.max()) + 6) if times.notna().any() else 150.0
            review_time = st.slider("Review elapsed time (h)", 0.0, max_time, 0.0, step=0.5, key="availability_time")
            cutoff = start + pd.Timedelta(hours=review_time)
            known = available_references(pairs[pairs["batch_id"] == review_batch], cutoff)
            st.caption(f"Cutoff: {cutoff}. Only assays whose result time is at or before this cutoff are shown.")
            st.dataframe(known, hide_index=True, width="stretch")
        else:
            st.warning("No unambiguous batch start time is available for the result-availability view.")

    with quality_tab:
        st.subheader("Rows retained, excluded, and flagged")
        st.dataframe(result.source_summary, hide_index=True, width="stretch")
        st.caption("Exclusion applies to matching only. All normalized rows remain in processed source tables, and original inputs remain in raw source tables. Range/outlier flags do not exclude a row.")
        st.subheader("Missing and invalid cells")
        only_missing = st.checkbox("Only show columns with missing or invalid cells", value=True, key="only_missing")
        missing = result.missing_summary
        st.dataframe(missing[missing["missing_after_parse"] > 0] if only_missing else missing, hide_index=True, width="stretch")
        st.caption("Missing raw cells and invalid parses are counted separately. NaN and non-finite numeric values are never interpolated or imputed.")
        st.subheader("Issue ledger")
        if result.issues.empty:
            st.success("No issues found by the configured checks. This does not establish that the data is correct or validated.")
        else:
            counts = result.issues.groupby(["source", "issue"]).size().reset_index(name="flags")
            chart = px.bar(counts, x="issue", y="flags", color="source", labels={"issue": "Check", "flags": "Flags", "source": "Source"})
            st.plotly_chart(chart, width="stretch")
        st.dataframe(result.issues, hide_index=True, width="stretch")
        st.caption("row_number is the 1-based data-row position in the original source, excluding the CSV header. Multiple issues can refer to one row.")
        with st.expander("Potential-outlier assessment by batch and variable"):
            st.dataframe(result.outlier_summary, hide_index=True, width="stretch")
            st.write("Modified z scores use each eligible batch's median and MAD, with at least eight finite values. Zero MAD means not evaluated. Scheduled feed uses range checks only. Real process changes can still be flagged in other variables; this is exploratory review, not a trained anomaly detector.")

    with raw_tab:
        st.subheader("Inspect the same source before and after preparation")
        source_name = st.selectbox("Source table", list(REQUIRED_COLUMNS), key="source_table")
        st.caption("Raw tables show all input batches. Processed tables reflect the selected analysis scope and retain excluded rows with _eligible=False. Uploaded original bytes remain in the upload widget; CSV exports serialize their cell contents.")
        st.write("Raw input snapshot")
        st.dataframe(result.raw.as_dict()[source_name], hide_index=True, width="stretch")
        st.write("Processed copy with quality and eligibility flags")
        st.dataframe(result.processed.as_dict()[source_name], hide_index=True, width="stretch")
        st.download_button("Download source CSVs (ZIP)", source_csv_zip(result.raw), file_name="pat_raw_sources.zip", mime="application/zip")
        with st.expander("CSV schema and import example"):
            show_schema()

    with learn_tab:
        st.subheader("Time proximity and information availability are separate questions")
        st.latex(r"|t_{sensor}-t_{spectrum}|\leq\tau,\qquad t_{sensor}\leq t_{spectrum}\text{ for backward matching}")
        st.latex(r"t_{collection}\leftrightarrow t_{spectrum},\qquad \text{reference known at }t\iff t_{result}\leq t")
        st.write("Example: a sample collected at 24 h and reported at 28 h labels a spectrum near 24 h retrospectively. At 26 h its laboratory value is unavailable. At 28 h it becomes available, but still describes the 24 h sample.")
        st.latex(r"\mathrm{MAD}=\mathrm{median}(|x_i-\tilde{x}|),\qquad z_i^*=0.6745(x_i-\tilde{x})/\mathrm{MAD}")
        st.write("For [36.8, 36.9, 37.0, 37.0, 37.0, 37.1, 37.2, 42.0], the median is 37.0 and MAD is 0.1. The score for 42.0 is 33.725, above the default review threshold of 3.5. That is a review flag, not proof of a sensor fault.")
        st.markdown("""
1. Enable **Add known data-quality issues**. Find the missing cells, duplicated row, impossible DO, and invalid timestamp in the report.
2. With the +10 minute Raman shift, compare a 5 minute and 30 minute sensor tolerance. Explain why match counts change.
3. At review time 0 h, no laboratory results are available. Move to 6 h to reveal the initial sample's results.
4. Download source CSVs, then upload all four on this page to reproduce the workflow.

Read `docs/phase_2_learning_guide.md` for the code walkthrough, assumptions, tests, and three interview questions with sample answers. Spectral Analytics can use the latest successful prepared Raman snapshot from this page.
""")
        st.markdown("Methods: [pandas time matching](https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html) · [NIST potential-outlier labeling](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm)")

    st.download_button("Download integration & quality report (ZIP)", report_archive, file_name="pat_integration_report.zip", mime="application/zip", type="primary")
    st.caption("Report includes all raw inputs, scoped processed tables, aligned measurements, assay pairs, issue/missingness summaries, and a configuration manifest. No preprocessing or prediction model is fitted in Data_Integration.")


def show_schema():
    for source, columns in REQUIRED_COLUMNS.items():
        suffix = ", raman_<wavenumber>, …" if source == "raman" else ""
        st.code(f"{source}.csv: {', '.join(columns)}{suffix}", language=None)
    st.write("Raman channel names encode increasing wavenumbers, for example raman_400, raman_405. Use at least two channels. Sensor units must match the schema names: °C, pH, % air saturation, L/h, and rpm. Concentrations are g/L. Batch scenario and description columns are optional. Other extra columns are preserved in raw inputs and omitted from processed tables.")


if __name__ == "__main__":
    main()
