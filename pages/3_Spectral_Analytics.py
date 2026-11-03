"""Phase 3: an interactive Raman transformation and exploratory PCA laboratory."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pywt
from scipy import fft
from scipy.signal import find_peaks
import streamlit as st

from src.app_data import load_dataset
from src.chemometrics import fit_spectral_pca, project_spectral_pca
from src.data_generation import SimulationConfig
from src.preprocessing import (
    MultiplicativeScatterCorrection, SpectralPreprocessor, estimate_baseline,
    savitzky_golay, select_region, snv, validate_axis,
)
from src.spectral_data import raman_from_integration, raman_from_simulation
from src.spectral_transformations import (
    discrete_fourier_transform, finite_difference_derivative, fourier_analysis,
    frequency_band_features, laplacian_1d, reconstruct_fourier,
    reconstruct_wavelet, wavelet_decompose, wavelet_denoise, wavelet_features,
)


RAMAN_LABEL = "Raman shift (cm⁻¹)"
FREQUENCY_LABEL = "Spatial frequency (cycles per cm⁻¹)"


def line_figure(axis, series, *, x_label=RAMAN_LABEL, y_label="Intensity (a.u.)", height=340):
    figure = go.Figure()
    for name, values in series.items():
        figure.add_trace(go.Scatter(x=axis, y=values, name=name, mode="lines"))
    figure.update_layout(xaxis_title=x_label, yaxis_title=y_label, height=height, margin=dict(l=10, r=10, t=25, b=10), legend=dict(orientation="h"))
    return figure


@st.cache_data(max_entries=8, show_spinner=False)
def spectrum_snapshot(signal, axis, reference, window_length, smoothness, asymmetry,
                      fft_window, fft_detrend, lower, upper, wavelet, level, threshold_scale):
    """Independent single-spectrum comparisons; invalid individual methods abstain."""
    transforms, errors = {}, {}

    def add(name, operation, unit="Intensity (a.u.)", x=None, x_label=RAMAN_LABEL):
        try:
            transforms[name] = (axis if x is None else x, operation(), x_label, unit)
        except ValueError as error:
            errors[name] = str(error)

    def correct_scatter():
        if not len(reference):
            raise ValueError("Select at least one reference batch for MSC.")
        return MultiplicativeScatterCorrection().fit(reference).transform(signal)

    baseline = estimate_baseline(signal, smoothness=smoothness, asymmetry=asymmetry)
    smoothed = savitzky_golay(signal, axis, window_length=window_length)
    fourier = fourier_analysis(signal, axis, window=fft_window, detrend=fft_detrend)
    filtered = reconstruct_fourier(fourier, lower, upper)
    add("Raw", lambda: signal)
    add("Baseline corrected", lambda: signal - baseline)
    add("Savitzky–Golay smoothing", lambda: smoothed)
    add("SNV", lambda: snv(signal), "SNV intensity (dimensionless)")
    add("MSC", correct_scatter)
    add("First derivative (finite difference)", lambda: finite_difference_derivative(signal, axis, 1), "dI/dν (a.u. per cm⁻¹)")
    add("Second derivative (finite difference)", lambda: finite_difference_derivative(signal, axis, 2), "d²I/dν² (a.u. per (cm⁻¹)²)")
    add("1-D Laplacian", lambda: laplacian_1d(signal, axis), "d²I/dν² (a.u. per (cm⁻¹)²)")
    add("Savitzky–Golay first derivative", lambda: savitzky_golay(signal, axis, window_length=window_length, derivative=1), "dI/dν (a.u. per cm⁻¹)")
    add("Savitzky–Golay second derivative", lambda: savitzky_golay(signal, axis, window_length=window_length, derivative=2), "d²I/dν² (a.u. per (cm⁻¹)²)")
    add("FFT magnitude", lambda: fourier.amplitude, "One-sided amplitude (a.u.)", fourier.frequencies, FREQUENCY_LABEL)
    add("Reconstructed Fourier band", lambda: filtered)
    wavelets, threshold = None, None
    if level is not None:
        wavelets = wavelet_decompose(signal, wavelet=wavelet, level=level)
        denoised, threshold = wavelet_denoise(wavelets, threshold_scale=threshold_scale)
        add("Wavelet denoised", lambda: denoised)
        coefficients = np.concatenate(wavelets.coefficients)
        add("Wavelet decomposition", lambda: coefficients, "Coefficient amplitude (a.u.)", np.arange(len(coefficients)), "Concatenated coefficient index (A then details)")
    else:
        errors["Wavelet denoised"] = errors["Wavelet decomposition"] = "Region too short for this wavelet; choose a shorter filter or a wider region."
    return transforms, errors, baseline, smoothed, fourier, wavelets, threshold


@st.cache_data(max_entries=4, show_spinner="Fitting exploratory PCA on reference batches only…")
def pca_analysis(values, axis, reference_mask, n_components, autoscale, baseline,
                 smoothness, asymmetry, scatter, window_length, derivative):
    preprocessing = SpectralPreprocessor(axis, baseline=baseline, baseline_smoothness=smoothness,
        baseline_asymmetry=asymmetry, scatter=scatter, window_length=window_length, derivative=derivative)
    pipeline = fit_spectral_pca(values[reference_mask], axis, n_components=n_components,
        autoscale=autoscale, preprocessor=preprocessing)
    scores, residuals = project_spectral_pca(pipeline, values)
    return pipeline, scores, residuals


def main():
    st.set_page_config(page_title="Vaccine PAT | Spectral Analytics", page_icon="🔬", layout="wide")
    st.caption("VACCINE PAT INTELLIGENCE PLATFORM · PHASE 3")
    st.title("Explore spectra. Understand each transformation.")
    st.write("A Raman laboratory for preprocessing, derivatives, Fourier analysis, wavelets, and reference-fitted PCA.")
    st.info("Educational exploration · No concentration predictor or validated alarm limits · Transformations can remove useful chemical information as well as nuisance variation.")
    with st.sidebar:
        st.header("Spectral data")
        source = st.radio("Raman source", ["Synthetic dataset", "Latest Phase 2 prepared Raman"], key="spectral_source")
    if source == "Synthetic dataset":
        config = st.session_state.get("simulation_config", SimulationConfig())
        collection = raman_from_simulation(load_dataset(config))
        st.sidebar.caption(f"Overview configuration: seed {config.seed}, {config.n_batches} batches, {config.duration_h:g} h.")
    else:
        result = st.session_state.get("integration_result")
        if result is None:
            st.warning("Run Data Integration first, then return here to use its latest successful prepared Raman snapshot.")
            return
        try:
            collection = raman_from_integration(result)
        except ValueError as error:
            st.error(str(error))
            return
        st.caption("Using the latest successful Phase 2 snapshot and its batch scope. Sensor/assay values are not spectral features.")
    if not collection.excluded.empty:
        st.warning(f"Excluded {len(collection.excluded)} ineligible or incomplete Raman rows. No intensities were interpolated or imputed.")
        with st.expander("Excluded Raman source rows"):
            st.dataframe(collection.excluded, hide_index=True, width="stretch")
    if len(collection.intensities) == 0:
        st.warning("No eligible, complete Raman spectra remain. Review the Phase 2 report.")
        return
    try:
        full_axis = validate_axis(collection.wavenumbers_cm1, collection.intensities.shape[1], uniform=True)
    except ValueError as error:
        st.error(f"This laboratory requires a uniform Raman axis for comparable SG/FFT methods: {error}")
        return

    batches = collection.metadata["batch_id"].unique().tolist()
    selected = st.sidebar.multiselect("Batches to analyze", batches, default=batches, key="spectral_batches")
    if not selected:
        st.warning("Select at least one batch to analyze.")
        return
    selected_rows = collection.metadata["batch_id"].isin(selected).to_numpy()
    metadata = collection.metadata.loc[selected_rows].reset_index(drop=True)
    values = collection.intensities[selected_rows].copy()
    references = st.sidebar.multiselect("Reference batches for MSC and PCA", selected, default=selected[:3], key="spectral_references")
    reference_mask = metadata["batch_id"].isin(references).to_numpy()
    st.sidebar.caption("References define the MSC mean spectrum, centering/scaling, and PCA axes. Other selected batches are projected exploration data, not an untouched predictive test set.")
    spectrum_batch = st.sidebar.selectbox("Spectrum batch", selected, key="spectral_batch")
    batch_indices = np.flatnonzero(metadata["batch_id"].eq(spectrum_batch))
    sample_number = st.sidebar.slider("Spectrum observation index", 0, len(batch_indices) - 1, 0, key="spectral_sample") if len(batch_indices) > 1 else 0
    sample_index = batch_indices[sample_number]

    spacing = float(full_axis[1] - full_axis[0])
    region = st.sidebar.slider("Raman shift range (cm⁻¹)", float(full_axis[0]), float(full_axis[-1]),
        (float(full_axis[0]), float(full_axis[-1])), step=spacing, key="spectral_region")
    try:
        values, axis = select_region(values, full_axis, *region)
    except ValueError as error:
        st.warning(str(error))
        return
    if len(axis) < 7:
        st.warning("Choose a wider spectral region with at least seven channels for the comparison laboratory.")
        return
    maximum_window = min(31, len(axis) if len(axis) % 2 else len(axis) - 1)
    windows = list(range(5, maximum_window + 1, 2))
    with st.sidebar:
        st.subheader("Transformation settings")
        window_length = st.selectbox("Savitzky–Golay window (channels)", windows, index=windows.index(min(11, maximum_window)), key="spectral_sg_window")
        st.caption("SG polynomial order = 3; window width in physical units depends on channel spacing.")
        log_lambda = st.slider("Baseline log₁₀ smoothness", 2, 8, 5, key="baseline_smoothness")
        asymmetry = st.number_input("Baseline asymmetry p", min_value=0.001, max_value=0.1, value=0.01, step=0.001, key="baseline_asymmetry")
        fft_window = st.selectbox("FFT window", ["none", "hann"], key="fft_window")
        fft_detrend = st.selectbox("FFT detrending", ["none", "constant", "linear"], key="fft_detrend")
        nyquist = float(1 / (2 * spacing))
        band = st.slider("Fourier pass band (cycles per cm⁻¹)", 0.0, nyquist, (0.0, nyquist * 0.4), step=nyquist / 100, format="%.5f", key="fft_band")
        wavelet = st.selectbox("Discrete wavelet", ["db2", "db4", "sym4", "haar"], key="spectral_wavelet")
        maximum_level = min(5, pywt.dwt_max_level(len(axis), pywt.Wavelet(wavelet).dec_len))
        level = st.selectbox("DWT levels", list(range(1, maximum_level + 1)), index=min(3, maximum_level) - 1, key="wavelet_level") if maximum_level else None
        threshold_scale = st.slider("Wavelet threshold multiplier", 0.0, 3.0, 1.0, step=0.1, key="wavelet_threshold")

    smoothness = 10.0 ** log_lambda
    signal = values[sample_index]
    transforms, errors, baseline, smoothed, fourier, wavelets, threshold = spectrum_snapshot(
        signal, axis, values[reference_mask], window_length, smoothness, asymmetry,
        fft_window, fft_detrend, *band, wavelet, level, threshold_scale,
    )
    metrics = st.columns(4)
    metrics[0].metric("Complete selected spectra", len(values))
    metrics[1].metric("Selected Raman channels", len(axis))
    metrics[2].metric("Reference spectra", int(reference_mask.sum()))
    metrics[3].metric("Spectral spacing (cm⁻¹)", f"{spacing:g}")
    sample = metadata.iloc[sample_index]
    st.caption(f"Selected spectrum: {sample['sample_id']} · {sample['elapsed_time_h']:g} h · {sample['measurement_timestamp']}")
    raw_tab, compare_tab, fft_tab, wavelet_tab, pca_tab, learn_tab = st.tabs([
        "Raw spectra", "Compare transformations", "Fourier laboratory", "Wavelet laboratory", "PCA exploration", "Learn Phase 3",
    ])

    with raw_tab:
        st.subheader("Spectral overlays by batch")
        figure = go.Figure()
        palette = px.colors.qualitative.Safe
        for batch_number, batch_id in enumerate(selected):
            indices = np.flatnonzero(metadata["batch_id"].eq(batch_id))
            for overlay_number, index in enumerate(indices[np.unique(np.linspace(0, len(indices) - 1, min(3, len(indices)), dtype=int))]):
                figure.add_trace(go.Scatter(x=axis, y=values[index], mode="lines", name=batch_id,
                    legendgroup=batch_id, showlegend=overlay_number == 0, line=dict(color=palette[batch_number % len(palette)], width=1),
                    text=f"{metadata.iloc[index]['sample_id']} · {metadata.iloc[index]['elapsed_time_h']:g} h", hovertemplate="%{text}<br>Shift: %{x}<br>Intensity: %{y}<extra></extra>"))
        figure.update_layout(xaxis_title=RAMAN_LABEL, yaxis_title="Intensity (a.u.)", height=380, legend=dict(orientation="h"))
        st.plotly_chart(figure, width="stretch", key="raw_overlay")
        st.caption("Up to three evenly spaced observations per selected batch are drawn. Statistics below use every selected complete spectrum.")
        st.subheader("Mean spectra and pooled variation")
        pooled_mean = values.mean(axis=0)
        pooled_std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(len(axis))
        summary = go.Figure()
        summary.add_trace(go.Scatter(x=axis, y=pooled_mean + pooled_std, line=dict(width=0), showlegend=False))
        summary.add_trace(go.Scatter(x=axis, y=pooled_mean - pooled_std, line=dict(width=0), fill="tonexty", fillcolor="rgba(15,118,110,0.15)", name="Pooled mean ± 1 SD"))
        for batch_id in selected:
            summary.add_trace(go.Scatter(x=axis, y=values[metadata["batch_id"].eq(batch_id)].mean(axis=0), name=f"{batch_id} mean"))
        summary.update_layout(xaxis_title=RAMAN_LABEL, yaxis_title="Intensity (a.u.)", height=340, legend=dict(orientation="h"))
        st.plotly_chart(summary, width="stretch", key="spectral_means")
        st.caption("Pooled SD includes real concentration, process-time, baseline, and batch variation. It is neither pure noise nor a confidence interval.")
        st.subheader("Peaks and estimated background in the selected spectrum")
        prominence = st.number_input("Peak prominence (a.u.)", min_value=0.0, value=0.05, step=0.01, key="peak_prominence")
        peaks, _ = find_peaks(signal, prominence=prominence)
        peak_plot = line_figure(axis, {"Raw": signal, "AsLS estimated baseline": baseline})
        peak_plot.add_trace(go.Scatter(x=axis[peaks], y=signal[peaks], mode="markers", name="Local peaks"))
        st.plotly_chart(peak_plot, width="stretch", key="spectral_peaks")
        st.caption("Local maxima are not molecular assignments. The estimated baseline can absorb broad chemical features; inspect the correction before using it.")
        st.subheader("High-frequency residual inspection")
        residual = signal - smoothed
        st.plotly_chart(line_figure(axis, {"Raw − SG smoothed": residual}, y_label="Residual intensity (a.u.)", height=250), width="stretch", key="noise_residual")
        st.caption(f"Residual SD = {residual.std(ddof=1):.5g} a.u. It can contain real narrow peaks as well as noise; it is not a validated instrument-noise estimate.")

    with compare_tab:
        choices = st.multiselect("Transformations to compare (up to four)", list(transforms) + list(errors), default=["Raw", "Baseline corrected", "SNV"], max_selections=4, key="transform_comparison")
        st.caption("Each method acts independently on the same raw selected region. Panels keep their own units and axes. PCA has a separate, explicitly ordered preprocessing chain.")
        if not choices:
            st.warning("Select at least one transformation to compare.")
        for name in choices:
            if name in errors:
                st.warning(f"{name}: {errors[name]}")
                continue
            x_values, y_values, x_label, y_label = transforms[name]
            st.plotly_chart(line_figure(x_values, {name: y_values}, x_label=x_label, y_label=y_label, height=260), width="stretch", key=f"compare_{name}")
        comparison = pd.DataFrame({"raman_shift_cm1": axis})
        for name, (x_values, y_values, x_label, _) in transforms.items():
            if x_label == RAMAN_LABEL:
                comparison[name] = y_values
        st.download_button("Download selected-spectrum transformations (CSV)", comparison.to_csv(index=False).encode("utf-8"), file_name="raman_transformations.csv", mime="text/csv")

    with fft_tab:
        st.subheader("Variation along the spectral axis")
        st.warning("These frequencies describe how intensity varies along Raman shift. They are not temporal Hz and do not convert the spectrum into molecular vibrational frequencies.")
        spectral_plot = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["One-sided amplitude (window-gain corrected)", "Mean-square contribution per frequency bin"])
        spectral_plot.add_trace(go.Scatter(x=fourier.frequencies, y=fourier.amplitude, name="Amplitude"), row=1, col=1)
        spectral_plot.add_trace(go.Scatter(x=fourier.frequencies, y=fourier.power, name="Per-bin mean square"), row=2, col=1)
        spectral_plot.update_yaxes(title_text="Amplitude (a.u.)", row=1, col=1)
        spectral_plot.update_yaxes(title_text="Mean square (a.u.²)", row=2, col=1)
        spectral_plot.update_xaxes(title_text=FREQUENCY_LABEL, row=2, col=1)
        spectral_plot.update_layout(height=530, showlegend=False)
        st.plotly_chart(spectral_plot, width="stretch", key="fft_spectrum")
        features = frequency_band_features(fourier, *band)
        st.dataframe(pd.DataFrame([features]), hide_index=True, width="stretch")
        if features["selected_bins"] == 0:
            st.warning("The selected band contains no discrete FFT bins; reconstruction is zero.")
        st.plotly_chart(line_figure(axis, {"Raw": signal, "Actually analyzed (detrend/window)": fourier.analyzed_signal, "Retained Fourier band": reconstruct_fourier(fourier, *band)}), width="stretch", key="fft_reconstruction")
        round_trip = np.max(np.abs(reconstruct_fourier(fourier) - fourier.analyzed_signal))
        st.caption(f"Nyquist = {fourier.nyquist:.6g} cycles per cm⁻¹; bin spacing = {fourier.frequencies[1]:.6g}; full-band reconstruction max error = {round_trip:.3g}. Power sums to mean(analyzed signal²), without window-power compensation.")
        st.write("Hann windowing reduces endpoint discontinuities but changes the signal. Reconstruction retains phase and returns the analyzed, possibly detrended/tapered signal. It does not undo the taper. Hard frequency cutoffs can ring; high-frequency content can include real narrow peaks. Sampling above Nyquist aliases and cannot be repaired by filtering afterward.")
        fft_table = pd.DataFrame({"frequency_cycles_per_cm1": fourier.frequencies, "real": fourier.coefficients.real, "imaginary": fourier.coefficients.imag, "amplitude_au": fourier.amplitude, "mean_square_per_bin_au2": fourier.power})
        st.download_button("Download Fourier values (CSV)", fft_table.to_csv(index=False).encode("utf-8"), file_name="raman_fourier.csv", mime="text/csv")
        with st.expander("DFT versus FFT: four-point numerical example"):
            toy = np.array([1, 0, -1, 0])
            direct = discrete_fourier_transform(toy)
            st.write("DFT and FFT compute the same coefficients. The FFT is a faster algorithm; the direct DFT is useful for small teaching examples.")
            st.dataframe(pd.DataFrame({"k": np.arange(4), "DFT real": direct.real, "DFT imaginary": direct.imag}), hide_index=True)
            st.caption(f"Input [1, 0, −1, 0] → [0, 2, 0, 2]. Max difference from scipy.fft.fft: {np.max(np.abs(direct - fft.fft(toy))):.3g}.")

    with wavelet_tab:
        st.subheader("Localized, multiscale spectral variation")
        if wavelets is None:
            st.warning(errors["Wavelet decomposition"])
        else:
            coefficient_plot = make_subplots(rows=len(wavelets.coefficients), cols=1, subplot_titles=wavelets.labels)
            for row, coefficient in enumerate(wavelets.coefficients, start=1):
                coefficient_plot.add_trace(go.Scatter(x=np.arange(len(coefficient)), y=coefficient, showlegend=False), row=row, col=1)
                coefficient_plot.update_yaxes(title_text="Coefficient (a.u.)", row=row, col=1)
                coefficient_plot.update_xaxes(title_text="Coefficient index", row=row, col=1)
            coefficient_plot.update_layout(height=180 * len(wavelets.coefficients))
            st.plotly_chart(coefficient_plot, width="stretch", key="wavelet_coefficients")
            denoised, _ = wavelet_denoise(wavelets, threshold_scale=threshold_scale)
            st.plotly_chart(line_figure(axis, {"Raw": signal, "Wavelet denoised": denoised}), width="stretch", key="wavelet_denoised")
            reconstruction_error = np.max(np.abs(reconstruct_wavelet(wavelets) - signal))
            st.caption(f"Symmetric boundary extension · Soft threshold = {threshold:.5g} a.u. · Unthresholded reconstruction max error = {reconstruction_error:.3g}.")
            st.write("A is the coarsest approximation; D bands are details from coarse to fine. Coefficient indices are not Raman shifts. Thresholding can remove genuine weak peaks. Symmetric padding changes coefficient counts and squared norms, so coefficient squared norms are not a conserved physical energy partition.")
            wavelet_table = wavelet_features(wavelets)
            st.dataframe(wavelet_table, hide_index=True, width="stretch")
            st.download_button("Download wavelet summaries (CSV)", wavelet_table.to_csv(index=False).encode("utf-8"), file_name="raman_wavelets.csv", mime="text/csv")

    with pca_tab:
        st.subheader("Fit reference batches; project the remaining batches")
        st.caption("This is exploratory PCA. Projected batches are not an independent predictive test, and largest reconstruction residuals are review candidates without calibrated alarm limits.")
        maximum_components = max(2, min(6, len(axis), int(reference_mask.sum()) - 1))
        with st.form("pca_settings"):
            components = st.selectbox("PCA components", list(range(2, maximum_components + 1)), index=0 if maximum_components == 2 else 1, key="pca_components")
            scatter = st.selectbox("PCA scatter correction", ["none", "snv", "msc"], key="pca_scatter")
            pca_baseline = st.checkbox("PCA baseline correction", key="pca_baseline")
            pca_smoothing = st.checkbox("PCA Savitzky–Golay smoothing", value=True, key="pca_smoothing")
            derivative = st.selectbox("PCA derivative order", [0, 1, 2], key="pca_derivative")
            autoscale = st.checkbox("PCA scale channels to unit variance", key="pca_autoscale")
            st.form_submit_button("Fit / update exploratory PCA", key="fit_pca", type="primary")
        st.caption("Order: selected region → optional baseline → scatter correction → optional SG smoothing/derivative → reference mean centering / optional autoscaling → PCA. Autoscaling can emphasize noisy low-variance channels.")
        if reference_mask.sum() < 3:
            st.warning("Choose reference batches containing at least three complete spectra for a two-component PCA view.")
        else:
            try:
                pipeline, scores, residuals = pca_analysis(values, axis, reference_mask, components, autoscale,
                    pca_baseline, smoothness, asymmetry, scatter, window_length if pca_smoothing else None, derivative)
            except ValueError as error:
                st.warning(f"PCA configuration needs review: {error}")
            else:
                pca = pipeline.named_steps["pca"]
                score_table = metadata.copy()
                score_table["role"] = np.where(reference_mask, "Reference", "Projected exploration")
                for index in range(scores.shape[1]):
                    score_table[f"PC{index + 1}"] = scores[:, index]
                score_table["reconstruction_rmse"] = residuals
                score_plot = px.scatter(score_table, x="PC1", y="PC2", color="batch_id", symbol="role",
                    hover_data=["sample_id", "elapsed_time_h", "scenario", "reconstruction_rmse"],
                    labels={"batch_id": "Batch", "role": "Role", "PC1": "PC1 score (PCA input units)", "PC2": "PC2 score (PCA input units)"})
                candidates = score_table.nlargest(min(5, len(score_table)), "reconstruction_rmse")
                if st.checkbox("Highlight five largest reconstruction residuals", value=True, key="pca_candidates"):
                    score_plot.add_trace(go.Scatter(x=candidates["PC1"], y=candidates["PC2"], mode="markers", marker=dict(symbol="diamond-open", size=14, color="black"), name="Largest residuals (review)", text=candidates["sample_id"]))
                score_plot.update_layout(height=430)
                st.plotly_chart(score_plot, width="stretch", key="pca_scores")
                st.dataframe(candidates[["sample_id", "batch_id", "role", "reconstruction_rmse"]], hide_index=True, width="stretch")
                variance = pd.DataFrame({"component": np.arange(1, components + 1), "explained_variance_ratio": pca.explained_variance_ratio_, "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_)})
                variance_plot = go.Figure(go.Bar(x=variance["component"], y=variance["explained_variance_ratio"], name="Per component"))
                variance_plot.add_trace(go.Scatter(x=variance["component"], y=variance["cumulative_explained_variance"], name="Cumulative"))
                variance_plot.update_layout(xaxis_title="Component", yaxis_title="Fraction of reference variance", yaxis_range=[0, 1.02], height=300)
                st.plotly_chart(variance_plot, width="stretch", key="pca_variance")
                loading_index = st.selectbox("Loading component", list(range(1, components + 1)), key="pca_loading") - 1
                st.plotly_chart(line_figure(axis, {f"PC{loading_index + 1} loading": pca.components_[loading_index]}, y_label="Loading (unit-vector coefficient)"), width="stretch", key="pca_loadings")
                st.caption("Loadings refer to the processed/scaled channels. PCA sign is arbitrary. Large scores or residuals do not prove a batch is abnormal; there are no control limits here.")
                st.dataframe(variance, hide_index=True, width="stretch")
                st.download_button("Download PCA scores (CSV)", score_table.to_csv(index=False).encode("utf-8"), file_name="pca_scores.csv", mime="text/csv")
                loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{index + 1}" for index in range(components)])
                loadings.insert(0, "raman_shift_cm1", axis)
                st.download_button("Download PCA loadings (CSV)", loadings.to_csv(index=False).encode("utf-8"), file_name="pca_loadings.csv", mime="text/csv")

    with learn_tab:
        st.latex(r"x_{SNV}=\frac{x-\bar{x}}{s_x},\qquad x_i=a_i+b_i r+e_i,\qquad x_{MSC}=\frac{x_i-a_i}{b_i}")
        st.latex(r"I''_i\approx\frac{I_{i+1}-2I_i+I_{i-1}}{(\Delta\nu)^2},\qquad X_k=\sum_{n=0}^{N-1}x_n e^{-2\pi i kn/N}")
        st.write("SNV removes each spectrum's offset and overall scale. MSC uses a fixed reference mean fitted from reference batches. Both can remove concentration-related amplitude. Derivatives suppress some smooth backgrounds but amplify noise. The 1-D Laplacian equals the second derivative; it is not the Laplace transform.")
        st.write("FFT gives global spatial-frequency content along Raman shift. Wavelets retain localized multiscale information. PCA scores locate samples in a reduced space, loadings describe its spectral directions, and explained variance describes reference-data variation—not glucose prediction accuracy.")
        st.markdown("""
1. Move through B001 and B004 spectra, inspecting baseline, means, and residuals.
2. Compare finite-difference and SG derivatives while changing the SG window.
3. Compare a full Fourier pass band with a narrow band, then enable Hann windowing.
4. Set the wavelet threshold multiplier to 0, then increase it and inspect weak peaks.
5. Fit PCA with B001–B003 as references and project B004–B006; change scaling and inspect the loadings.

Read `docs/phase_3_learning_guide.md` for numerical examples, code walkthrough, assumptions, regression checks, and three interview questions with sample answers. Predictive Modeling is available in the sidebar for training-only feature comparisons and a frozen batch holdout test.
""")
        st.markdown("Methods: [SciPy SG filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) · [SciPy FFT](https://docs.scipy.org/doc/scipy/reference/fft.html) · [PyWavelets](https://pywavelets.readthedocs.io/en/stable/ref/dwt-discrete-wavelet-transform.html) · [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html)")


if __name__ == "__main__":
    main()
