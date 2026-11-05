"""Phase 4: training-only model comparison, then a frozen independent batch test."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app_data import load_dataset
from src.navigation import pause_monitoring_on_navigation
from src.data_generation import SimulationConfig
from src.data_integration import integrate_sources, sources_from_dataset
from src.feature_engineering import FEATURE_SETS
from src.modeling import DEFAULT_CASES, MODELS, ModelCase, TrainingConfig, pls_diagnostics, train_study
from src.prediction_reliability_ui import render_study_reliability
from src.modeling_data import TARGETS, fixed_batch_partition, modeling_dataset
from src.modeling_storage import (
    DEFAULT_STATE_ROOT, cv_predictions, evaluate_once, holdout_is_sealed,
    load_holdout, load_study, save_study, stored_partition, study_report_zip,
)
from src.validation import regression_metrics
from src.llm_bootstrap import refresh_legacy_llm_modules

refresh_legacy_llm_modules()
from src.llm_results import modeling_snapshot
from src.llm_ui import render_explanation
from src.ui_labels import workflow_label


@st.cache_data(max_entries=4, show_spinner="Preparing the common assay-labeled modeling cohort…")
def synthetic_modeling_data(config, target):
    return modeling_dataset(integrate_sources(sources_from_dataset(load_dataset(config))), target)


def experiment_label(case):
    return f"{MODELS[case.model]} · {FEATURE_SETS[case.feature_set]}"


def prediction_plot(rows, title):
    figure = px.scatter(rows, x="reference_g_l", y="prediction_g_l", color="batch_id",
        hover_data=["assay_sample_id", "elapsed_time_h"], title=title,
        labels={"reference_g_l": "Offline reference (g/L)", "prediction_g_l": "Model prediction (g/L)", "batch_id": "Batch"})
    limits = [float(min(rows.reference_g_l.min(), rows.prediction_g_l.min())), float(max(rows.reference_g_l.max(), rows.prediction_g_l.max()))]
    figure.add_trace(go.Scatter(x=limits, y=limits, mode="lines", line=dict(dash="dash", color="gray"), name="Identity"))
    figure.update_layout(height=390)
    return figure


def residual_plot(rows, title):
    figure = px.scatter(rows, x="elapsed_time_h", y="residual_g_l", color="batch_id",
        hover_data=["assay_sample_id", "reference_g_l", "prediction_g_l"], title=title,
        labels={"elapsed_time_h": "Sample collection elapsed time (h)", "residual_g_l": "Prediction − reference (g/L)", "batch_id": "Batch"})
    figure.add_hline(y=0, line_dash="dash")
    figure.update_layout(height=330)
    return figure


def main():
    st.set_page_config(page_title="Vaccine PAT | Predictive Modeling", page_icon="📈", layout="wide")
    pause_monitoring_on_navigation()
    st.caption("VACCINE PAT INTELLIGENCE PLATFORM")
    st.title("Compare soft sensors. Test independent batches.")
    st.write("Train from offline glucose or lactate assays, compare fixed spectral feature recipes, and preserve whole-batch separation.")
    st.info("Synthetic/public-data learning workflow · Assay references are labels; simulated truth is not a model input · No potency, purity, or manufacturing-validation claim.")
    with st.sidebar:
        st.header("Modeling cohort")
        source = st.radio("Modeling data source", ["Synthetic modeling cohort", "Latest Phase 2 prepared sources"], format_func=workflow_label, key="model_source")
        target = st.selectbox("Prediction target", list(TARGETS), key="model_target")
    if source == "Synthetic modeling cohort":
        overview = st.session_state.get("simulation_config", SimulationConfig())
        config = replace(overview, n_batches=max(8, overview.n_batches))
        dataset = synthetic_modeling_data(config, target)
        st.sidebar.caption(f"Seed {config.seed} · {config.n_batches} batches · {config.duration_h:g} h. The default six-batch Overview is extended to eight, preserving its earlier batches. Overview settings are unchanged.")
    else:
        integration = st.session_state.get("integration_result")
        if integration is None:
            st.warning("Run Data Integration first to supply prepared spectra, sensors, references, and batch metadata.")
            return
        try:
            dataset = modeling_dataset(integration, target)
        except ValueError as error:
            st.error(str(error))
            return
        st.caption("Using the latest successful Data_Integration snapshot. Previously inspected batches are not automatically a blind test; whole-batch separation alone cannot undo prior data inspection.")
    if not dataset.excluded.empty:
        st.warning(f"Excluded {len(dataset.excluded)} assay rows from the common cohort. All experiments use the same remaining rows, including sensor-only and spectral-only cases.")
        with st.expander("Modeling input exclusions"):
            st.dataframe(dataset.excluded, hide_index=True, width="stretch")
    state_root = Path(st.session_state.get("modeling_state_root", DEFAULT_STATE_ROOT))
    try:
        partition = stored_partition(state_root, dataset.cohort_id) or fixed_batch_partition(dataset.metadata.batch_id)
        training, held_out = partition.masks(dataset.metadata)
        sealed = holdout_is_sealed(state_root, dataset.cohort_id)
        study = load_study(state_root, dataset.cohort_id)
    except (ValueError, OSError) as error:
        st.error(f"Modeling setup needs review: {error}")
        return
    metrics = st.columns(4)
    metrics[0].metric("Labeled assay rows", len(dataset.X))
    metrics[1].metric("Training assay rows", int(training.sum()))
    metrics[2].metric("Held-out assay rows", int(held_out.sum()))
    metrics[3].metric("Training batches", len(partition.training_batches))
    st.caption(f"Cohort {dataset.cohort_id[:12]} · Target: {target} (g/L) · One reference pair per accepted assay sample; no target interpolation.")
    assignments = dataset.metadata.groupby("batch_id").size().reset_index(name="assay_rows")
    assignments["role"] = np.where(assignments.batch_id.isin(partition.training_batches), "Training / CV", "Held-out final test")
    with st.expander("Fixed batch assignments and input schema", expanded=True):
        st.dataframe(assignments, hide_index=True, width="stretch")
        st.caption("The same row cohort, training batches, held-out batches, and CV folds are used across every feature/model comparison. Features contain only Raman intensities and five declared sensors; IDs, times, assay results, and scenario labels are excluded.")
    matching_study = study is not None and study.target == target
    if study is not None and not matching_study:
        st.warning(f"The saved study predicts {study.target}. Select that target to view it. A new target can replace the study only before this cohort's holdout is revealed.")
    catalog = {experiment_label(ModelCase(feature, model)): ModelCase(feature, model) for feature in FEATURE_SETS for model in MODELS}
    defaults = [experiment_label(case) for case in (study.cases if matching_study else DEFAULT_CASES)]
    existing = study.config if matching_study else TrainingConfig()
    with st.form("model_training_plan"):
        experiments = st.multiselect("Feature/model experiments", list(catalog), default=defaults, disabled=sealed, key="model_experiments")
        splits = list(range(2, len(partition.training_batches) + 1))
        folds = st.selectbox("Training-only GroupKFold splits", splits, index=splits.index(min(existing.n_splits, len(partition.training_batches))), disabled=sealed, key="model_folds")
        max_components = st.slider("Maximum PLS latent variables to compare", 1, 12, min(12, existing.max_pls_components), disabled=sealed, key="model_components")
        trees = st.selectbox("Trees per forest / boosting fit", [40, 80, 160], index=[40, 80, 160].index(existing.tree_count) if existing.tree_count in (40, 80, 160) else 1, disabled=sealed, key="model_trees")
        custom_range = st.checkbox("Declare concentration and decision ranges", value=existing.declared_range is not None or existing.decision_range is not None, disabled=sealed, key="model_custom_range")
        training_y = dataset.y.loc[training]
        supported = (float(training_y.min()), float(training_y.max()))
        declared_defaults = existing.declared_range or supported
        decision_defaults = existing.decision_range or declared_defaults
        st.caption(f"Training-reference extent: {supported[0]:.4g}–{supported[1]:.4g} g/L. Default policy uses this provisional extent. Custom ranges are saved before test disclosure and cannot extend it.")
        bounds = st.columns(4)
        range_low = bounds[0].number_input("Declared lower (g/L)", min_value=0.0, value=declared_defaults[0], disabled=sealed, key=f"model_range_low_{target}")
        range_high = bounds[1].number_input("Declared upper (g/L)", min_value=0.0, value=declared_defaults[1], disabled=sealed, key=f"model_range_high_{target}")
        decision_low = bounds[2].number_input("Decision lower (g/L)", min_value=0.0, value=decision_defaults[0], disabled=sealed, key=f"model_decision_low_{target}")
        decision_high = bounds[3].number_input("Decision upper (g/L)", min_value=0.0, value=decision_defaults[1], disabled=sealed, key=f"model_decision_high_{target}")
        train_clicked = st.form_submit_button("Train and compare (CV only)", type="primary", disabled=sealed, key="train_models")
    st.caption("PLS compares valid component counts from 1 to the chosen cap. RF and XGBoost compare compact fixed grids; Linear Regression is an untuned baseline. Features and scalers are fitted separately inside each training fold. No early stopping uses the holdout.")
    if sealed:
        st.info("This cohort's holdout is sealed: the saved comparison is frozen. Revisits show stored results. New experiments require a genuinely new cohort rather than retuning against revealed test scores.")
    if train_clicked:
        if holdout_is_sealed(state_root, dataset.cohort_id):
            st.error("Holdout already revealed; this comparison cannot be retrained.")
        else:
            progress = st.progress(0.0, text="Preparing training-only comparisons…")
            try:
                config = TrainingConfig(folds, max_components, trees,
                    declared_range=(range_low, range_high) if custom_range else None,
                    decision_range=(decision_low, decision_high) if custom_range else None)
                study = train_study(dataset, partition, [catalog[label] for label in experiments], config,
                    progress=lambda index, total, case: progress.progress(index / total, text=f"Training {index + 1}/{total}: {case}"))
                save_study(study, state_root)
                matching_study = True
                progress.progress(1.0, text="Training-only comparison saved. Held-out labels remain unevaluated.")
                st.success("CV study saved. Review errors, feature stability, and robustness before freezing the final comparison.")
            except (ValueError, OSError) as error:
                st.error(f"Training could not complete: {error}")
    active = study if matching_study else None
    evaluation = load_holdout(state_root, dataset.cohort_id) if active is not None else None
    cv_tab, predictions_tab, interpretation_tab, sensitivity_tab, test_tab, learn_tab = st.tabs([
        "Cross-validation comparison", "Predictions & residuals", "PLS & feature stability",
        "Noise & drift sensitivity", "Independent batch test", "Information",
    ])
    if active is not None:
        case_ids = active.comparison.case_id.tolist()
        case_id = st.selectbox("Experiment to inspect", case_ids,
            format_func=lambda value: experiment_label(active.results[value].case), key="inspect_model_case")
        selected = active.results[case_id]
        all_cv = cv_predictions(active)
        rows = all_cv[all_cv.case_id == case_id]
        st.caption(f"Saved study {active.study_id[:12]} · CV-selected winner: {experiment_label(active.results[active.winner_id].case)} · Current form edits do not change the saved results until training is submitted.")
        render_study_reliability(active, evaluation, case_id=case_id, key="model_prediction_reliability")

    with cv_tab:
        if active is None:
            st.write("Choose experiments and click Train and compare. No model training occurs during ordinary page rendering.")
        else:
            st.subheader("Compare pooled out-of-fold errors")
            st.dataframe(active.comparison, hide_index=True, width="stretch")
            chart = px.bar(active.comparison, x="case_id", y="RMSECV", color="model", labels={"case_id": "Experiment", "RMSECV": f"{target} RMSECV (g/L)"})
            st.plotly_chart(chart, width="stretch", key="cv_comparison_plot")
            st.caption("RMSECV is pooled over one OOF prediction per training assay row. These folds also selected settings, so this is a tuning estimate with selection optimism. RMSE_training is a resubstitution error, not independent performance. R² can be negative and is undefined for constant references.")
            if not active.failures.empty:
                st.warning("Some declared experiments failed and are excluded from ranking; their errors remain in the report.")
                st.dataframe(active.failures, hide_index=True, width="stretch")
            with st.expander("Audit the shared CV folds"):
                st.dataframe(active.folds, hide_index=True, width="stretch")
            with st.expander("Candidate search for the inspected experiment"):
                st.dataframe(selected.tuning, hide_index=True, width="stretch")

    with predictions_tab:
        if active is None:
            st.caption("Out-of-fold predictions appear after a training-only comparison.")
        else:
            st.plotly_chart(prediction_plot(rows, "Training batches: out-of-fold predictions"), width="stretch", key="cv_prediction_plot")
            st.plotly_chart(residual_plot(rows, "Out-of-fold residuals across process time"), width="stretch", key="cv_residual_plot")
            st.dataframe(rows[["batch_id", "assay_sample_id", "elapsed_time_h", "reference_g_l", "prediction_g_l", "residual_g_l"]], hide_index=True, width="stretch")
            batch_rows = [{"batch_id": batch_id, **regression_metrics(group.reference_g_l, group.prediction_g_l)} for batch_id, group in rows.groupby("batch_id")]
            st.dataframe(pd.DataFrame(batch_rows).rename(columns={"RMSE": "OOF RMSE"}), hide_index=True, width="stretch")

    with interpretation_tab:
        if active is None:
            st.caption("PLS latent variables and fold feature stability appear after training.")
        else:
            st.subheader("Feature stability across training CV folds")
            important = selected.feature_stability.nlargest(20, "mean_normalized_importance")
            st.plotly_chart(px.bar(important, x="feature", y="mean_normalized_importance", error_y="sd_normalized_importance",
                labels={"mean_normalized_importance": "Mean normalized importance", "feature": "Engineered feature"}), width="stretch", key="feature_stability_plot")
            st.caption("Linear/PLS importance is normalized absolute coefficient in standardized feature coordinates. Tree importance is normalized split-based importance. Fold SD describes instability, not a confidence interval or causal effect. These definitions should not be compared as one universal importance scale across models.")
            pls_cases = [value for value in case_ids if active.results[value].case.model == "pls"]
            if pls_cases:
                pls_id = st.selectbox("PLS experiment", pls_cases, key="pls_case")
                pls_result = active.results[pls_id]
                tuning = pls_result.tuning.copy()
                tuning["latent_variables"] = tuning.parameters.map(lambda value: json.loads(value)["n_components"])
                st.plotly_chart(px.line(tuning, x="latent_variables", y="RMSECV", markers=True,
                    labels={"latent_variables": "PLS latent variables", "RMSECV": "Training-only RMSECV (g/L)"}), width="stretch", key="pls_selection_plot")
                scores, loadings, coefficients = pls_diagnostics(pls_result, active.training_metadata)
                if "LV2" in scores:
                    score_plot = px.scatter(scores, x="LV1", y="LV2", color="batch_id", hover_data=["assay_sample_id"])
                else:
                    score_plot = px.scatter(scores, x="elapsed_time_h", y="LV1", color="batch_id", hover_data=["assay_sample_id"])
                    st.caption("The selected model has one latent variable, so LV1 is shown against collection time.")
                st.plotly_chart(score_plot, width="stretch", key="pls_scores_plot")
                latent = st.selectbox("PLS loading", loadings.columns.tolist(), key="pls_loading")
                loading_table = loadings.reset_index()
                loading_table["feature_index"] = np.arange(len(loading_table))
                st.plotly_chart(px.line(loading_table, x="feature_index", y=latent, hover_data=["feature"],
                    labels={"feature_index": "Engineered feature index (ordered)", latent: f"{latent} loading"}), width="stretch", key="pls_loadings_plot")
                top_coefficients = coefficients.loc[coefficients.coefficient_per_standardized_feature.abs().nlargest(20).index]
                st.plotly_chart(px.bar(top_coefficients, x="feature", y="coefficient_per_standardized_feature"), width="stretch", key="pls_coefficients_plot")
                st.caption("PLS directions seek X–y covariance. Extra latent variables can fit noise; the selected count comes from whole-batch CV. Scores/loadings shown here are the final training fit, not held-out diagnostics. Coefficients refer to engineered standardized features, not automatically to molecular assignments.")
            else:
                st.caption("Include a PLS experiment to inspect latent-variable selection, scores, loadings, and coefficients.")

    with sensitivity_tab:
        if active is None:
            st.caption("Prespecified perturbations are evaluated on OOF validation inputs after training.")
        else:
            st.subheader("Training-CV sensitivity of the inspected experiment")
            st.dataframe(selected.robustness, hide_index=True, width="stretch")
            st.plotly_chart(px.bar(selected.robustness, x="protocol", y="RMSE", labels={"RMSE": "OOF RMSE (g/L)"}), width="stretch", key="noise_sensitivity_plot")
            st.caption("The same deterministic spectral perturbations are used across experiments: Gaussian intensity noise and a constant baseline offset. Sensors and labels are unchanged. A sensor-only model is therefore unaffected. These limited synthetic challenges do not establish broad robustness, and these scores never choose test settings.")

    with test_tab:
        if active is None:
            st.write("Complete a training-only comparison before evaluating held-out batches.")
        else:
            st.write(f"Frozen CV winner: **{experiment_label(active.results[active.winner_id].case)}**. Final batches: **{', '.join(active.partition.holdout_batches)}**. The final step evaluates all successful declared experiments on the same test rows and retains the CV-selected winner.")
            if evaluation is None:
                st.caption("This step reveals held-out scores and permanently seals further tuning for this local cohort. Repeated visits reuse stored results. No test metric has been calculated by the training step.")
                if st.button("Freeze comparison and evaluate held-out batches once", key="evaluate_holdout", disabled=sealed, type="primary"):
                    try:
                        with st.spinner("Scoring the frozen experiment set…"):
                            evaluate_once(active, dataset, state_root)
                        st.rerun()
                    except (ValueError, OSError) as error:
                        st.error(f"Final evaluation could not complete: {error}")
                if sealed:
                    st.warning("An evaluation claim exists without a completed result. Inspect the local audit record; automatic retuning or reevaluation is disabled.")
            else:
                merged = active.comparison.merge(evaluation["metrics"], on="case_id", validate="one_to_one")
                st.subheader("Independent-batch performance — original CV ranking retained")
                st.dataframe(merged, hide_index=True, width="stretch")
                comparison = merged.melt(id_vars=["case_id"], value_vars=["RMSECV", "RMSEP"], var_name="partition", value_name="RMSE_g_l")
                st.plotly_chart(px.bar(comparison, x="case_id", y="RMSE_g_l", color="partition", barmode="group", labels={"RMSE_g_l": "RMSE (g/L)"}), width="stretch", key="independent_comparison_plot")
                final_rows = evaluation["predictions"].loc[lambda frame: frame.case_id == case_id]
                st.plotly_chart(prediction_plot(final_rows, "Held-out batches: model prediction versus assay"), width="stretch", key="test_prediction_plot")
                st.plotly_chart(residual_plot(final_rows, "Held-out residuals across process time"), width="stretch", key="test_residual_plot")
                st.dataframe(evaluation["batch_metrics"], hide_index=True, width="stretch")
                st.success("Frozen evaluation saved with the CV-selected fitted pipeline and model metadata. Test scores did not change the selected model.")

    with learn_tab:
        st.latex(r"RMSE=\sqrt{\frac{1}{n}\sum_i(\hat y_i-y_i)^2},\qquad Bias=\frac{1}{n}\sum_i(\hat y_i-y_i)")
        st.write("RMSECV uses out-of-fold training predictions; RMSEP uses the independent held-out batches. MAE is average absolute error. R² compares squared error with reference variance and may be negative. Bias is signed prediction minus reference.")
        st.write("For references [1, 2, 3] and predictions [2, 2, 2], errors are [1, 0, −1]: RMSE = √(2/3) = 0.8165 g/L, MAE = 0.6667 g/L, bias = 0, and R² = 0. Zero bias does not imply low error.")
        st.markdown("""
1. Start with raw-spectra PLS, then compare preprocessing and derivatives using training CV.
2. Inspect latent-variable selection: lower training error alone does not justify more components.
3. Compare FFT/wavelet features and tree models with the conventional spectral baseline.
4. Review fold stability and prespecified noise/baseline sensitivity before freezing the plan.
5. Evaluate the fixed holdout once, inspect batch/time errors, and explain whether differences are practically meaningful.

References remain sparse offline assays. No IDs, timestamps, latent glucose/lactate, or batch scenario labels enter predictors. Common-row filtering avoids giving different feature sets different evaluation samples. Projected PCA exploration in Spectral_Analytics is separate from predictive validation here.

The local evaluation record discourages repeated test-driven tuning; it is not an access-control system and cannot guarantee that data was never viewed elsewhere. Use a genuinely new cohort for further final-validation claims. Read `docs/phase_4_learning_guide.md` for the architecture, equations, examples, limitations, tests, and three interview questions with sample answers. Process Monitoring can replay finalized saved pipelines or separately prepared demo calibration models without modifying this cohort's test record.
""")
        st.markdown("Methods: [PLSRegression](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html) · [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) · [XGBoost sklearn interface](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html)")

    if active is not None:
        st.download_button("Download modeling study report (ZIP)", study_report_zip(active, evaluation), file_name=f"pat_modeling_{active.study_id[:12]}.zip", mime="application/zip")
        st.caption("Report contains CV/tuning/fold/stability/robustness diagnostics and the CV-selected fitted pipeline. Holdout tables are included only after final evaluation. Local artifacts and the audit record are in .pat_state/modeling/; only load trusted generated model artifacts.")
    render_explanation(modeling_snapshot(active, evaluation, inspected_case=case_id,
        source_kind="synthetic" if source == "Synthetic modeling cohort" else "prepared_sources")
        if active is not None else None, key="modeling")


if __name__ == "__main__":
    main()
