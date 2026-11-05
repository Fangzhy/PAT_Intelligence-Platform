"""Phase 8: trusted immutable model bundles, batch inference, and deployment guide."""

from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_generation import SimulationConfig
from src.data_integration import read_source_csv
from src.deployment import (
    DEFAULT_RELEASE_ROOT, create_release, discover_deployment_models, evaluation_report,
    example_inputs, inference_schema, model_card, predict_frame, release_zip, verify_release,
)
from src.deployment_check import deployment_preflight
from src.model_registry import DEFAULT_MONITORING_MODELS, load_model, prepare_demo_models
from src.modeling_storage import DEFAULT_STATE_ROOT, _json_text
from src.monitoring import MonitoringSettings
from src.navigation import pause_monitoring_on_navigation
from src.reliability_storage import DEFAULT_RELIABILITY_ROOT
from src.runtime import apply_page_runtime, public_demo
from src.ui_labels import workflow_label


@st.cache_resource(max_entries=8, show_spinner=False)
def cached_pipeline(reference):
    return load_model(reference)


def main():
    st.set_page_config(page_title="PAT | Deployment & Reporting", page_icon="📦", layout="wide")
    apply_page_runtime()
    pause_monitoring_on_navigation()
    st.caption("PAT INTELLIGENCE PLATFORM")
    st.title("Deploy reproducible models. Keep the evidence.")
    st.info("Educational deployment · Trusted app-created pipelines · Synthetic evidence · No manufacturing validation")
    if public_demo():
        st.badge("Public demo · temporary session storage", color="orange")
    config = st.session_state.get("simulation_config", SimulationConfig())
    modeling_root = Path(st.session_state.get("modeling_state_root", DEFAULT_STATE_ROOT))
    monitoring_root = Path(st.session_state.get("monitoring_models_root", DEFAULT_MONITORING_MODELS))
    reliability_root = Path(st.session_state.get("reliability_state_root", DEFAULT_RELIABILITY_ROOT))
    release_root = Path(st.session_state.get("deployment_state_root", DEFAULT_RELEASE_ROOT))
    st.write("Select a finalized Predictive_Modeling or Model_Reliability pipeline, or a separate CV-only demo model. Packaging copies its fitted feature/scaler/model pipeline and recorded evidence; it never evaluates a holdout or trains during routine rendering.")
    if st.button("Prepare / reuse deployment demo models", key="deployment_prepare"):
        try:
            with st.spinner("Preparing separate glucose/lactate demo calibration…"):
                prepared = prepare_demo_models(config, monitoring_root)
            st.session_state["deployment_preferred_identity"] = next(reference.identity for reference in prepared if reference.target == "Glucose")
            st.session_state.pop("deployment_model", None)
        except (ValueError, OSError) as error:
            st.error(f"Demo preparation failed: {error}")
    references, issues = discover_deployment_models(modeling_root, monitoring_root, reliability_root)
    if issues:
        st.warning("Some saved artifacts are unavailable and were excluded.")
        st.dataframe(pd.DataFrame(issues), hide_index=True, width="stretch")
    guide_tab, model_tab, inference_tab, release_tab = st.tabs(["Deployment & demonstration guide", "Model evidence", "Prediction interface", "Release bundle"])
    with guide_tab:
        preflight = deployment_preflight()
        st.success("Local runtime and deployment files pass preflight.") if preflight["ready"] else st.warning("Runtime/file preflight needs review. This does not measure hosted deployment status.")
        st.dataframe(pd.DataFrame(preflight["dependencies"]), hide_index=True, width="stretch")
        if not preflight["ready"]:
            st.json({"python": preflight["python"], "required_files": preflight["files"]})
        st.markdown("""
Start locally with `streamlit run app.py`. The Cloud entrypoint is also `app.py`,
using Python 3.11 and the pinned requirements. Read `deployment/README.md` for
the GitHub/Streamlit Cloud release steps, `deployment/DEMO_GUIDE.md` for a timed
walkthrough, and `deployment/INTERVIEW_PREPARATION.md` for technical practice.

Cloud configuration uses `[deployment]` with `mode = "public_demo"` in Advanced
settings. Public sessions use separate temporary model folders and in-app
sources. Download reports/models you want to keep; hosted local storage and
session history are not durable lifecycle records.
""")
        st.caption("A passing preflight is not evidence that an app has been published. Public URL and deployment status are recorded only after the hosted app is actually checked.")
        st.latex(r"\hat y_{before}=f_{saved}(x),\qquad |\hat y_{reload}-\hat y_{before}|\leq atol+rtol|\hat y_{before}|")
        st.write("The default parity tolerance is atol = 10⁻⁹ and rtol = 10⁻⁷. This tests serialization/inference consistency, not independent predictive accuracy. File hashes detect corruption but do not establish the trustworthiness of a model.")
    if not references:
        with model_tab:
            st.info("No trusted saved models found. Prepare separate demo models above, or finalize a study in Predictive Modeling / Model Reliability.")
        return
    labels = {f"{reference.target} · {reference.version} · {reference.source}": reference for reference in references}
    preferred = st.session_state.get("deployment_preferred_identity")
    default = next((index for index, reference in enumerate(labels.values()) if reference.identity == preferred), 0)
    selected = st.selectbox("Trusted saved pipeline", list(labels), index=default, format_func=workflow_label, key="deployment_model")
    reference = labels[selected]
    try:
        bundle = cached_pipeline(reference)
    except (ValueError, OSError) as error:
        st.error(f"Saved pipeline unavailable: {error}")
        return
    with model_tab:
        st.markdown(model_card(bundle))
        st.json(evaluation_report(bundle))
        st.download_button("Download recorded model evaluation (JSON)", _json_text(evaluation_report(bundle)), file_name=f"{bundle.target.lower()}_evaluation.json", mime="application/json", key="deployment_evaluation")
    with inference_tab:
        st.write("Historical batch prediction with exact schema, units, relative sensor timing, quality/domain checks, and the frozen saved pipeline. CSV data only; executable model uploads are not accepted.")
        schema = inference_schema(bundle)
        example = example_inputs(bundle)
        st.download_button("Download inference CSV example / headers", example.to_csv(index=False).encode("utf-8"), file_name=f"{bundle.target.lower()}_inference.csv", mime="text/csv", key="deployment_example")
        st.download_button("Download input schema (JSON)", _json_text(schema), file_name="input_schema.json", mime="application/json")
        mode = st.radio("Inference input", ["Synthetic example", "Upload input CSV"], key="deployment_input")
        inputs = example
        if mode == "Upload input CSV":
            uploaded = st.file_uploader("Prediction input CSV", type=["csv"], key="deployment_csv")
            if uploaded is not None:
                try:
                    inputs = read_source_csv(uploaded.getvalue())
                except ValueError as error:
                    st.error(str(error))
                    inputs = None
            else:
                inputs = None
        domain = st.number_input("Batch inference maximum |z|", min_value=0.5, max_value=100.0, value=6.0, step=0.5, key="deployment_domain")
        age = st.number_input("Batch inference maximum sensor age (minutes)", min_value=0.0, max_value=240.0, value=30.0, step=5.0, key="deployment_sensor_age")
        context = sha256((reference.identity + str(reference.stamp) + str(domain) + str(age) +
            (inputs.to_csv(index=False) if inputs is not None else "no input")).encode()).hexdigest()
        if st.button("Run saved-pipeline inference", key="deployment_predict", disabled=inputs is None or inputs.empty):
            try:
                result = predict_frame(bundle, inputs, MonitoringSettings(domain_z_limit=domain, max_sensor_age_minutes=age))
                st.session_state["deployment_prediction_result"] = {"context": context, "frame": result}
            except (ValueError, TypeError) as error:
                st.session_state.pop("deployment_prediction_result", None)
                st.error(str(error))
        result = st.session_state.get("deployment_prediction_result")
        if result is not None and result["context"] == context:
            st.dataframe(result["frame"], hide_index=True, width="stretch", key="deployment_predictions")
            st.download_button("Download versioned predictions (CSV)", result["frame"].to_csv(index=False), file_name="pipeline_predictions.csv", mime="text/csv")
            st.caption("Results reflect the explicit inference action and retain model version and inference time. Invalid rows are gaps; no model training or reference-label substitution occurs.")
    with release_tab:
        st.write("Create a local immutable release snapshot with fitted pipeline, original metadata, schema, pinned environment, model card, checksums, and reproducibility examples where the model axis supports them.")
        if st.button("Create / verify release bundle", key="deployment_release"):
            try:
                folder = create_release(reference, release_root)
                st.session_state["deployment_release_result"] = {"identity": reference.identity, "folder": str(folder)}
            except (ValueError, OSError) as error:
                st.session_state.pop("deployment_release_result", None)
                st.error(f"Release snapshot failed: {error}")
        saved = st.session_state.get("deployment_release_result")
        if saved is not None and saved["identity"] == reference.identity:
            try:
                verification = verify_release(saved["folder"], trusted=True)
                st.json(verification)
                st.download_button("Download trusted model release (ZIP)", release_zip(saved["folder"]), file_name=f"{verification['release_id']}.zip", mime="application/zip", key="deployment_release_zip")
            except (ValueError, OSError) as error:
                st.error(f"Release integrity needs review: {error}")


if __name__ == "__main__":
    main()
