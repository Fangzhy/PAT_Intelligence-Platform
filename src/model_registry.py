"""Discover trusted local pipelines and explicitly prepare separate demo models."""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.data_generation import generate_dataset
from src.data_integration import integrate_sources, sources_from_dataset
from src.modeling import ModelCase, TrainingConfig, train_study
from src.modeling_data import TARGETS, fixed_batch_partition, modeling_dataset
from src.modeling_storage import _atomic_joblib, _atomic_json, study_manifest


DEFAULT_MONITORING_MODELS = Path(__file__).resolve().parents[1] / ".pat_state" / "monitoring" / "models"
DEMO_PROTOCOL = "phase5-demo-v2-reliability"


@dataclass(frozen=True)
class ModelReference:
    identity: str
    target: str
    version: str
    source: str
    pipeline_path: str
    metadata_path: str
    stamp: tuple[int, int, int]


@dataclass
class ModelBundle:
    pipeline: Pipeline
    metadata: dict

    @property
    def target(self):
        return self.metadata["target"]

    @property
    def version(self):
        return self.metadata["model_version"]


def _reference(folder, pipeline_name, metadata_name, source, root):
    folder = folder.resolve()
    if not folder.is_relative_to(Path(root).resolve()):
        raise ValueError("Model artifacts must remain inside the application's model directory.")
    pipeline_path, metadata_path = folder / pipeline_name, folder / metadata_name
    if not pipeline_path.resolve().is_relative_to(folder) or not metadata_path.resolve().is_relative_to(folder):
        raise ValueError("Linked model files outside the artifact directory are not supported.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("target") not in TARGETS or metadata.get("target_unit") != "g/L" or not metadata.get("model_version"):
        raise ValueError("Model metadata needs a supported target, g/L units, and version.")
    stamp = (pipeline_path.stat().st_mtime_ns, pipeline_path.stat().st_size, metadata_path.stat().st_mtime_ns)
    identity = sha256((str(pipeline_path) + metadata["model_version"]).encode()).hexdigest()
    return ModelReference(identity, metadata["target"], metadata["model_version"], source, str(pipeline_path), str(metadata_path), stamp)


def discover_models(modeling_root, monitoring_root=DEFAULT_MONITORING_MODELS):
    """List metadata first. No pickle is loaded and no model is trained here."""
    references, issues = [], []
    for root, pipeline_name, metadata_name, source in (
        (Path(modeling_root), "winner_pipeline.joblib", "winner_metadata.json", "Phase 4 final model"),
        (Path(monitoring_root), "pipeline.joblib", "model_metadata.json", "Separate demo calibration"),
    ):
        if not root.exists():
            continue
        for metadata_path in sorted(root.glob(f"*/{metadata_name}")):
            folder = metadata_path.parent
            if source == "Phase 4 final model" and not (folder / "holdout_results.joblib").exists():
                continue  # A claim or partially completed evaluation is not a model release.
            try:
                references.append(_reference(folder, pipeline_name, metadata_name, source, root))
            except (ValueError, OSError, KeyError) as error:
                issues.append({"artifact": folder.name, "issue": str(error)})
    references.sort(key=lambda ref: (ref.source != "Phase 4 final model", -ref.stamp[0], ref.version))
    return references, issues


def load_model(reference: ModelReference) -> ModelBundle:
    """Load only a selected app-generated local artifact; never accept model uploads."""
    metadata = json.loads(Path(reference.metadata_path).read_text(encoding="utf-8"))
    if metadata.get("target") != reference.target or metadata.get("model_version") != reference.version:
        raise ValueError("Selected model metadata changed; refresh the model list.")
    import sklearn
    if metadata.get("versions", {}).get("scikit_learn") != sklearn.__version__:
        raise ValueError("The saved model needs its pinned scikit-learn version; synchronize requirements.txt.")
    try:
        pipeline = joblib.load(reference.pipeline_path)
        if not isinstance(pipeline, Pipeline) or set(pipeline.named_steps) != {"features", "scale", "model"}:
            raise ValueError("Expected the saved feature/scaler/model pipeline.")
        features, scaler = pipeline.named_steps["features"], pipeline.named_steps["scale"]
        check_is_fitted(scaler)
        check_is_fitted(pipeline.named_steps["model"])
        names = features.get_feature_names_out()
        if len(names) != len(scaler.mean_) or not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
            raise ValueError("Model scaling metadata is invalid.")
        axis = np.asarray(metadata["spectral_axis_cm1"], dtype=float)
        if axis.shape != features.axis_.shape or not np.allclose(axis, features.axis_, rtol=1e-10, atol=1e-10):
            raise ValueError("Model axis and saved feature transformer disagree.")
        expected = [f"raman_{value:.12g}" for value in axis]
        if not set(expected).issubset(metadata["input_columns"]):
            raise ValueError("Model input schema does not match its Raman axis.")
    except Exception as error:
        raise ValueError(f"Saved pipeline could not be loaded: {error}") from error
    metadata = dict(metadata)
    metadata.setdefault("validation_status", "Independent synthetic-batch evaluation completed; not manufacturing validation")
    metadata["artifact_source"] = reference.source
    metadata["artifact_identity"] = reference.identity
    metadata["artifact_stamp"] = reference.stamp
    return ModelBundle(pipeline, metadata)


def _demo_identity(config, target):
    payload = {"protocol": DEMO_PROTOCOL, "calibration": asdict(config), "target": target}
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def prepare_demo_models(replay_config, root=DEFAULT_MONITORING_MODELS):
    """Explicit calibration action, separate from monitoring and Phase 4 test state.

    Calibration uses a different seed. Each target fits raw-spectrum PLS through
    training-only grouped CV. B007/B008 of that calibration cohort are unused;
    no independent test metrics are claimed. Existing matching models are reused.
    """
    calibration = replace(replay_config, n_batches=8, seed=replay_config.seed + 1_000_003)
    root = Path(root).resolve()
    references = []
    pending = []
    for target in TARGETS:
        folder = root / _demo_identity(calibration, target)
        if (folder / "model_metadata.json").exists() and (folder / "pipeline.joblib").exists():
            try:
                reference = _reference(folder, "pipeline.joblib", "model_metadata.json", "Separate demo calibration", root)
                load_model(reference)  # Check the pinned artifact before reusing it.
                references.append(reference)
            except (ValueError, OSError):
                pending.append((target, folder))  # Explicit preparation can repair its own artifacts.
        else:
            pending.append((target, folder))
    if pending:
        integrated = integrate_sources(sources_from_dataset(generate_dataset(calibration)))
        for target, folder in pending:
            dataset = modeling_dataset(integrated, target)
            study = train_study(dataset, fixed_batch_partition(dataset.metadata.batch_id),
                cases=[ModelCase("raw", "pls")], config=TrainingConfig(max_pls_components=4))
            metadata = study_manifest(study)
            metadata.update({
                "model_version": f"phase5-demo-{target.lower()}-{study.study_id[:12]}",
                "selected_case": study.winner_id, "cv_metrics": study.results[study.winner_id].cv_metrics,
                "calibration_config": asdict(calibration), "demo_protocol": DEMO_PROTOCOL,
                "validation_status": "Demo calibration: grouped CV only; no independent test or manufacturing validation",
                "calibration_scope": "Training batches only, with a different seed from replay; calibration holdout batches are unused.",
            })
            folder.mkdir(parents=True, exist_ok=True)
            _atomic_joblib(study.results[study.winner_id].estimator, folder / "pipeline.joblib")
            _atomic_json(metadata, folder / "model_metadata.json")
            references.append(_reference(folder, "pipeline.joblib", "model_metadata.json", "Separate demo calibration", root))
    return references
