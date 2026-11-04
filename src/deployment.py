"""Trusted model release snapshots and guarded batch prediction.

Only app-created local pipelines can become releases. CSV is the only upload
format; no uploaded pickle/joblib/ZIP is deserialized. Hashes detect corruption,
not authenticity. Evaluation records are read, never generated or modified here.
"""

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from importlib import import_module
import json
from pathlib import Path
import platform
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import numpy as np
import pandas as pd

from src.data_generation import SimulationConfig, generate_dataset
from src.model_registry import discover_models, load_model
from src.modeling_storage import _atomic_json, _json_text, _operation_lock
from src.monitoring import MonitoringSettings, predict_observation
from src.replay import replay_batch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_ROOT = ROOT / ".pat_state" / "releases"
IDENTITY_COLUMNS = ["sample_id", "batch_id", "measurement_timestamp", "sensor_timestamp"]
RELEASE_PROTOCOL = "pat-release-v1"


def discover_deployment_models(modeling_root, monitoring_root, reliability_root):
    references, issues = discover_models(modeling_root, monitoring_root)
    for target in ("glucose", "lactate"):
        extra, errors = discover_models(Path(reliability_root) / target, Path(reliability_root) / "unused-models")
        references.extend(replace(reference, source="Phase 7 frozen demo") for reference in extra)
        issues.extend(errors)
    return references, issues


def inference_schema(bundle):
    return {"target": bundle.target, "target_unit": "g/L", "model_version": bundle.version,
        "columns": IDENTITY_COLUMNS + list(bundle.metadata["input_columns"]),
        "spectral_axis_cm1": bundle.metadata["spectral_axis_cm1"],
        "units": {"temperature_c": "degC", "ph": "dimensionless", "dissolved_oxygen_pct": "% air saturation",
            "feed_rate_l_h": "L/h", "agitation_rpm": "rpm", "raman_*": "arbitrary units"},
        "time_policy": "Explicit timezone required. Sensor must be at/before spectrum within configured age; this is historical batch inference, not a live freshness check.",
        "column_policy": "Exact model input headers; extra columns are rejected. No interpolation or inferred unit conversions."}


def example_inputs(bundle):
    """Synthetic input example only when its physical axis matches the model."""
    axis = np.asarray(bundle.metadata["spectral_axis_cm1"], dtype=float)
    if len(axis) < 32 or not np.allclose(axis, np.linspace(400, 1800, len(axis)), rtol=1e-10, atol=1e-10):
        return pd.DataFrame(columns=inference_schema(bundle)["columns"])
    data = generate_dataset(SimulationConfig(n_batches=1, duration_h=24, n_wavenumbers=len(axis), seed=733))
    batch = replay_batch(data, "B001")
    positions = [0, 6, 12, 18, 24]
    frame = batch.metadata.loc[positions, IDENTITY_COLUMNS].reset_index(drop=True)
    values = batch.inputs.iloc[positions].reset_index(drop=True)
    return pd.concat([frame, values[list(bundle.metadata["input_columns"])]], axis=1)


def predict_frame(bundle, inputs, settings=None, *, max_rows=1000):
    """Guard every CSV row through the same fitted monitoring inference path."""
    settings = settings or MonitoringSettings()
    required = inference_schema(bundle)["columns"]
    if not isinstance(inputs, pd.DataFrame) or not inputs.columns.is_unique:
        raise ValueError("Supply a DataFrame with unique input headers.")
    if inputs.empty or len(inputs) > max_rows:
        raise ValueError(f"Supply between one and {max_rows} observations.")
    missing, extra = set(required) - set(inputs), set(inputs) - set(required)
    if missing or extra:
        raise ValueError(f"Input schema mismatch. Missing: {', '.join(sorted(missing)) or 'none'}. Unexpected: {', '.join(sorted(extra)) or 'none'}.")
    frame = inputs.copy(deep=True).reset_index(drop=True)
    duplicates = frame.duplicated(["batch_id", "sample_id"], keep=False)
    records = []
    for position, row in frame.iterrows():
        invalid_identity = any(not isinstance(row[key], str) or not row[key].strip() for key in ("sample_id", "batch_id"))
        if duplicates.iloc[position] or invalid_identity:
            records.append({"source_row": position + 1, "sample_id": row.sample_id, "batch_id": row.batch_id,
                "prediction_g_l": np.nan, "status": "invalid_input", "flags": "Ambiguous or missing batch/sample identity",
                "model_version": bundle.version, "prediction_time_utc": pd.NaT, "max_abs_z": np.nan})
            continue
        values = frame.loc[[position], list(bundle.metadata["input_columns"])].copy()
        for column in values:
            values[column] = pd.to_numeric(values[column], errors="coerce")
        result = predict_observation(bundle, bundle.target, values, bundle.metadata["spectral_axis_cm1"],
            row.measurement_timestamp, row.sensor_timestamp, settings)
        records.append({"source_row": position + 1, "sample_id": row.sample_id, "batch_id": row.batch_id,
            "prediction_g_l": result.prediction_g_l, "status": result.status, "flags": "; ".join(result.flags),
            "model_version": result.model_version, "prediction_time_utc": result.prediction_time_utc, "max_abs_z": result.max_abs_z})
    return pd.DataFrame(records)


def model_card(bundle):
    metadata = bundle.metadata
    return f"""# Vaccine PAT model card

- Version: {bundle.version}
- Target: {bundle.target}, g/L
- Selected case: {metadata.get('selected_case', 'Unavailable')}
- Training cohort fingerprint: {metadata.get('cohort_id', 'Unavailable')}
- Study ID: {metadata.get('study_id', 'Unavailable')}
- Training batch IDs: {', '.join(metadata.get('partition', {}).get('training_batches', []))}
- Held-out batch IDs: {', '.join(metadata.get('partition', {}).get('holdout_batches', []))}
- Evidence: {metadata.get('validation_status', 'Independent synthetic-batch test recorded' if metadata.get('holdout_metrics') else 'Grouped CV only; no independent test recorded')}

## Intended use

Educational predictions using synthetic or appropriately licensed inputs with
the exact declared schema and units. The fitted features, scaler, and estimator
are saved together. Inference does not fit or tune any component.

## Evaluation

Training labels are offline assay references. Grouped CV supports selection
and can be optimistic after tuning. Independent-test metrics are included only
when already recorded; no test is evaluated by this release operation.

CV metrics: {json.dumps(metadata.get('cv_metrics', {}), sort_keys=True)}

Independent metrics: {json.dumps(metadata.get('holdout_metrics', {}), sort_keys=True)}

## Limits

Synthetic evaluation does not establish manufacturing validation, potency,
purity, or a confirmed CPP/CQA relationship. Quality/domain screens do not
guarantee accuracy. Missing/invalid inputs produce gaps. Historical batch
inference checks relative input timing, not current wall-clock freshness.
Only load this executable joblib artifact as a trusted application-created
artifact in its pinned environment. File hashes detect corruption, not trust.
"""


def _stamp(reference):
    path, metadata = Path(reference.pipeline_path), Path(reference.metadata_path)
    return path.stat().st_mtime_ns, path.stat().st_size, metadata.stat().st_mtime_ns


def _check_fitted_environment(versions):
    """Do not label the current lock as an older artifact's training environment."""
    for name, module in {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
                         "scikit_learn": "sklearn", "xgboost": "xgboost"}.items():
        actual = import_module(module).__version__
        if versions.get(name) != actual:
            raise ValueError(f"Release needs its pinned fitted environment: {name} expected {versions.get(name)}, installed {actual}.")


def create_release(reference, root=DEFAULT_RELEASE_ROOT):
    """Save an immutable copy of the selected artifact, schema, and parity cases."""
    if _stamp(reference) != reference.stamp:
        raise ValueError("The selected artifact changed. Refresh model discovery before releasing.")
    bundle = load_model(reference)
    _check_fitted_environment(bundle.metadata.get("versions", {}))
    pipeline_bytes = Path(reference.pipeline_path).read_bytes()
    metadata_bytes = Path(reference.metadata_path).read_bytes()
    requirements = (ROOT / "requirements.txt").read_bytes()
    if _stamp(reference) != reference.stamp:
        raise ValueError("The selected artifact changed while taking its snapshot.")
    identity = sha256(pipeline_bytes + metadata_bytes + requirements + RELEASE_PROTOCOL.encode()).hexdigest()
    folder = Path(root).resolve() / f"pat-{identity[:24]}"
    with _operation_lock(folder):
        if (folder / "release_manifest.json").exists():
            verify_release(folder, trusted=True)
            return folder
        files = {"pipeline.joblib": pipeline_bytes, "model_metadata.json": metadata_bytes,
            "input_schema.json": _json_text(inference_schema(bundle)).encode(),
            "requirements.txt": requirements, "MODEL_CARD.md": model_card(bundle).encode("utf-8"),
            "LICENSE": (ROOT / "LICENSE").read_bytes()}
        example = example_inputs(bundle)
        if not example.empty:
            raw = np.asarray(bundle.pipeline.predict(example[list(bundle.metadata["input_columns"])]), dtype=float).reshape(-1)
            if len(raw) != len(example) or not np.isfinite(raw).all():
                raise ValueError("Cannot record reproducibility examples from invalid raw model predictions.")
            files["example_inputs.csv"] = example.to_csv(index=False).encode()
            files["parity_predictions.json"] = _json_text({"raw_model_prediction_g_l": raw.tolist(),
                "interpretation": "Serialization parity cases, not guarded operational predictions or independent test metrics"}).encode()
        for name, content in files.items():
            (folder / name).write_bytes(content)
        manifest = {"protocol": RELEASE_PROTOCOL, "release_id": folder.name, "model_version": bundle.version,
            "target": bundle.target, "source": reference.source, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_cohort_id": bundle.metadata.get("cohort_id"), "study_id": bundle.metadata.get("study_id"),
            "selected_case": bundle.metadata.get("selected_case"), "fitted_versions": bundle.metadata.get("versions", {}),
            "packaging_runtime": {"python": platform.python_version(), "system": platform.system()},
            "hashes": {name: sha256(content).hexdigest() for name, content in files.items()},
            "parity_examples": len(example), "rtol": 1e-7, "atol": 1e-9,
            "limitations": "Trusted app-created artifacts only; hashes are integrity checks, not digital signatures. Synthetic evidence is not manufacturing validation."}
        _atomic_json(manifest, folder / "release_manifest.json")
        verify_release(folder, trusted=True)
    return folder


def verify_release(folder, *, trusted=False):
    """Check integrity before deserializing an explicitly trusted local release."""
    if not trusted:
        raise ValueError("Only verify trusted application-created local releases; untrusted model loading is disabled.")
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "release_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != RELEASE_PROTOCOL or manifest.get("release_id") != folder.name:
        raise ValueError("Release identity/protocol mismatch.")
    required = {"pipeline.joblib", "model_metadata.json", "input_schema.json", "requirements.txt", "MODEL_CARD.md", "LICENSE"}
    if not required.issubset(manifest.get("hashes", {})):
        raise ValueError("Release is missing required integrity records.")
    for name, digest in manifest["hashes"].items():
        path = folder / name
        if Path(name).name != name or path.resolve().parent != folder or not path.is_file() or sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Release integrity check failed: {name}")
    _check_fitted_environment(manifest.get("fitted_versions", {}))
    rows = int(manifest.get("parity_examples", 0))
    if rows:
        if not {"example_inputs.csv", "parity_predictions.json"}.issubset(manifest["hashes"]):
            raise ValueError("Release parity files are missing.")
        example = pd.read_csv(folder / "example_inputs.csv")
        metadata = json.loads((folder / "model_metadata.json").read_text(encoding="utf-8"))
        pipeline = joblib.load(folder / "pipeline.joblib")
        actual = np.asarray(pipeline.predict(example[metadata["input_columns"]]), dtype=float).reshape(-1)
        expected = np.asarray(json.loads((folder / "parity_predictions.json").read_text())["raw_model_prediction_g_l"])
        if actual.shape != (rows,) or expected.shape != actual.shape or not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=manifest["rtol"], atol=manifest["atol"]):
            raise ValueError("Reloaded predictions differ from the recorded serialization parity cases.")
    return {"release_id": folder.name, "integrity": "passed", "parity": "passed" if rows else "not assessed: custom input examples needed", "parity_rows": rows}


def release_zip(folder):
    folder = Path(folder)
    verify_release(folder, trusted=True)
    manifest = json.loads((folder / "release_manifest.json").read_text(encoding="utf-8"))
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name in ["release_manifest.json", *manifest["hashes"]]:
            archive.writestr(name, (folder / name).read_bytes())
    return output.getvalue()


def evaluation_report(bundle):
    """JSON-safe recorded evidence; no inference/tuning/test disclosure occurs."""
    return {"model_version": bundle.version, "target": bundle.target, "unit": "g/L",
        "cohort_id": bundle.metadata.get("cohort_id"), "partition": bundle.metadata.get("partition", {}),
        "cv_metrics": bundle.metadata.get("cv_metrics", {}), "independent_test_metrics": bundle.metadata.get("holdout_metrics"),
        "validation_status": bundle.metadata.get("validation_status", "Independent synthetic test recorded" if bundle.metadata.get("holdout_metrics") else "Grouped CV only"),
        "winner": bundle.metadata.get("selected_case"), "training_config": bundle.metadata.get("training_config", {}),
        "fitted_versions": bundle.metadata.get("versions", {}), "limitations": "Synthetic evidence; not manufacturing validation or CQA prediction."}
