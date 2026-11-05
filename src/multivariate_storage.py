"""Explicit save/load of trusted local historical PCA reference artifacts."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re

import joblib
import sklearn

from src.modeling_storage import _atomic_joblib, _atomic_json, _operation_lock
from src.multivariate_monitoring import ProcessMonitoringModel, monitoring_model_manifest


DEFAULT_PCA_ROOT = Path(__file__).resolve().parents[1] / ".pat_state" / "multivariate"
PCA_REFERENCE_PROTOCOL = "historical-pca-reference-v1"


def save_pca_reference(model, historical_config, root):
    metadata = {"protocol": PCA_REFERENCE_PROTOCOL, "historical_config": asdict(historical_config),
                "scikit_learn": sklearn.__version__, "reference": monitoring_model_manifest(model)}
    identity = sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    root = Path(root).resolve()
    folder = (root / identity).resolve()
    if not folder.is_relative_to(root):
        raise ValueError("PCA reference directory must stay inside the local artifact directory.")
    with _operation_lock(folder):
        _atomic_joblib(model, folder / "reference.joblib")
        metadata["pipeline_sha256"] = sha256((folder / "reference.joblib").read_bytes()).hexdigest()
        _atomic_json(metadata, folder / "reference.json")
    return identity


def list_pca_references(root):
    records = {}
    root = Path(root).resolve()
    if not root.exists():
        return records
    for path in sorted(root.glob("*/reference.json")):
        if not path.resolve().is_relative_to(root) or not re.fullmatch(r"[0-9a-f]{64}", path.parent.name):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("protocol") == PCA_REFERENCE_PROTOCOL:
                records[path.parent.name] = data
        except (OSError, ValueError):
            continue
    return records


def load_pca_reference(root, identity):
    if not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise ValueError("Select an application-created PCA reference identity.")
    root = Path(root).resolve()
    folder = (root / identity).resolve()
    pipeline = folder / "reference.joblib"
    metadata_path = folder / "reference.json"
    if not folder.is_relative_to(root) or not pipeline.resolve().is_relative_to(folder) or not metadata_path.resolve().is_relative_to(folder):
        raise ValueError("PCA reference files must stay inside the local artifact directory.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("protocol") != PCA_REFERENCE_PROTOCOL or metadata.get("scikit_learn") != sklearn.__version__:
        raise ValueError("PCA reference protocol or scikit-learn version is incompatible.")
    if sha256(pipeline.read_bytes()).hexdigest() != metadata.get("pipeline_sha256"):
        raise ValueError("Saved PCA reference failed its integrity check.")
    model = joblib.load(pipeline)
    if not isinstance(model, ProcessMonitoringModel) or json.loads(json.dumps(monitoring_model_manifest(model), allow_nan=False)) != metadata["reference"]:
        raise ValueError("Saved PCA model and metadata disagree.")
    return model, metadata
