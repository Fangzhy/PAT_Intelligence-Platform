"""Explicit training-only baseline preparation for an exact monitoring artifact."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling_data import BatchPartition
from src.modeling_storage import _atomic_json, _json_text, _operation_lock
from src.reliability import input_drift_features


DEFAULT_ACQUIRED_RELIABILITY_ROOT = Path(__file__).resolve().parents[1] / ".pat_state" / "acquired_reliability"
PROTOCOL = "acquired-reliability-reference-v1"


def artifact_context(bundle):
    metadata = bundle.metadata
    return {"artifact_identity": metadata["artifact_identity"], "artifact_stamp": list(metadata["artifact_stamp"]),
            "model_version": bundle.version, "study_id": metadata["study_id"],
            "cohort_id": metadata["cohort_id"], "target": bundle.target}


def reference_path(bundle, root):
    identity = sha256(_json_text(artifact_context(bundle)).encode()).hexdigest()
    root = Path(root).resolve()
    path = (root / identity / "reference.json").resolve()
    if not path.is_relative_to(root):
        raise ValueError("Reliability reference must stay inside its artifact directory.")
    return path


def prepare_acquired_reference(bundle, cohort, root):
    """Predict original training rows once on explicit request; never fit/test."""
    metadata = bundle.metadata
    if (cohort.cohort_id != metadata["cohort_id"] or cohort.target != bundle.target
            or list(cohort.X.columns) != metadata["input_columns"]):
        raise ValueError("Development cohort does not match this monitoring model. Restore its original development source/settings.")
    partition = BatchPartition(tuple(metadata["partition"]["training_batches"]), tuple(metadata["partition"]["holdout_batches"]))
    training, _ = partition.masks(cohort.metadata)
    inputs = cohort.X.loc[training]
    if len(inputs) != metadata["training_rows"] or inputs.empty:
        raise ValueError("Original training-row coverage differs from the saved model.")
    existing = load_acquired_reference(bundle, root)
    if existing is not None:
        return existing
    predictions = np.asarray(bundle.pipeline.predict(inputs), dtype=float).reshape(-1)
    if predictions.shape != (len(inputs),) or not np.isfinite(predictions).all():
        raise ValueError("Training baseline predictions must be finite and match the training rows.")
    features = input_drift_features(inputs)
    reference = {"protocol": PROTOCOL, "artifact": artifact_context(bundle),
        "training_row_ids": list(map(str, inputs.index)), "input_features": features.to_dict(orient="list"),
        "predictions_g_l": predictions[predictions >= 0].tolist(),
        "scope": "Original training rows only; saved model inference, no fitting or holdout evaluation."}
    reference["sha256"] = sha256(_json_text(reference).encode()).hexdigest()
    path = reference_path(bundle, root)
    with _operation_lock(path.parent):
        _atomic_json(reference, path)
    return reference


def load_acquired_reference(bundle, root):
    path = reference_path(bundle, root)
    if not path.exists():
        return None
    reference = json.loads(path.read_text(encoding="utf-8"))
    digest = reference.pop("sha256", None)
    if (digest != sha256(_json_text(reference).encode()).hexdigest() or reference.get("protocol") != PROTOCOL
            or reference.get("artifact") != artifact_context(bundle)):
        raise ValueError("Training reliability reference failed its identity/integrity check.")
    features = pd.DataFrame(reference["input_features"])
    if len(features) != len(reference["training_row_ids"]) or len(features) != bundle.metadata["training_rows"]:
        raise ValueError("Training reliability reference has inconsistent coverage.")
    reference["sha256"] = digest
    return reference
