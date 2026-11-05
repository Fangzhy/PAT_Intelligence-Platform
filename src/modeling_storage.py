"""Local study persistence and a durable one-time holdout evaluation gate.

Only application-created local artifacts are loaded. User uploads are CSV data,
never pickle/joblib models. This is a teaching audit trail, not a regulated
access-control or tamper-proof record system.
"""

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost

from src.modeling import PROTOCOL_VERSION, TrainingStudy, evaluate_holdout, pls_diagnostics
from src.modeling_data import BatchPartition
from src.modeling_compat import normalize_training_config
from src.prediction_reliability import study_range_performance


DEFAULT_STATE_ROOT = Path(__file__).resolve().parents[1] / ".pat_state" / "modeling"


def _folder(root, cohort_id):
    if not re.fullmatch(r"[0-9a-f]{64}", cohort_id):
        raise ValueError("Cohort IDs must be SHA-256 hexadecimal identifiers.")
    return Path(root).resolve() / cohort_id


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _json_text(value):
    return json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False)


def _atomic_joblib(value, path):
    with NamedTemporaryFile(dir=path.parent, prefix="study-", suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        joblib.dump(value, temporary_path, compress=3)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json(value, path):
    with NamedTemporaryFile(dir=path.parent, prefix="metadata-", suffix=".tmp", delete=False, mode="w", encoding="utf-8") as temporary:
        temporary.write(_json_text(value))
        temporary_path = Path(temporary.name)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


@contextmanager
def _operation_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "operation.lock"
    try:
        with path.open("x", encoding="utf-8") as lock:
            lock.write("Application modeling operation in progress.\n")
    except FileExistsError as error:
        raise ValueError("Another modeling operation is in progress for this cohort.") from error
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def study_manifest(study):
    policy = getattr(study.results[study.winner_id].estimator, "prediction_policy_", None)
    return {
        "protocol": PROTOCOL_VERSION, "study_id": study.study_id, "cohort_id": study.cohort_id,
        "target": study.target, "target_unit": "g/L", "partition": asdict(study.partition),
        "data_provenance": getattr(study, "data_provenance", {}) or {"status": "not_recorded", "sources": []},
        "input_policy": getattr(study, "input_policy", {}) or None,
        "training_config": asdict(study.config), "experiments": [asdict(case) for case in study.cases],
        "cv_selected_winner": study.winner_id, "input_columns": list(study.input_columns),
        "spectral_axis_cm1": study.axis.tolist(), "training_rows": len(study.training_reference),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "cv_interpretation": "Pooled OOF errors used for tuning; selection makes these optimistic. Independent batch test is a separate frozen step.",
        "winner_policy": "Choose by RMSECV before test disclosure, never by RMSEP.",
        "label_source": "Sparse offline assay references paired at collection time, not simulated truth.",
        "prediction_reliability": policy.summary() if policy is not None else {"status": "not_assessed"},
        "limitations": "Synthetic/appropriately licensed public-data demonstration; not manufacturing validation or CQA prediction.",
    }


def stored_partition(root, cohort_id):
    path = _folder(root, cohort_id) / "partition.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return BatchPartition(tuple(data["training_batches"]), tuple(data["holdout_batches"]))


def holdout_is_sealed(root, cohort_id):
    return (_folder(root, cohort_id) / "holdout_claim.json").exists()


def save_study(study: TrainingStudy, root=DEFAULT_STATE_ROOT):
    folder = _folder(root, study.cohort_id)
    with _operation_lock(folder):
        if holdout_is_sealed(root, study.cohort_id):
            raise ValueError("This cohort's holdout has been revealed or claimed. Further tuning is locked; retain its results as a final report.")
        existing = stored_partition(root, study.cohort_id)
        if existing is not None and existing != study.partition:
            raise ValueError("This cohort already has a different fixed batch partition.")
        _atomic_json(asdict(study.partition), folder / "partition.json")
        _atomic_joblib(study, folder / "training_study.joblib")
        _atomic_json(study_manifest(study), folder / "study_manifest.json")
    return folder


def load_study(root, cohort_id):
    folder = _folder(root, cohort_id)
    path = folder / "training_study.joblib"
    if not path.exists():
        return None
    manifest = json.loads((folder / "study_manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol"] != PROTOCOL_VERSION:
        raise ValueError("Saved study uses a different modeling protocol version.")
    study = joblib.load(path)
    if not isinstance(study, TrainingStudy) or study.cohort_id != cohort_id or study.study_id != manifest["study_id"]:
        raise ValueError("Saved study identity does not match its manifest.")
    study.config = normalize_training_config(study.config)
    return study


def load_holdout(root, cohort_id):
    path = _folder(root, cohort_id) / "holdout_results.joblib"
    return joblib.load(path) if path.exists() else None


def evaluate_once(study, dataset, root=DEFAULT_STATE_ROOT):
    """Evaluate once, then return the stored result on later page visits.

    The claim is written before calculating any test scores. Failure leaves that
    claim visible and freezes tuning rather than quietly reopening the holdout.
    """
    if dataset.cohort_id != study.cohort_id or dataset.target != study.target:
        raise ValueError("Holdout inputs do not match the frozen study.")
    folder = _folder(root, study.cohort_id)
    with _operation_lock(folder):
        previous = load_holdout(root, study.cohort_id)
        if previous is not None:
            if previous["study_id"] != study.study_id:
                raise ValueError("The stored holdout belongs to a different frozen study.")
            return previous
        if holdout_is_sealed(root, study.cohort_id):
            raise ValueError("Holdout evaluation was already claimed but did not finish. Inspect the local audit files; tuning remains locked.")
        saved = load_study(root, study.cohort_id)
        if saved is None or saved.study_id != study.study_id:
            raise ValueError("Save this exact CV study before evaluating held-out batches.")
        claim = {"study_id": study.study_id, "target": study.target, "cv_selected_winner": study.winner_id,
            "claimed_at_utc": datetime.now(timezone.utc).isoformat(), "partition": asdict(study.partition)}
        _atomic_json(claim, folder / "holdout_claim.json")
        result = evaluate_holdout(study, dataset)
        winner = study.results[study.winner_id]
        _atomic_joblib(winner.estimator, folder / "winner_pipeline.joblib")
        metadata = study_manifest(study)
        metadata.update({"model_version": f"phase4-{study.study_id[:12]}", "selected_case": study.winner_id,
            "cv_metrics": winner.cv_metrics,
            "holdout_metrics": result["metrics"].set_index("case_id").loc[study.winner_id].to_dict(),
            "claim": claim})
        _atomic_json(metadata, folder / "winner_metadata.json")
        _atomic_json({"study_id": study.study_id, "winner_id": study.winner_id, "metrics": result["metrics"].to_dict(orient="records")}, folder / "holdout_summary.json")
        # The completed result is the final marker: a successful reload also
        # guarantees that the selected pipeline and metadata were written.
        _atomic_joblib(result, folder / "holdout_results.joblib")
    return result


def cv_predictions(study):
    frames = []
    for case_id, result in study.results.items():
        rows = study.training_metadata.copy()
        rows["reference_g_l"] = study.training_reference.to_numpy()
        rows["prediction_g_l"] = result.oof_prediction
        rows["residual_g_l"] = result.oof_prediction - study.training_reference.to_numpy()
        rows["case_id"] = case_id
        diagnostics = getattr(result, "oof_reliability", None)
        if diagnostics is not None:
            for column in diagnostics:
                rows[column] = diagnostics[column]
        frames.append(rows)
    return pd.concat(frames)


def study_report_zip(study, evaluation=None):
    if evaluation is not None and evaluation["study_id"] != study.study_id:
        raise ValueError("Evaluation and training study identities must match.")
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("study_manifest.json", _json_text(study_manifest(study)))
        archive.writestr("cv_comparison.csv", study.comparison.to_csv(index=False))
        archive.writestr("cv_predictions.csv", cv_predictions(study).to_csv(index=True))
        archive.writestr("cv_folds.csv", study.folds.to_csv(index=False))
        archive.writestr("prediction_reliability/range_performance.csv", study_range_performance(study, evaluation).to_csv(index=False))
        archive.writestr("failed_experiments.csv", study.failures.to_csv(index=False))
        for case_id, result in study.results.items():
            for name in ("tuning", "fold_scores", "feature_stability", "robustness"):
                archive.writestr(f"experiments/{case_id}/{name}.csv", getattr(result, name).to_csv(index=False))
            if result.case.model == "pls":
                for name, frame in zip(("pls_training_scores", "pls_loadings", "pls_coefficients"), pls_diagnostics(result, study.training_metadata)):
                    archive.writestr(f"experiments/{case_id}/{name}.csv", frame.to_csv(index=True))
        if evaluation is not None:
            for name in ("metrics", "predictions", "batch_metrics"):
                archive.writestr(f"holdout_{name}.csv", evaluation[name].to_csv(index=True))
        winner_bytes = BytesIO()
        joblib.dump(study.results[study.winner_id].estimator, winner_bytes, compress=3)
        archive.writestr("cv_selected_pipeline.joblib", winner_bytes.getvalue())
        archive.writestr("README.txt", "Labels are offline assay references. CV scores are tuning estimates. Holdout tables appear only after the one-time frozen evaluation. Load the generated joblib pipeline only as a trusted local artifact in the pinned environment.\n")
    return output.getvalue()
