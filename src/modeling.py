"""Training-only feature/model search, pooled batch CV, and frozen holdout scoring."""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from numbers import Integral
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from src.feature_engineering import FEATURE_SETS, SpectralFeatures
from src.modeling_data import BatchPartition, ModelingDataset
from src.validation import batch_folds, fold_manifest, regression_metrics
from src.prediction_reliability import (RELIABILITY_PROTOCOL, checked_bounds,
    fit_prediction_policy, assess_prediction_reliability)


PROTOCOL_VERSION = "phase4-v1"
MODELS = {"pls": "PLS", "linear": "Linear Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}


@dataclass(frozen=True)
class ModelCase:
    feature_set: str
    model: str

    def __post_init__(self):
        if self.feature_set not in FEATURE_SETS or self.model not in MODELS:
            raise ValueError("Choose a supported feature recipe and regression model.")

    @property
    def case_id(self):
        return f"{self.feature_set}__{self.model}"


DEFAULT_CASES = tuple(ModelCase(feature, model) for feature, model in (
    ("raw", "pls"), ("preprocessed", "pls"), ("preprocessed", "linear"),
    ("first_derivative", "pls"), ("second_derivative", "pls"),
    ("fft", "pls"), ("fft", "random_forest"),
    ("wavelet", "pls"), ("wavelet", "xgboost"),
    ("summary", "linear"), ("sensors", "random_forest"),
    ("combined", "pls"), ("combined", "xgboost"),
))


@dataclass(frozen=True)
class TrainingConfig:
    n_splits: int = 3
    max_pls_components: int = 8
    tree_count: int = 80
    seed: int = 42
    declared_range: tuple | None = None
    decision_range: tuple | None = None

    def __post_init__(self):
        for name, minimum in (("n_splits", 2), ("max_pls_components", 1), ("tree_count", 1), ("seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name in ("declared_range", "decision_range"):
            object.__setattr__(self, name, checked_bounds(getattr(self, name), name))
        if self.declared_range and self.decision_range and not (
            self.declared_range[0] <= self.decision_range[0] < self.decision_range[1] <= self.declared_range[1]):
            raise ValueError("Decision range must be inside the declared concentration range.")


@dataclass
class CaseResult:
    case: ModelCase
    estimator: Pipeline
    best_params: dict
    cv_metrics: dict
    training_metrics: dict
    oof_prediction: np.ndarray
    tuning: pd.DataFrame
    fold_scores: pd.DataFrame
    feature_stability: pd.DataFrame
    robustness: pd.DataFrame
    oof_reliability: pd.DataFrame | None = field(default=None)


@dataclass
class TrainingStudy:
    study_id: str
    cohort_id: str
    target: str
    partition: BatchPartition
    config: TrainingConfig
    cases: tuple[ModelCase, ...]
    results: dict[str, CaseResult]
    failures: pd.DataFrame
    comparison: pd.DataFrame
    training_metadata: pd.DataFrame
    training_reference: pd.Series
    folds: pd.DataFrame
    winner_id: str
    input_columns: tuple[str, ...]
    axis: np.ndarray


def _base_model(name, config):
    if name == "pls":
        return PLSRegression(scale=False, max_iter=1000, tol=1e-8)
    if name == "linear":
        return LinearRegression()
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=config.tree_count, random_state=config.seed, n_jobs=1)
    return XGBRegressor(n_estimators=config.tree_count, objective="reg:squarederror", tree_method="hist", random_state=config.seed, n_jobs=1, verbosity=0)


def _candidate_parameters(name, config, prepared_folds):
    if name == "pls":
        limit = min(config.max_pls_components, min(min(len(fold["training"]) - 1, fold["train_values"].shape[1]) for fold in prepared_folds))
        return list(ParameterGrid({"n_components": list(range(1, limit + 1))}))
    if name == "random_forest":
        return list(ParameterGrid({"max_depth": [6, None], "min_samples_leaf": [1, 3]}))
    if name == "xgboost":
        return list(ParameterGrid({"max_depth": [2, 3], "learning_rate": [0.05, 0.1]}))
    return [{}]


def _prepare_folds(X, axis, recipe, folds):
    prepared = []
    for fold in folds:
        features = SpectralFeatures(axis, recipe).fit(X.iloc[fold.training_positions])
        train_features = features.transform(X.iloc[fold.training_positions])
        scaler = StandardScaler().fit(train_features)
        prepared.append({
            "fold": fold, "features": features, "scaler": scaler,
            "training": fold.training_positions, "validation": fold.validation_positions,
            "train_values": scaler.transform(train_features),
            "validation_values": scaler.transform(features.transform(X.iloc[fold.validation_positions])),
        })
    return prepared


def _prediction(estimator, X):
    prediction = np.asarray(estimator.predict(X), dtype=float).reshape(-1)
    if len(prediction) != len(X) or not np.isfinite(prediction).all():
        raise ValueError("Model predictions must be finite and have one value per row.")
    return prediction


def _importance(model):
    values = np.asarray(model.feature_importances_ if hasattr(model, "feature_importances_") else np.abs(model.coef_).reshape(-1), dtype=float)
    total = np.abs(values).sum()
    return np.abs(values) / total if total > 0 else np.zeros_like(values)


def _robustness(models, prepared, X, y, seed):
    """Prespecified spectral perturbations on OOF validation inputs, never test rows."""
    spectral_columns = [column for column in X if column.startswith("raman_")]
    records = []
    for number, (name, noise, offset) in enumerate((
        ("Clean OOF inputs", 0.0, 0.0), ("Spectral noise SD 0.01 a.u.", 0.01, 0.0),
        ("Spectral noise SD 0.03 a.u.", 0.03, 0.0), ("Baseline offset +0.05 a.u.", 0.0, 0.05),
    )):
        perturbed = X.copy()
        if noise or offset:
            perturbed[spectral_columns] += np.random.default_rng(seed + number).normal(0, noise, (len(X), len(spectral_columns))) + offset
        prediction = np.full(len(X), np.nan)
        for estimator, fold in zip(models, prepared):
            positions = fold["validation"]
            prediction[positions] = _prediction(estimator, perturbed.iloc[positions])
        records.append({"protocol": name, **regression_metrics(y, prediction)})
    return pd.DataFrame(records)


def train_case(X, y, axis, case, config, folds, prepared=None):
    """Search model-only parameters with fixed, fold-local fitted features/scalers.

    Reusing fold features across candidate estimators is safe because feature
    recipe parameters are fixed and each reference/scaler sees only that fold's
    training rows. Final refit uses all training batches, never held-out batches.
    """
    prepared = _prepare_folds(X, axis, case.feature_set, folds) if prepared is None else prepared
    tuning, best_score, best = [], np.inf, None
    reference = np.asarray(y, dtype=float)
    for parameters in _candidate_parameters(case.model, config, prepared):
        predictions = np.full(len(X), np.nan)
        estimators, fold_scores = [], []
        try:
            for entry in prepared:
                positions, validation = entry["training"], entry["validation"]
                model = clone(_base_model(case.model, config)).set_params(**parameters)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(entry["train_values"], reference[positions])
                predictions[validation] = _prediction(model, entry["validation_values"])
                estimators.append(Pipeline([("features", entry["features"]), ("scale", entry["scaler"]), ("model", model)]))
                fold_scores.append({"fold": entry["fold"].number, **regression_metrics(reference[validation], predictions[validation])})
            metrics = regression_metrics(reference, predictions)
            tuning.append({"parameters": json.dumps(parameters, sort_keys=True), "RMSECV": metrics["RMSE"], "status": "ok", "error": ""})
            if metrics["RMSE"] < best_score:
                best_score = metrics["RMSE"]
                best = parameters, metrics, predictions, estimators, fold_scores
        except (ValueError, FloatingPointError, ConvergenceWarning) as error:
            tuning.append({"parameters": json.dumps(parameters, sort_keys=True), "RMSECV": np.nan, "status": "failed", "error": str(error)})
    if best is None:
        raise ValueError("Every candidate failed; review input variation, spectral recipe, and component limits.")
    parameters, metrics, predictions, estimators, fold_scores = best
    names = prepared[0]["features"].get_feature_names_out()
    importances = np.vstack([_importance(estimator.named_steps["model"]) for estimator in estimators])
    stability = pd.DataFrame({"feature": names, "mean_normalized_importance": importances.mean(axis=0), "sd_normalized_importance": importances.std(axis=0, ddof=1)})
    final = Pipeline([
        ("features", SpectralFeatures(axis, case.feature_set)),
        ("scale", StandardScaler()),
        ("model", _base_model(case.model, config).set_params(**parameters)),
    ]).fit(X, reference)
    fit_prediction_policy(final, X, reference, config.declared_range, config.decision_range)
    diagnostics = []
    for estimator, entry in zip(estimators, prepared):
        training, validation = entry["training"], entry["validation"]
        fit_prediction_policy(estimator, X.iloc[training], reference[training], config.declared_range, config.decision_range)
        diagnostics.append(assess_prediction_reliability(estimator, X.iloc[validation], predictions[validation]))
    oof_reliability = pd.concat(diagnostics).loc[X.index]
    return CaseResult(case, final, parameters, metrics, regression_metrics(reference, _prediction(final, X)), predictions,
        pd.DataFrame(tuning), pd.DataFrame(fold_scores), stability,
        _robustness(estimators, prepared, X, reference, config.seed), oof_reliability)


def train_study(dataset: ModelingDataset, partition: BatchPartition, cases=DEFAULT_CASES,
                config=TrainingConfig(), progress=None):
    """Only training rows/labels enter CV, ranking, fit, and noise diagnostics."""
    cases = tuple(cases)
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Choose at least one experiment, with no duplicate feature/model cases.")
    training, _ = partition.masks(dataset.metadata)
    X, y = dataset.X.loc[training].copy(), dataset.y.loc[training].copy()
    metadata = dataset.metadata.loc[training].copy()
    if not np.isfinite(y).all() or y.nunique() < 2:
        raise ValueError("Training references must be finite and have nonzero variation.")
    declared = config.declared_range or (float(y.min()), float(y.max()))
    if declared[0] < y.min() or declared[1] > y.max():
        raise ValueError("Declared concentration range must stay within the training-reference extent.")
    if config.decision_range and not (declared[0] <= config.decision_range[0] < config.decision_range[1] <= declared[1]):
        raise ValueError("Decision range must be inside the declared concentration range.")
    folds = batch_folds(metadata["batch_id"], config.n_splits)
    payload = {"protocol": PROTOCOL_VERSION, "reliability_protocol": RELIABILITY_PROTOCOL, "cohort_id": dataset.cohort_id, "target": dataset.target,
        "partition": asdict(partition), "config": asdict(config), "cases": [asdict(case) for case in cases]}
    study_id = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    results, failures, comparison, prepared = {}, [], [], {}
    # Bound BLAS and tree threads for predictable local dashboard responsiveness.
    with threadpool_limits(limits=1):
        for index, case in enumerate(cases):
            if progress is not None:
                progress(index, len(cases), case.case_id)
            try:
                if case.feature_set not in prepared:
                    prepared[case.feature_set] = _prepare_folds(X, dataset.axis, case.feature_set, folds)
                result = train_case(X, y, dataset.axis, case, config, folds, prepared[case.feature_set])
                results[case.case_id] = result
                comparison.append({"case_id": case.case_id, "feature_set": case.feature_set, "model": case.model,
                    "RMSECV": result.cv_metrics["RMSE"], "MAE_CV": result.cv_metrics["MAE"], "R2_CV": result.cv_metrics["R2"], "Bias_CV": result.cv_metrics["Bias"],
                    "RMSE_training": result.training_metrics["RMSE"], "features": len(result.feature_stability), "parameters": json.dumps(result.best_params, sort_keys=True)})
            except (ValueError, FloatingPointError, ConvergenceWarning) as error:
                failures.append({"case_id": case.case_id, "error": str(error)})
    if not results:
        raise ValueError("No experiment could be trained successfully. Review the modeling inputs.")
    table = pd.DataFrame(comparison).sort_values(["RMSECV", "case_id"]).reset_index(drop=True)
    return TrainingStudy(study_id, dataset.cohort_id, dataset.target, partition, config, cases, results,
        pd.DataFrame(failures, columns=["case_id", "error"]), table, metadata, y, fold_manifest(folds, X.index),
        table.iloc[0]["case_id"], tuple(dataset.X.columns), dataset.axis.copy())


def evaluate_holdout(study: TrainingStudy, dataset: ModelingDataset):
    """Score the complete frozen experiment set, without tuning or winner changes.

    The app calls this through the durable one-time evaluation gate. Keeping the
    numerical function pure lets tests verify predictions and metrics directly.
    """
    if dataset.cohort_id != study.cohort_id or dataset.target != study.target or tuple(dataset.X.columns) != study.input_columns:
        raise ValueError("Holdout data must match the study's cohort, target, and input schema.")
    _, held_out = study.partition.masks(dataset.metadata)
    X, reference = dataset.X.loc[held_out], dataset.y.loc[held_out]
    metrics, predictions, batches = [], [], []
    with threadpool_limits(limits=1):
        for case_id, result in study.results.items():
            prediction = _prediction(result.estimator, X)
            values = regression_metrics(reference, prediction)
            metrics.append({"case_id": case_id, "RMSEP": values["RMSE"], "MAE": values["MAE"], "R2": values["R2"], "Bias": values["Bias"]})
            rows = dataset.metadata.loc[held_out].copy()
            rows["reference_g_l"], rows["prediction_g_l"] = reference.to_numpy(), prediction
            rows["residual_g_l"] = prediction - reference.to_numpy()
            rows["case_id"] = case_id
            diagnostics = assess_prediction_reliability(result.estimator, X, prediction)
            for column in diagnostics:
                rows[column] = diagnostics[column]
            predictions.append(rows)
            for batch_id, group in rows.groupby("batch_id", sort=True):
                scores = regression_metrics(group["reference_g_l"], group["prediction_g_l"])
                batches.append({"case_id": case_id, "batch_id": batch_id, "rows": len(group), "RMSEP": scores["RMSE"], "MAE": scores["MAE"], "R2": scores["R2"], "Bias": scores["Bias"]})
    return {"study_id": study.study_id, "winner_id": study.winner_id, "metrics": pd.DataFrame(metrics), "predictions": pd.concat(predictions), "batch_metrics": pd.DataFrame(batches)}


def pls_diagnostics(result: CaseResult, metadata):
    if result.case.model != "pls":
        raise ValueError("Latent-variable diagnostics require a PLS experiment.")
    model = result.estimator.named_steps["model"]
    names = result.estimator.named_steps["features"].get_feature_names_out()
    scores = metadata.copy()
    for index in range(model.x_scores_.shape[1]):
        scores[f"LV{index + 1}"] = model.x_scores_[:, index]
    loadings = pd.DataFrame(model.x_loadings_, index=pd.Index(names, name="feature"), columns=[f"LV{index + 1}" for index in range(model.x_loadings_.shape[1])])
    coefficients = pd.DataFrame({"feature": names, "coefficient_per_standardized_feature": np.asarray(model.coef_).reshape(-1)})
    return scores, loadings, coefficients
