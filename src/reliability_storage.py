"""Read frozen study evidence; explicitly prepare a separate reliability demo.

Normal page visits never train, claim an unrevealed holdout, or modify a study.
Demo studies use the existing modeling protocol in a separate per-target root.
"""

from dataclasses import replace
from pathlib import Path

from src.data_generation import generate_dataset
from src.data_integration import integrate_sources, sources_from_dataset
from src.model_registry import _reference, load_model
from src.modeling import ModelCase, TrainingConfig, train_study
from src.modeling_data import fixed_batch_partition, modeling_dataset
from src.modeling_storage import evaluate_once, holdout_is_sealed, load_holdout, load_study, save_study


DEFAULT_RELIABILITY_ROOT = Path(__file__).resolve().parents[1] / ".pat_state" / "reliability"


def reliability_demo_config(overview_config):
    return replace(overview_config, n_batches=max(8, overview_config.n_batches), seed=overview_config.seed + 2_000_007)


def reliability_demo_cohort(overview_config, target):
    config = reliability_demo_config(overview_config)
    return modeling_dataset(integrate_sources(sources_from_dataset(generate_dataset(config))), target)


def load_reliability_evidence(cohort, root):
    """Load a matching trusted study and only an already completed test result."""
    study = load_study(root, cohort.cohort_id)
    if study is None:
        return None, None, None
    if study.target != cohort.target or tuple(cohort.X.columns) != study.input_columns:
        raise ValueError("The saved study target/input schema differs from this cohort. Select its original Phase 4 target and source.")
    evaluation = load_holdout(root, cohort.cohort_id)
    if evaluation is None:
        return study, None, None
    if evaluation["study_id"] != study.study_id or evaluation["winner_id"] != study.winner_id:
        raise ValueError("Saved evaluation and CV-selected study identities disagree.")
    folder = Path(root) / cohort.cohort_id
    reference = _reference(folder, "winner_pipeline.joblib", "winner_metadata.json", "Frozen reliability study", root)
    bundle = load_model(reference)
    if bundle.metadata["study_id"] != study.study_id or bundle.target != study.target:
        raise ValueError("Released pipeline and study identities disagree.")
    return study, evaluation, bundle


def prepare_reliability_demo(overview_config, target, root=DEFAULT_RELIABILITY_ROOT):
    """Explicit one-time CV selection/test, isolated from Phase 4 and Phase 5.

    Matching completed records are reused. A failed holdout claim stays sealed
    under the existing gate; it is never silently reopened or retrained.
    """
    cohort = reliability_demo_cohort(overview_config, target)
    target_root = Path(root) / target.lower()
    study, evaluation, bundle = load_reliability_evidence(cohort, target_root)
    if study is None:
        if holdout_is_sealed(target_root, cohort.cohort_id):
            raise ValueError("This sealed demo cohort is missing its original study. Inspect the local audit records; automatic retraining is disabled.")
        study = train_study(cohort, fixed_batch_partition(cohort.metadata.batch_id),
                            cases=[ModelCase("raw", "pls")], config=TrainingConfig(max_pls_components=4))
        save_study(study, target_root)
    if evaluation is None:
        evaluate_once(study, cohort, target_root)
    study, evaluation, bundle = load_reliability_evidence(cohort, target_root)
    return cohort, study, evaluation, bundle
