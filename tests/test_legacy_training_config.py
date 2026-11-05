"""Read pre-range configurations and cached legacy classes without changing evidence."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import joblib
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.data_generation import SimulationConfig, generate_dataset
from src.data_integration import integrate_sources, sources_from_dataset
from src.modeling import ModelCase, TrainingConfig, train_study
from src.modeling_compat import normalize_training_config
from src.modeling_data import fixed_batch_partition, modeling_dataset
from src.modeling_storage import evaluate_once, load_holdout, load_study, save_study, study_report_zip


@dataclass(frozen=True)
class LegacyTrainingConfig:
    n_splits: int = 3
    max_pls_components: int = 2
    tree_count: int = 40
    seed: int = 42


CONFIG = SimulationConfig(n_batches=8, duration_h=72, n_wavenumbers=71, seed=351)


@pytest.fixture(scope="module")
def legacy_study():
    cohort = modeling_dataset(integrate_sources(sources_from_dataset(generate_dataset(CONFIG))))
    study = train_study(cohort, fixed_batch_partition(cohort.metadata.batch_id),
        [ModelCase("raw", "pls")], TrainingConfig(max_pls_components=2, tree_count=40))
    return study, cohort


def write_legacy_study(study, cohort, root, *, sealed):
    folder = save_study(study, root)
    if sealed:
        evaluate_once(study, cohort, root)
    legacy = deepcopy(study)
    legacy.config = LegacyTrainingConfig(**{name: getattr(study.config, name) for name in LegacyTrainingConfig.__dataclass_fields__})
    joblib.dump(legacy, folder / "training_study.joblib")
    path = folder / "study_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["training_config"] = asdict(legacy.config)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def test_normalization_preserves_old_settings_and_existing_range_choices():
    old = LegacyTrainingConfig(n_splits=4, max_pls_components=7, tree_count=160, seed=19)
    current = normalize_training_config(old)
    assert (current.n_splits, current.max_pls_components, current.tree_count, current.seed) == (4, 7, 160, 19)
    assert current.declared_range is None and current.decision_range is None
    configured = TrainingConfig(declared_range=(1, 15), decision_range=(3, 6))
    assert normalize_training_config(configured) == configured
    assert not hasattr(old, "declared_range")


def test_loading_legacy_config_and_exporting_does_not_modify_or_retest_frozen_artifacts(legacy_study, tmp_path, monkeypatch):
    study, cohort = legacy_study
    folder = write_legacy_study(study, cohort, tmp_path, sealed=True)
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    monkeypatch.setattr("src.modeling_storage.evaluate_holdout", lambda *args: pytest.fail("No repeated test evaluation"))
    loaded = load_study(tmp_path, cohort.cohort_id)
    assert loaded.study_id == study.study_id and loaded.partition == study.partition
    assert loaded.config.declared_range is None and loaded.config.decision_range is None
    assert loaded.config.max_pls_components == study.config.max_pls_components
    frozen = load_holdout(tmp_path, cohort.cohort_id)
    repeated = evaluate_once(loaded, cohort, tmp_path)
    pd.testing.assert_frame_equal(frozen["metrics"], repeated["metrics"])
    with ZipFile(BytesIO(study_report_zip(loaded, frozen))) as archive:
        assert json.loads(archive.read("study_manifest.json"))["training_config"]["declared_range"] is None
    assert before == {path.name: path.read_bytes() for path in folder.iterdir()}


@pytest.mark.parametrize("sealed", [False, True])
def test_modeling_page_reads_a_cached_legacy_class_and_requests_restart_before_mutation(legacy_study, tmp_path, monkeypatch, sealed):
    study, cohort = legacy_study
    folder = write_legacy_study(study, cohort, tmp_path, sealed=sealed)
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    import src.modeling as modeling
    monkeypatch.setattr(modeling, "TrainingConfig", LegacyTrainingConfig)
    monkeypatch.setattr(modeling, "train_study", lambda *args, **kwargs: pytest.fail("Stale runtime must not train"))
    monkeypatch.setattr("src.modeling_storage.evaluate_holdout", lambda *args: pytest.fail("Stale runtime must not evaluate"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "pages/4_Predictive_Modeling.py"), default_timeout=90)
    app.session_state["simulation_config"] = CONFIG
    app.session_state["modeling_state_root"] = str(tmp_path)
    app.run()
    assert not app.exception
    assert app.button(key="train_models").disabled
    if not sealed:
        assert app.button(key="evaluate_holdout").disabled
    assert any("older modeling code loaded" in warning.value for warning in app.warning)
    assert any("RMSECV" in frame.value for frame in app.dataframe)
    assert before == {path.name: path.read_bytes() for path in folder.iterdir()}
