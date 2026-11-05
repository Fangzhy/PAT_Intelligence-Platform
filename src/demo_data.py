"""Separate reproducible development and monitoring simulation populations.

The physical simulator is unchanged. Roles, seeds and IDs are explicit metadata;
they never become model features. Monitoring is a synthetic exercise, not a new
prospective validation claim for a model selected after reviewing that exercise.
"""

from dataclasses import asdict, replace
from hashlib import sha256
import json

from src.data_generation import SimulationConfig, generate_dataset


MONITORING_SEED_OFFSET = 3_000_017
DATA_PROTOCOL = "separate-demo-populations-v1"
PROVENANCE_COLUMNS = ("data_role", "data_source_id", "generation_seed", "development_seed", "generator_protocol", "generation_config_json")


def monitoring_demo_config(development_config: SimulationConfig):
    return replace(development_config, seed=development_config.seed + MONITORING_SEED_OFFSET)


def demo_provenance(development_config, *, monitoring=False):
    config = monitoring_demo_config(development_config) if monitoring else development_config
    role = "monitoring_demo" if monitoring else "historical_development"
    identity = sha256(json.dumps({"protocol": DATA_PROTOCOL, "role": role, "config": asdict(config)}, sort_keys=True).encode()).hexdigest()
    return {"data_role": role, "data_source_id": identity, "generation_seed": config.seed,
            "development_seed": development_config.seed, "generator_protocol": DATA_PROTOCOL,
            "generation_config_json": json.dumps(asdict(config), sort_keys=True),
            "generation_config": asdict(config), "id_namespace": "MON-" if monitoring else "B"}


def _tag_dataset(dataset, provenance, *, monitoring=False):
    if monitoring:
        # Every modality, including the spectral index and delayed references,
        # must share the new identities. Never rename just a display label.
        fields = {}
        for name in ("sensors", "truth", "spectral_metadata", "offline_assays", "batches"):
            frame = getattr(dataset, name).copy(deep=True)
            for column in ("batch_id", "sample_id"):
                if column in frame:
                    frame[column] = "MON-" + frame[column].astype(str)
            fields[name] = frame
        spectra = dataset.spectra.copy(deep=True)
        spectra.index = spectra.index.map(lambda value: f"MON-{value}")
        fields["spectra"] = spectra
        dataset = replace(dataset, **fields)
    return replace(dataset, batches=dataset.batches.assign(**{key: provenance[key] for key in PROVENANCE_COLUMNS}))


def generate_development_dataset(config):
    return _tag_dataset(generate_dataset(config), demo_provenance(config))


def generate_monitoring_dataset(development_config):
    config = monitoring_demo_config(development_config)
    return _tag_dataset(generate_dataset(config), demo_provenance(development_config, monitoring=True), monitoring=True)


def source_provenance(batches):
    """Read declared source metadata; never infer an external source's seed.

    Uploaded/HTTP metadata is source-declared, not verified instrument identity.
    Missing or partial provenance stays explicitly unrecorded.
    """
    present = [key for key in PROVENANCE_COLUMNS if key in batches]
    if batches.empty or not present:
        return {"status": "not_recorded", "sources": []}
    rows = batches[present].drop_duplicates().astype(object)
    rows = rows.where(rows.notna(), None)
    return {"status": "source_declared", "sources": rows.to_dict(orient="records")}
