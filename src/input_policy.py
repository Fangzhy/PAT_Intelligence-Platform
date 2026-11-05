"""Serializable development integration/quality rules reused for monitoring."""

from dataclasses import asdict
import numpy as np

from src.data_integration import IntegrationConfig, SENSOR_RANGES


INPUT_POLICY_VERSION = "saved-input-policy-v1"


def make_input_policy(config=None, sensor_ranges=None):
    config = config or IntegrationConfig()
    ranges = SENSOR_RANGES if sensor_ranges is None else sensor_ranges
    return {"version": INPUT_POLICY_VERSION, "integration": asdict(config),
            "sensor_ranges": {name: list(bounds) for name, bounds in ranges.items()},
            "missing_values": "reject; no imputation", "spectral_axis": "exact saved physical axis; no resampling",
            "statistical_flags": "batch-local modified z scores; descriptive warnings, not an exclusion gate"}


def read_input_policy(value):
    if not isinstance(value, dict) or value.get("version") != INPUT_POLICY_VERSION:
        raise ValueError("Unsupported saved integration/quality policy.")
    config = IntegrationConfig(**value["integration"])
    ranges = value["sensor_ranges"]
    # Reuse the same finite/increasing range validation as integration.
    if set(ranges) != set(SENSOR_RANGES) or any(len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1] for bounds in ranges.values()):
        raise ValueError("Saved sensor quality ranges are invalid.")
    return config, {name: tuple(bounds) for name, bounds in ranges.items()}


def shared_input_policy(models):
    policies = [bundle.metadata.get("input_policy") for bundle in models.values() if bundle is not None]
    recorded = [policy for policy in policies if policy]
    if recorded and (len(recorded) != len(policies) or any(policy != recorded[0] for policy in recorded[1:])):
        raise ValueError("Choose pipelines with matching saved integration and quality rules for one monitoring run.")
    policy = recorded[0] if recorded else make_input_policy()
    config, ranges = read_input_policy(policy)
    if config.sensor_direction != "backward":
        raise ValueError("This model saved retrospective nearest matching. Prepare a development study with backward matching before online acquisition.")
    return policy, bool(recorded)
