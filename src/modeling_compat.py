"""Read old training settings without changing saved study evidence."""

from dataclasses import fields


def normalize_training_config(config):
    """Rebuild known settings in memory; newly added optional fields use defaults.

    Import at call time so a cached storage module does not pin a class alias.
    An older running modeling module may itself have only the original fields;
    the page handles that read-only case and asks for a server restart.
    """
    from src.modeling import TrainingConfig

    values = {item.name: getattr(config, item.name, item.default) for item in fields(TrainingConfig)}
    return TrainingConfig(**values)


def supports_range_settings(config_class):
    return {"declared_range", "decision_range"}.issubset(getattr(config_class, "__dataclass_fields__", {}))
