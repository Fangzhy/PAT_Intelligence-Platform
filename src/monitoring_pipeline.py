"""Received-data integration and saved-transform evidence; never train here."""

import numpy as np
import pandas as pd

from src.data_integration import integrate_sources
from src.input_policy import read_input_policy
from src.modeling_data import SENSOR_COLUMNS


def integrate_received_sources(sources, policy):
    config, ranges = read_input_policy(policy)
    if config.sensor_direction != "backward":
        raise ValueError("Online integration requires the saved backward matching policy.")
    return integrate_sources(sources, config, sensor_ranges=ranges)


def transformed_observation(session, bundle, spectrum_event_id):
    """Inspect one original inference input without predicting or refitting.

    Failed/ambiguous inputs cannot become a fresh usable row in this view.
    Feature values may be inspectable even when an applicability gate withheld
    a concentration; displaying them does not reinstate that prediction.
    """
    history = session.history()
    selected = history.loc[history.spectrum_event_id.eq(spectrum_event_id)]
    if len(selected) != 1 or selected.iloc[0].alignment_status != "aligned" or selected.iloc[0].invalidation_reason:
        raise ValueError("This observation has no valid aligned inference inputs.")
    record = selected.iloc[0]
    spectrum = session.acquisition.accepted.get(spectrum_event_id)
    sensor = session.acquisition.accepted.get(record.sensor_event_id)
    if spectrum is None or sensor is None:
        raise ValueError("An original inference input is unavailable.")
    inputs = pd.DataFrame([{f"raman_{float(key.removeprefix('raman_')):.12g}": value for key, value in spectrum[1].items() if key.startswith("raman_")} |
                           {key: sensor[1][key] for key in SENSOR_COLUMNS}])
    required = bundle.metadata["input_columns"]
    inputs = inputs[required]
    pipeline = bundle.pipeline
    features = pipeline.named_steps["features"].transform(inputs)
    scaled = pipeline.named_steps["scale"].transform(features)
    if not np.isfinite(np.asarray(features)).all() or not np.isfinite(np.asarray(scaled)).all():
        raise ValueError("Saved preprocessing did not produce finite features.")
    return pd.DataFrame({"feature": pipeline.named_steps["features"].get_feature_names_out(),
                         "preprocessed_value": np.asarray(features).reshape(-1),
                         "standardized_value": np.asarray(scaled).reshape(-1)})
