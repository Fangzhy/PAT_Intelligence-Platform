"""Paced, fresh-timestamp test producer for the existing acquisition contract."""

from bisect import bisect_right
import json
from time import monotonic

import numpy as np
import pandas as pd

from src.acquisition import AcquisitionError, AcquisitionPage, InMemoryTestAdapter, _integer
from src.data_integration import _utc_timestamp


class PacedTestAdapter(InMemoryTestAdapter):
    """Release a finite synthetic batch over real time, without pre-exposing it.

    One original process hour takes seconds_per_hour real seconds. Chemical
    measurements are unchanged; all valid source timestamps are compressed onto
    a fresh UTC clock. New instances have a new source identity/checkpoint.
    """

    def __init__(self, records, *, seconds_per_hour=1.0, started_at=None, clock=monotonic):
        if not np.isfinite(seconds_per_hour) or not 0.01 <= seconds_per_hour <= 60:
            raise AcquisitionError("Test pace must be between 0.01 and 60 seconds per original process hour.")
        started_at = pd.Timestamp.now(tz="UTC") if started_at is None else _utc_timestamp(started_at)
        if pd.isna(started_at):
            raise AcquisitionError("The test source start must be timezone-aware.")
        copied = json.loads(json.dumps(records, allow_nan=False))
        starts = [_utc_timestamp(event["data"].get("start_timestamp")) for event in copied if event.get("source") == "batches"]
        starts = [value for value in starts if pd.notna(value)]
        if not starts:
            raise AcquisitionError("A paced test source needs batch start metadata.")
        original_start = min(starts)
        self.release_seconds = []
        for event in copied:
            row = event["data"]
            source = event["source"]
            release_column = "start_timestamp" if source == "batches" else "result_timestamp" if source == "assays" else "measurement_timestamp"
            original_release = _utc_timestamp(row.get(release_column))
            delay = (original_release - original_start).total_seconds() / 3600 * seconds_per_hour if pd.notna(original_release) else (self.release_seconds[-1] if self.release_seconds else 0)
            self.release_seconds.append(max(0, float(delay)))
            for column in list(row):
                if column.endswith("timestamp"):
                    value = _utc_timestamp(row[column])
                    if pd.notna(value):
                        row[column] = (started_at + pd.Timedelta(seconds=(value - original_start).total_seconds() / 3600 * seconds_per_hour)).isoformat()
        if self.release_seconds != sorted(self.release_seconds):
            raise AcquisitionError("Paced test deliveries must be in availability order.")
        super().__init__(copied)
        self.clock = clock
        self.origin = float(clock())
        self.started_at = started_at

    def poll(self, cursor, limit):
        _integer(cursor, 0, len(self.records), "Cursor")
        _integer(limit, 1, 100, "Page size")
        released = bisect_right(self.release_seconds, max(0, float(self.clock()) - self.origin))
        records = json.loads(json.dumps(self.records[cursor:min(cursor + limit, released)], allow_nan=False))
        following = cursor + len(records)
        more = following < len(self.records)
        return AcquisitionPage(self.source_id, records, following, more, idle=not records and more)
