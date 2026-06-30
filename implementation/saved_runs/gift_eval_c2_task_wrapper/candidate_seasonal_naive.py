"""Baseline candidate: Seasonal-Naive forecast.

Repeats the last full season of observations across the horizon. Picks a season period from
the frequency string (weekly -> 52). Falls back to plain naive (repeat last value) when the
context is shorter than one season or no season can be inferred.
"""

import numpy as np

_SEASON_BY_FREQ = {
    "S": 60,
    "T": 60,
    "MIN": 60,
    "H": 24,
    "D": 7,
    "W": 52,
    "M": 12,
    "Q": 4,
    "A": 1,
    "Y": 1,
}


def _infer_season(freq, metadata):
    base = str(freq).upper().split("-")[0].lstrip("0123456789") or str(freq).upper()
    for key, period in _SEASON_BY_FREQ.items():
        if base.startswith(key):
            return period
    if metadata and metadata.get("season_length", 1) > 1:
        return int(metadata["season_length"])
    return 1


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    if context.size == 0:
        return np.zeros(prediction_length, dtype=float)

    season = _infer_season(freq, metadata)
    if season > 1 and context.size >= season:
        last_season = context[-season:]
        reps = int(np.ceil(prediction_length / season))
        return np.tile(last_season, reps)[:prediction_length].astype(float)

    # Fallback: plain naive
    return np.full(prediction_length, context[-1], dtype=float)
