"""Baseline: seasonal naive. Tiles the last full season; period inferred from the frequency.

For hourly data the period is 24 (daily seasonality), which is the strong baseline on
m4_hourly/H/short (C5: MASE ~1.19 vs naive ~11.6). Falls back to last-value naive when the
context is shorter than one season or no period applies. All indexing is bound-checked.
"""

import numpy as np

_SEASON_BY_FREQ = {"S": 60, "T": 60, "MIN": 60, "H": 24, "D": 7, "W": 52, "M": 12,
                   "Q": 4, "A": 1, "Y": 1}


def _season(freq, metadata):
    base = str(freq).upper().split("-")[0].lstrip("0123456789") or str(freq).upper()
    for key, period in _SEASON_BY_FREQ.items():
        if base.startswith(key):
            return period
    return 1


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = context.size
    if n == 0:
        return np.zeros(prediction_length)
    season = _season(freq, metadata)
    if season > 1 and n >= season:
        last_season = context[-season:]
        reps = int(np.ceil(prediction_length / season))
        return np.tile(last_season, reps)[:prediction_length].astype(float)
    return np.full(prediction_length, float(context[-1]))
