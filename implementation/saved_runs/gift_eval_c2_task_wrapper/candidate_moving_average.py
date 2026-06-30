"""Baseline candidate: Moving-average forecast.

Predicts a flat forecast equal to the mean of the last `window` observations. The window
shrinks for short series; falls back to the last value (or 0.0) when context is tiny.
"""

import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    if context.size == 0:
        return np.zeros(prediction_length, dtype=float)

    window = metadata.get("ma_window") if metadata else None
    if not window:
        window = 8
    window = int(min(window, context.size))
    if window < 1:
        window = 1

    avg = float(np.mean(context[-window:]))
    return np.full(prediction_length, avg, dtype=float)
