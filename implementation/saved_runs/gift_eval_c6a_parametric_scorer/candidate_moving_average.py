"""Baseline: moving average of the last `window` observations (flat forecast)."""

import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    if context.size == 0:
        return np.zeros(prediction_length)
    window = int(min(8, context.size))
    return np.full(prediction_length, float(np.mean(context[-window:])))
