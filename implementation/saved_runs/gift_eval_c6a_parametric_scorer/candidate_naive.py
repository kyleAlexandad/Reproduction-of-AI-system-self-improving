"""Baseline: last-value naive (repeat the last observation)."""

import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    if context.size == 0:
        return np.zeros(prediction_length)
    return np.full(prediction_length, float(context[-1]))
