"""Baseline candidate: Naive forecast (repeat the last observed value)."""

import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    last = context[-1] if context.size > 0 else 0.0
    return np.full(prediction_length, last, dtype=float)
