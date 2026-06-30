import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    if len(context) == 0:
        return np.zeros(prediction_length)
    return np.repeat(context[-1], prediction_length)
