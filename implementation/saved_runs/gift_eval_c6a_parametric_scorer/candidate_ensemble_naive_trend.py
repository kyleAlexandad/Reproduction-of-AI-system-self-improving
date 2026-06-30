"""Baseline: ensemble of mostly naive (0.8) + a small damped-trend correction (0.2)."""

import numpy as np


def _damped_trend(context, prediction_length):
    n = context.size
    last = float(context[-1])
    if n < 4:
        return np.full(prediction_length, last)
    slope = float(np.mean(np.diff(context[-4:])))
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    out = last + slope * np.cumsum(phi ** steps)
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    span = hi - lo
    return np.nan_to_num(np.clip(out, lo - 0.25 * span, hi + 0.25 * span),
                         nan=last, posinf=hi, neginf=lo)


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    if context.size == 0:
        return np.zeros(prediction_length)
    last = float(context[-1])
    naive = np.full(prediction_length, last)
    trend = _damped_trend(context, prediction_length)
    return (0.8 * naive + 0.2 * trend).astype(float)
