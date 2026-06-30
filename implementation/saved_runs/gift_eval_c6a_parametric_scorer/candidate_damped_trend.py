"""Baseline: last value + a small, capped, heavily-damped trend, with robust clipping."""

import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = context.size
    if n == 0:
        return np.zeros(prediction_length)
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
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)
