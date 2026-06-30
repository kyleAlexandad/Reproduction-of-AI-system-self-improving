"""Optional conservative seed candidate for GIFT-Eval C3 (m4_weekly/W/short).

Stays VERY close to the strong last-value naive baseline, adding only a tiny, heavily-damped
trend correction (capped to a small fraction of the recent scale) and robust range clipping.
The intent is a safe starting point that should be ~naive and never diverge far above it.

NOT the default seed: the built-in naive is kept as the default because it is the documented,
exactly-reproduced anchor (MASE 2.7773) and gives the cleanest before/after comparison. Use this
seed only via:  python gift_eval_era_search.py --initial_candidate initial_candidate_conservative.py
"""

import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    last = float(context[-1])
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Tiny, capped slope from the last few points only.
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Heavily damped trend so the forecast stays anchored near the last value.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping to a band around recent observations.
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    span = hi - lo
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)
