import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Handle very short contexts by returning naive (last value) forecast
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate recent trend (slope) from the last 4 points
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for damping and clipping. Use recent 13 points or all context if shorter.
    scale_context = context[-13:] if n >= 13 else context
    scale = float(np.std(scale_context))

    # Damp the slope significantly and clip it to a small fraction of the recent scale.
    # If scale is 0 (constant series), slope will be clipped to 0, which is desired.
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Damping factor for the trend contribution over the forecast horizon
    phi = 0.6
    # Steps for the forecast horizon
    steps = np.arange(1, prediction_length + 1)

    # Calculate forecasts with damped trend
    # The cumulative sum of phi**steps ensures the trend effect diminishes over time
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping based on recent historical min/max
    clip_context = context[-13:] if n >= 13 else context
    lo = float(np.min(clip_context))
    hi = float(np.max(clip_context))
    span = hi - lo

    # Clip forecasts within a band around recent observations.
    # If span is 0 (constant history), lo - 0.25 * span becomes lo, and hi + 0.25 * span becomes hi.
    # So, forecasts are clipped to be exactly lo (which equals hi and last) for constant series.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Ensure no NaN or inf values are returned.
    # Fallback to last_value for NaN, and historical hi/lo for inf.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)