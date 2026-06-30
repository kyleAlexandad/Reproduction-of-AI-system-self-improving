import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Fallback to naive (last value) for very short series, as recommended
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a short-term damped trend
    # Use the last 4 points for slope calculation, making it very recent and local
    recent_for_slope = context[-4:]
    # Mean difference over the last 3 steps (from 4 points)
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Determine a scale for clipping the slope and output
    # Use std dev of last 13 points if available, otherwise of the entire context
    # This makes scale robust even for short series (n < 13 but >= 4)
    if n >= 13:
        scale = float(np.std(context[-13:]))
    else:
        scale = float(np.std(context))

    # Handle cases where scale might be zero (e.g., constant series)
    if scale == 0:
        # If the series is constant, the best prediction is the last value, and slope should be 0.
        slope = 0.0
    else:
        # Clip the slope to a small fraction of the scale to prevent aggressive extrapolation
        # This is a crucial conservative step
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply a damping factor to the trend component over the prediction horizon
    # phi = 0.6 means the trend influence quickly diminishes
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    # The cumulative sum ensures the trend effect accumulates but is damped for each future step
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping of forecasts to stay within a band around recent observations
    # Use min/max of last 13 points or entire context if shorter
    if n >= 13:
        lo = float(np.min(context[-13:]))
        hi = float(np.max(context[-13:]))
    else:
        lo = float(np.min(context))
        hi = float(np.max(context))

    span = hi - lo
    if span == 0:
        # If recent context is constant, clip to that constant value
        out = np.clip(out, lo, hi)
    else:
        # Clip within a band (e.g., 25% of the recent range)
        out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Ensure no NaN or inf values are returned.
    # Fallback to `last` for NaN, and `hi`/`lo` for inf, maintaining bounds.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)