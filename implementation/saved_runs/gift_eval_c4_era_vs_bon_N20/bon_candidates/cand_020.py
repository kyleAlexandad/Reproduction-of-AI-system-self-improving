import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle SHORT context arrays robustly
    # If context is empty, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # For very short series (less than 4 points), fall back to simple Naive (last value).
    # This ensures robustness for `recent = context[-4:]` and `np.diff(recent)`.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a damped trend for longer series
    # Use the last 4 points to estimate a recent slope
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for clipping the slope and forecasts
    # Use the standard deviation of the last 13 points if available, otherwise use the whole context.
    scale_data = context[-13:] if n >= 13 else context
    scale = float(np.std(scale_data))

    # If the scale is very small (e.g., all recent values are identical),
    # treat it as 0 for clipping purposes, which effectively clips slope to 0.
    # No explicit `if scale < epsilon` check needed here as `np.clip` handles scale=0 correctly.

    # Hard requirement: keep corrections small. Clip the slope significantly.
    # This prevents the forecast from running away.
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply a heavily damped trend extrapolation (phi = 0.6)
    phi = 0.6
    steps = np.arange(1, prediction_length + 1) # Steps: [1, 2, ..., prediction_length]
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a band around recent observations.
    # Use min/max of the last 13 points if available, otherwise use the whole context.
    clip_data = context[-13:] if n >= 13 else context
    lo = float(np.min(clip_data))
    hi = float(np.max(clip_data))
    
    # Calculate the span of recent values. If lo == hi (e.g., all recent values are the same),
    # span will be 0, and clipping will effectively set all forecasts to `lo`.
    span = hi - lo

    # Clip the extrapolated forecasts within a conservative range around recent min/max.
    # This prevents extreme values and keeps forecasts anchored to recent history.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Hard requirement 4: Never output NaN or inf.
    # Convert any potential NaNs, positive infinities, or negative infinities
    # to safe float values (last observed, recent max, recent min respectively).
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)